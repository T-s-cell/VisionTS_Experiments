#!/usr/bin/env python3
"""S3 preflight: re-forward a frozen sample of cached windows and compare with
the old cache under the two frozen criteria:
  c1: max_i|new-old| / max(1e-8, max_i|old|) <= 1e-5
  c2: max_i|new-old| / d      <= 1e-4   (d = formal scoring denominator incl. fallback)
Both must pass; predictions finite. A failing window re-inferences its WHOLE
variable. Thresholds are never loosened based on results.

Modes:
  --sample-only  freeze the sampling list to preflight_sample.json (no GPU)
  --run          execute the comparison -> preflight_report.json + reuse_final.json
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

import common as C
import protocol as P
from infer_fill import load_model, forward_rows

SAMPLE_PATH = os.path.join(C.HERE, "preflight_sample.json")
REPORT_PATH = os.path.join(C.HERE, "preflight_report.json")
FINAL_PATH = os.path.join(C.HERE, "reuse_final.json")


def windows_hash(windows):
    return hashlib.sha256(
        json.dumps(windows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_sample():
    inv = C.load_json(C.INV_PATH)
    zero_ids = {z["sample_id"] for z in inv["meta"]["zero_variance_windows"]}
    by_domain = {}
    for w in inv["windows"]:
        by_domain.setdefault(w["domain"], {}).setdefault(w["var_key"], []).append(w)

    sample_windows = []
    per_domain = {}
    for domain in sorted(by_domain):
        vars_sorted = sorted(by_domain[domain])
        rng = random.Random(f"{P.PROTOCOL_VERSION}|{domain}")
        chosen = rng.sample(vars_sorted, min(P.PF_VARS_PER_DOMAIN, len(vars_sorted)))
        per_domain[domain] = sorted(chosen)
        for vk in sorted(chosen):
            # compare mode needs old-cache coverage => sample only test/val windows
            rows = sorted((w for w in by_domain[domain][vk]
                           if w["split_in_old_protocol"] in ("test", "val")),
                          key=lambda w: w["flat_index"])
            n = len(rows)
            idxs = sorted(set(int(round(i)) for i in np.linspace(0, n - 1, P.PF_WINDOWS_PER_VAR)))
            for i in idxs:
                w = rows[i]
                sample_windows.append({"flat_index": w["flat_index"], "sample_id": w["sample_id"],
                                       "var_key": vk, "domain": domain, "mode": "compare"})
    # all constant-history windows; comparable only when the old cache covers them
    for w in inv["windows"]:
        if w["sample_id"] in zero_ids:
            mode = "compare" if w["split_in_old_protocol"] in ("test", "val") else "new_only"
            sample_windows.append({"flat_index": w["flat_index"], "sample_id": w["sample_id"],
                                   "var_key": w["var_key"], "domain": w["domain"], "mode": mode})
    sample_windows.sort(key=lambda r: r["flat_index"])
    # dedupe (a sampled var may already include a zero-var window)
    seen, dedup = set(), []
    for r in sample_windows:
        key = (r["sample_id"], r["mode"])
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        dedup.append(r)
    sample = {
        "protocol_version": P.PROTOCOL_VERSION,
        "rule": (f"per domain (sorted) {P.PF_VARS_PER_DOMAIN} vars via random.Random("
                 f"'<protocol>|<domain>') from sorted var list; per var {P.PF_WINDOWS_PER_VAR} "
                 "evenly spaced windows incl. first/last; plus ALL zero-variance windows "
                 "(compare when old-cached, else new_only sanity)"),
        "inventory_sha256": C.sha256_file(C.INV_PATH),
        "n_compare": sum(1 for r in dedup if r["mode"] == "compare"),
        "n_new_only": sum(1 for r in dedup if r["mode"] == "new_only"),
        "domains": per_domain,
        "windows": dedup,
    }
    sample["windows_sha256"] = windows_hash(sample["windows"])
    C.dump_json(sample, SAMPLE_PATH)
    print(f"[preflight] sample frozen: {len(dedup)} windows "
          f"(compare={sample['n_compare']}, new_only={sample['n_new_only']}) -> {SAMPLE_PATH}")


def load_sample():
    s = C.load_json(SAMPLE_PATH)
    if s["windows_sha256"] != windows_hash(s["windows"]):
        raise SystemExit("preflight sample hash mismatch — sampling list was modified")
    if s["inventory_sha256"] != C.sha256_file(C.INV_PATH):
        raise SystemExit("preflight sample was frozen against a different inventory")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="execute the preflight comparison (GPU host)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=P.RUN_CONFIG["batch_size"])
    args = ap.parse_args()

    if args.sample_only == args.run:
        ap.error("exactly one of --sample-only / --run is required")

    if args.sample_only:
        build_sample()
        C.mark_done("s3_sample_frozen",
                    {"sample_sha256": C.sha256_file(SAMPLE_PATH),
                     "sha": {"preflight_sample.json": C.sha256_file(SAMPLE_PATH)},
                     "input_guards": C.stage_input_guards("sample")})
        return

    t0 = time.time()
    sample = load_sample()
    # bind this preflight verdict to the exact data/model/code/config/env state;
    # infer_fill re-derives the same fingerprint and refuses mismatches (S3->S4)
    fp = C.forward_fingerprint()
    plan = C.load_json(C.REUSE_PLAN)
    reuse_cand = {vk for vk, v in plan["variables"].items() if v["verdict"] == "reuse_candidate"}
    if reuse_cand != set(plan["variables"]):
        print(f"[preflight] note: {len(plan['variables']) - len(reuse_cand)} vars were already "
              "non-candidate at S2")

    inv = C.load_json(C.INV_PATH)
    win_by_sid = {w["sample_id"]: w for w in inv["windows"]}
    vars_meta = {v["var_key"]: v for v in C.load_json(C.VARS_PATH)}

    # load old cache rows for compared windows
    old_pred_by_sid, old_file = {}, {}
    for vk, v in plan["variables"].items():
        if v["verdict"] != "reuse_candidate":
            continue
        with np.load(os.path.join(P.OLD_CACHE, v["file"]), allow_pickle=False) as z:
            sids = [str(s) for s in z["sample_ids"]]
            for sid, row in zip(sids, z["pred"]):
                old_pred_by_sid[sid] = row
                old_file[sid] = v["file"]

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda unavailable, falling back to cpu", file=sys.stderr)
        args.device = "cpu"
    model = load_model(args.device)
    arrays = np.load(C.ARRAYS_PATH)
    past_all = arrays["past"]

    def formal_d(w):
        if w["win_std"] >= P.EPS_STD:
            return w["win_std"], "window"
        return float(vars_meta[w["var_key"]]["calib_std"]), "calib_std"

    order = sorted(sample["windows"], key=lambda r: r["flat_index"])
    # forward in batches of the same variable to mimic fill batching
    preds = {}
    todo = [r for r in order if r["mode"] in ("compare", "new_only")]
    sids = [r["sample_id"] for r in todo]
    rows = [past_all[win_by_sid[s]["flat_index"]] for s in sids]
    all_preds = forward_rows(model, rows, args.device, args.batch_size)
    for s, p in zip(sids, all_preds):
        preds[s] = p

    per_var = {}
    for r in todo:
        sid = r["sample_id"]
        w = win_by_sid[sid]
        new = preds[sid]
        rec = {"flat_index": r["flat_index"], "sample_id": sid, "mode": r["mode"]}
        if not np.isfinite(new).all():
            rec.update(passed=False, reason="new pred non-finite")
        elif r["mode"] == "new_only":
            rec.update(passed=True, reason="new_only sanity (finite)")
        else:
            old = old_pred_by_sid[sid]
            diff = np.abs(new - old)
            c1 = float(diff.max() / max(1e-8, float(np.abs(old).max())))
            d, dsrc = formal_d(w)
            c2 = float(diff.max() / d)
            ok = (c1 <= P.TOL_PRED_REL) and (c2 <= P.TOL_PRED_OVER_D) and np.isfinite(old).all()
            rec.update(passed=bool(ok), c1=c1, c2=c2, d=d, d_source=dsrc,
                       old_file=old_file[sid],
                       reason=None if ok else f"c1={c1:.3e} c2={c2:.3e}")
        per_var.setdefault(r["var_key"], []).append(rec)

    variables, fail_vars = {}, {}
    for vk, recs in sorted(per_var.items()):
        bad = [x for x in recs if not x["passed"]]
        if bad:
            fail_vars[vk] = bad
            variables[vk] = {"verdict": "reinfer", "n_checked": len(recs),
                             "failed_windows": [b["sample_id"] for b in bad],
                             "detail": bad}
        else:
            variables[vk] = {"verdict": "reuse", "n_checked": len(recs), "detail": recs}

    # final verdicts: S2 candidates that passed preflight stay "reuse";
    # S2 pending or preflight-failed => reinfer; vars never sampled stay per S2
    final_vars = {}
    for vk, v in plan["variables"].items():
        if v["verdict"] != "reuse_candidate":
            final_vars[vk] = {"verdict": "reinfer", "reason": "S2 verification failed"}
        elif vk in variables:
            final_vars[vk] = {"verdict": variables[vk]["verdict"],
                              "reason": None if variables[vk]["verdict"] == "reuse"
                              else "preflight dual-criteria failure",
                              "n_checked": variables[vk]["n_checked"]}
        else:
            final_vars[vk] = {"verdict": "reuse", "reason": "not sampled (S2 verified)"}
    # reinfer any var sharing a variable-level anomaly class beyond its sample?
    # no — scope is exactly the sampled variable, per frozen rule.
    reuse_n = sum(1 for v in final_vars.values() if v["verdict"] == "reuse")
    reuse_windows = sum(plan["variables"][vk]["n"] for vk, v in final_vars.items()
                        if v["verdict"] == "reuse")

    checked = [x for recs in per_var.values() for x in recs]
    c1s = [x["c1"] for x in checked if "c1" in x]
    c2s = [x["c2"] for x in checked if "c2" in x]
    report = {
        "stage": "S3_preflight",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": args.device,
        "sample_sha256": C.sha256_file(SAMPLE_PATH),
        "forward_binding": fp,
        "criteria": {"c1_max_rel": P.TOL_PRED_REL, "c2_max_over_d": P.TOL_PRED_OVER_D},
        "stats": {
            "n_checked": len(checked),
            "n_compare": sum(1 for x in checked if x["mode"] == "compare"),
            "n_new_only": sum(1 for x in checked if x["mode"] == "new_only"),
            "n_failed": len(fail_vars),
            "c1_max": max(c1s) if c1s else None,
            "c2_max": max(c2s) if c2s else None,
        },
        "per_var_detail": variables,
        "elapsed_s": round(time.time() - t0, 1),
    }
    final = {
        "stage": "S3_preflight_final",
        "created": report["created"],
        "forward_binding": fp,
        "variables": final_vars,
        "totals": {
            "reuse_vars": reuse_n, "reinfer_vars": P.N_VARS - reuse_n,
            "reuse_windows": reuse_windows,
            "fill_windows": P.TOTAL_WINDOWS - reuse_windows,
        },
    }
    C.dump_json(report, REPORT_PATH)
    C.dump_json(final, FINAL_PATH)
    C.mark_done("s3_preflight", {"reuse_vars": reuse_n, "reinfer_vars": P.N_VARS - reuse_n,
                                 "fill_windows": final["totals"]["fill_windows"],
                                 "sha": {"preflight_report.json": C.sha256_file(REPORT_PATH),
                                         "reuse_final.json": C.sha256_file(FINAL_PATH)},
                                 "input_guards": C.stage_input_guards("preflight")})
    print(f"[preflight] checked={report['stats']['n_checked']} "
          f"(compare={report['stats']['n_compare']}) failed_vars={len(fail_vars)} "
          f"c1_max={report['stats']['c1_max']:.3e} c2_max={report['stats']['c2_max']:.3e}")
    print(f"[preflight] final: reuse_vars={reuse_n} reinfer_vars={P.N_VARS - reuse_n} "
          f"reuse_windows={reuse_windows} fill_windows={final['totals']['fill_windows']}")
    if fail_vars:
        print("failed variables (whole var re-inferred):")
        for vk in sorted(fail_vars):
            print("  -", vk)


if __name__ == "__main__":
    main()
