#!/usr/bin/env python3
"""S5a assemble: merge cached (reusable vars) + newly inferred predictions into
one per-window record stream. Coverage must be exactly the frozen 8,106 with no
gap and no duplicate (fill set was built by set difference; asserted here).
"""
import argparse
import gzip
import json
import os
import time

import numpy as np

import common as C
import protocol as P
import sys
if C.OLD_DIR not in sys.path:
    sys.path.insert(0, C.OLD_DIR)


def iter_preds_jsonl(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_path = os.path.join(C.RESULTS_DIR, "preds.jsonl.gz")
    if os.path.isfile(out_path) and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force")

    t0 = time.time()
    inv = C.load_json(C.INV_PATH)
    windows = inv["windows"]
    final = C.load_json(os.path.join(C.HERE, "reuse_final.json"))
    reuse_vars = {vk for vk, v in final["variables"].items() if v["verdict"] == "reuse"}
    plan = C.load_json(C.REUSE_PLAN)
    mm = C.load_json(os.path.join(C.CACHE_DIR, "manifest.json"))

    arrays = np.load(C.ARRAYS_PATH)
    past_all, target_all = arrays["past"], arrays["target"]
    d_frozen = arrays["d"]
    dsrc_frozen = [str(x) for x in arrays["denom_source"]]
    vars_meta = {v["var_key"]: v for v in C.load_json(C.VARS_PATH)}

    cache_rows = {}   # sid -> (pred row, cache_file, sha)
    for vk in sorted(reuse_vars):
        v = plan["variables"][vk]
        fsha = v["sha256"]
        fpath = os.path.join(P.OLD_CACHE, v["file"])
        # re-verify the ACTUAL file hash at merge time (not just the S2 record)
        got = C.sha256_file(fpath)
        if got != fsha:
            raise SystemExit(f"cache file {v['file']}: sha256 now {got} != recorded {fsha}")
        with np.load(fpath, allow_pickle=False) as z:
            for sid, row in zip([str(s) for s in z["sample_ids"]], z["pred"]):
                if sid in cache_rows:
                    raise SystemExit(f"cache duplicate window {sid}")
                cache_rows[sid] = (row, v["file"], fsha)

    new_rows = {}
    new_meta = {}
    for vk, ent in mm["entries"].items():
        var = vk[:-3] if vk.endswith("__P1") else vk
        fpath = os.path.join(C.CACHE_DIR, ent["file"])
        got = C.sha256_file(fpath)
        if got != ent["sha256"]:
            raise SystemExit(f"new cache file {ent['file']}: sha256 now {got} != manifest {ent['sha256']}")
        with np.load(fpath, allow_pickle=False) as z:
            for sid, row in zip([str(s) for s in z["sample_ids"]], z["pred"]):
                if sid in new_rows:
                    raise SystemExit(f"new-cache duplicate window {sid}")
                new_rows[sid] = row
                new_meta[sid] = (ent["file"], ent["sha256"])

    overlap = set(cache_rows) & set(new_rows)
    if overlap:
        raise SystemExit(f"cache/new overlap on {len(overlap)} windows e.g. {sorted(overlap)[:5]}")

    # fill set must be EXACTLY the set difference: all windows − actually cached
    all_sids = {w["sample_id"] for w in windows}
    expected_fill = all_sids - set(cache_rows)
    if set(new_rows) != expected_fill:
        missing = sorted(expected_fill - set(new_rows))[:5]
        extra = sorted(set(new_rows) - expected_fill)[:5]
        raise SystemExit(f"fill set mismatch: missing {missing} extra {extra}")

    n_cache = n_new = 0
    records = []
    with gzip.open(out_path, "wt") as f:
        for w in windows:
            sid = w["sample_id"]
            fi = w["flat_index"]
            past = past_all[fi]
            target = target_all[fi]
            ws = float(past.std(ddof=0))
            if ws != w["win_std"]:
                raise SystemExit(f"{sid}: frozen win_std mismatch {ws} != {w['win_std']}")
            if ws >= P.EPS_STD:
                d, dsrc = ws, "window"
            else:
                d = float(vars_meta[w["var_key"]]["calib_std"])
                dsrc = "calib_std"
            # cross-check against the frozen scoring denominator
            if abs(d - float(d_frozen[fi])) > P.RECOMPUTE_TOL_D * max(1e-12, abs(d)) \
                    or dsrc != dsrc_frozen[fi]:
                raise SystemExit(f"{sid}: frozen d/denom_source mismatch "
                                 f"({d}/{dsrc} vs {float(d_frozen[fi])}/{dsrc_frozen[fi]})")
            std_defined = d >= P.EPS_STD
            if sid in cache_rows:
                pred, cfile, csha = cache_rows[sid]
                src = "cache"
                n_cache += 1
            elif sid in new_rows:
                pred = new_rows[sid]
                cfile, csha = new_meta[sid]
                src = "new"
                n_new += 1
            else:
                raise SystemExit(f"window {sid} has no prediction source")
            pred = np.asarray(pred, dtype=np.float64)
            if pred.shape != (P.PRED_LEN,) or not np.isfinite(pred).all():
                raise SystemExit(f"{sid}: pred invalid")
            if not (np.isfinite(target).all() and np.isfinite(d) and d > 0):
                raise SystemExit(f"{sid}: target/d invalid")
            rec = {
                "flat_index": w["flat_index"], "var_key": w["var_key"],
                "sample_id": sid, "domain": w["domain"],
                "source": src, "pred_file": cfile, "pred_file_sha256": csha,
                "pred": [float(x) for x in pred],
                "target": [float(x) for x in target],
                "d": d, "denom_source": dsrc, "std_defined": std_defined,
                "win_std": ws,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)

    total = len(records)
    if total != P.TOTAL_WINDOWS or n_cache + n_new != total:
        raise SystemExit(f"coverage mismatch: {total} records (cache={n_cache}, new={n_new})")
    report = {
        "stage": "S5a_assemble",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "totals": {"windows": total, "cache": n_cache, "new": n_new,
                   "reuse_vars": len(reuse_vars), "reinfer_vars": P.N_VARS - len(reuse_vars)},
        "expected_fill_windows": final["totals"]["fill_windows"],
        "preds_sha256": C.sha256_file(out_path),
        "cache_all_manifest_sha256": C.sha256_file(os.path.join(C.CACHE_DIR, "manifest.json")),
        "elapsed_s": round(time.time() - t0, 1),
    }
    if report["totals"]["cache"] != final["totals"]["reuse_windows"]:
        raise SystemExit(f"cache windows {report['totals']['cache']} != "
                         f"reuse_final {final['totals']['reuse_windows']}")
    C.dump_json(report, os.path.join(C.RESULTS_DIR, "assemble_report.json"))
    C.mark_done("s5a_assemble", {**report["totals"],
                                 "sha": {"results/preds.jsonl.gz": report["preds_sha256"],
                                         "results/assemble_report.json":
                                             C.sha256_file(os.path.join(C.RESULTS_DIR,
                                                                        "assemble_report.json"))},
                                 "input_guards": C.stage_input_guards("assemble")})
    print(f"[assemble] OK: {total} windows (cache={n_cache}, new={n_new}) -> {out_path}")


if __name__ == "__main__":
    main()
