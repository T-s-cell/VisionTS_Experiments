#!/usr/bin/env python3
"""S5b evaluate: per-window metrics (metrics_util reused read-only), then
window -> variable -> domain (equal weight) -> overall (equal weight over 19).
Raw scale only at variable level. Subsets: 1,677 old-test (pure-cache
reproduction MUST match the reference within 5e-7; final-subset delta is
reported with reasons when variables were re-inferred) and 2,474 LN-test.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import common as C
import protocol as P
if C.OLD_DIR not in sys.path:
    sys.path.insert(0, C.OLD_DIR)
from metrics_util import sample_metrics  # frozen old scoring rule (read-only)

from assemble import iter_preds_jsonl


def aggregate(win_rows):
    """win_rows: dicts with var_key, domain, std_mse/std_mae/raw_mse/raw_mae (None ok).
    Returns (variable_level rows, domain_level rows, overall dict)."""
    by_var = defaultdict(list)
    for r in win_rows:
        by_var[r["var_key"]].append(r)
    var_rows = []
    for vk, rs in sorted(by_var.items()):
        def m(field):
            vals = [x[field] for x in rs if x[field] is not None]
            return sum(vals) / len(vals) if vals else None
        var_rows.append({
            "var_key": vk, "domain": rs[0]["domain"], "n": len(rs),
            "n_std_undefined": sum(1 for x in rs if not x["std_defined"]),
            "raw_mse": m("raw_mse"), "raw_mae": m("raw_mae"),
            "std_mse": m("std_mse"), "std_mae": m("std_mae"),
        })
    by_domain = defaultdict(list)
    for r in var_rows:
        by_domain[r["domain"]].append(r)
    if set(by_domain) != P.EXPECTED_DOMAINS:
        raise SystemExit(f"domain coverage mismatch: {set(by_domain) ^ P.EXPECTED_DOMAINS}")
    dom_rows = []
    for d, rs in sorted(by_domain.items()):
        mses = [r["std_mse"] for r in rs if r["std_mse"] is not None]
        maes = [r["std_mae"] for r in rs if r["std_mae"] is not None]
        dom_rows.append({
            "domain": d, "n_vars": len(rs),
            "std_mse": sum(mses) / len(mses) if mses else None,
            "std_mae": sum(maes) / len(maes) if maes else None,
            "n_std_undefined": sum(r["n_std_undefined"] for r in rs),
        })
    if len(dom_rows) != P.N_DOMAINS:
        raise SystemExit(f"domain count {len(dom_rows)} != {P.N_DOMAINS}")
    d_mses = [r["std_mse"] for r in dom_rows if r["std_mse"] is not None]
    d_maes = [r["std_mae"] for r in dom_rows if r["std_mae"] is not None]
    overall = {
        "n_domains": len(dom_rows),
        "std_mse": sum(d_mses) / len(d_mses) if d_mses else None,
        "std_mae": sum(d_maes) / len(d_maes) if d_maes else None,
    }
    return var_rows, dom_rows, overall


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    preds_path = os.path.join(C.RESULTS_DIR, "preds.jsonl.gz")
    win_rows = list(iter_preds_jsonl(preds_path))
    if len(win_rows) != P.TOTAL_WINDOWS:
        raise SystemExit(f"preds rows {len(win_rows)} != {P.TOTAL_WINDOWS}")

    for r in win_rows:
        diffs = [p - t for p, t in zip(r["pred"], r["target"])]
        sm = sample_metrics(r["pred"], r["target"], r["win_std"],
                            None if r["denom_source"] == "window" else r["d"])
        if sm["denom_source"] != r["denom_source"]:
            raise SystemExit(f"{r['sample_id']}: denom_source {sm['denom_source']} != {r['denom_source']}")
        r.update({"raw_mse": sm["raw_mse"], "raw_mae": sm["raw_mae"],
                  "std_mse": sm["std_mse"], "std_mae": sm["std_mae"],
                  "std_defined": sm["std_defined"]})

    var_rows, dom_rows, overall = aggregate(win_rows)

    # ---------------- write main tables
    var_fields = ["var_key", "domain", "n", "n_std_undefined", "raw_mse", "raw_mae", "std_mse", "std_mae"]
    write_csv(os.path.join(C.RESULTS_DIR, "variable_level.csv"), var_rows, var_fields)
    dom_fields = ["domain", "n_vars", "std_mse", "std_mae", "n_std_undefined"]
    write_csv(os.path.join(C.RESULTS_DIR, "domain_level.csv"), dom_rows, dom_fields)
    with open(os.path.join(C.RESULTS_DIR, "overall.json"), "w") as f:
        json.dump({"config": "P1_fixed_all_native", **overall}, f, indent=1)

    # ---------------- subset 1,677 (old test)
    final = C.load_json(os.path.join(C.HERE, "reuse_final.json"))
    reinfer_vars = {vk for vk, v in final["variables"].items() if v["verdict"] == "reinfer"}
    test_ids = {w["sample_id"] for w in C.load_json(C.INV_PATH)["windows"]
                if w["split_in_old_protocol"] == "test"}
    sub_final = [r for r in win_rows if r["sample_id"] in test_ids]
    _, _, ov_final = aggregate(sub_final)

    # pure-cache recompute: old npz values only, old scoring chain, old fallbacks
    import numpy as np
    plan = C.load_json(C.REUSE_PLAN)
    splits = C.load_json(P.OLD_SPLITS)
    domain_of_sid = {w["sample_id"]: w["domain"] for w in C.load_json(C.INV_PATH)["windows"]}
    pure_rows = []
    for vk, v in plan["variables"].items():
        if v["verdict"] != "reuse_candidate":
            continue
        with np.load(os.path.join(P.OLD_CACHE, v["file"]), allow_pickle=False) as z:
            sids = [str(s) for s in z["sample_ids"]]
            calib = float(splits["variables"][vk]["calib_std"])
            for sid, pred, true, wstd in zip(sids, z["pred"], z["true"], z["win_std"]):
                if sid not in test_ids:
                    continue
                sm = sample_metrics(list(pred), list(true), float(wstd), calib)
                pure_rows.append({"var_key": vk, "domain": domain_of_sid[sid],
                                  "raw_mse": sm["raw_mse"], "raw_mae": sm["raw_mae"],
                                  "std_mse": sm["std_mse"], "std_mae": sm["std_mae"],
                                  "std_defined": sm["std_defined"]})
    _, _, ov_pure = aggregate(pure_rows)
    delta = {"std_mse": abs(ov_pure["std_mse"] - P.REF_1677_STD_MSE),
             "std_mae": abs(ov_pure["std_mae"] - P.REF_1677_STD_MAE)}
    subset_1677 = {
        "n_windows": len(test_ids),
        "pure_cache": {"n_rows": len(pure_rows), **{k: ov_pure[k] for k in ("std_mse", "std_mae")},
                       "delta_vs_ref": delta,
                       "within_5e-7": bool(delta["std_mse"] <= P.REF_TOL and delta["std_mae"] <= P.REF_TOL),
                       "reference": {"std_mse": P.REF_1677_STD_MSE, "std_mae": P.REF_1677_STD_MAE}},
        "final_preds": {k: ov_final[k] for k in ("std_mse", "std_mae")},
        "reinfer_vars_touching_subset": sorted(vk for vk in reinfer_vars
                                               if any(r["var_key"] == vk for r in sub_final)),
        "note": "≤5e-7 is enforced on the PURE-cache recompute only (frozen amendment "
                "2026-10-10); the final-subset value legitimately shifts when variables "
                "were re-inferred — delta and reasons are reported.",
    }
    with open(os.path.join(C.RESULTS_DIR, "subset_1677.json"), "w") as f:
        json.dump(subset_1677, f, indent=1)

    # ---------------- subset 2,474 (LN test manifest)
    ln = C.load_json(P.LN_MANIFEST)
    ln_ids = set()
    for vk, branches in ln["variables"].items():
        node = branches.get("native", {}).get("test", [])
        for sid in node:
            ln_ids.add(sid if isinstance(sid, str) else sid["sample_id"])
    sub_ln = [r for r in win_rows if r["sample_id"] in ln_ids]
    var_ln, dom_ln, ov_ln = aggregate(sub_ln)
    n_cache_ln = sum(1 for r in sub_ln if r["source"] == "cache")
    subset_2474 = {
        "n_windows_manifest": len(ln_ids),
        "n_windows_found": len(sub_ln),
        "n_cache": n_cache_ln, "n_new": len(sub_ln) - n_cache_ln,
        "overall": {"std_mse": ov_ln["std_mse"], "std_mae": ov_ln["std_mae"]},
        "manifest": P.LN_MANIFEST,
        "note": "informational: fixed P=1 on the LN fine-tune test split, same scoring rules",
    }
    with open(os.path.join(C.RESULTS_DIR, "subset_2474.json"), "w") as f:
        json.dump(subset_2474, f, indent=1)
    write_csv(os.path.join(C.RESULTS_DIR, "subset_2474_variable_level.csv"), var_ln, var_fields)

    C.mark_done("s5b_evaluate", {"overall": overall,
                                 "subset_1677_pure_within_tol": subset_1677["pure_cache"]["within_5e-7"],
                                 "sha": {"results/preds.jsonl.gz":
                                             C.sha256_file(os.path.join(C.RESULTS_DIR,
                                                                        "preds.jsonl.gz")),
                                         **{f"results/{n}": C.sha256_file(os.path.join(C.RESULTS_DIR, n))
                                            for n in ("variable_level.csv", "domain_level.csv",
                                                      "overall.json", "subset_1677.json",
                                                      "subset_2474.json")}},
                                 "input_guards": C.stage_input_guards("evaluate")})
    print(f"[evaluate] overall std-MSE={overall['std_mse']:.6f} std-MAE={overall['std_mae']:.6f}")
    print(f"[evaluate] 1,677 pure-cache: {ov_pure['std_mse']:.6f}/{ov_pure['std_mae']:.6f} "
          f"(delta {delta['std_mse']:.2e}/{delta['std_mae']:.2e}, within_5e-7="
          f"{subset_1677['pure_cache']['within_5e-7']})")
    print(f"[evaluate] 1,677 final: {ov_final['std_mse']:.6f}/{ov_final['std_mae']:.6f} | "
          f"2,474 final: {ov_ln['std_mse']:.6f}/{ov_ln['std_mae']:.6f} "
          f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
