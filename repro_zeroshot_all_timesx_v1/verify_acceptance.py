#!/usr/bin/env python3
"""S5c acceptance (R1-R7). Read-only; writes results/acceptance.json.
R1 identity coverage & chronology; R2 validity; R3 constant-history handling;
R4 cache provenance; R5 independent recompute; R6 1,677 pure-cache reproduction;
R7 frozen totals. Any failure => exit nonzero, no silent relaxation.
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

import common as C
import protocol as P


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    t0 = time.time()
    checks = {}

    inv = C.load_json(C.INV_PATH)
    windows = inv["windows"]
    preds = []
    with gzip.open(os.path.join(C.RESULTS_DIR, "preds.jsonl.gz"), "rt") as f:
        for line in f:
            preds.append(json.loads(line))

    # R1 identity coverage + chronology (from frozen inventory)
    inv_ids = [w["sample_id"] for w in windows]
    pred_ids = [r["sample_id"] for r in preds]
    checks["R1_unique_inventory_ids"] = len(set(inv_ids)) == len(inv_ids) == P.TOTAL_WINDOWS
    checks["R1_pred_set_equals_inventory"] = set(pred_ids) == set(inv_ids) and len(pred_ids) == len(inv_ids)
    checks["R1_chronology_all_past_lt_future"] = all(
        w["past_end"] < w["future_start"] for w in windows)
    sid_fi = {w["sample_id"]: w["flat_index"] for w in windows}
    checks["R1_flat_index_alignment"] = all(
        r["flat_index"] == sid_fi[r["sample_id"]] for r in preds)

    # R2 validity
    checks["R2_all_finite"] = all(
        np.isfinite(r["pred"]).all() and np.isfinite(r["target"]).all()
        and np.isfinite(r["d"]) and r["d"] > 0 for r in preds)
    checks["R2_std_defined_all"] = all(r["std_defined"] for r in preds)
    checks["R2_pred_shape_all"] = all(len(r["pred"]) == P.PRED_LEN and
                                      len(r["target"]) == P.PRED_LEN for r in preds)

    # R3 constant-history windows handled via fallback, none zero-filled/silent
    zero_ids = {z["sample_id"] for z in inv["meta"]["zero_variance_windows"]}
    zrecs = [r for r in preds if r["sample_id"] in zero_ids]
    checks["R3_zero_var_count"] = len(zrecs) == len(zero_ids) == P.ZERO_VAR_EXPECTED
    checks["R3_all_fallback_calib"] = all(r["denom_source"] == "calib_std" and r["d"] >= P.EPS_STD
                                          for r in zrecs)
    checks["R3_no_other_fallback"] = all(r["denom_source"] == "window"
                                         for r in preds if r["sample_id"] not in zero_ids)

    # R4 provenance
    final = C.load_json(os.path.join(C.HERE, "reuse_final.json"))
    reuse_vars = {vk for vk, v in final["variables"].items() if v["verdict"] == "reuse"}
    plan = C.load_json(C.REUSE_PLAN)
    mm = C.load_json(os.path.join(C.CACHE_DIR, "manifest.json"))
    cache_ok = new_ok = True
    for r in preds:
        if r["source"] == "cache":
            if (r["var_key"] not in reuse_vars
                    or r["pred_file"] != plan["variables"][r["var_key"]]["file"]
                    or r["pred_file_sha256"] != plan["variables"][r["var_key"]]["sha256"]):
                cache_ok = False
        else:
            ent = mm["entries"].get(f"{r['var_key']}__P1")
            if (ent is None or r["pred_file"] != ent["file"]
                    or r["pred_file_sha256"] != ent["sha256"]):
                new_ok = False
    checks["R4_cache_provenance"] = cache_ok
    checks["R4_new_provenance"] = new_ok

    # R5 independent recompute
    rc_path = os.path.join(C.RESULTS_DIR, "recompute_check.json")
    rc = C.load_json(rc_path) if os.path.isfile(rc_path) else {}
    checks["R5_independent_recompute"] = bool(rc.get("pass")) and rc.get(
        "inputs", {}).get("preds.jsonl.gz") == C.sha256_file(
        os.path.join(C.RESULTS_DIR, "preds.jsonl.gz"))

    # R6 subset 1,677 pure-cache reproduction
    s1677 = C.load_json(os.path.join(C.RESULTS_DIR, "subset_1677.json"))
    checks["R6_pure_cache_1677_within_5e-7"] = bool(s1677["pure_cache"]["within_5e-7"])

    # R7 totals
    n_cache = sum(1 for r in preds if r["source"] == "cache")
    n_new = sum(1 for r in preds if r["source"] == "new")
    checks["R7_totals"] = (len(preds) == P.TOTAL_WINDOWS
                           and len({w["domain"] for w in windows}) == P.N_DOMAINS
                           and len({w["var_key"] for w in windows}) == P.N_VARS)
    checks["R7_fill_equals_set_difference"] = (
        n_new == P.TOTAL_WINDOWS - final["totals"]["reuse_windows"]
        and n_cache == final["totals"]["reuse_windows"])

    passed = all(bool(v) for v in checks.values())
    out = {
        "stage": "S5c_acceptance",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checks": {k: bool(v) for k, v in checks.items()},
        "pass": passed,
        "counts": {"windows": len(preds), "cache": n_cache, "new": n_new},
        "subset_1677_final": {"std_mse": s1677["final_preds"]["std_mse"],
                              "std_mae": s1677["final_preds"]["std_mae"],
                              "reinfer_vars_touching_subset": s1677["reinfer_vars_touching_subset"]},
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(C.RESULTS_DIR, "acceptance.json"), "w") as f:
        json.dump(out, f, indent=1)
    if passed:
        C.mark_done("accept", {"pass": True,
                               "sha": {"results/acceptance.json":
                                           C.sha256_file(os.path.join(C.RESULTS_DIR,
                                                                      "acceptance.json")),
                                       "results/recompute_check.json":
                                           C.sha256_file(os.path.join(C.RESULTS_DIR,
                                                                      "recompute_check.json")),
                                       "results/preds.jsonl.gz":
                                           C.sha256_file(os.path.join(C.RESULTS_DIR,
                                                                      "preds.jsonl.gz"))},
                               "input_guards": C.stage_input_guards("accept")})
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"[acceptance] {'ALL PASS' if passed else 'FAILED'} ({out['elapsed_s']}s)")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
