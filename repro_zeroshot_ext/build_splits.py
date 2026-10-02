#!/usr/bin/env python3
"""Recompute the frozen split protocol and write splits.json.

Protocol (frozen, do not tune):
  - test  = last ceil(N/5) samples per variable, by future_start
  - val   = most recent pre-test samples whose target window does not overlap
            any test target window (future_end < min test future_start), cap 32
  - fallback to fixed P=1 if val < 8
  - calib_std = std(ddof=0) of the 96 past values of the FIRST sample
            (sorted by future_start); denominator fallback only.
            Verified to lie entirely before the first val target time.

Expected totals (must match exactly or abort without writing):
  val=4736 test=1677 cross_boundary_removed=242 fallback_vars=0
  food_wine_festivals: val=14 test=4 removed=2
"""
import argparse
import json
import math
import os
import sys

from timesx_data import EPS_STD, find_zip, iter_variables, parse_ts, pstdev

HERE = os.path.dirname(os.path.abspath(__file__))

PROTOCOL = {
    "name": "TimesX-ext-protocol",
    "sort_key": "(future_start, idx)",
    "test_rule": "last ceil(N/5) samples by future_start",
    "val_rule": "most recent pre-test samples with future_end < min(test future_start), cap 32",
    "val_cap": 32,
    "fallback_min_val": 8,
    "tie_break": "P=1",
    "calib_std_rule": ("std(ddof=0) of past_time (96 values) of the FIRST sample "
                       "sorted by future_start; denominator fallback only when "
                       "window std < 1e-8; never modifies pred/true"),
    "frozen_on": "2026-09-27",
}

EXPECTED = {
    "val": 4736, "test": 1677, "cross_boundary_removed": 242, "fallback_vars": 0,
    "food_wine_festivals": {"val": 14, "test": 4, "removed": 2},
}


def split_variable(v):
    samples = v.samples
    n = len(samples)
    n_test = (n + 4) // 5  # exact ceil(N*0.2) since 0.2 = 1/5
    test = samples[n - n_test:]
    test_min_start = test[0].future_start
    candidates = samples[:n - n_test]
    eligible = [s for s in candidates if s.future_end < test_min_start]
    removed = [s for s in candidates if s.future_end >= test_min_start]
    val = eligible[-PROTOCOL["val_cap"]:]
    calib = eligible[:len(eligible) - len(val)]
    fallback = len(val) < PROTOCOL["fallback_min_val"]

    first = samples[0]
    calib_std = pstdev([float(x) for x in first.past_val])
    val_targets = [s.future_start for s in val] or [s.future_start for s in test]
    hist_end = max(parse_ts(t) for t in first.past_ts)
    calib_before = hist_end < min(val_targets)

    def row(s):
        return {"sample_id": s.sample_id, "idx": s.idx, "date": s.date,
                "future_start": s.future_start.isoformat(),
                "future_end": s.future_end.isoformat()}

    return {
        "group": v.group, "domain": v.domain, "n": n,
        "n_test": n_test, "n_val": len(val),
        "n_cross_boundary_removed": len(removed), "fallback_p1": fallback,
        "calib_std": calib_std, "calib_std_rule": PROTOCOL["calib_std_rule"],
        "calib_source_sample_id": first.sample_id,
        "calib_hist_end": hist_end.isoformat(),
        "calib_hist_before_first_val_target": calib_before,
        "calib_remaining_ids": [s.sample_id for s in calib],
        "val": [row(s) for s in val],
        "test": [row(s) for s in test],
        "cross_boundary_removed_ids": [s.sample_id for s in removed],
    }, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "splits.json"))
    ap.add_argument("--force", action="store_true",
                    help="regenerate splits.json even when it already exists (deliberate refreeze)")
    args = ap.parse_args()
    zip_path = find_zip(args.zip)

    if os.path.isfile(args.out) and not args.force:
        raise SystemExit(f"{args.out} already exists (frozen protocol record); "
                         f"pass --force to regenerate deliberately")

    variables = {}
    tot = {"val": 0, "test": 0, "cross_boundary_removed": 0, "fallback_vars": 0}
    fwf = None
    for v in iter_variables(zip_path):
        entry, first = split_variable(v)
        variables[v.var_key] = entry
        tot["val"] += entry["n_val"]
        tot["test"] += entry["n_test"]
        tot["cross_boundary_removed"] += entry["n_cross_boundary_removed"]
        tot["fallback_vars"] += int(entry["fallback_p1"])
        if v.var.startswith("food_wine_festivals"):
            fwf = entry

    check = {
        "val": tot["val"] == EXPECTED["val"],
        "test": tot["test"] == EXPECTED["test"],
        "cross_boundary_removed": tot["cross_boundary_removed"] == EXPECTED["cross_boundary_removed"],
        "fallback_vars": tot["fallback_vars"] == EXPECTED["fallback_vars"],
    }
    if fwf is not None:
        e = EXPECTED["food_wine_festivals"]
        check["food_wine_festivals"] = (fwf["n_val"] == e["val"] and fwf["n_test"] == e["test"]
                                        and fwf["n_cross_boundary_removed"] == e["removed"])
    else:
        check["food_wine_festivals"] = False

    print(json.dumps({"totals": tot, "expected": EXPECTED, "check": check}, indent=2))

    bad_calib = [k for k, e in variables.items() if not e["calib_hist_before_first_val_target"]]
    degenerate_calib = [k for k, e in variables.items() if e["calib_std"] < EPS_STD]
    print("calib_hist_before_first_val_target violations:", bad_calib)
    print("degenerate calib_std (<1e-8):", degenerate_calib)
    if fwf is not None:
        print("food_wine_festivals:", {k: fwf[k] for k in
                                       ("n", "n_val", "n_test", "n_cross_boundary_removed", "fallback_p1")})

    if not all(check.values()):
        print("MISMATCH with frozen expected numbers — splits.json NOT written.", file=sys.stderr)
        sys.exit(1)
    if bad_calib:
        print("calib_std provenance violated — splits.json NOT written.", file=sys.stderr)
        sys.exit(1)

    doc = {
        "protocol": PROTOCOL,
        "expected_check": {"expected": EXPECTED, "totals": tot, "passed": True},
        "zip_path_used": zip_path,
        "config": {"p_candidates": {"daily": [1, 5], "weekly": [1, 52]}},
        "variables": variables,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print("frozen ->", args.out, f"({len(variables)} variables)")


if __name__ == "__main__":
    main()
