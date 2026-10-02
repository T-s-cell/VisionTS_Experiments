#!/usr/bin/env python3
"""Stage: aggregate test metrics  sample -> variable -> domain -> overall.

Two reporting configs, both covering all 19 domains:
  - P1_fixed : every variable scored under P=1
  - selected : every variable scored under its validation-selected P
- domain value = equal-weight mean over its variables (standardized scale only)
- overall value = equal-weight mean over the 19 domain values (NOT pooled vars)
- raw scale is reported only at variable level
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

EXPECTED_DOMAINS = {
    "CropsAndStaples", "EnergyAndFuels", "LivestockAndFoodProducts",
    "RawMaterialsAndConstruction", "SpecialtyAndAdvancedMaterials",
    "StrategicAndHighValueMaterials", "Currency",
    "arts", "climate", "economy", "electronic_technology", "finance", "pets",
    "public_health", "public_policy", "science", "shopping", "society", "traffic",
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def domain_and_overall(var_rows, label):
    per_domain = defaultdict(list)
    for r in var_rows:
        per_domain[r["domain"]].append(r)
    assert set(per_domain) == EXPECTED_DOMAINS, \
        f"{label}: domain coverage mismatch: {set(per_domain) ^ EXPECTED_DOMAINS}"
    dt = {}
    for d, drs in per_domain.items():
        dt[d] = {"n_vars": len(drs), "std_mse": mean([r["std_mse"] for r in drs]),
                 "std_mae": mean([r["std_mae"] for r in drs]),
                 "n_std_undefined": sum(r["n_std_undefined"] for r in drs)}
    ov = {"config": label, "n_domains": len(dt),
          "std_mse": mean([v["std_mse"] for v in dt.values()]),
          "std_mae": mean([v["std_mae"] for v in dt.values()])}
    return dt, ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=os.path.join(HERE, "P_selection.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    with open(os.path.join(args.results, "test_variable_level.csv")) as f:
        rows = list(csv.DictReader(f))
    sel = json.load(open(args.selection))

    for r in rows:
        r["P"] = int(r["P"])
        r["is_selected"] = int(r["is_selected"])
        r["n_test"] = int(r["n_test"])
        r["n_std_undefined"] = int(r["n_std_undefined"])
        r["std_mse"] = float(r["std_mse"]) if r["std_mse"] not in ("", "None") else None
        r["std_mae"] = float(r["std_mae"]) if r["std_mae"] not in ("", "None") else None
        r["raw_mse"] = float(r["raw_mse"])
        r["raw_mae"] = float(r["raw_mae"])

    # every variable exactly one row per reporting config
    p1_rows = [r for r in rows if r["P"] == 1]
    sel_rows = [r for r in rows if r["is_selected"] == 1]
    assert len({r["var_key"] for r in p1_rows}) == len(p1_rows) == 190, "P=1 rows not 1 per var"
    assert len({r["var_key"] for r in sel_rows}) == len(sel_rows) == 190, "selected rows not 1 per var"

    tables, overalls = {}, {}
    tables["P1_fixed"], overalls["P1_fixed"] = domain_and_overall(p1_rows, "P1_fixed")
    tables["selected"], overalls["selected"] = domain_and_overall(sel_rows, "selected")

    with open(os.path.join(args.results, "domain_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "domain", "n_vars", "std_mse", "std_mae", "n_std_undefined"])
        for cfg, dt in tables.items():
            for d in sorted(dt):
                v = dt[d]
                w.writerow([cfg, d, v["n_vars"], v["std_mse"], v["std_mae"], v["n_std_undefined"]])
    with open(os.path.join(args.results, "overall.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "n_domains", "std_mse", "std_mae"])
        for cfg in ("P1_fixed", "selected"):
            v = overalls[cfg]
            w.writerow([cfg, v["n_domains"], v["std_mse"], v["std_mae"]])

    stats = sel["stats"]
    fp = sel["fingerprint"]
    lines = [
        "# VisionTS zero-shot on TimesX — stage-1 results",
        "",
        f"Protocol: **{sel['protocol']['name']}** — per-window z-score standardization "
        "(denominator falls back to the frozen calib_std when the window std degenerates); "
        "chronological split: test = last ceil(N/5), val = pre-test non-overlapping capped 32; "
        "MSE/MAE only. Not comparable with tables built under other protocols.",
        "",
        f"- variables: {stats['n_vars']} | selected-P counts: {stats['by_selected_P']} | "
        f"fallback: {stats['fallback_vars']} | exact ties: {stats['exact_ties']}",
        f"- selection reasons: {stats['by_reason']}",
        f"- fingerprints: splits={fp['splits_sha256'][:12]}… zip={fp['zip_sha256'][:12]}… "
        f"ckpt={fp['ckpt_sha256'][:12]}… | r(norm_const)=c(align_const)=0.4, ctx=96→12, "
        f"arch={fp['run_config']['arch']}, fp32",
        "",
        "## Overall (standardized scale; equal weight over 19 domains)",
        "",
        "| config | MSE | MAE |",
        "|---|-----|-----|",
    ]
    for cfg in ("P1_fixed", "selected"):
        v = overalls[cfg]
        lines.append(f"| {cfg} | {v['std_mse']:.4f} | {v['std_mae']:.4f} |")

    lines += ["", "## Per domain (standardized scale, variables equal-weight)", "",
              "| Domain | P=1 MSE | P=1 MAE | sel MSE | sel MAE | selected-P counts | vars |",
              "|---|---|---|---|---|---|---|"]
    for d in sorted(EXPECTED_DOMAINS):
        r1, rs = tables["P1_fixed"][d], tables["selected"][d]
        dsel = [s for s in sel["variables"].values() if s["domain"] == d]
        cnt = dict(Counter(str(s["selected_P"]) for s in dsel))
        lines.append(f"| {d} | {r1['std_mse']:.4f} | {r1['std_mae']:.4f} | "
                     f"{rs['std_mse']:.4f} | {rs['std_mae']:.4f} | {cnt} | {r1['n_vars']} |")
    lines += ["",
              "Raw-scale metrics are reported only at variable level "
              "(test_variable_level.csv); they are never averaged across units.",
              ""]
    with open(os.path.join(args.results, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
