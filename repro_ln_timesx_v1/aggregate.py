#!/usr/bin/env python3
"""Stage aggregate (spec sec 7.3): unified tables from frozen predictions.

Aggregation: window mean -> variable -> merge ALL domains of one
(method, seed) [assert: no duplicate vars, 190 vars / 19 domains] ->
domain equal-weight -> overall; then 3-seed mean +/- sample std (ddof=1).
Z0/N0 are deterministic singles read from their own full-domain npz.
Raw-scale metrics ONLY at variable level. Fallback (selected_with_fallback)
is reported separately and never merged into the main fine-tuning table.
Non-finite metrics, missing domains or variables abort with a clear error.

Pure helpers (merge_var_tables / dom_overall / seed_stats) are exercised by
preflight section K on synthetic data with known closed-form answers.
"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DOMAIN_NAMES, METHODS_TRAIN, PREDICTIONS, RESULTS,
                    atomic_write_json)

F_METHODS = tuple(METHODS_TRAIN)
METRICS = ("mse", "mae", "raw_mse", "raw_mae")


class AggregateError(RuntimeError):
    pass


def load_set(path):
    with np.load(path, allow_pickle=False) as z:
        pred, target, d = z["pred"], z["target"], z["d"]
        vks = [str(x) for x in z["var_keys"]]
    err = pred - target
    return {"var": vks,
            "std_mse": np.mean((err / d[:, None]) ** 2, axis=1),
            "std_mae": np.mean(np.abs(err) / d[:, None], axis=1),
            "raw_mse": np.mean(err ** 2, axis=1),
            "raw_mae": np.mean(np.abs(err), axis=1)}


def var_table_of(entry):
    """Per-window arrays -> {var_key: {metric: mean, n: count}}."""
    per_var = {}
    for vk, sm, sa, rm, ra in zip(entry["var"], entry["std_mse"],
                                  entry["std_mae"], entry["raw_mse"],
                                  entry["raw_mae"]):
        pv = per_var.setdefault(vk, {k: [] for k in METRICS})
        pv["mse"].append(sm)
        pv["mae"].append(sa)
        pv["raw_mse"].append(rm)
        pv["raw_mae"].append(ra)
    return {vk: {k: float(np.mean(v[k])) for k in METRICS} | {"n": len(v["mse"])}
            for vk, v in per_var.items()}


def merge_var_tables(tables, expect_domains=len(DOMAIN_NAMES),
                     expect_vars=190, strict=True):
    """Merge per-domain var tables of one (method, seed). Duplicates and
    (in strict mode) missing domains/vars are hard errors."""
    merged = {}
    for t in tables:
        for vk, v in t.items():
            if vk in merged:
                raise AggregateError(f"duplicate var {vk} across domain runs")
            merged[vk] = v
    if strict:
        doms = {vk.split("__", 1)[0] for vk in merged}
        missing = sorted(set(DOMAIN_NAMES) - doms)
        extra = sorted(doms - set(DOMAIN_NAMES))
        if missing or extra:
            raise AggregateError(f"domain coverage broken: missing={missing[:3]} "
                                 f"extra={extra[:3]}")
        if len(merged) != expect_vars:
            raise AggregateError(f"{len(merged)} vars merged, expected "
                                 f"{expect_vars}")
    return merged


def dom_overall(merged, strict=True):
    """Merged var table -> (dom_t, overall) with domain equal-weight."""
    dom_t = {}
    for d in DOMAIN_NAMES:
        vs = [merged[vk] for vk in merged if vk.split("__", 1)[0] == d]
        if not vs:
            if strict:
                raise AggregateError(f"no variables for domain {d}")
            continue
        dom_t[d] = {k: float(np.mean([v[k] for v in vs])) for k in METRICS}
    _assert_finite(dom_t, "domain table")
    overall = {k: float(np.mean([dom_t[d][k] for d in dom_t])) for k in METRICS}
    _assert_finite(overall, "overall")
    return dom_t, overall


def _assert_finite(obj, what):
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, float) and not math.isfinite(o):
            raise AggregateError(f"non-finite {what} at {path}")
    walk(obj, what)


def seed_stats(values):
    """values: list of seed-level floats -> (mean, sample_std ddof=1)."""
    if len(values) == 1:
        return values[0], None
    return float(np.mean(values)), float(np.std(values, ddof=1))


def fmt(mean, std):
    return f"{mean:.6f}" if std is None else f"{mean:.6f}+/-{std:.6f}"


def main():
    sel = json.load(open(os.path.join(RESULTS, "selection.json")))
    sets = {"Z0": load_set(os.path.join(PREDICTIONS, "Z0__test.npz")),
            "N0": load_set(os.path.join(PREDICTIONS, "N0__test.npz"))}
    for rid in sel["per_run"]:
        sets[rid] = load_set(os.path.join(PREDICTIONS, f"{rid}__test.npz"))

    # ---------- var tables per source ----------
    z0_var = var_table_of(sets["Z0"])
    n0_var = var_table_of(sets["N0"])
    z0_dom, z0_ov = dom_overall(z0_var)
    n0_dom, n0_ov = dom_overall(n0_var)

    # ---------- per (method, seed): merge 19 domain runs ----------
    seed_merged = {}   # (m, seed) -> merged var table
    seed_dom = {}      # (m, seed) -> dom_t
    seed_ov = {}       # (m, seed) -> overall
    for m in F_METHODS:
        for seed in (2021, 2022, 2023):
            rids = [rid for rid, r in sel["per_run"].items()
                    if r["method"] == m and r["seed"] == seed]
            if len(rids) != len(DOMAIN_NAMES):
                raise AggregateError(f"{m} s{seed}: {len(rids)} runs, "
                                     f"expected {len(DOMAIN_NAMES)}")
            merged = merge_var_tables([var_table_of(sets[rid]) for rid in rids])
            seed_merged[(m, seed)] = merged
            seed_dom[(m, seed)], seed_ov[(m, seed)] = dom_overall(merged)

    # ---------- main results ----------
    method_overall = {
        "Z0": {"mse": (z0_ov["mse"], None), "mae": (z0_ov["mae"], None)},
        "N0": {"mse": (n0_ov["mse"], None), "mae": (n0_ov["mae"], None)}}
    overall_rows = [["Z0", "single", fmt(z0_ov["mse"], None),
                     fmt(z0_ov["mae"], None)],
                    ["N0", "single", fmt(n0_ov["mse"], None),
                     fmt(n0_ov["mae"], None)]]
    for m in F_METHODS:
        mo = {}
        for k in ("mse", "mae"):
            vals = [seed_ov[(m, s)][k] for s in (2021, 2022, 2023)]
            mean, std = seed_stats(vals)
            mo[k] = (mean, std)
        method_overall[m] = mo
        overall_rows.append([m, "mean+-std over 3 seeds",
                             fmt(*mo["mse"]), fmt(*mo["mae"])])

    # ---------- domain summary ----------
    domain_rows = []
    for m, dt in (("Z0", z0_dom), ("N0", n0_dom)):
        for d in DOMAIN_NAMES:
            domain_rows.append([m, d, "single", fmt(dt[d]["mse"], None),
                                fmt(dt[d]["mae"], None)])
    for m in F_METHODS:
        for d in DOMAIN_NAMES:
            mv = [seed_dom[(m, s)][d]["mse"] for s in (2021, 2022, 2023)]
            av = [seed_dom[(m, s)][d]["mae"] for s in (2021, 2022, 2023)]
            domain_rows.append([m, d, "mean+-std", fmt(*seed_stats(mv)),
                                fmt(*seed_stats(av))])

    # ---------- variable level (standardized AND raw) ----------
    all_vars = sorted(z0_var)
    if len(all_vars) != 190:
        raise AggregateError(f"Z0 covers {len(all_vars)} vars, expected 190")
    var_rows = []
    for m, merged in (("Z0", z0_var), ("N0", n0_var)):
        for vk in all_vars:
            t = merged[vk]
            var_rows.append([m, vk, t["n"], fmt(t["mse"], None),
                             fmt(t["mae"], None), fmt(t["raw_mse"], None),
                             fmt(t["raw_mae"], None)])
    for m in F_METHODS:
        for vk in all_vars:
            per_seed = [seed_merged[(m, s)][vk] for s in (2021, 2022, 2023)]
            ms = {k: [t[k] for t in per_seed] for k in METRICS}
            var_rows.append([m, vk, per_seed[0]["n"]] +
                            [fmt(*seed_stats(ms[k])) for k in METRICS])

    # ---------- fallback (selected_with_fallback) ----------
    fb_counts = 0
    fallback_rows = []
    for m in F_METHODS:
        for seed in (2021, 2022, 2023):
            merged = seed_merged[(m, seed)]
            dom_scores = {}
            for rid, r in sel["per_run"].items():
                if r["method"] != m or r["seed"] != seed:
                    continue
                src = z0_dom if r["fallback_to_zeroshot"] else dom_overall(
                    var_table_of(sets[rid]), strict=False)[0]
                dom_scores[r["domain"]] = src[r["domain"]]
                if r["fallback_to_zeroshot"]:
                    fb_counts += 1
            if set(dom_scores) != set(DOMAIN_NAMES):
                raise AggregateError(f"{m} s{seed} fallback: domains "
                                     f"{sorted(set(DOMAIN_NAMES) - set(dom_scores))} missing")
            fallback_rows.append([m, seed,
                                  fmt(float(np.mean([dom_scores[d]["mse"]
                                                     for d in DOMAIN_NAMES])), None),
                                  fmt(float(np.mean([dom_scores[d]["mae"]
                                                     for d in DOMAIN_NAMES])), None),
                                  sum(1 for r in sel["per_run"].values()
                                      if r["method"] == m and r["seed"] == seed
                                      and r["fallback_to_zeroshot"])])

    # ---------- improvement/degradation vs Z0 ----------
    impr_rows = []
    for m in F_METHODS:
        for metric in ("mse", "mae"):
            improved = degraded = equal = 0
            for d in DOMAIN_NAMES:
                v = float(np.mean([seed_dom[(m, s)][d][metric]
                                   for s in (2021, 2022, 2023)]))
                ref = z0_dom[d][metric]
                improved += v < ref
                degraded += v > ref
                equal += v == ref
            impr_rows.append([m, "domain(19)", metric, improved, degraded,
                              equal, 19])
            improved = degraded = equal = 0
            for vk in all_vars:
                v = float(np.mean([seed_merged[(m, s)][vk][metric]
                                   for s in (2021, 2022, 2023)]))
                ref = z0_var[vk][metric]
                improved += v < ref
                degraded += v > ref
                equal += v == ref
            impr_rows.append([m, "var(190)", metric, improved, degraded,
                              equal, 190])

    # ---------- pre-registered comparisons ----------
    def overall_metric(m, metric):
        if m in ("Z0", "N0"):
            src = z0_ov if m == "Z0" else n0_ov
            return src[metric]
        return float(np.mean([seed_ov[(m, s)][metric]
                              for s in (2021, 2022, 2023)]))

    comp_rows = []
    for a, b in (("F1", "F0"), ("F2", "F1"), ("F3", "F1"), ("F4", "F2"),
                 ("F4", "F3"), ("F0", "Z0"), ("F1", "Z0"), ("F2", "Z0"),
                 ("F3", "Z0"), ("F4", "Z0")):
        for metric in ("mse", "mae"):
            va, vb = overall_metric(a, metric), overall_metric(b, metric)
            delta = va - vb
            rel = f"{-delta / vb * 100:+.2f}%" if vb != 0 \
                else "ref=0, ratio undefined"
            comp_rows.append([f"{a}-{b}", metric, f"{delta:+.6f}", rel])

    # ---------- write ----------
    def w_csv(name, header, rows):
        with open(os.path.join(RESULTS, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    w_csv("overall.csv", ["method", "seed_stat", "MSE", "MAE"], overall_rows)
    w_csv("domain_summary.csv",
          ["method", "domain", "seed_stat", "MSE", "MAE"], domain_rows)
    w_csv("test_variable_level.csv",
          ["method", "var_key", "n_windows", "std_MSE", "std_MAE",
           "raw_MSE", "raw_MAE"], var_rows)
    w_csv("fallback_summary.csv",
          ["method", "seed", "overall_MSE", "overall_MAE",
           "n_fallback_domains"], fallback_rows)
    w_csv("improvement_counts.csv",
          ["method", "level", "metric", "improved", "degraded", "equal",
           "total"], impr_rows)
    w_csv("pre_registered_comparisons.csv",
          ["comparison", "metric", "delta", "relative_improvement"], comp_rows)

    report = make_report(sel, overall_rows, impr_rows, comp_rows,
                         fallback_rows)
    with open(os.path.join(RESULTS, "report.md"), "w") as f:
        f.write(report)
    atomic_write_json({"methods_overall": {m: {"mse": list(v["mse"]),
                                               "mae": list(v["mae"])}
                                           for m, v in method_overall.items()}},
                      os.path.join(RESULTS, "aggregate_summary.json"))
    print(f"[aggregate] wrote tables + report.md (fallback domains: {fb_counts})")


def make_report(sel, overall_rows, impr_rows, comp_rows, fallback_rows):
    lines = [
        "# TimesX LN fine-tuning (protocol timesx_ln_v1) — results",
        "",
        f"- Runs: {sel['n_runs']}/285 frozen; selection rule: best_ft "
        "(val std-MSE, epochs 1-10, earliest on tie).",
        "- Aggregation: window mean -> variable -> 19 domains merged per "
        "(method, seed) -> domain equal-weight -> overall; 3-seed "
        "mean±std(ddof=1); standardized by per-window input std.",
        "- Z0 = fixed-P1 zero-shot recomputed on the NEW split; N0 = "
        "last-value. Both deterministic singles.",
        "",
        "## Overall (standardized)",
        "",
        "| method | MSE | MAE |",
        "|---|---|---|",
    ]
    for r in overall_rows:
        lines.append(f"| {r[0]} | {r[2]} | {r[3]} |")
    lines += ["", "## Improvement/degradation vs Z0", "",
              "| method | level | metric | improved | degraded | equal |",
              "|---|---|---|---|---|---|"]
    for r in impr_rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
    lines += ["", "## Pre-registered comparisons (overall)", "",
              "| comparison | metric | delta | relative |",
              "|---|---|---|---|"]
    for r in comp_rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    n_fb = sum(r[4] for r in fallback_rows)
    lines += ["", "## Fallback (selected_with_fallback, separate table)", "",
              f"- fallback_to_zeroshot triggered in {n_fb} of 285 runs "
              f"(per-domain breakdown in fallback_summary.csv).", "",
              "## Interpretation limits", "",
              "- Three-seed std describes training randomness only; overlapping"
              " windows are not independent samples.",
              "- The new split shares data with the earlier zero-shot study; "
              "this is an exploratory method comparison, not a blind test.",
              "- No causal attribution beyond the pre-registered contrasts.",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except AggregateError as e:
        print(f"[aggregate] FAILED: {e}")
        sys.exit(1)
