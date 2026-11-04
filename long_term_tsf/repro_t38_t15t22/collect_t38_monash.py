"""EXP-038 task A collector: build logs/summary_t38_monash.csv and
logs/RESULT_t38_monash.md from the archived evidence only (logs/*.json,
verify json, naive/monash CSVs). Also renders the paper-Figure-5
comparison chart (measured vs paper-cited VTS/Naive ratio per row) to
logs/fig5_monash_ratio.png. Every number cites its evidence file.
Local-only (numpy/json); matplotlib is optional (chart skipped with a
note if unavailable). --selfcheck exercises all output paths with
fabricated records in temp_selfcheck_t38/ without touching real files.
"""
import argparse
import csv
import math
import os

import numpy as np

import t38_common as C


def fmt(v, nd=6):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return "n/a"
        return ("%%.%df" % nd) % v
    return str(v)


def load_subs():
    subs = {}
    for ds in C.MONASH_ORDER:
        mp, np_ = C.monash_json(ds), C.naive_json(ds)
        if os.path.isfile(mp) and os.path.isfile(np_):
            subs[ds] = {"monash": C.load_json(mp), "naive": C.load_json(np_)}
    return subs


def load_verify():
    if os.path.isfile(C.VERIFY_JSON):
        return C.load_json(C.VERIFY_JSON).get("monash", {})
    return {}


def row_status(ds, m, n):
    flags = []
    if m.get("autotune", {}).get("accepted_p") != C.EXPECTED_P[ds]:
        flags.append("P!=T6")
    if not m.get("anchor", {}).get("pass"):
        flags.append("anchor>1e-3")
    n1v = n.get("n1_vs_paper_naive", {})
    if n1v.get("paper_naive") is not None and not n1v.get("round_match"):
        flags.append("N1_round")
    if not m.get("checks", {}).get("rule_matches_meta"):
        flags.append("rule!=meta")
    ctx = m.get("context", {})
    exp_ctx = ctx.get("expected_from_accepted_p")
    if exp_ctx is not None and any(int(k) < exp_ctx for k in
                                   ctx.get("actual_update_config_final", {})):
        flags.append("ctx_trunc")
    return flags


def paper_ratio(key):
    p = C.PAPER_T15[key]
    return float(p["vts"]) / float(p["naive"])


def build_rows(subs):
    rows = []
    pooled = None
    for ds in C.MONASH_ORDER:
        m = subs.get(ds, {}).get("monash", {})
        n = subs.get(ds, {}).get("naive", {})
        split = m.get("split", {})
        rule = m.get("pred_rule", {})
        gen = m.get("generation", {})
        t = m.get("timing", {})
        ph = t.get("phases_sec", {})
        vm = m.get("result_csv", {}).get("metrics", {}).get("MAE[0.5]")
        n1 = n.get("n1", {}).get("metrics", {}).get("MAE[0.5]")
        n2 = n.get("n2", {}).get("metrics", {}).get("MAE[0.5]")
        paper = C.PAPER_T15.get(ds)
        n1v = n.get("n1_vs_paper_naive", {})
        v_ok = quantum = None
        if paper is not None and vm is not None:
            v_ok, quantum, _ = C.paper_round_match(vm, paper["vts"])
        row = {
            "task": "A_monash_t15", "item": ds,
            "pred_len": split.get("meta_prediction_length"),
            "pred_rule_source": rule.get("source"),
            "n_series": split.get("n_series"),
            "autotune_P": m.get("autotune", {}).get("accepted_p"),
            "P_expect": C.EXPECTED_P[ds],
            "ctx_configured": m.get("context", {}).get("configured"),
            "ctx_expected": m.get("context", {}).get("expected_from_accepted_p"),
            "ctx_actual_max": max([int(k) for k in
                                   m.get("context", {}).get(
                                       "actual_update_config_final", {})],
                                  default=None),
            "vts_mae": vm, "anchor_mae": m.get("anchor", {}).get("mae_anchor"),
            "anchor_rel": m.get("anchor", {}).get("rel_diff"),
            "anchor_pass": m.get("anchor", {}).get("pass"),
            "n1_mae": n1, "n2_mae": n2,
            "ratio_vts_n1": (vm / n1) if (vm and n1) else None,
            "ratio_vts_n2": (vm / n2) if (vm and n2) else None,
            "paper_vts": paper["vts"] if paper else "POOLED",
            "paper_naive": paper["naive"] if paper else "POOLED",
            "n1_round_match": n1v.get("round_match"),
            "vts_round_match": v_ok,
            "model_init_s": ph.get("model_init"),
            "download_s": ph.get("download"),
            "prep_s": ph.get("prep"),
            "autotune_s": ph.get("autotune"),
            "inference_s": ph.get("inference"),
            "eval_s": ph.get("eval"),
            "wall_s": t.get("wall_sec"),
            "residual_frac": t.get("residual_frac"),
            "gpu_peak_mb": (m.get("gpu_peak") or {}).get("allocated_mb"),
            "download_sha256_16": (gen.get("download_sha256") or "")[:16]
            if gen.get("download_sha256") else None,
            "flags": "|".join(row_status(ds, m, n)) if m else "MISSING",
        }
        rows.append(row)
        if ds in ("cif_2016_6", "cif_2016_12") and vm and n1:
            pooled = pooled or {}
            pooled[ds] = {"vts": vm, "n1": n1, "n_series":
                          split.get("n_series")}
    if pooled and len(pooled) == 2:
        vts = C.cif_pool({k: v["vts"] for k, v in pooled.items()})
        n1 = C.cif_pool({k: v["n1"] for k, v in pooled.items()})
        ok_v, _, wv = C.paper_round_match(vts, "570907.24")
        ok_n, _, wn = C.paper_round_match(n1, "578596.5")
        anchor = C.cif_pool(
            {k: C.AUTHOR_ANCHOR[k]["mae"] for k in pooled})
        rel = C.rel_diff(vts, anchor)
        rows.append({
            "task": "A_monash_t15", "item": "CIF_2016_POOLED",
            "pred_len": "6|12", "pred_rule_source": "cif_meta_forced",
            "n_series": "%d|%d" % (pooled["cif_2016_6"]["n_series"],
                                   pooled["cif_2016_12"]["n_series"]),
            "autotune_P": "3|6", "P_expect": "3|6",
            "ctx_configured": None, "ctx_expected": None, "ctx_actual_max": None,
            "vts_mae": vts, "anchor_mae": anchor,
            "anchor_rel": rel,
            "anchor_pass": bool(rel <= C.ANCHOR_TOL),
            "n1_mae": n1, "n2_mae": None,
            "ratio_vts_n1": vts / n1 if n1 else None, "ratio_vts_n2": None,
            "paper_vts": "570907.24", "paper_naive": "578596.5",
            "n1_round_match": ok_n, "vts_round_match": ok_v,
            "model_init_s": None, "download_s": None, "prep_s": None,
            "autotune_s": None, "inference_s": None, "eval_s": None,
            "wall_s": None, "residual_frac": None, "gpu_peak_mb": None,
            "download_sha256_16": None, "flags": "",
        })
    return rows


FIELDS = ["task", "item", "pred_len", "pred_rule_source", "n_series",
          "autotune_P", "P_expect", "ctx_configured", "ctx_expected",
          "ctx_actual_max", "vts_mae", "anchor_mae", "anchor_rel",
          "anchor_pass", "n1_mae", "n2_mae", "ratio_vts_n1", "ratio_vts_n2",
          "paper_vts", "paper_naive", "n1_round_match", "vts_round_match",
          "model_init_s", "download_s", "prep_s", "autotune_s",
          "inference_s", "eval_s", "wall_s", "residual_frac", "gpu_peak_mb",
          "download_sha256_16", "flags"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fig5(rows, out_png):
    """Paper Figure 5 comparison: measured VTS/N1 ratio vs the ratio
    implied by the paper's printed values, log scale."""
    names, meas, pap = [], [], []
    for r in rows:
        key = r["item"]
        if key not in C.PAPER_T15_ORDER or r["ratio_vts_n1"] is None:
            continue
        names.append(key)
        meas.append(r["ratio_vts_n1"])
        pap.append(paper_ratio(key))
    if not names:
        return "no rows available for fig5"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return "matplotlib unavailable (%s); fig5 skipped" % e
    x = np.arange(len(names))
    w = 0.4
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(x - w / 2, meas, w, label="this run VTS/N1", color="#3b6db3")
    ax.bar(x + w / 2, pap, w, label="paper Table15 VTS/Naive",
           color="#c44e52", alpha=0.75)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.axhline(C.GEOMEAN_PAPER, ls="--", lw=0.8, color="gray",
               label="paper Normalized MAE 0.729")
    g = math.exp(sum(math.log(v) for v in meas) / len(meas))
    ax.axhline(g, ls=":", lw=0.8, color="black",
               label="this-run geomean %.3f" % g)
    ax.set_ylabel("VisionTS / Naive (MAE ratio, log)")
    ax.set_title("VisionTS paper Fig.5 comparison: measured vs paper-cited "
                 "(per Monash subset)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return "fig5 written (%d rows, geomean %.4f)" % (len(names), g)


def result_md(rows, subs, vj):
    L = []
    L.append("# EXP-038 RESULT A: Monash Table 15 full reproduction "
             "(author-pinned env, 30 code subsets -> 29 paper rows)\n")
    L.append("Generated by collect_t38_monash.py from archived evidence "
             "only (logs/monash_<ds>.json, logs/naive_<ds>.json, "
             "logs/verify_t38.json section monash, "
             "save_t38_monash/*_monash_*.csv).\n")
    L.append("Definitions: N1 = last imputed value repeated (Monash "
             "benchmark 'Naive', primary denominator; Woo et al. 2024 / "
             "Godahewa et al. 2021); N2 = seasonal-naive tiled with the "
             "autotuned P (sensitivity). VisionTS and Naive share the same "
             "test split objects, imputation, evaluator and metric list. "
             "Env: author-pinned python/torch/gluonts/pandas per "
             "monash_<ds>.json env field.\n")

    L.append("## Table 15 rows (paper order; ratio = VTS/N1)\n")
    L.append("| row | P | pred_len(src) | n_series | VTS MAE | N1 MAE | "
             "ratio | paper VTS | paper Naive | paper ratio | anchor rel | "
             "flags |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    by = {r["item"]: r for r in rows}
    for key in C.PAPER_T15_ORDER:
        if key == "CIF_2016_POOLED":
            r = by.get("CIF_2016_POOLED")
            if r:
                L.append("| CIF_2016 (pooled) | 3\\|6 | 6\\|12(cif_meta) | %s "
                         "| %s | %s | %s | 570907.24 | 578596.5 | %.4f | %s "
                         "| %s |" %
                         (r["n_series"], fmt(r["vts_mae"], 2),
                          fmt(r["n1_mae"], 2), fmt(r["ratio_vts_n1"], 4),
                          paper_ratio(key), fmt(r["anchor_rel"], 2),
                          r["flags"] or "-"))
            continue
        r = by.get(key)
        if not r:
            L.append("| %s | - | - | - | - | - | - | %s | %s | %.4f | - | "
                     "MISSING |" % (key, C.PAPER_T15[key]["vts"],
                                    C.PAPER_T15[key]["naive"],
                                    paper_ratio(key)))
            continue
        L.append("| %s | %s | %s(%s) | %s | %s | %s | %s | %s | %s | %.4f | "
                 "%s | %s |" %
                 (key, fmt(r["autotune_P"], 0), fmt(r["pred_len"], 0),
                  r["pred_rule_source"], fmt(r["n_series"], 0),
                  fmt(r["vts_mae"], 2), fmt(r["n1_mae"], 2),
                  fmt(r["ratio_vts_n1"], 4), r["paper_vts"], r["paper_naive"],
                  paper_ratio(key), fmt(r["anchor_rel"], 2),
                  r["flags"] or "-"))
    geo = (vj.get("geomean") or {})
    if geo.get("valid"):
        L.append("\nGeometric mean of the 29 measured ratios: **%.4f** vs "
                 "paper Normalized MAE **%.3f** (rel %.4f%s). Evidence: "
                 "G10 in logs/verify_t38.json.\n"
                 % (geo["geomean"], geo["paper"], geo["rel_diff"],
                    " — >2%% WARN" if geo.get("warn") else ""))
    else:
        L.append("\nGeomean NOT computed: only %s/29 paper rows valid "
                 "(coverage count reported per user correction #2).\n"
                 % geo.get("n_valid_rows"))

    L.append("## Paper Figure 5 comparison chart\n")
    L.append("Chart: `logs/fig5_monash_ratio.png` (log-scale bars, "
             "measured VTS/N1 vs paper-printed VTS/Naive per row).\n")

    L.append("## Per-subset timing and peak memory (phase split)\n")
    L.append("| subset | model_init s | download s | prep s | autotune s | "
             "inference s | eval s | wall s | residual frac | GPU peak MB |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["item"] == "CIF_2016_POOLED" or r["wall_s"] is None:
            continue
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["item"], fmt(r["model_init_s"], 1), fmt(r["download_s"], 1),
            fmt(r["prep_s"], 1), fmt(r["autotune_s"], 1),
            fmt(r["inference_s"], 1), fmt(r["eval_s"], 1),
            fmt(r["wall_s"], 1), fmt(r["residual_frac"], 4),
            fmt(r["gpu_peak_mb"], 0)))
    L.append("")
    L.append("Download=gluonts test-split getter (first-time raw download "
             "or cache load); prep=imputation+autotune-set-up up to the "
             "first forecast call; autotune=validation forecast calls; "
             "inference=final test forecast + forecast-object building; "
             "eval=gluonts evaluator. "
             "residual<1%% asserted by gate G11.\n")

    L.append("## Anomaly list\n")
    anomalies = vj.get("anomalies", [])
    flagged = [g for g in vj.get("gates", [])
               if not g["st"].startswith("PASS") and g["gate"] != "G10_geomean"]
    if not anomalies and not flagged:
        L.append("(none)\n")
    for a in anomalies:
        L.append("- [%s] %s: %s" % (a["type"], a["ds"], a["detail"]))
    for g in flagged:
        L.append("- [gate %s] %s %s: %s" % (g["st"], g["ds"], g["gate"],
                                            g["detail"]))
    if anomalies or flagged:
        L.append("")
    L.append("Over-tolerance rows are KEPT and explained (user correction "
             "#2); troubleshooting evidence: per-series MAE arrays in "
             "monash_<ds>.json, final forecast matrices in "
             "save_t38_monash/forecast_monash_<ds>.npz (not in the pack).\n")

    L.append("## Code-vs-paper anomalies documented\n")
    L.append("- run.py candidate-P space lacks H=168 hourly (paper Table 5 "
             "lists {24/x, 168/x, 1}); the code is used as-is, documented "
             "only.")
    L.append("- australian_electricity_demand (336) and pedestrian_counts "
             "(24) pred_len come from the official hard overrides "
             "(dataset.py:403-406), not raw .tsf horizons.")
    L.append("- CIF 2016 paper row = two code subsets pooled by series "
             "count (15 @ H=6, 57 @ H=12), pooling verified in G9. The "
             "paper's printed pooled VTS 570907.24 does not strictly "
             "round-match the pooled unrounded anchor (570907.2287 -> "
             "570907.23): last-digit print inconsistency, documented; "
             "the G9 hard check uses the pooled-anchor rel diff.")
    L.append("- M4 subsets (m4_monthly/weekly/daily/hourly) download from "
             "GitHub M4-methods via the gluonts _m4 generator (no zenodo "
             "zip): provenance + cache diff recorded instead of zip sha.\n")

    L.append("## EXP-037 current-env FRED-MD (kept separately, NOT re-run)\n")
    L.append("EXP-037 (2026-11-04, current env python3.12/torch2.12) "
             "FRED-MD zero-shot MAE **%.6f** vs author anchor 1893.668242 "
             "= rel %.3e -> G4 FAIL kept ('reproducible in this env, "
             "author value not reproduced'); in the author-pinned env the "
             "anchor reproduced bitwise (MAE) / to -1.19e-7 (MSE), "
             "attributed to the whole software environment. EXP-038 "
             "re-runs FRED-MD in the author-pinned env above; this "
             "current-env value stays archived only.\n"
             % (C.T37_FRED_CURRENT_ENV_MAE, C.T37_FRED_CURRENT_ENV_REL))
    return "\n".join(L) + "\n"


def selfcheck():
    subs = {}
    for ds in C.MONASH_ORDER:
        subs[ds] = {
            "monash": {
                "split": {"meta_prediction_length": C.PINNED_PRED.get(ds, 24),
                          "n_series": 42, "label_len_unique":
                          [C.PINNED_PRED.get(ds, 24)]},
                "pred_rule": {"expected": C.PINNED_PRED.get(ds, 24),
                              "source": "freq_fallback"},
                "checks": {"rule_matches_meta": True,
                           "pinned_pred_match": True},
                "autotune": {"accepted_p": C.EXPECTED_P[ds],
                             "events": [{"p": C.EXPECTED_P[ds],
                                         "valid_mae": 1.0, "accept": True}]},
                "context": {"configured": 1000,
                            "expected_from_accepted_p": 1001,
                            "actual_update_config_final": {"1001": 42},
                            "n_update_config_calls": 5},
                "timing": {"wall_sec": 100.0, "residual_frac": 0.001,
                           "phases_sec": {"model_init": 2.0, "download": 3.0,
                                          "prep": 5.0, "autotune": 40.0,
                                          "inference": 45.0, "eval": 4.0}},
                "gpu_peak": {"allocated_mb": 1234.0},
                "result_csv": {"metrics": {"MAE[0.5]": 1.0, "MSE[mean]": 2.0}},
                "anchor": {"mae_anchor": 1.001, "rel_diff": 1e-4, "pass": True},
                "generation": {"download_sha256": "ab" * 32,
                               "downloaded_bytes": 123,
                               "recipe_provenance": "x" if ds in C.M4_SUBSETS
                               else None},
                "data_version": {"n_files_new_or_changed": 1},
            },
            "naive": {
                "n1": {"metrics": {"MAE[0.5]": 2.0}},
                "n2": {"metrics": {"MAE[0.5]": 2.5}},
                "ratios": {"visionts_over_n1": 0.5, "visionts_over_n2": 0.4},
                "n1_vs_paper_naive": {"paper_naive": "2.0", "quantum": 0.01,
                                      "round_match": True, "rel_diff": 0.0},
            },
        }
    out = "temp_selfcheck_t38"
    os.makedirs(out, exist_ok=True)
    rows = build_rows(subs)
    write_csv(os.path.join(out, "summary_t38_monash.csv"), rows)
    md = result_md(rows, subs, {"anomalies": [], "gates": [],
                                "geomean": {"valid": False, "n_valid_rows": 0}})
    with open(os.path.join(out, "RESULT_t38_monash.md"), "w") as f:
        f.write(md)
    print("[t38] selfcheck fig5:", fig5(rows, os.path.join(
        out, "fig5_monash_ratio.png")))
    print("[t38] selfcheck OK -> %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    subs = load_subs()
    vj = load_verify()
    rows = build_rows(subs)
    write_csv(C.SUMMARY_A_CSV, rows)
    md = result_md(rows, subs, vj)
    with open(C.RESULT_A_MD, "w") as f:
        f.write(md)
    print("[t38] fig5:", fig5(rows, C.FIG5_PNG))
    print("[t38] wrote %s and %s" % (C.SUMMARY_A_CSV, C.RESULT_A_MD))


if __name__ == "__main__":
    main()
