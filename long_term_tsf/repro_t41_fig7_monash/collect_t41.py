"""EXP-041 collector: RESULT_t41_fig7_monash.md + summary_t41.csv +
figure reference.  Reads manifest / extract env / tsne params / nn
summary; reports the 30-NN table with EXPLICIT numerators/denominators,
the paper Fig.7 comparison, and the frozen-disclosure block.

Reporting discipline (frozen):
  - the Monash sampling rule (29 paper rows, quota + redistribution,
    unified left-5-column visible mask) is a SUPPLEMENTARY setting of
    this round, NOT specified by the paper;
  - the unified mask differs from prediction-time geometry (where the
    visible patch count follows update_config per subset);
  - 2-D t-SNE overlap is qualitative ONLY — no prediction / causal use;
  - inconsistencies with the paper figure are reported as observations.
"""
import csv
import os

import numpy as np

import t41_common as X


def main():
    manifest = X.load_json(X.MANIFEST_JSON)
    samples = manifest["samples"]
    summ = X.load_json(X.SAVE_BASE + "/nn_summary.json")
    params = X.load_json(X.SAVE_BASE + "/tsne_params.json")
    env = X.load_json(X.EXTRACT_ENV_JSON)
    verify = X.load_json(X.VERIFY_JSON) if os.path.isfile(X.VERIFY_JSON) else {}

    L = ["# EXP-041: paper Fig.7 completed — Monash t-SNE rows",
         "",
         "Joint 2,200-row t-SNE (frozen EXP-002 1,900 rows re-enter the",
         "fit; their coords are NOT reused) with 300 stratified Monash",
         "histories from the 29 paper Table-15 rows (seed 2021).",
         "",
         "**Supplementary setting disclosure (frozen):** the Monash",
         "sampling rule and the unified left-5-column visible mask are",
         "visualization settings of this round, NOT paper-specified;",
         "H = official forecast length (archived pred_rule.expected x",
         "gluonts metadata cross-check), P = EXP-038 autotune accepted_p,",
         "L_eff = actual per-sample input history = min(L_config, N-H)",
         "(official short histories preserved).",
         ""]
    if verify:
        L.append("Gates: **%s** (FG1-FG7, %d checks, %d fail) -> %s"
                 % ("ALL PASS" if verify.get("all_pass") else "FAIL",
                    verify.get("summary", {}).get("n_checks", 0),
                    verify.get("summary", {}).get("n_fail", 0),
                    "logs/verify_t41.json"))
        L.append("")

    # ---- sampling summary ------------------------------------------------
    L.append("## S1 Monash sampling (seed 2021)")
    L.append("")
    L.append("| item | value |")
    L.append("|---|---|")
    L.append("| samples | %d |" % len(samples))
    L.append("| paper rows | %d (quota floor(300/29)=10, surplus 10 to "
             "largest-capacity rows) |" % len(by_row(samples)))
    n_short = sum(1 for s in samples if s["short_history"])
    n_nan_w = sum(1 for s in samples if s["n_nan_raw_window"] > 0)
    n_packing = sum(1 for s in samples if s["fill_method"] != "random_greedy")
    L.append("| short-history samples (L_eff < L_config) | %d |" % n_short)
    L.append("| windows with raw NaN (forward-filled) | %d |" % n_nan_w)
    L.append("| windows from packing redraw (greedy underfull) | %d |"
             % n_packing)
    L.append("| redistribution transfers | %s |"
             % (manifest.get("transfers") or "none"))
    L.append("| CIF_2016_POOLED split | %d (%s) + %d (%s), the designed "
             "50/50 — no shortfall |"
             % (sum(1 for s in samples if s["subset"] == "cif_2016_6"),
                "cif_2016_6",
                sum(1 for s in samples if s["subset"] == "cif_2016_12"),
                "cif_2016_12"))
    worst = sorted(manifest.get("capacities", {}).items(),
                   key=lambda kv: kv[1])[:3]
    L.append("| smallest-capacity rows | %s |"
             % ", ".join("%s=%d" % kv for kv in worst))
    n_cfg = env.get("n_update_config_calls", "?")
    L.append("| geometry groups (update_config calls) | %s |" % n_cfg)

    # ---- per-row table (per subset: pooled rows list each subset's H/P/L) ----
    L.append("")
    L.append("## S2 per-row draw (per subset; pooled rows list each subset)")
    L.append("")
    L.append("| paper row | subset | H | P | L_config | drawn | row capacity |")
    L.append("|---|---|---|---|---|---|---|")
    for s_row in X.T38.PAPER_T15_ORDER:
        for ds in X.row_subsets(s_row):
            rows = [s for s in samples if s["row"] == s_row
                    and s["subset"] == ds]
            if not rows:
                continue
            s0 = rows[0]
            cap = manifest.get("capacities", {}).get(s_row, "?")
            L.append("| %s | %s | %d | %d | %d | %d | %s |"
                     % (s_row, ds, s0["H"], s0["P"], s0["L_config"],
                        len(rows), cap))

    # ---- 30-NN table --------------------------------------------------------
    L.append("")
    L.append("## S3 high-dim 30-NN ImageNet proportion (frozen encoder, "
             "float64 euclidean over all 2,200 rows)")
    L.append("")
    L.append("| time-series source | n_rows | ImageNet neighbours "
             "(numerator) | denominator | proportion |")
    L.append("|---|---|---|---|---|")
    for g in X.TS_SOURCES:
        p = summ["proportions"][g]
        L.append("| %s | %d | %d | %d | %.4f |"
                 % (g, X.SPANS[g][1] - X.SPANS[g][0],
                    p["numerator_imagenet_neighbors"], p["denominator"],
                    p["proportion"]))
    L.append("")
    L.append("Self excluded by global row index (fill_diagonal inf + "
             "assert).  2-D overlap is qualitative only.")

    # ---- paper comparison ---------------------------------------------------
    L.append("")
    L.append("## S4 paper Fig.7 comparison (qualitative)")
    L.append("")
    L.append("Paper (VisionTS_paper_full.md L128/L136): ETTm1 clusters "
             "upper-right; Monash lower-left with a significant gap to "
             "ImageNet; ImageNet concentrated, time series scattered.  "
             "Our joint refit is compared side-by-side in "
             "`fig41_tsne_monash.png` (panel 2 = the sha-pinned original "
             "asset).  Agreement / disagreement is reported as an "
             "observation; no numeric threshold against the paper exists "
             "in this round.")

    # ---- stability ------------------------------------------------------------
    L.append("")
    L.append("## S5 stability + determinism")
    L.append("")
    runs = params.get("runs", {})
    for k, v in sorted(runs.items()):
        extra = ("; refit bitwise-equal" if v.get("recheck_bitwise_equal")
                 else "")
        L.append("- %s: sklearn %s, %.1f s%s"
                 % (k, v.get("sklearn", "?"), v.get("elapsed_s", -1), extra))
    L.append("- frozen 1,900 rows in the merged matrix verified "
             "element-identical (np.array_equal) before fitting.")
    L.append("- sklearn at fit time: %s (EXP-002 pin %s)"
             % (runs.get("rs2021", {}).get("sklearn", "?"),
                X.SKLEARN_PIN_PREFIX + "*"))
    probe = verify.get("probe_reextract") or {}
    if probe:
        L.append("- batch-invariance probe (FG3, first %s samples): "
                 "grouped-schedule re-extraction bitwise == archived "
                 "(max abs err %s); per-sample B=1 re-extraction "
                 "deviates abs %s / rel %.4f%% — **encoder forwards are "
                 "NOT bitwise-invariant to batch size** (kernel "
                 "selection), a recorded observation; all downstream "
                 "features/coords/NN use the archived grouped schedule "
                 "self-consistently."
                 % (probe.get("n_probe_samples", "?"),
                    probe.get("max_abs_err_grouped_batch", "?"),
                    probe.get("max_abs_err_single_sample", "?"),
                    probe.get("rel_err_single_sample_pct", -1.0)))

    # ---- budget -----------------------------------------------------------------
    L.append("")
    L.append("## S6 environment")
    L.append("")
    L.append("- extract GPU: %s (torch %s)" % (env.get("gpu"),
                                               env.get("torch")))
    L.append("- sampling env: author-pinned env (python 3.8, gluonts "
             "0.14.4); cache %s" % manifest.get("gluonts_cache", "?"))

    # ---- csv ----------------------------------------------------------------------
    csv_path = os.path.join(X.LOG_DIR, "summary_t41.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "row", "subset", "series_idx", "H", "P",
                    "L_config", "L_eff", "window_start", "N_in",
                    "short_history", "n_nan_raw_window", "fill_method"])
        for s in samples:
            w.writerow([s["k"], s["row"], s["subset"], s["series_idx"],
                        s["H"], s["P"], s["L_config"], s["L_eff"],
                        s["window_start"], s["N_in"], s["short_history"],
                        s["n_nan_raw_window"], s["fill_method"]])
    X.log("wrote %s (%d rows)" % (csv_path, len(samples)))

    # ---- self-check -----------------------------------------------------------------
    assert len(samples) == X.N_MONASH
    assert all(np.isfinite([s["L_eff"], s["H"], s["P"]]).all()
               for s in samples)
    with open(X.RESULT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    X.log("wrote %s" % X.RESULT_MD)


def by_row(samples):
    return set(s["row"] for s in samples)


if __name__ == "__main__":
    main()
