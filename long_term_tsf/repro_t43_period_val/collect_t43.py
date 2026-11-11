"""EXP-043 collector: summary_t43_val.csv + summary_t43_test.csv +
RESULT_t43_period_val.md + 2 figures.

Reporting discipline (frozen):
  - the equal-weight mean-over-H selection rule and its tie-break are a
    SUPPLEMENTARY setting of this round, fixed before any test evaluation;
    the paper specifies validation selection but not the aggregation;
  - deviations from paper numbers are REPORTED OBSERVATIONS ONLY — no
    numeric threshold against the paper is a gate anywhere (user
    correction #4);
  - whether the validation-selected P matches the paper P, and whether
    the selected arm improves or regresses on test vs the default-P arm,
    are both reported honestly; neither is a pass/fail criterion.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t43_common as X

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def mets_of(path):
    m = np.load(os.path.join(path, "results", "_", "metrics.npy"))
    return float(m[1]), float(m[0])  # mse, mae


def sel_arm_group(ds, h, sel):
    """Returns (P_chosen, source, metrics_dir or None)."""
    P = sel["per_ds"][ds]["chosen_P"]
    if P == X.PAPER_P[ds]:
        return P, "exp001_reused", X.base_dir(ds, h)
    if P == 1:
        return P, "exp030_reused", X.p1_dir(ds, h)
    return P, "t43_new", X.t43_run_dir(ds, h)


def main():
    sel = X.load_json(X.LEDGER_JSON)
    os.makedirs(X.LOG_DIR, exist_ok=True)

    # ---- S1 validation table (48 rows) ----
    val_rows = []
    for ds in X.ETT4:
        for P in X.CANDIDATE_P[ds]:
            per_h = []
            for h in X.HORIZONS:
                j = X.load_json(X.vjson_path(ds, P, h))
                val_rows.append((ds, P, h, j["n_windows"],
                                 "%.12f" % j["mse_from_sse"],
                                 "%.12f" % j["mae_from_sae"]))
                per_h.append(j["mse_from_sse"])
    val_csv = os.path.join(X.LOG_DIR, "summary_t43_val.csv")
    with open(val_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ds", "P", "H", "n_windows", "mse_val_f64", "mae_val_f64"])
        w.writerows(val_rows)

    # ---- test arms ----
    test = {}  # ds -> h -> {"default":(mse,mae), "p1":..., "sel":...}
    for ds, h in X.group_ids():
        mse_d, mae_d = mets_of(X.base_dir(ds, h))
        mse_p, mae_p = mets_of(X.p1_dir(ds, h))
        P, src, mdir = sel_arm_group(ds, h, sel)
        mse_s, mae_s = mets_of(mdir)
        test[ds] = test.get(ds, {})
        test[ds][h] = {"default": (mse_d, mae_d), "p1": (mse_p, mae_p),
                       "sel": (mse_s, mae_s), "sel_P": P, "sel_src": src}

    test_csv = os.path.join(X.LOG_DIR, "summary_t43_test.csv")
    with open(test_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ds", "H", "P_default", "P_p1", "P_selected",
                    "sel_source",
                    "mse_default", "mse_p1", "mse_selected",
                    "mae_default", "mae_p1", "mae_selected",
                    "delta_mse_sel_vs_default", "rel_mse_sel_vs_default_pct",
                    "delta_mae_sel_vs_default", "rel_mae_sel_vs_default_pct"])
        for ds in X.ETT4:
            for h in X.HORIZONS:
                r = test[ds][h]
                dms = r["sel"][0] - r["default"][0]
                dma = r["sel"][1] - r["default"][1]
                w.writerow([ds, h, X.PAPER_P[ds], 1, r["sel_P"],
                            r["sel_src"],
                            "%.10f" % r["default"][0], "%.10f" % r["p1"][0],
                            "%.10f" % r["sel"][0],
                            "%.10f" % r["default"][1], "%.10f" % r["p1"][1],
                            "%.10f" % r["sel"][1],
                            "%+.10f" % dms, "%+.4f" % (100 * dms / r["default"][0]),
                            "%+.10f" % dma, "%+.4f" % (100 * dma / r["default"][1])])

    # ---- RESULT md ----
    L = ["# EXP-043: VisionTS periodic-P validation-set selection (ETT4)",
         "",
         "Paper A.2 claim under test: P is selected from the Table-5 "
         "candidates (hourly {24,168,1}, 15T {96,672,1}) on the validation "
         "set; Table 6 ships P=24/96.  The zero-shot lineage (EXP-001/030) "
         "never produced validation scores; this round computes them with "
         "the official Dataset_ETT_* classes (flag='val') under official "
         "test-loader semantics, per the frozen protocol.",
         "",
         "Split definition (explicit): month rows K = 720 (hourly) / 2880 "
         "(15T); train [0,12K); **val target [12K,16K)**; test target "
         "[16K,20K).  Standardization fits the train block only.  All "
         "windows enumerated (no shuffle, no tail-drop), predictions and "
         "targets never inverse-transformed (paper-normalized metrics).",
         ""]
    if os.path.isfile(X.VERIFY_JSON):
        v = X.load_json(X.VERIFY_JSON)
        s = v.get("summary", v)
        L.append("Gates: **%s** (VG3-VG13, %d checks, %d fail) -> %s"
                 % ("ALL PASS" if s.get("all_pass") else "FAIL",
                    s.get("n_checks", 0), s.get("n_fail", 0),
                    "logs/verify_t43.json"))
        L.append("")

    L.append("## S1 validation scores (48 configs; float64 SSE/cnt, "
             "unrounded)")
    L.append("")
    L.append("| dataset | P | H=96 | H=192 | H=336 | H=720 | equal-weight "
             "mean MSE |")
    L.append("|---|---|---|---|---|---|---|")
    for ds in X.ETT4:
        for P in X.CANDIDATE_P[ds]:
            cells = []
            ms = []
            for h in X.HORIZONS:
                j = X.load_json(X.vjson_path(ds, P, h))
                cells.append("%.6f" % j["mse_from_sse"])
                ms.append(j["mse_from_sse"])
            rec = sel["per_ds"][ds]["candidates"][str(P)]
            L.append("| %s | %d | %s | %s | %s | %s | %.10f%s |"
                     % (ds, P, cells[0], cells[1], cells[2], cells[3],
                        rec["mean_mse"],
                        "  **<- chosen**" if sel["per_ds"][ds]["chosen_P"] == P
                        else ""))
    L.append("")
    L.append("(per-H window counts and float64 detail: "
             "`logs/summary_t43_val.csv`; truth bitwise vs deterministic "
             "recompute on all 48 configs.)")

    L.append("")
    L.append("## S2 selection (ledger, generated BEFORE any test evaluation)")
    L.append("")
    L.append("Rule (supplementary setting, frozen in advance): per dataset "
             "argmin of the equal-weight mean of the 4 per-H validation "
             "MSEs; ties -> prefer the paper P, then the smaller P.")
    L.append("")
    L.append("| dataset | chosen P | paper P | agrees | tie-break | margin "
             "2nd-best − best (MSE) |")
    L.append("|---|---|---|---|---|---|")
    for ds in X.ETT4:
        p = sel["per_ds"][ds]
        L.append("| %s | %d | %d | %s | %s | %.3e |"
                 % (ds, p["chosen_P"], p["paper_P"],
                    "yes" if p["agrees_with_paper"] else "**no**",
                    p["tie_break"], p["margin_second_minus_best_mse"]))
    L.append("")
    L.append("Ledger: `logs/selection.json` + `val_table.csv` + `LEDGER_OK` "
             "(sha256 chain); every new test fingerprint embeds the ledger "
             "sha256; ordering acceptance->ledger->tests is gated "
             "(VG9/VG11).")

    L.append("")
    L.append("## S3 test comparison: selected-P vs default-P vs P=1")
    L.append("")
    L.append("Default-P arm reuses EXP-001 archives; P=1 arm reuses EXP-030 "
             "archives (both re-verified before use); the selected arm "
             "reuses an archive wherever it coincides with those P values, "
             "and otherwise runs new (ledger-bound) jobs.")
    L.append("")
    for ds in X.ETT4:
        L.append("### %s" % ds)
        L.append("")
        L.append("| H | default P=%d mse/mae | P=1 mse/mae | selected P=%d "
                 "mse/mae | Δmse sel−def | Δmae sel−def |"
                 % (X.PAPER_P[ds], sel["per_ds"][ds]["chosen_P"]))
        L.append("|---|---|---|---|---|---|")
        for h in X.HORIZONS:
            r = test[ds][h]
            dms = r["sel"][0] - r["default"][0]
            dma = r["sel"][1] - r["default"][1]
            L.append("| %d | %.4f / %.4f | %.4f / %.4f | %.4f / %.4f | "
                     "%+.4f | %+.4f |"
                     % (h, r["default"][0], r["default"][1],
                        r["p1"][0], r["p1"][1],
                        r["sel"][0], r["sel"][1], dms, dma))
        dm = np.mean([test[ds][h]["sel"][0] for h in X.HORIZONS]) \
            - np.mean([test[ds][h]["default"][0] for h in X.HORIZONS])
        dma = np.mean([test[ds][h]["sel"][1] for h in X.HORIZONS]) \
            - np.mean([test[ds][h]["default"][1] for h in X.HORIZONS])
        srcs = sorted(set(test[ds][h]["sel_src"] for h in X.HORIZONS))
        L.append("")
        L.append("dataset mean (4H equal-weight): Δmse %+.5f, Δmae %+.5f "
                 "(selected − default); selected-arm sources: %s."
                 % (dm, dma, ", ".join(srcs)))
        L.append("")

    # overall (16-group equal weight)
    ov = {}
    for arm in ("default", "p1", "sel"):
        ov[arm] = {
            "mse": float(np.mean([test[ds][h][arm][0]
                                  for ds in X.ETT4 for h in X.HORIZONS])),
            "mae": float(np.mean([test[ds][h][arm][1]
                                  for ds in X.ETT4 for h in X.HORIZONS]))}
    L.append("### Overall (16 (ds,H) groups, equal weight)")
    L.append("")
    L.append("| arm | MSE | MAE | ΔMSE vs default | ΔMAE vs default |")
    L.append("|---|---|---|---|---|")
    for arm, label in (("default", "default P (paper Table 6; EXP-001)"),
                       ("sel", "validation-selected P"),
                       ("p1", "P=1 (EXP-030)")):
        L.append("| %s | %.6f | %.6f | %+.6f | %+.6f |"
                 % (label, ov[arm]["mse"], ov[arm]["mae"],
                    ov[arm]["mse"] - ov["default"]["mse"],
                    ov[arm]["mae"] - ov["default"]["mae"]))
    L.append("")
    L.append("CSV: `logs/summary_t43_test.csv`.")

    L.append("")
    L.append("## S4 observations (report-only; no paper threshold is a gate)")
    L.append("")
    agree = [ds for ds in X.ETT4 if sel["per_ds"][ds]["agrees_with_paper"]]
    disagree = [ds for ds in X.ETT4 if not sel["per_ds"][ds]["agrees_with_paper"]]
    L.append("- Selection vs paper: agrees on %s; differs on %s.  Both "
             "outcomes are recorded honestly; the rule is a supplementary "
             "aggregation, so a difference is not a paper contradiction "
             "by itself." % (agree or "none", disagree or "none"))
    better = sum(1 for ds in X.ETT4 for h in X.HORIZONS
                 if test[ds][h]["sel"][0] < test[ds][h]["default"][0])
    L.append("- Selected-vs-default per (ds,H): test MSE improves in %d/16 "
             "groups, regresses in %d (ties count as regress)."
             % (better, 16 - better))
    L.append("- P=1 reference (EXP-030 anchor): overall MSE %.6f vs default "
             "%.6f (%+.2f%%) — consistent with EXP-030's frozen conclusion "
             "that P=1 is substantially worse; shown here only as the "
             "third arm of the comparison."
             % (ov["p1"]["mse"], ov["default"]["mse"],
                100 * (ov["p1"]["mse"] - ov["default"]["mse"])
                / ov["default"]["mse"]))
    L.append("- Paper-value deviations: EXP-001's own summary "
             "(`repro_zeroshot/summary.csv`, max |Δ| 0.0006 vs paper Table "
             "9) remains the frozen reference for the default arm; this "
             "round adds no new paper threshold and treats all paper "
             "comparisons as observations only.")

    L.append("")
    L.append("## S5 environment + provenance")
    L.append("")
    env = X.load_json(os.path.join(X.LOG_DIR, "env_t43.json"))
    L.append("- eta GPU: %s (torch %s, python %s); val/test loaders: "
             "batch_size 32, shuffle=False, drop_last=False, "
             "num_workers=10 (official test semantics)."
             % (env.get("gpu_name", "?"), env.get("torch", "?"),
                env.get("python", "?")))
    L.append("- Frozen weights: MAE-base ckpt sha256 %s (%d B)."
             % (X.CKPT_SHA256, X.CKPT_BYTES))
    L.append("- Framework md5s == EXP-030 lineage fingerprint (preflight "
             "P6); EXP-001/EXP-030 reuse re-verified per group (VG7/VG8).")
    L.append("- Val/test targets verified bitwise against deterministic "
             "recomputation from raw CSV + train-block StandardScaler "
             "([12K,16K) / [16K,20K) target regions, split fields recorded "
             "explicitly per config).")

    # ---- figures ----
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, ds in zip(axes, X.ETT4):
        Ps = X.CANDIDATE_P[ds]
        vals = [sel["per_ds"][ds]["candidates"][str(p)]["mean_mse"]
                for p in Ps]
        cols = ["#2a7" if p == sel["per_ds"][ds]["chosen_P"] else "#89c"
                for p in Ps]
        ax.bar([str(p) for p in Ps], vals, color=cols)
        ax.set_title("%s (chosen P=%d)" % (ds, sel["per_ds"][ds]["chosen_P"]))
        ax.set_xlabel("candidate P")
        ax.set_ylabel("equal-weight mean val MSE")
    fig.suptitle("EXP-043 validation: mean MSE per candidate P "
                 "(chosen = green)")
    fig.tight_layout()
    fig1 = os.path.join(X.LOG_DIR, "fig43_val_means.png")
    fig.savefig(fig1, dpi=130)

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, ds in zip(axes, X.ETT4):
        hs = X.HORIZONS
        for arm, lab, c in (("default", "default P", "#888"),
                            ("sel", "selected P", "#2a7"),
                            ("p1", "P=1", "#c66")):
            ax.plot(hs, [test[ds][h][arm][0] for h in hs],
                    marker="o", color=c, label=lab)
        ax.set_title(ds)
        ax.set_xlabel("H")
        ax.set_ylabel("test MSE")
        ax.legend(fontsize=8)
    fig.suptitle("EXP-043 test MSE: three arms")
    fig.tight_layout()
    fig2 = os.path.join(X.LOG_DIR, "fig43_test_arms.png")
    fig.savefig(fig2, dpi=130)

    # ---- self-check ----
    assert len(val_rows) == 48
    assert all(os.path.isfile(p) for p in (val_csv, test_csv, fig1, fig2))
    with open(X.RESULT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("[collect] wrote %s, %s, figures, %s"
          % (val_csv, test_csv, X.RESULT_MD))


if __name__ == "__main__":
    main()
