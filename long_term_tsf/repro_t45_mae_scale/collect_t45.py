"""EXP-045 collector: logs/summary_t45.csv (48 rows) + logs/means_t45.json
+ 3 figures + RESULT_t45_mae_scale.md.

Reporting discipline (frozen):
  - paper Table 17 anchors are 3-decimal, ETT cells only (L448-450);
    per-cell / per-ds comparisons are REPORT-ONLY (never a gate);
  - the table's bottom-row six-dataset overall numbers are deliberately
    NOT quoted anywhere (review correction #5: OCR-misalignment risk,
    e.g. 0.34617 vs Table 2 Huge overall ~0.315); the ETT-only overall
    is SELF-COMPUTED (per-ds 4H equal mean -> 4-ds equal mean);
  - hard gate: the self-computed Base overall must equal the EXP-018
    pinned anchor 0.34439535439014435/0.37042006477713585 (<=1e-6) --
    it is the denominator of every Delta;
  - cost layer (a) wall_time_s: Base reuse rows carry EXP-001 historical
    values from the OLD environment -- reference only, NEVER compared
    against this round's Large/Huge wall times (correction in plan);
    the fair 3-way comparison is cost layer (b), timing_t45.json
    (same batch, same bs, same precision, same GPU, CUDA-synced);
  - observations preserve local reversals; no preset direction
    ("bigger is better/worse") and no attribution to parameter count or
    image overfitting is claimed.
"""
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t45_common as X

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULT_MD = os.path.join(X.PKG, "RESULT_t45_mae_scale.md")

ARCH_COL = {"mae_base": "#2a7", "mae_large": "#468", "mae_huge": "#a46"}


def wall_time_of(d):
    sp = os.path.join(d, "SUCCESS.txt")
    if not os.path.isfile(sp):
        return None
    for ln in X.read_fingerprint(sp):
        m = re.match(r"^wall_time_s=(\d+)$", ln)
        if m:
            return int(m.group(1))
    return None


def geometry_ok_of(arch, ds, h):
    if arch == "mae_base":
        return "archived(exp001)"
    srp = os.path.join(X.run_dir(ds, h, arch), "shape_record.jsonl")
    if not os.path.isfile(srp):
        return "missing"
    recs = [json.loads(ln) for ln in open(srp) if '"update_config"' in ln]
    if not recs:
        return "missing"
    ok, _ = X.geometry_matches(
        recs[-1], X.expected_geometry(ds, h, X.PAPER_P[ds], X.PAPER_L[ds],
                                      arch))
    return "ok" if ok else "MISMATCH"


def main():
    os.makedirs(X.LOG_DIR, exist_ok=True)

    timing_peak = {}
    tp = os.path.join(X.LOG_DIR, "timing_t45.json")
    if os.path.isfile(tp):
        for r in X.load_json(tp)["rows"]:
            timing_peak[(r["ds"], r["h"], r["arch"])] = r["peak_gb"]

    rows, means = [], {a: [] for a in X.ARCHS}
    for ds in X.ETT4:
        for h in X.HORIZONS:
            for arch in X.ARCHS:
                if arch == "mae_base":
                    d, source = X.base_dir(ds, h), "exp001_reused"
                else:
                    d, source = X.run_dir(ds, h, arch), "t45_%s" \
                        % X.ARCH_SHORT[arch]
                rdir = os.path.join(d, "results", "_")
                mets = np.load(os.path.join(rdir, "metrics.npy"))
                p = np.load(os.path.join(rdir, "pred.npy"))
                t = np.load(os.path.join(rdir, "true.npy"))
                p64, t64 = p.astype(np.float64), t.astype(np.float64)
                del p, t
                mse_r = float(np.mean((p64 - t64) ** 2))
                mae_r = float(np.mean(np.abs(p64 - t64)))
                del p64, t64
                wt = wall_time_of(d)
                rows.append([
                    ds, h, arch, source, os.path.relpath(d, X.ROOT),
                    "%.10f" % float(mets[1]), "%.10f" % float(mets[0]),
                    "%.10f" % mse_r, "%.10f" % mae_r,
                    "%.2e" % abs(mse_r - float(mets[1])),
                    "%.2e" % abs(mae_r - float(mets[0])),
                    X.expected_n_windows(ds, h),
                    geometry_ok_of(arch, ds, h),
                    "0.4(%s)" % ("archived" if arch == "mae_base"
                                 else "recorded"),
                    wt if wt is not None else "",
                    ("exp001_historical(old-env, reference only)"
                     if arch == "mae_base" else "t45_this_round"),
                    "%.4f" % timing_peak[(ds, h, arch)]
                    if (ds, h, arch) in timing_peak else ""])
                means[arch].append([ds, h, float(mets[1]), float(mets[0])])

    path = os.path.join(X.LOG_DIR, "summary_t45.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ds", "H", "arch", "source", "save_dir",
                    "metrics_mse", "metrics_mae",
                    "recompute_mse_f64", "recompute_mae_f64",
                    "diff_mse", "diff_mae", "n_windows", "geometry_ok",
                    "norm_const", "wall_time_s", "wall_time_source",
                    "timing_peak_gb"])
        w.writerows(rows)
    print("[collect] wrote %s (%d rows)" % (path, len(rows)))
    assert len(rows) == 48

    X.dump_json(os.path.join(X.LOG_DIR, "means_t45.json"), means)

    # ---- three-layer aggregation (unrounded) ----
    def ds_avg_of(arch):
        out = {}
        for ds, h, mse, mae in means[arch]:
            out.setdefault(ds, []).append((mse, mae))
        return {ds: (float(np.mean([m for m, _ in v])),
                     float(np.mean([a for _, a in v])))
                for ds, v in out.items()}

    ds_avg = {a: ds_avg_of(a) for a in X.ARCHS}
    overall = {a: (float(np.mean([ds_avg[a][ds][0] for ds in X.ETT4])),
                   float(np.mean([ds_avg[a][ds][1] for ds in X.ETT4])))
               for a in X.ARCHS}

    # ---- HARD gate: Base overall anchor (denominator of every Delta) ----
    dmse = abs(overall["mae_base"][0] - X.BASE_OVERALL["overall_mse"])
    dmae = abs(overall["mae_base"][1] - X.BASE_OVERALL["overall_mae"])
    if dmse > X.SELF_TOL or dmae > X.SELF_TOL:
        raise SystemExit(
            "[collect] Base overall self-check FAIL: %.17g/%.17g vs pinned "
            "%.17g/%.17g (tol %.0e) -- the denominator of every Delta is "
            "wrong, stop and locate the cause"
            % (overall["mae_base"][0], overall["mae_base"][1],
               X.BASE_OVERALL["overall_mse"], X.BASE_OVERALL["overall_mae"],
               X.SELF_TOL))
    print("[collect] Base overall anchor OK: %.17g / %.17g"
          % (overall["mae_base"][0], overall["mae_base"][1]))

    # ---- figures: accuracy ----
    for metric, mi, figname in (("MSE", 0, "fig45_accuracy_mse.png"),
                                ("MAE", 1, "fig45_accuracy_mae.png")):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        for ax, ds in zip(axes.ravel(), X.ETT4):
            for arch in X.ARCHS:
                pts = sorted((h, m[mi]) for d, h, m, a in means[arch]
                             if d == ds)
                xs = [str(p[0]) for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, "-o", color=ARCH_COL[arch], ms=4,
                        label=X.ARCH_SHORT[arch])
            ax.set_title("%s (%s)" % (ds, metric))
            ax.set_xlabel("prediction length H")
            ax.set_ylabel("test %s" % metric)
            ax.grid(alpha=0.3)
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle("EXP-045 accuracy: test %s by MAE variant (zero-shot, "
                     "frozen EXP-001 protocol)" % metric)
        fig.tight_layout()
        fig.savefig(os.path.join(X.LOG_DIR, figname), dpi=130)

    # ---- figure: cost ----
    if os.path.isfile(tp):
        tj = X.load_json(tp)
        trows = tj["rows"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for arch in X.ARCHS:
            vs = [r["mean_s"] for r in trows if r["arch"] == arch]
            ax1.bar(X.ARCH_SHORT[arch], float(np.mean(vs)),
                    yerr=float(np.std(vs)), color=ARCH_COL[arch])
            vs2 = [r["peak_gb"] for r in trows if r["arch"] == arch]
            ax2.bar(X.ARCH_SHORT[arch], float(np.mean(vs2)),
                    yerr=float(np.std(vs2)), color=ARCH_COL[arch])
        ax1.set_ylabel("mean forward wall s (bs=32, CUDA-synced)")
        ax1.set_title("per-forward inference time (mean±std over 16 cfgs)")
        ax2.set_ylabel("peak allocated GB")
        ax2.set_title("peak GPU memory (torch allocator)")
        fig.suptitle("EXP-045 cost layer (b): same batch, same bs=32, "
                     "fp32, eval()+inference_mode, sequential release")
        fig.tight_layout()
        fig.savefig(os.path.join(X.LOG_DIR, "fig45_cost.png"), dpi=130)

    # ---- RESULT md ----
    L = ["# EXP-045: VisionTS MAE-Base / Large / Huge zero-shot comparison "
         "(accuracy + cost), ETT4", "",
         "Question: reproduce paper Table 2 / Table 17 (Appendix B.7) MAE "
         "variant comparison on the four ETT datasets, against the frozen "
         "EXP-001 Base archives; report accuracy (MSE/MAE, three layers) "
         "AND cost (full-run wall time + same-batch CUDA-synced "
         "microbenchmark + peak memory).",
         "",
         "Anchor provenance: %s" % X.PAPER_T17_NOTE,
         "",
         "Fixed settings (frozen): eta single GPU; frozen MAE reconstruction "
         "ckpts (visualize), inference only; ONLY --vm_arch changes "
         "(mae_base/mae_large/mae_huge -> patch 16/16/14, patch grid "
         "14x14/14x14/16x16, strict full state-dict load); L/P = paper "
         "table 8/6 (2880/24, 1728/24, 2304/96, 4032/96); r=c=0.4; "
         "bilinear; original orientation; features=M; bs=32; seed=2021; "
         "train_epochs=0; no inverse (paper-normalized metrics); Base 16 "
         "groups are EXP-001 archives (read-only reuse); 32 new groups "
         "(Large 16 + Huge 16).  NO tuning from test results.",
         ""]
    if os.path.isfile(os.path.join(X.LOG_DIR, "verify_t45_all.json")):
        v = X.load_json(os.path.join(X.LOG_DIR, "verify_t45_all.json"))
        L.append("Gates (pre-collect pass): **%s** (MG1-MG10 + MG11a "
                 "coverage, %d checks, %d fail) -> logs/verify_t45_all.json"
                 % ("ALL PASS" if v.get("all_pass") else "FAIL",
                    v.get("n_checks", 0), v.get("n_fail", 0)))
        L.append("")
    if os.path.isfile(os.path.join(X.LOG_DIR, "verify_t45_mg11.json")):
        v = X.load_json(os.path.join(X.LOG_DIR, "verify_t45_mg11.json"))
        L.append("MG11 (final, post-timing/post-collect): **%s** (%d checks,"
                 " %d fail) -> logs/verify_t45_mg11.json"
                 % ("ALL PASS" if v.get("all_pass") else "FAIL",
                    v.get("n_checks", 0), v.get("n_fail", 0)))
        L.append("")

    # layer 1: per-(ds,H) with Delta vs Base and paper anchors
    L.append("## Layer 1: per (dataset, H)")
    L.append("")
    for ds in X.ETT4:
        L.append("### %s" % ds)
        L.append("")
        L.append("| H | arch | MSE | MAE | dMSE vs Base | dMAE vs Base | "
                 "paper MSE | dMSE vs paper | paper MAE | dMAE vs paper |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for h in X.HORIZONS:
            vals = {}
            for a in X.ARCHS:
                for d, hh, mse, mae in means[a]:
                    if d == ds and hh == h:
                        vals[a] = (mse, mae)
            pb = vals["mae_base"]
            for a in X.ARCHS:
                v = vals[a]
                pm, pa = X.PAPER_T17[ds][h][a]
                L.append("| %d | %s | %.4f | %.4f | %+.4f | %+.4f | %.3f | "
                         "%+.4f | %.3f | %+.4f |"
                         % (h, X.ARCH_SHORT[a], v[0], v[1],
                            v[0] - pb[0], v[1] - pb[1],
                            pm, v[0] - pm, pa, v[1] - pa))
        L.append("")

    # layer 2: per-ds averages
    L.append("## Layer 2: per-dataset mean (4H equal weight)")
    L.append("")
    L.append("| dataset | arch | MSE | MAE | dMSE vs Base | dMAE vs Base | "
             "paper avg MSE | paper avg MAE |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds in X.ETT4:
        for a in X.ARCHS:
            v = ds_avg[a][ds]
            pm, pa = X.PAPER_T17_DS_AVG[ds][a]
            L.append("| %s | %s | %.4f | %.4f | %+.4f | %+.4f | %.3f | %.3f |"
                     % (ds, X.ARCH_SHORT[a], v[0], v[1],
                        v[0] - ds_avg["mae_base"][ds][0],
                        v[1] - ds_avg["mae_base"][ds][1], pm, pa))
    L.append("")

    # layer 3: ETT-only overall (self-computed)
    L.append("## Layer 3: ETT-only overall (self-computed: per-ds 4H equal "
             "weight -> 4-ds equal weight)")
    L.append("")
    L.append("| arch | overall MSE | overall MAE | dMSE vs Base | dMAE vs "
             "Base |")
    L.append("|---|---|---|---|---|")
    for a in X.ARCHS:
        v = overall[a]
        L.append("| %s (%s) | %.8f | %.8f | %+.8f | %+.8f |"
                 % (X.ARCH_SHORT[a],
                    {112: "112M", 330: "330M", 657: "657M"}[
                        X.ARCH_FACTS[a]["paper_params_m"]],
                    v[0], v[1],
                    v[0] - overall["mae_base"][0],
                    v[1] - overall["mae_base"][1]))
    L.append("")
    L.append("Base overall self-check vs EXP-018 pinned anchor: %.17g / "
             "%.17g (tol %.0e) -- PASS (hard gate in collect; re-checked in"
             " MG11)." % (overall["mae_base"][0], overall["mae_base"][1],
                          X.SELF_TOL))
    L.append("")
    L.append("NOTE (correction #5): the paper table's bottom-row six-dataset"
             " overall numbers are NOT quoted or compared anywhere (OCR "
             "misalignment risk, e.g. 0.34617 vs Table 2 Huge overall "
             "~0.315); the ETT-only overall above is self-computed.")
    L.append("")

    # observations (report-only; local reversals preserved)
    L.append("## Observations (report-only; no preset direction, no "
             "attribution)")
    L.append("")
    for ds in X.ETT4:
        cells_lb, cells_hb = 0, 0
        for h in X.HORIZONS:
            get = lambda a: next((m for m in means[a]
                                  if m[0] == ds and m[1] == h), None)
            lb = get("mae_large")[2] - get("mae_base")[2]
            hb = get("mae_huge")[2] - get("mae_base")[2]
            cells_lb += lb < 0
            cells_hb += hb < 0
        L.append("- **%s**: Large beats Base on %d/4 H (MSE); Huge beats "
                 "Base on %d/4 H (MSE); per-cell values and reversals in "
                 "Layer 1." % (ds, cells_lb, cells_hb))
    L.append("- Local reversals are preserved as-is (see Layer 1); no "
             "'bigger is better/worse' prior is applied and no difference "
             "is attributed to parameter count or image-domain "
             "overfitting.")
    L.append("- %s" % X.PAPER_T17_NOTE)
    L.append("")

    # cost section
    L.append("## Cost")
    L.append("")
    L.append("Layer (a) full-run wall_time_s (per-group SUCCESS.txt; "
             "includes data loading + inference + saving): per-config "
             "values in summary_t45.csv.  Base reuse rows carry EXP-001 "
             "HISTORICAL values from the old environment -- reference "
             "only, NEVER compared against this round's Large/Huge wall "
             "times.  This round's Large/Huge groups are directly "
             "comparable among themselves (same env, sequential, same "
             "GPU).")
    L.append("")
    if os.path.isfile(tp):
        tj = X.load_json(tp)
        meta = tj["meta"]
        L.append("Layer (b) microbenchmark (fair 3-way comparison): "
                 "logs/timing_t45.json + timing_t45.csv + fig45_cost.png.  "
                 "Protocol: the SAME test batch (first batch of the "
                 "official test loader, bs=32, identical tensor) per "
                 "(ds,H) for all three archs; fp32; eval() + "
                 "torch.inference_mode(); warmup %d; %d timed repeats "
                 "around torch.cuda.synchronize(); peak memory from "
                 "max_memory_allocated (reset per arch); archs load and "
                 "run SEQUENTIALLY with del + empty_cache between (no "
                 "grad tracking, no cross-model residue); timed region "
                 "contains ONLY the forward on the GPU-resident batch -- "
                 "no loader IO / model load / saving (model load timed "
                 "separately as load_s)." % (meta["warmup"],
                                             meta["repeats"]))
        L.append("")
        L.append("| arch | mean forward s (avg over 16 cfgs) | mean peak GB |"
                 " |")
        L.append("|---|---|---|---|")
        for a in X.ARCHS:
            vs = [r["mean_s"] for r in tj["rows"] if r["arch"] == a]
            vs2 = [r["peak_gb"] for r in tj["rows"] if r["arch"] == a]
            L.append("| %s | %.4f (std %.4f) | %.2f |"
                     % (X.ARCH_SHORT[a], float(np.mean(vs)),
                        float(np.std(vs)), float(np.mean(vs2))))
        L.append("")
    L.append("Per-config detail: `logs/summary_t45.csv` (source column "
             "marks the 16 exp001_reused groups); accuracy figures "
             "`logs/fig45_accuracy_mse.png` / `_mae.png`; cost figure "
             "`logs/fig45_cost.png`.")
    L.append("")

    L.append("## Environment + provenance")
    L.append("")
    env = X.load_json(os.path.join(X.LOG_DIR, "env_t45.json"))
    L.append("- eta GPU: %s (torch %s, python %s); loaders: batch_size 32,"
             " shuffle=False, drop_last=False, num_workers=10 (official"
             " test semantics); every run went through run.py verbatim via"
             " infer_one_t45.py observation wrapper (framework files"
             " untouched)."
             % (env.get("gpu_name", "?"), env.get("torch", "?"),
                env.get("python", "?")))
    pf = X.load_json(os.path.join(X.LOG_DIR, "preflight_t45.json"))
    p2 = pf["checks"].get("P2_ckpts", {})
    p2b = pf["checks"].get("P2b_strict_load_params", {})
    for a in X.ARCHS:
        v, vb = p2.get(a, {}), p2b.get(a, {})
        L.append("- %s ckpt: %s sha256 %s (%d B); strict load OK; params "
                 "%d vs paper %dM (rel %.4f%%)."
                 % (X.ARCH_SHORT[a], X.ARCH_FACTS[a]["ckpt"],
                    str(v.get("sha256", "?"))[:16] + "...",
                    v.get("bytes", -1), vb.get("params", -1),
                    X.ARCH_FACTS[a]["paper_params_m"],
                    100 * vb.get("rel_diff", -1)))
    L.append("- Checkpoint provenance (correction #4): official URL %s; "
             "TLS STATUS AT DOWNLOAD: the CDN leaf certificate "
             "*.fbaipublicfiles.com (DigiCert Global G2 TLS RSA SHA256 "
             "2020 CA1) EXPIRED 2026-11-11, two days before this round's "
             "download; curl/OpenSSL and wget/GnuTLS both correctly "
             "refused, so no CA-verifiable fetch was possible; per user "
             "decision (2026-11-13) the files were fetched with "
             "certificate verification SKIPPED from the official URL and "
             "bound by (i) measured sha256+bytes hard-gated in preflight,"
             " (ii) strict full state-dict load on the official entry "
             "path, (iii) parameter count vs the paper Table 17 header "
             "(112M/330M/657M, +-2%%).  No substitute weights were used."
             % X.MAE_URL)
    L.append("- Framework md5s == EXP-030 lineage fingerprint (MG1/"
             "preflight P5); truth arrays bitwise-verified against "
             "deterministic recomputation for every config (MG4; EXP-001 "
             "arrays cross-validate the pipeline, MG7).")

    with open(RESULT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("[collect] wrote %s" % RESULT_MD)

    # self-check: means recompute from per-config values <=1e-12
    for arch in X.ARCHS:
        by = {}
        for ds, h, mse, mae in means[arch]:
            by.setdefault(ds, []).append((mse, mae))
        for ds, lst in by.items():
            v = ds_avg[arch][ds]
            assert abs(v[0] - sum(m for m, _ in lst) / 4) <= 1e-12
            assert abs(v[1] - sum(a for _, a in lst) / 4) <= 1e-12
    print("[collect] self-check: ds means recompute <=1e-12 OK; 48 rows")


if __name__ == "__main__":
    main()
