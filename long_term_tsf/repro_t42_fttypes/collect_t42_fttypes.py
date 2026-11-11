"""EXP-042 collector: logs/summary_t42.csv (144 rows) +
RESULT_t42_fttypes.md + fig_t42_fttypes.png, from archived evidence only
(this round's train_<DS>_<H>_<ft>_seed<s>.json, EXP-003 LN48 pins,
EXP-038/039 All/Rand archives, EXP-033 Freeze16).  Local-only.

Aggregation convention (per plan): per seed -> per (ft, ds, seed) 4H
equal mean -> per-ft per-seed overall -> cross-seed mean +/- std
(ddof=1).  Paired diffs vs LN at the same seed / same ds / 4H-mean
layer.  Paper Table 22 anchors are compared ONLY at the dataset-4H-mean
and overall layers; the ETTm2 Bias-vs-LN counterexample direction and
the 1st-place counts are reported as observations of the CURRENT
one-round recipe, separated from any attribution.
"""
import csv
import os

import numpy as np

import t42_common as X   # first: fixes sys.path for the t39/t38/t22 pkgs
import t38_common as C

METHODS = ["bias", "mlp", "attn", "ln", "all", "rand", "freeze"]
REF_METHODS = ["ln", "all", "rand"]      # per-seed archives
PAPER_1ST_COUNTS = {"ln": 7, "bias": 2, "freeze": 1}   # paper Table 22 (10 cells incl. its Average row)
PAPER_1ST_COUNTS_8 = {"ln": 5, "bias": 2, "freeze": 1}  # unified 8-cell subset (dataset rows only)


def fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), (float(arr.std(ddof=1))
                               if len(vals) > 1 else 0.0)


FIELDS = ["task", "ds", "h", "seed", "ft", "source", "mse", "mae",
          "train_pure_s", "vali_s", "test_s", "train_peak_mb",
          "train_batches", "ckpt_sha16",
          "ln_mse", "ln_mae", "d_mse_vs_ln", "d_mae_vs_ln"]


def build_rows():
    rows = []
    for ds, h in X.CONFIGS:
        for s in X.LN_KEYS_SEEDS:
            for ft in X.FT_TYPES:
                p = X.train_json(ds, h, s, ft)
                row = {"task": "t42_fttypes", "ds": ds, "h": h, "seed": s,
                       "ft": ft, "source": "missing"}
                if os.path.isfile(p):
                    rec = C.load_json(p)
                    row["source"] = "this run"
                    row["mse"] = rec["success"]["mse"]
                    row["mae"] = rec["success"]["mae"]
                    ph = rec.get("phases", {})
                    tw, vv = ph.get("train_wall"), ph.get("vali_wall")
                    row["train_pure_s"] = (tw - vv) \
                        if (tw is not None and vv is not None) else None
                    row["vali_s"] = vv
                    row["test_s"] = ph.get("test_wall")
                    row["train_peak_mb"] = ph.get("train_peak_mb")
                    row["train_batches"] = rec.get(
                        "sampler_len_check", {}).get("expected_train_batches")
                    row["ckpt_sha16"] = rec.get("final_ckpt_sha256",
                                                "?")[:16]
                else:
                    for k in ("mse", "mae", "train_pure_s", "vali_s",
                              "test_s", "train_peak_mb", "train_batches",
                              "ckpt_sha16"):
                        row[k] = None
                row["ln_mse"], row["ln_mae"] = \
                    X.LN48["%s_%d_seed%d" % (ds, h, s)]
                if row.get("mse") is not None:
                    row["d_mse_vs_ln"] = row["mse"] - row["ln_mse"]
                    row["d_mae_vs_ln"] = row["mae"] - row["ln_mae"]
                else:
                    row["d_mse_vs_ln"] = None
                    row["d_mae_vs_ln"] = None
                rows.append(row)
    return rows


def ref_layer():
    """refs[k][ds] = {"mean": (mse, mae), "std": (mse_std, mae_std),
    "per_seed": {s: (mse4h, mae4h)}} for ln/all/rand; freeze[ds]=(m,a)."""
    freeze16 = C.load_freeze16()
    refs = {}
    for k in REF_METHODS:
        refs[k] = {}
        for ds in C.ETT4:
            per_seed = {}
            for s in X.LN_KEYS_SEEDS:
                if k == "ln":
                    per_seed[s] = (
                        sum(X.LN48["%s_%d_seed%d" % (ds, h, s)][0]
                            for h in C.HORIZONS) / 4.0,
                        sum(X.LN48["%s_%d_seed%d" % (ds, h, s)][1]
                            for h in C.HORIZONS) / 4.0)
                else:
                    vm, va = [], []
                    for h in C.HORIZONS:
                        p = X.t38_train_json(ds, h, k) if s == 2021 else \
                            X.t39_train_json(ds, h, k, s)
                        rec = C.load_json(p)
                        vm.append(rec["success"]["mse"])
                        va.append(rec["success"]["mae"])
                    per_seed[s] = (sum(vm) / 4.0, sum(va) / 4.0)
            mm, msd = ms([v[0] for v in per_seed.values()])
            am, asd = ms([v[1] for v in per_seed.values()])
            refs[k][ds] = {"mean": (mm, am), "std": (msd, asd),
                           "per_seed": per_seed}
    refs["freeze"] = {ds: (sum(freeze16[(ds, h)][0] for h in C.HORIZONS) / 4.0,
                           sum(freeze16[(ds, h)][1] for h in C.HORIZONS) / 4.0)
                      for ds in C.ETT4}
    return refs


def ref_overall(refs):
    """refs overall: per-seed 4-dataset mean -> 3-seed mean +/- std
    (ddof=1) — the SAME pipeline as the t42 arms' agg_ov.  The overall
    std is the three-seed spread, never the cross-dataset spread."""
    out = {}
    for k in REF_METHODS:
        per_seed = {}
        for s in X.LN_KEYS_SEEDS:
            per_seed[s] = (
                sum(refs[k][ds]["per_seed"][s][0] for ds in C.ETT4) / 4.0,
                sum(refs[k][ds]["per_seed"][s][1] for ds in C.ETT4) / 4.0)
        mm, msd = ms([v[0] for v in per_seed.values()])
        am, asd = ms([v[1] for v in per_seed.values()])
        out[k] = {"per_seed": per_seed, "mse_mean": mm, "mse_std": msd,
                  "mae_mean": am, "mae_std": asd}
    return out


def check_exp039_consistency(ref_ov):
    """Hard gate: recomputed ln/all/rand overall (mean, std) must match
    the EXP-039 published T3 pins within EXP039_T3_TOL (5e-5 = the
    rounding step of the published 4-decimal values)."""
    recs = []
    for k in REF_METHODS:
        v = ref_ov[k]
        for met in ("mse", "mae"):
            m, sd = v["%s_mean" % met], v["%s_std" % met]
            pm, psd = X.EXP039_T3_OVERALL[k][met]
            dm, dsd = abs(m - pm), abs(sd - psd)
            ok = dm <= X.EXP039_T3_TOL and dsd <= X.EXP039_T3_TOL
            assert ok, \
                "EXP-039 T3 consistency FAILED: %s %s ours %.6f+-%.6f " \
                "vs published %.4f+-%.4f" % (k, met, m, sd, pm, psd)
            recs.append("%s %s ours %.6f+-%.6f vs t39 %.4f+-%.4f OK"
                        % (k, met, m, sd, pm, psd))
    return recs


def collect_layer(rows):
    """-> per_seed_ds[ft][s][ds], per_seed_ov[ft][s], agg_ds[ft][ds],
    agg_ov[ft]."""
    per_seed_ds = {ft: {s: {} for s in X.LN_KEYS_SEEDS}
                   for ft in X.FT_TYPES}
    for ft in X.FT_TYPES:
        for s in X.LN_KEYS_SEEDS:
            for ds in C.ETT4:
                cells = [r for r in rows if r["ft"] == ft and r["ds"] == ds
                         and r["seed"] == s and r["mse"] is not None]
                if len(cells) != 4:
                    continue
                per_seed_ds[ft][s][ds] = {
                    "mse": sum(r["mse"] for r in cells) / 4.0,
                    "mae": sum(r["mae"] for r in cells) / 4.0}
    per_seed_ov = {}
    for ft in X.FT_TYPES:
        per_seed_ov[ft] = {}
        for s in X.LN_KEYS_SEEDS:
            dsm = per_seed_ds[ft][s]
            if len(dsm) == 4:
                per_seed_ov[ft][s] = {
                    "mse": sum(dsm[ds]["mse"] for ds in C.ETT4) / 4.0,
                    "mae": sum(dsm[ds]["mae"] for ds in C.ETT4) / 4.0}
    agg_ds, agg_ov = {}, {}
    for ft in X.FT_TYPES:
        agg_ds[ft] = {}
        for ds in C.ETT4:
            mses = [per_seed_ds[ft][s][ds]["mse"] for s in X.LN_KEYS_SEEDS
                    if ds in per_seed_ds[ft][s]]
            maes = [per_seed_ds[ft][s][ds]["mae"] for s in X.LN_KEYS_SEEDS
                    if ds in per_seed_ds[ft][s]]
            mm, msd = ms(mses)
            am, asd = ms(maes)
            agg_ds[ft][ds] = {"mse_mean": mm, "mse_std": msd,
                              "mae_mean": am, "mae_std": asd,
                              "per_seed_mse": mses, "per_seed_mae": maes}
        mses = [per_seed_ov[ft][s]["mse"] for s in X.LN_KEYS_SEEDS
                if s in per_seed_ov[ft]]
        maes = [per_seed_ov[ft][s]["mae"] for s in X.LN_KEYS_SEEDS
                if s in per_seed_ov[ft]]
        mm, msd = ms(mses)
        am, asd = ms(maes)
        agg_ov[ft] = {"mse_mean": mm, "mse_std": msd, "mae_mean": am,
                      "mae_std": asd, "per_seed_mse": mses,
                      "per_seed_mae": maes}
    return per_seed_ds, per_seed_ov, agg_ds, agg_ov


def paired_vs_ln(rows, refs, per_seed_ov):
    """paired[ft][ds] (+ 'Overall'): per-seed (ft value − LN value at
    the same seed; 4H-mean layer per ds, 4-dataset mean for Overall)."""
    paired = {}
    for ft in X.FT_TYPES:
        paired[ft] = {}
        for ds in C.ETT4:
            per_seed = {}
            for s in X.LN_KEYS_SEEDS:
                cells = [r for r in rows if r["ft"] == ft and r["ds"] == ds
                         and r["seed"] == s and r["mse"] is not None]
                if len(cells) != 4:
                    continue
                fm = sum(r["mse"] for r in cells) / 4.0
                fa = sum(r["mae"] for r in cells) / 4.0
                per_seed[s] = {"mse": fm - refs["ln"][ds]["per_seed"][s][0],
                               "mae": fa - refs["ln"][ds]["per_seed"][s][1]}
            pm, psd = ms([v["mse"] for v in per_seed.values()])
            pam, pasd = ms([v["mae"] for v in per_seed.values()])
            n_better = sum(1 for v in per_seed.values() if v["mse"] < 0)
            paired[ft][ds] = {"per_seed": per_seed, "mse_mean": pm,
                              "mse_std": psd, "mae_mean": pam,
                              "mae_std": pasd,
                              "n_seeds_mse_better": n_better}
        if ft in per_seed_ov and len(per_seed_ov[ft]) == 3:
            per_seed = {}
            for s in X.LN_KEYS_SEEDS:
                ln_ov = (
                    sum(refs["ln"][ds]["per_seed"][s][0]
                        for ds in C.ETT4) / 4.0,
                    sum(refs["ln"][ds]["per_seed"][s][1]
                        for ds in C.ETT4) / 4.0)
                per_seed[s] = {
                    "mse": per_seed_ov[ft][s]["mse"] - ln_ov[0],
                    "mae": per_seed_ov[ft][s]["mae"] - ln_ov[1]}
            pm, psd = ms([v["mse"] for v in per_seed.values()])
            pam, pasd = ms([v["mae"] for v in per_seed.values()])
            paired[ft]["Overall"] = {
                "per_seed": per_seed, "mse_mean": pm, "mse_std": psd,
                "mae_mean": pam, "mae_std": pasd,
                "n_seeds_mse_better": sum(
                    1 for v in per_seed.values() if v["mse"] < 0)}
    return paired


def first_counts(agg_ds, refs):
    counts = {m: 0 for m in METHODS}
    table = {}
    for ds in C.ETT4:
        for met, idx in (("mse", 0), ("mae", 1)):
            vals = {}
            for ft in X.FT_TYPES:
                vals[ft] = agg_ds[ft][ds]["%s_mean" % met]
            for k in REF_METHODS:
                vals[k] = refs[k][ds]["mean"][idx]
            vals["freeze"] = refs["freeze"][ds][idx]
            best = min(vals, key=vals.get)
            counts[best] += 1
            table["%s_%s" % (ds, met)] = {"best": best,
                                          "vals": vals}
    return counts, table


def result_md(rows, refs, ref_ov, cons, per_seed_ds, per_seed_ov,
              agg_ds, agg_ov, paired, counts, first_table):
    L = []
    L.append("# EXP-042 RESULT: Table 22 remaining finetune types "
             "(Bias / MLP / Attention on the ORIGINAL MAE-base pretrained "
             "weights; LN = EXP-003 per-seed archives; All/Rand-VM = "
             "EXP-038 seed-2021 + EXP-039; Freeze = EXP-033)\n")
    L.append("Generated by collect_t42_fttypes.py from archived evidence "
             "only.  144 new trainings = 4 ETT datasets x H={96,192,336,"
             "720} x seeds {2021,2022,2023} x ft {bias,mlp,attn}; recipe "
             "= EXP-003 verbatim (bilinear, r=c=0.4, FP32, Adam 1e-4, "
             "bs 256, 1 epoch, full-val element-weighted MSE selection, "
             "strict best-ckpt reload then test) executed zero-change "
             "through repro_fullshot/run_train.py main().  Aggregation = "
             "per-seed 4H mean -> per-dataset (3-seed mean +/- std "
             "ddof=1) -> overall.  Acceptance = pipeline and numeric "
             "self-consistency; proximity to paper values is NOT an "
             "acceptance criterion.\n")
    L.append("**Scope disclosure (frozen):** finetune scope follows the "
             "official visionts/model.py:46-57 predicates audited at "
             "preflight (scope_audit_t42.json, 3-way verified).  The "
             "paper's C.3 text calls Bias `BitFit` (bias terms of linear "
             "layers); the official code's `bias` predicate is BROADER "
             "(all LayerNorm biases + patch_embed Conv2d bias + linear "
             "biases).  We adopt the official-code scope, record the "
             "difference, and neither widen nor narrow it.  All three "
             "arms start from the frozen ckpt bit-for-bit (VG3) and "
             "every frozen key stays bitwise unchanged after training "
             "(VG4).\n")

    # ---- T1 per-horizon -------------------------------------------------
    L.append("## T1: per-horizon results (48 (ds,H,seed) triples x 3 ft; "
             "LN per seed)\n")
    L.append("train* = train_wall - vali_wall (pure training; the official "
             "train() calls vali() inside the epoch loop).\n")
    by_key = {}
    for r in rows:
        by_key.setdefault((r["ds"], r["h"], r["seed"]), {})[r["ft"]] = r
    L.append("| ds | H | seed | Bias MSE/MAE | MLP MSE/MAE | Attn MSE/MAE "
             "| LN MSE/MAE | Bias train*/vali/test s |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, h in X.CONFIGS:
        for s in X.LN_KEYS_SEEDS:
            cell = by_key.get((ds, h, s), {})
            b = cell.get("bias", {})
            L.append("| %s | %d | %d | %s/%s | %s/%s | %s/%s | %s/%s | "
                     "%s/%s/%s |" %
                     (ds, h, s, fmt(b.get("mse")), fmt(b.get("mae")),
                      fmt(cell.get("mlp", {}).get("mse")),
                      fmt(cell.get("mlp", {}).get("mae")),
                      fmt(cell.get("attn", {}).get("mse")),
                      fmt(cell.get("attn", {}).get("mae")),
                      fmt(X.LN48["%s_%d_seed%d" % (ds, h, s)][0]),
                      fmt(X.LN48["%s_%d_seed%d" % (ds, h, s)][1]),
                      fmt(b.get("train_pure_s"), 1),
                      fmt(b.get("vali_s"), 1), fmt(b.get("test_s"), 1)))
    L.append("")

    # ---- T2 dataset layer ------------------------------------------------
    L.append("## T2: dataset layer (per-seed 4H mean, then 3-seed "
             "mean +/- std, ddof=1)\n")
    L.append("| ds | metric | Bias | MLP | Attn | LN | All | Rand-VM | "
             "Freeze |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ds in C.ETT4:
        for met in ("mse", "mae"):
            cells = []
            for ft in X.FT_TYPES:
                a = agg_ds[ft][ds]["%s_mean" % met]
                sd = agg_ds[ft][ds]["%s_std" % met]
                cells.append("%s+-%s" % (fmt(a), fmt(sd, 4))
                             if a is not None else "n/a")
            for k in REF_METHODS:
                m, sd = refs[k][ds]["mean"][0 if met == "mse" else 1], \
                    refs[k][ds]["std"][0 if met == "mse" else 1]
                cells.append("%s+-%s" % (fmt(m), fmt(sd, 4)))
            fz = refs["freeze"][ds][0 if met == "mse" else 1]
            cells.append(fmt(fz))
            L.append("| %s | %s | %s |" % (ds, met.upper(), " | ".join(cells)))
    L.append("")

    # ---- T3 overall -------------------------------------------------------
    L.append("## T3: overall (per-seed 4-dataset mean -> 3-seed mean +/- "
             "std, ddof=1; the std is the THREE-SEED spread for every "
             "method, including the LN/All/Rand-VM references)\n")
    L.append("| method | MSE | MAE |")
    L.append("|---|---|---|")
    for ft in X.FT_TYPES:
        v = agg_ov[ft]
        L.append("| %s | %s+-%s | %s+-%s |" %
                 (ft, fmt(v["mse_mean"]), fmt(v["mse_std"]),
                  fmt(v["mae_mean"]), fmt(v["mae_std"])))
    for k in REF_METHODS:
        v = ref_ov[k]
        L.append("| %s | %s+-%s | %s+-%s |" %
                 (k, fmt(v["mse_mean"]), fmt(v["mse_std"]),
                  fmt(v["mae_mean"]), fmt(v["mae_std"])))
    L.append("| freeze | %s | %s |" %
             (fmt(sum(refs["freeze"][ds][0] for ds in C.ETT4) / 4.0),
              fmt(sum(refs["freeze"][ds][1] for ds in C.ETT4) / 4.0)))
    L.append("")
    L.append("- EXP-039 T3 statistical consistency (hard): recomputed "
             "ln/all/rand overall from the same archives match "
             "RESULT_t39_ett.md L87-89 within %.0e: %s."
             % (X.EXP039_T3_TOL, "; ".join(cons)))

    # ---- T4 paired diffs vs LN -------------------------------------------
    L.append("## T4: paired diffs vs LN (same seed; 4H-mean layer per "
             "ds and 4-dataset mean for Overall; negative = the ft arm "
             "is BETTER)\n")
    L.append("| ft | ds | dMSE mean +/- std | dMAE mean +/- std | seeds "
             "with MSE better |")
    L.append("|---|---|---|---|---|")
    for ft in X.FT_TYPES:
        for ds in C.ETT4 + ["Overall"]:
            p = paired[ft].get(ds)
            if p is None:
                continue
            L.append("| %s | %s | %s+-%s | %s+-%s | %d/3 |" %
                     (ft, ds, fmt(p["mse_mean"]), fmt(p["mse_std"]),
                      fmt(p["mae_mean"]), fmt(p["mae_std"]),
                      p["n_seeds_mse_better"]))
    L.append("")

    # ---- T5 paper comparison ----------------------------------------------
    L.append("## T5: paper Table 22 comparison (anchors ONLY at dataset "
             "and overall layers; observation, no acceptance threshold)\n")
    L.append("| ds | metric | Bias (ours/paper) | MLP (ours/paper) | Attn "
             "(ours/paper) |")
    L.append("|---|---|---|---|---|")
    for ds in C.ETT4 + ["Overall"]:
        for met in ("mse", "mae"):
            cells = []
            for ft in X.FT_TYPES:
                if ds == "Overall":
                    ours = agg_ov[ft]["%s_mean" % met]
                    paper = X.PAPER_T42_OVERALL[ft][0 if met == "mse" else 1]
                else:
                    ours = agg_ds[ft][ds]["%s_mean" % met]
                    paper = X.PAPER_T42[ds][ft][0 if met == "mse" else 1]
                cells.append("%s / %s" % (fmt(ours), fmt(paper, 3)))
            L.append("| %s | %s | %s |" % (ds, met.upper(),
                                           " | ".join(cells)))
    L.append("")

    # ---- T6 ETTm2 counterexample -------------------------------------------
    L.append("## T6: ETTm2 Bias-vs-LN counterexample (paper: Bias "
             "0.256/0.318 < LN 0.261/0.319 — the only dataset where Bias "
             "beats LN)\n")
    b = agg_ds["bias"]["ETTm2"]
    ln_m = ms([refs["ln"]["ETTm2"]["per_seed"][s][0]
               for s in X.LN_KEYS_SEEDS])[0]
    ln_a = ms([refs["ln"]["ETTm2"]["per_seed"][s][1]
               for s in X.LN_KEYS_SEEDS])[0]
    mse_dir = "REPRODUCED" if b["mse_mean"] is not None and \
        b["mse_mean"] < ln_m else "NOT reproduced"
    mae_dir = "REPRODUCED" if b["mae_mean"] is not None and \
        b["mae_mean"] < ln_a else "NOT reproduced"
    L.append("- ours: Bias ETTm2 4H-mean MSE %s vs LN %s -> %s; MAE %s "
             "vs %s -> %s (per-seed Bias MSE: %s)"
             % (fmt(b["mse_mean"]), fmt(ln_m), mse_dir,
                fmt(b["mae_mean"]), fmt(ln_a), mae_dir,
                ", ".join("%.4f" % v for v in b["per_seed_mse"])))
    L.append("- observation of the CURRENT one-round recipe only; no "
             "attribution, no causal claim.\n")

    # ---- T7 1st-place counts ------------------------------------------------
    L.append("## T7: 1st-place counts over the 8 (ds x metric) cells, "
             "7 methods (paper Table 22 counts over its 10 cells = 8 "
             "dataset cells + the Average row x {MSE, MAE}; counts "
             "unified here to the same 8-cell subset)\n")
    L.append("| cell (ds/metric) | best method | best value |")
    L.append("|---|---|---|")
    for cell, v in sorted(first_table.items()):
        L.append("| %s | %s | %s |" % (cell, v["best"],
                                       fmt(min(v["vals"].values()))))
    L.append("")
    L.append("- ours (8 cells): %s" % ", ".join("%s %d" % (m, c)
                                                for m, c in counts.items()
                                                if c > 0))
    L.append("- paper unified to the same 8 cells (its Average-row wins "
             "removed from LN 7/Bias 2/Freeze 1): %s; paper full "
             "10-cell: %s — observation only, different column set "
             "(no Rand-VM in the paper's count)."
             % (", ".join("%s %d" % (m, c)
                          for m, c in PAPER_1ST_COUNTS_8.items() if c),
                ", ".join("%s %d" % (m, c)
                          for m, c in PAPER_1ST_COUNTS.items() if c)))

    # ---- budget / env ---------------------------------------------------------
    L.append("## T8: budget + environment\n")
    pure = [r["train_pure_s"] for r in rows if r["train_pure_s"]]
    wall = [r for r in rows if r["mse"] is not None]
    L.append("- completed tasks: %d/144; pure-train sum %s s; mean per "
             "task %s s" % (len(wall), fmt(sum(pure), 1) if pure else "n/a",
                            fmt(sum(pure) / len(pure), 1) if pure else "n/a"))
    envs = set()
    for ds, h in X.CONFIGS:
        for s in X.LN_KEYS_SEEDS:
            for ft in X.FT_TYPES:
                p = X.train_json(ds, h, s, ft)
                if os.path.isfile(p):
                    envs.add(C.load_json(p)["env"]["torch"])
    L.append("- torch versions seen: %s; timing basis: per-task phases "
             "in train jsons (train_wall - vali_wall = pure train)."
             % ", ".join(sorted(envs)) if envs else "- no envs recorded")
    return L


def main():
    os.makedirs(X.LOG_DIR, exist_ok=True)
    rows = build_rows()
    n_done = sum(1 for r in rows if r["mse"] is not None)
    assert n_done == len(rows) == 144, \
        "collect requires ALL 144 tasks (%d done)" % n_done
    with open(X.SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    refs = ref_layer()
    ref_ov = ref_overall(refs)
    cons = check_exp039_consistency(ref_ov)
    per_seed_ds, per_seed_ov, agg_ds, agg_ov = collect_layer(rows)
    paired = paired_vs_ln(rows, refs, per_seed_ov)
    counts, first_table = first_counts(agg_ds, refs)
    L = result_md(rows, refs, ref_ov, cons, per_seed_ds, per_seed_ov,
                  agg_ds, agg_ov, paired, counts, first_table)
    with open(X.RESULT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    make_figure(agg_ds, agg_ov, refs, ref_ov)
    print("[t42-collect] wrote %s (%d rows) and %s" %
          (X.SUMMARY_CSV, len(rows), X.RESULT_MD))
    for r in cons:
        print("[t42-collect] EXP-039 consistency: %s" % r)
    for ft in X.FT_TYPES:
        v = agg_ov[ft]
        print("[t42-collect] overall %s: %.4f+-%.4f / %.4f+-%.4f" %
              (ft, v["mse_mean"], v["mse_std"], v["mae_mean"],
               v["mae_std"]))


def make_figure(agg_ds, agg_ov, refs, ref_ov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    panels = [(ds, agg_ds) for ds in C.ETT4] + [("Overall", agg_ov)]
    for ax, (title, src) in zip(axes, panels):
        names, vals, errs = [], [], []
        for m in METHODS:
            if title == "Overall":
                if m in X.FT_TYPES:
                    v = src[m]["mse_mean"]
                    e = src[m]["mse_std"]
                elif m in REF_METHODS:
                    v, e = ref_ov[m]["mse_mean"], ref_ov[m]["mse_std"]
                else:
                    v = sum(refs["freeze"][ds][0] for ds in C.ETT4) / 4.0
                    e = 0.0
            else:
                if m in X.FT_TYPES:
                    v = src[m][title]["mse_mean"]
                    e = src[m][title]["mse_std"]
                elif m in REF_METHODS:
                    v, e = refs[m][title]["mean"][0], \
                        refs[m][title]["std"][0]
                else:
                    v, e = refs["freeze"][title][0], 0.0
            names.append(m)
            vals.append(v)
            errs.append(e)
        colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd",
                  "#8c564b", "#7f7f7f"]
        bars = ax.bar(names, vals, yerr=errs, color=colors, capsize=3)
        ax.set_title("%s (MSE, 3-seed mean+-std)" % title, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    "%.3f" % v, ha="center", va="bottom", fontsize=7)
        if title == "ETTm2":
            ax.annotate("paper counterexample:\nBias < LN",
                        xy=(0.5, 0.02), xycoords="axes fraction",
                        fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(X.FIG_PNG, dpi=150)
    print("[t42-collect] wrote %s" % X.FIG_PNG)


if __name__ == "__main__":
    main()
