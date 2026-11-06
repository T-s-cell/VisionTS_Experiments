"""EXP-039 collector: logs/summary_t39_ett.csv + RESULT_t39_ett.md +
fig_t39_seed_effect.png, from archived evidence only (this round's
train_<DS>_<H>_<arm>_seed<s>.json, the frozen EXP-038 seed-2021 train
jsons incl. the EXP-037 reuse pair, EXP-003 LN48 pins, EXP-033 Freeze16
via summary_t22.csv, verify_t39.json section fullshot). Local-only;
--selfcheck exercises output paths with fabricated records.

Aggregation convention (per plan): per seed -> per (ds, seed) 4H equal
mean -> per-seed overall -> cross-seed mean +/- std (ddof=1). Paper
Table 22 is compared only at the dataset-4H-mean and overall layers.
"""
import argparse
import csv
import math
import os

import t39_common as X   # first: fixes sys.path for the t38/t22 packages
import t38_common as C


def fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def ms(vals):
    """(mean, std ddof=1) over a list of floats, or None."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    import numpy as np
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) \
        if len(vals) > 1 else (float(arr.mean()), 0.0)


def arm_rec(ds, h, arm, s):
    if s == 2021:
        p = X.t38_train_json(ds, h, arm)
        if os.path.isfile(p):
            return C.load_json(p), "EXP-038 seed2021 archive"
        return None, "t38 archive missing"
    p = X.train_json(ds, h, arm, s)
    if os.path.isfile(p):
        return C.load_json(p), "this run"
    return None, "missing"


FIELDS = ["task", "ds", "h", "seed",
          "all_source", "all_mse", "all_mae", "all_train_s",
          "all_train_pure_s", "all_vali_s",
          "all_test_s", "all_train_peak_mb", "all_train_batches",
          "all_ckpt_sha16",
          "rand_source", "rand_mse", "rand_mae", "rand_train_s",
          "rand_train_pure_s", "rand_vali_s",
          "rand_test_s", "rand_train_peak_mb",
          "rand_train_batches", "rand_ckpt_sha16",
          "ln_mse", "ln_mae", "ln_source",
          "freeze_mse", "freeze_mae", "freeze_source"]


def build_rows():
    freeze16 = C.load_freeze16()
    rows = []
    for ds, h in X.CONFIGS:
        for s in X.LN_KEYS_SEEDS:
            row = {"task": "t22_3seeds", "ds": ds, "h": h, "seed": s}
            fz = freeze16.get((ds, h))
            row["freeze_mse"] = fz[0] if fz else None
            row["freeze_mae"] = fz[1] if fz else None
            row["freeze_source"] = fz[2] if fz else "missing"
            row["ln_mse"], row["ln_mae"] = \
                X.LN48["%s_%d_seed%d" % (ds, h, s)]
            row["ln_source"] = "EXP-003 seed%d archive (pinned)" % s
            for arm in C.B_ARMS:
                rec, src = arm_rec(ds, h, arm, s)
                row["%s_source" % arm] = src
                if rec:
                    row["%s_mse" % arm] = rec["success"]["mse"]
                    row["%s_mae" % arm] = rec["success"]["mae"]
                    ph = rec.get("phases", {})
                    row["%s_train_s" % arm] = ph.get("train_wall")
                    row["%s_vali_s" % arm] = ph.get("vali_wall")
                    row["%s_train_pure_s" % arm] = train_pure(row, arm)
                    tw, vv = row["%s_train_s" % arm], row["%s_vali_s" % arm]
                    if row["%s_train_pure_s" % arm] is not None:
                        assert abs(row["%s_train_pure_s" % arm]
                                   - (tw - vv)) <= 1e-9, (ds, h, arm, s)
                    row["%s_test_s" % arm] = ph.get("test_wall")
                    row["%s_train_peak_mb" % arm] = ph.get("train_peak_mb")
                    row["%s_train_batches" % arm] = \
                        rec.get("sampler_len_check", {}).get(
                            "expected_train_batches",
                            rec.get("expected_train_batches"))
                    row["%s_ckpt_sha16" % arm] = \
                        rec["final_ckpt_sha256"][:16]
                else:
                    for k in ("mse", "mae", "train_s", "train_pure_s",
                              "vali_s", "test_s",
                              "train_peak_mb", "train_batches",
                              "ckpt_sha16"):
                        row["%s_%s" % (arm, k)] = None
            rows.append(row)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def train_pure(r, arm):
    w, v = r.get("%s_train_s" % arm), r.get("%s_vali_s" % arm)
    if w is None or v is None:
        return None
    return w - v


def collect_layer(rows):
    """-> per_seed_ds[s][ds][col], per_seed_ov[s][col], agg_ds, agg_ov,
    paired diffs. Uses ONLY row values (archived evidence)."""
    per_seed_ds = {s: {} for s in X.LN_KEYS_SEEDS}
    for s in X.LN_KEYS_SEEDS:
        for ds in C.ETT4:
            cells = [r for r in rows if r["ds"] == ds and r["seed"] == s]
            if len(cells) != 4:
                continue
            out = {}
            for k in ("all", "rand", "ln", "freeze"):
                out[k] = {
                    "mse": sum(r["%s_mse" % k] for r in cells) / 4.0,
                    "mae": sum(r["%s_mae" % k] for r in cells) / 4.0}
            per_seed_ds[s][ds] = out
    per_seed_ov = {}
    for s, dsm in per_seed_ds.items():
        if len(dsm) == 4:
            per_seed_ov[s] = {
                k: {"mse": sum(dsm[ds][k]["mse"] for ds in C.ETT4) / 4.0,
                    "mae": sum(dsm[ds][k]["mae"] for ds in C.ETT4) / 4.0}
                for k in ("all", "rand", "ln", "freeze")}
    agg_ds = {}
    for ds in C.ETT4:
        agg_ds[ds] = {}
        for k in ("all", "rand", "ln", "freeze"):
            mses = [per_seed_ds[s][ds][k]["mse"] for s in X.LN_KEYS_SEEDS
                    if ds in per_seed_ds[s]]
            maes = [per_seed_ds[s][ds][k]["mae"] for s in X.LN_KEYS_SEEDS
                    if ds in per_seed_ds[s]]
            mm, msd = ms(mses)
            am, asd = ms(maes)
            agg_ds[ds][k] = {"mse_mean": mm, "mse_std": msd,
                             "mae_mean": am, "mae_std": asd,
                             "per_seed_mse": mses, "per_seed_mae": maes}
    agg_ov = {}
    for k in ("all", "rand", "ln", "freeze"):
        mses = [per_seed_ov[s][k]["mse"] for s in X.LN_KEYS_SEEDS
                if s in per_seed_ov]
        maes = [per_seed_ov[s][k]["mae"] for s in X.LN_KEYS_SEEDS
                if s in per_seed_ov]
        mm, msd = ms(mses)
        am, asd = ms(maes)
        agg_ov[k] = {"mse_mean": mm, "mse_std": msd, "mae_mean": am,
                     "mae_std": asd, "per_seed_mse": mses,
                     "per_seed_mae": maes}
    # paired diffs per (ds, seed) at the 4H-mean layer
    paired = {}
    for pair in ("all_rand", "ln_all", "ln_rand"):
        a, b = pair.split("_")
        paired[pair] = {}
        for ds in C.ETT4:
            per_seed = {}
            for s in X.LN_KEYS_SEEDS:
                if ds in per_seed_ds[s]:
                    per_seed[s] = {
                        "mse": per_seed_ds[s][ds][a]["mse"]
                        - per_seed_ds[s][ds][b]["mse"],
                        "mae": per_seed_ds[s][ds][a]["mae"]
                        - per_seed_ds[s][ds][b]["mae"]}
            pm, psd = ms([v["mse"] for v in per_seed.values()])
            pma, pasd = ms([v["mae"] for v in per_seed.values()])
            paired[pair][ds] = {"per_seed": per_seed, "mse_mean": pm,
                                "mse_std": psd, "mae_mean": pma,
                                "mae_std": pasd}
    return per_seed_ds, per_seed_ov, agg_ds, agg_ov, paired


def pct(a, b):
    """% change of a relative to b (signed)."""
    if b is None or b == 0:
        return None
    return 100.0 * (a - b) / b


def result_md(rows, vj):
    L = []
    L.append("# EXP-039 RESULT: Table 22 B-line across three seeds "
             "(All / Rand-VM full finetune at seeds 2021/2022/2023; "
             "LN = EXP-003 per-seed archives; Freeze = EXP-033)\n")
    L.append("Generated by collect_t39_fullshot.py from archived evidence "
             "only (logs/train_<DS>_<H>_<arm>_seed<s>.json for seeds "
             "2022/2023, EXP-038 seed-2021 train jsons incl. the EXP-037 "
             "ETTh1_96 reuse pair, EXP-003 LN48 pins, "
             "repro_t22_freeze_ln/summary_t22.csv, verify_t39.json "
             "section fullshot).\n")
    L.append("Scope: All = verified ImageNet MAE-base full finetune; "
             "Rand-VM = same architecture, ALL weights randomly "
             "initialized, exactly the All recipe (this round's "
             "declared setting; the paper does not disclose the "
             "Rand-VM trainable set). LN column = EXP-003 per-seed "
             "SUCCESS values (the paper's LN column is a 3-seed mean; "
             "our LN aggregation is also a 3-seed mean of the same "
             "three seeds 2021/2022/2023). Freeze column = EXP-033 "
             "single frozen inference (seed-independent; shared by all "
             "seed rows). Recipe: EXP-003 (bilinear, r=c=0.4, FP32, "
             "Adam 1e-4, bs 256, 1 epoch, full-val MSE selection, "
             "reload then test). Seeds: 2021 archived from EXP-038, "
             "2022/2023 trained this round; aggregation = per-seed "
             "4H-mean then cross-seed mean +/- std (ddof=1).\n")

    # ---- T1: per-horizon, three seeds side by side -------------------
    L.append("## T1: per-horizon results (16 configs; All/Rand per seed; "
             "LN per seed; Freeze shared)\n")
    L.append("Timing columns for seed rows: train* = train_wall − vali_wall "
             "(pure training; the official train() calls vali() inside the "
             "epoch loop). In summary_t39_ett.csv, <arm>_train_s is the RAW "
             "train_wall and therefore INCLUDES the in-loop validation time; "
             "<arm>_train_pure_s = train_wall − vali_wall is the pure "
             "training figure used in this report. The seed-2021 All/Rand "
             "pair of ETTh1_96 comes from the frozen EXP-037 archive (no "
             "phase timing, n/a). Freeze is a single frozen inference "
             "(seed-independent) and is therefore shown as ONE value, not "
             "a seed mean±std.\n")
    by_key = {(r["ds"], r["h"], r["seed"]): r for r in rows}
    L.append("| ds | H | seed | All MSE/MAE | Rand-VM MSE/MAE | "
             "LN MSE/MAE | Freeze MSE/MAE | All train*/vali/test s | "
             "Rand train*/vali/test s |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ds, h in X.CONFIGS:
        for s in X.LN_KEYS_SEEDS:
            r = by_key[(ds, h, s)]
            L.append("| %s | %d | %d | %s/%s | %s/%s | %s/%s | %s/%s | "
                     "%s/%s/%s | %s/%s/%s |" %
                     (ds, h, s, fmt(r["all_mse"]), fmt(r["all_mae"]),
                      fmt(r["rand_mse"]), fmt(r["rand_mae"]),
                      fmt(r["ln_mse"]), fmt(r["ln_mae"]),
                      fmt(r["freeze_mse"]), fmt(r["freeze_mae"]),
                      fmt(train_pure(r, "all"), 1), fmt(r["all_vali_s"], 1),
                      fmt(r["all_test_s"], 1),
                      fmt(train_pure(r, "rand"), 1), fmt(r["rand_vali_s"], 1),
                      fmt(r["rand_test_s"], 1)))
    L.append("")

    per_seed_ds, per_seed_ov, agg_ds, agg_ov, paired = collect_layer(rows)

    # ---- T2: dataset 4H-mean layer ------------------------------------
    if agg_ds:
        L.append("## T2: dataset 4H-mean layer, cross-seed mean±std "
                 "(ddof=1, n=3) vs paper Table 22\n")
        L.append("| ds | col | ours MSE mean±std | ours MAE mean±std | "
                 "paper MSE/MAE | dMSE(mean−paper) | per-seed MSE "
                 "(2021/2022/2023) |")
        L.append("|---|---|---|---|---|---|---|")
        for ds in C.ETT4:
            for k in ("all", "rand", "ln", "freeze"):
                a = agg_ds[ds][k]
                pm, pa = C.PAPER_T22[ds][k]
                d = a["mse_mean"] - pm if a["mse_mean"] is not None else None
                if k == "freeze":
                    mse_cell = fmt(a["mse_mean"])
                    mae_cell = fmt(a["mae_mean"])
                    per_seed = "single frozen inference"
                else:
                    mse_cell = "%s±%s" % (fmt(a["mse_mean"]),
                                          fmt(a["mse_std"]))
                    mae_cell = "%s±%s" % (fmt(a["mae_mean"]),
                                          fmt(a["mae_std"]))
                    per_seed = "/".join("%.4f" % v
                                        for v in a["per_seed_mse"])
                L.append("| %s | %s | %s | %s | %.3f/%.3f | %s | %s |" %
                         (ds, k, mse_cell, mae_cell, pm, pa,
                          fmt(d, 4) if d is not None else "n/a",
                          per_seed))
        L.append("")

    # ---- T3: overall layer --------------------------------------------
    if agg_ov.get("all", {}).get("mse_mean") is not None:
        L.append("## T3: overall layer (4 datasets equal-weight), "
                 "cross-seed mean±std (ddof=1, n=3) vs paper\n")
        L.append("| col | ours MSE mean±std | ours MAE mean±std | "
                 "paper MSE/MAE | dMSE | per-seed MSE (2021/2022/2023) |")
        L.append("|---|---|---|---|---|---|")
        for k in ("all", "rand", "ln", "freeze"):
            a = agg_ov[k]
            pm, pa = C.PAPER_T22_OVERALL[k]
            if k == "freeze":
                mse_cell = fmt(a["mse_mean"])
                mae_cell = fmt(a["mae_mean"])
                per_seed = "single frozen inference"
            else:
                mse_cell = "%s±%s" % (fmt(a["mse_mean"]), fmt(a["mse_std"]))
                mae_cell = "%s±%s" % (fmt(a["mae_mean"]), fmt(a["mae_std"]))
                per_seed = "/".join("%.4f" % v for v in a["per_seed_mse"])
            L.append("| %s | %s | %s | %.3f/%.3f | %+.4f | %s |" %
                     (k, mse_cell, mae_cell, pm, pa,
                      a["mse_mean"] - pm, per_seed))
        L.append("")

    # ---- paired diffs ---------------------------------------------------
    if paired:
        L.append("## T4: paired differences at the dataset-4H-mean layer "
                 "(per seed, then cross-seed mean±std, ddof=1)\n")
        for pair, name in (("all_rand", "All − Rand-VM"),
                           ("ln_all", "LN − All"),
                           ("ln_rand", "LN − Rand-VM")):
            L.append("### %s (MSE; negative = first column better)\n" % name)
            L.append("| ds | seed2021 | seed2022 | seed2023 | mean±std | "
                     "direction consistent? |")
            L.append("|---|---|---|---|---|---|")
            for ds in C.ETT4:
                p = paired[pair][ds]
                seeds = [p["per_seed"].get(s, {}).get("mse")
                         for s in X.LN_KEYS_SEEDS]
                if all(v is not None for v in seeds):
                    consistent = all(v < 0 for v in seeds) or \
                        all(v > 0 for v in seeds)
                    L.append("| %s | %+.4f | %+.4f | %+.4f | %s±%s | %s |" %
                             (ds, *seeds, fmt(p["mse_mean"]),
                              fmt(p["mse_std"]),
                              "yes" if consistent else "NO"))
                else:
                    L.append("| %s | %s | %s | %s | %s | - |" %
                             tuple(["%s" % ds] + [fmt(v) for v in seeds] +
                                   [fmt(p["mse_mean"])] + ["-"]))
            L.append("")

    # ---- stability analysis ---------------------------------------------
    if per_seed_ds:
        L.append("## T5: cross-seed stability of the EXP-038 anomalies\n")
        L.append("Two anomalies from the single-seed EXP-038 round are "
                 "re-examined with per-seed direction and range at the "
                 "dataset-4H-mean layer. Observed outcomes are stated as "
                 "evidence; anything beyond these three seeds remains a "
                 "hypothesis.\n")
        # ETTh1 Rand deviation vs paper
        pm_rand = C.PAPER_T22["ETTh1"]["rand"][0]
        vals = [(s, per_seed_ds[s].get("ETTh1", {}).get("rand", {})
                 .get("mse")) for s in X.LN_KEYS_SEEDS]
        vals = [(s, v) for s, v in vals if v is not None]
        if vals:
            devs = [v - pm_rand for _, v in vals]
            L.append("- **ETTh1 Rand-VM vs paper 4H-mean (%.3f)**: per-seed "
                     "MSE %s -> deviations %s. %s" %
                     (pm_rand,
                      ", ".join("s%d %.4f" % (s, v) for s, v in vals),
                      ", ".join("%+.4f" % d for d in devs),
                      "Deviation direction: %s across the %d seeds; "
                      "cross-seed stability %s."
                      % ("consistent (%s)" % ("all positive" if
                         all(d > 0 for d in devs) else "all negative")
                         if all(d > 0 for d in devs) or
                         all(d < 0 for d in devs) else "INCONSISTENT",
                         len(vals),
                         "supported within these 3 seeds" if
                         all(d > 0 for d in devs) else
                         "NOT supported — the EXP-038 ETTh1 Rand gap "
                         "does not persist across seeds")))
        # ETTm1 All vs Rand
        vals = [(s, pct(per_seed_ds[s].get("ETTm1", {}).get("all", {})
                        .get("mse"),
                        per_seed_ds[s].get("ETTm1", {}).get("rand", {})
                        .get("mse"))) for s in X.LN_KEYS_SEEDS]
        vals = [(s, v) for s, v in vals if v is not None]
        if vals:
            L.append("- **ETTm1 All vs Rand (All %% vs Rand at 4H mean)**: "
                     "per-seed %s -> %s. %s" %
                     (", ".join("s%d %+.2f%%" % (s, v) for s, v in vals),
                      "All worse than Rand in %d/%d seeds" %
                      (sum(1 for _, v in vals if v > 0), len(vals)),
                      "cross-seed persistence %s." %
                      ("confirmed within these 3 seeds"
                       if all(v > 0 for _, v in vals)
                       else "REJECTED — not persistent across seeds")))
        # overall All vs Rand across seeds
        vals = [(s, pct(per_seed_ov[s]["all"]["mse"],
                        per_seed_ov[s]["rand"]["mse"]))
                for s in X.LN_KEYS_SEEDS if s in per_seed_ov]
        if vals:
            L.append("- **Overall pretrained-vs-random (All %% vs Rand)**: "
                     "per-seed %s; LN − All per-seed %s. Seed scatter "
                     "quantifies the single-seed EXP-038 numbers; the "
                     "cause of the remaining gap vs paper stays "
                     "undetermined in this round." %
                     (", ".join("s%d %+.2f%%" % (s, v) for s, v in vals),
                      ", ".join(
                          "s%d %+.2f%%" % (s, 100.0 *
                                           (per_seed_ov[s]["ln"]["mse"]
                                            - per_seed_ov[s]["all"]["mse"])
                                           / per_seed_ov[s]["all"]["mse"])
                          for s in X.LN_KEYS_SEEDS if s in per_seed_ov)))
        L.append("")

    L.extend(budget_md(rows))

    # ---- gates -----------------------------------------------------------
    L.append("## Gates and anomalies\n")
    gates = (vj or {}).get("gates", [])
    fails = [g for g in gates if g["verdict"] == "FAIL"]
    if gates and not fails:
        L.append("All %d gates pass (%d/%d PASS, 0 FAIL; BG1-BG10 over "
                 "64 new arms + LN48/Freeze16/EXP-038 provenance); see "
                 "verify_t39.json.\n" %
                 (len(gates), sum(1 for g in gates
                                  if g["verdict"] == "PASS"), len(gates)))
    else:
        for g in fails:
            L.append("- [FAIL] %s: %s" % (g["gate"], g["detail"]))
        if not gates:
            L.append("(verify_t39.json section fullshot not found — "
                     "run verify_t39_fullshot.py first.)")
        L.append("")
    L.append("GPU peak memory: summary_t39_ett.csv carries per-phase "
             "train/vali/test peaks from the observation-only wrappers; "
             "verify_t39.json fullshot.gpu_peak carries the same three "
             "phases plus overall_max_mb per arm (per-phase values are "
             "authoritative; the run-level counter reflects the last "
             "phase due to per-phase resets).\n")
    return "\n".join(L) + "\n"


def budget_md(rows):
    """Measured PURE-train s/batch per dataset from this round's arms
    (train_wall − vali_wall) -> remaining-arms estimate (skip-if-done
    already-completed groups)."""
    L = ["## Budget (measured pure-train s/batch per dataset, "
         "train_wall − vali_wall, this round)\n"]
    L.append("| dataset | measured pure-train s/batch | from | completed "
             "arms (this round) | est remaining train h |")
    L.append("|---|---|---|---|---|")
    total_left_h = 0.0
    done_tasks = set()
    for r in rows:
        for arm in C.B_ARMS:
            if r.get("%s_mse" % arm) is not None and r["seed"] in X.SEEDS:
                done_tasks.add((r["ds"], r["h"], arm, r["seed"]))
    for ds in C.ETT4:
        sb, src = None, None
        for r in rows:
            if r["ds"] != ds or r["seed"] not in X.SEEDS:
                continue
            for arm in C.B_ARMS:
                w = r.get("%s_train_s" % arm)
                v = r.get("%s_vali_s" % arm)
                b = r.get("%s_train_batches" % arm)
                if w and v is not None and b:
                    sb, src = (w - v) / b, "%s_%d_s%d_%s" % \
                        (ds, r["h"], r["seed"], arm)
                    break
            if sb:
                break
        left = [(h, arm, s) for (d, h, arm, s) in X.TASKS
                if d == ds and (d, h, arm, s) not in done_tasks]
        if sb:
            est = sum(C.expected_train_batches(ds, h)
                      for (h, _, _) in left) * sb / 3600.0
            total_left_h += est
            L.append("| %s | %.3f | %s | %d | %.2f h |" %
                     (ds, sb, src, len(left), est))
        else:
            L.append("| %s | n/a | - | %d | n/a |" % (ds, len(left)))
    L.append("\nTotal estimated remaining train time: **%.2f h** "
             "(pure training only; excludes vali/test phases and "
             "I/O; s/batch already amortized over the whole run — do "
             "not multiply by epochs).\n" % total_left_h)
    return L


def fig_seed_effect(agg_ds, agg_ov, out_png):
    """Overall + per-dataset mean±std (ddof=1) error bars, groups
    All/Rand/LN, Freeze as a single-value marker line."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return "matplotlib unavailable (%s); figure skipped" % e
    names = C.ETT4 + ["Overall"]
    cols = [("all", "#3b6db3"), ("rand", "#c44e52"), ("ln", "#4c9f70")]
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.26
    for gi, (k, color) in enumerate(cols):
        xs, means, stds = [], [], []
        for ni, name in enumerate(names):
            src = agg_ov if name == "Overall" else agg_ds.get(name)
            if not src or src.get(k, {}).get("mse_mean") is None:
                continue
            xs.append(ni + (gi - 1) * w)
            means.append(src[k]["mse_mean"])
            stds.append(src[k]["mse_std"])
        ax.bar(xs, means, w, yerr=stds, capsize=3,
               label={"all": "All (ImageNet init)", "rand": "Rand-VM",
                      "ln": "LN"}[k], color=color,
                       alpha=0.85 if k != "ln" else 1.0)
    fz = [agg_ds.get(ds, {}).get("freeze", {}).get("mse_mean")
          for ds in C.ETT4] + \
         [agg_ov.get("freeze", {}).get("mse_mean")]
    fz_x = [i for i, v in enumerate(fz) if v is not None]
    fz_v = [v for v in fz if v is not None]
    if fz_v:
        ax.scatter(fz_x, fz_v, marker="_", s=1200, color="#555555",
                   label="Freeze (seed-independent)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("MSE (dataset-4H / overall mean ± std, n=3 seeds)")
    ax.set_title("VisionTS Table 22 B-line across seeds 2021/2022/2023")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return "fig written (%d groups)" % len(names)


def selfcheck():
    out = "temp_selfcheck_t39"
    os.makedirs(out, exist_ok=True)
    rows = []
    for ds, h in X.CONFIGS:
        for s in X.LN_KEYS_SEEDS:
            r = {"task": "t22_3seeds", "ds": ds, "h": h, "seed": s,
                 "freeze_mse": 0.4, "freeze_mae": 0.41,
                 "freeze_source": "x", "ln_mse": 0.3, "ln_mae": 0.37,
                 "ln_source": "pin"}
            for arm in C.B_ARMS:
                r.update({"%s_source" % arm: "this run",
                          "%s_mse" % arm: 0.45 + 0.01 * (s - 2021),
                          "%s_mae" % arm: 0.42,
                          "%s_train_s" % arm: 800.0,
                          "%s_train_pure_s" % arm: 740.0,
                          "%s_vali_s" % arm: 60.0, "%s_test_s" % arm: 40.0,
                          "%s_train_peak_mb" % arm: 5000.0,
                          "%s_train_batches" % arm: 200,
                          "%s_ckpt_sha16" % arm: "ab" * 8})
            rows.append(r)
    write_csv(os.path.join(out, "summary_t39_ett.csv"), rows)
    md = result_md(rows, {"gates": []})
    with open(os.path.join(out, "RESULT_t39_ett.md"), "w") as f:
        f.write(md)
    per_seed_ds, per_seed_ov, agg_ds, agg_ov, paired = collect_layer(rows)
    msg = fig_seed_effect(agg_ds, agg_ov,
                          os.path.join(out, "fig_t39_seed_effect.png"))
    print("[t39] selfcheck OK -> %s (%s)" % (out, msg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    rows = build_rows()
    write_csv(X.SUMMARY_CSV, rows)
    vj = C.load_json(X.VERIFY_JSON).get("fullshot", {}) \
        if os.path.isfile(X.VERIFY_JSON) else {}
    md = result_md(rows, vj)
    with open(X.RESULT_MD, "w") as f:
        f.write(md)
    per_seed_ds, per_seed_ov, agg_ds, agg_ov, paired = collect_layer(rows)
    msg = fig_seed_effect(agg_ds, agg_ov, X.FIG_PNG)
    print("[t39] wrote %s, %s, %s (%s)" %
          (X.SUMMARY_CSV, X.RESULT_MD, X.FIG_PNG, msg))


if __name__ == "__main__":
    main()
