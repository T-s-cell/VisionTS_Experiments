#!/usr/bin/env python3
"""Three-group scoring under ONE ladder + environment-drift comparison
(float64, original units).

Groups:
  VTS-OLD  : frozen EXP-016 predictions (visionts_ref/, zeta environment)
  VTS-NEW  : this run's predictions (predictions/, dt/Aurora-final env)
  AURORA   : frozen EXP-017 predictions (aurora_ref/), per seed scored
             separately, then mean +- std(ddof=1) — seeds never averaged
             at prediction level, no best-seed selection.

Ladder (identical code path to EXP-016): prediction points -> windows ->
sequences (equal weight inside each subset); windows in CSV row order,
sequences ascending Idx, numpy float64 means. Three subsets reported
separately; NO cross-subset total.

Comparison per subset:
  env_drift     = VTS-NEW - VTS-OLD                  (the measured effect
                of the environment change on the frozen VisionTS pipeline)
  same_env_gap  = Aurora_mean - VTS-NEW              (model gap after
                aligning the environment)
  orig_gap      = Aurora_mean - VTS-OLD              (EXP-017 headline gap)
  drift_ratio   = env_drift / orig_gap               (per metric; None if
                orig_gap == 0)
  sign verdicts : does Aurora remain MSE-lower AND MAE-higher vs VTS-NEW?
  pred diff     : per-window |VTS-NEW - VTS-OLD| aligned by identity —
                mean/max abs, argmax window (Idx, Date, point offset)."""
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (HERE, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

import common as C
import protocol as P


def rows_from_npz(paths):
    rows = []
    for npz in paths:
        with np.load(npz, allow_pickle=False) as z:
            idx = z["idx"]
            date = z["date"]
            pred = z["pred"]
            tgt = z["target"]
        for i in range(len(idx)):
            rows.append((int(idx[i]), str(date[i]), pred[i], tgt[i]))
    return rows


def load_vts_old(ds):
    return rows_from_npz(
        [os.path.join(P.VTS_REF_DIR, "predictions",
                      f"{ds}__shard{s:03d}.npz")
         for s in range(P.n_shards(ds))])


def load_vts_new(ds):
    return rows_from_npz(
        [P.shard_paths(ds, s)[0] for s in range(P.n_shards(ds))])


def load_aurora_seed(ds, seed):
    return rows_from_npz(
        [P.aurora_shard_paths(os.path.join(P.AURORA_REF_DIR,
                                           "predictions_dt"), ds, seed, s)
         for s in range(P.n_shards(ds))])


def ladder(rows):
    """rows in fixed order -> (subset mse/mae via sequence-equal-weight,
    point-level reference, per-sequence metrics list)."""
    seq_m = defaultdict(list)   # idx -> [(mse_w, mae_w)] in row order
    all_sq, all_ab = [], []
    for ix, _date, pred, tgt in rows:
        d = pred - tgt
        mse_w = float(np.mean(d ** 2))
        mae_w = float(np.mean(np.abs(d)))
        seq_m[ix].append((mse_w, mae_w))
        all_sq.append(d ** 2)
        all_ab.append(np.abs(d))
    seqs = []
    for ix in sorted(seq_m):
        arr = np.array(seq_m[ix], dtype=np.float64)   # [n_win, 2]
        seqs.append((ix, len(seq_m[ix]),
                     float(arr[:, 0].mean()), float(arr[:, 1].mean())))
    mse = float(np.mean([s[2] for s in seqs]))
    mae = float(np.mean([s[3] for s in seqs]))
    sq = np.concatenate(all_sq)
    ab = np.concatenate(all_ab)
    return mse, mae, (float(sq.mean()), float(ab.mean())), seqs


def write_sequence_csv(path, seqs):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Idx", "n_windows", "MSE", "MAE"])
        for ix, n, m, a in seqs:
            w.writerow([ix, n, repr(m), repr(a)])


def group_metrics(rows, n_expect):
    if len(rows) != n_expect:
        sys.exit(f"windows {len(rows)} != expected {n_expect}")
    mse, mae, point, seqs = ladder(rows)
    return {"n_windows": len(rows),
            "n_sequences": len(seqs),
            "mse_seq_equal_weight": mse, "mae_seq_equal_weight": mae,
            "mse_point_level_ref": point[0], "mae_point_level_ref": point[1],
            "_seqs": seqs}


def pred_diff_stats(rows_old, rows_new):
    """Rows already verified identity-aligned (same length, same order).
    Returns stats of pred_new - pred_old over all points."""
    assert len(rows_old) == len(rows_new)
    max_abs, arg = -1.0, None
    sq_sum, ab_sum, n_pts = 0.0, 0.0, 0
    win_mad = []
    for (ix, dt, po, _to), (_i2, _d2, pn, _t2) in zip(rows_old, rows_new):
        d = np.abs(pn.astype(np.float64) - po.astype(np.float64))
        s = float(d.sum())
        ab_sum += s
        sq_sum += float((d ** 2).sum())
        n_pts += d.size
        win_mad.append(s / d.size)
        j = int(np.argmax(d))
        if float(d[j]) > max_abs:
            max_abs = float(d[j])
            arg = (ix, dt, j)
    return {"mean_abs_diff": ab_sum / n_pts,
            "rmse_diff": (sq_sum / n_pts) ** 0.5,
            "max_abs_diff": max_abs,
            "argmax_window_idx": arg[0], "argmax_window_date": arg[1],
            "argmax_point_offset": arg[2],
            "mean_window_mean_abs_diff": float(np.mean(win_mad))}


def main():
    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    summary = {"protocol_name": P.PROTOCOL_NAME,
               "ladder": "point->window->sequence-equal-weight (float64, "
                         "original units; no cross-subset total)",
               "groups": {"vts_old": "EXP-016 frozen predictions (zeta env)",
                          "vts_new": "this run (Aurora-final dt env)",
                          "aurora": "EXP-017 frozen predictions, "
                                    "per-seed mean+-std(ddof=1)"},
               "subsets": {}}
    for ds in P.ORDER:
        n_exp = P.EXPECTED_WINDOWS[ds]
        Pl = P.SUBSETS[ds]["pred"]

        old_rows = load_vts_old(ds)
        old = group_metrics(old_rows, n_exp)
        new_rows = load_vts_new(ds)
        new = group_metrics(new_rows, n_exp)

        per_seed = {}
        seed_rows = {}
        for seed in P.AURORA_SEEDS:
            r = load_aurora_seed(ds, seed)
            per_seed[seed] = group_metrics(r, n_exp)
            seed_rows[seed] = r
        mses = [per_seed[s]["mse_seq_equal_weight"] for s in P.AURORA_SEEDS]
        maes = [per_seed[s]["mae_seq_equal_weight"] for s in P.AURORA_SEEDS]
        aurora = {"n_windows": n_exp,
                  "n_sequences": per_seed[P.AURORA_SEEDS[0]]["n_sequences"],
                  "seeds": list(P.AURORA_SEEDS),
                  "mse_seq_equal_weight_mean": float(np.mean(mses)),
                  "mse_seq_equal_weight_std": float(np.std(mses, ddof=1)),
                  "mae_seq_equal_weight_mean": float(np.mean(maes)),
                  "mae_seq_equal_weight_std": float(np.std(maes, ddof=1)),
                  "per_seed": {str(s): {
                      "mse_seq_equal_weight":
                          per_seed[s]["mse_seq_equal_weight"],
                      "mae_seq_equal_weight":
                          per_seed[s]["mae_seq_equal_weight"]}
                      for s in P.AURORA_SEEDS}}

        env_drift = {"mse": new["mse_seq_equal_weight"]
                     - old["mse_seq_equal_weight"],
                     "mae": new["mae_seq_equal_weight"]
                     - old["mae_seq_equal_weight"]}
        orig_gap = {"mse": aurora["mse_seq_equal_weight_mean"]
                    - old["mse_seq_equal_weight"],
                    "mae": aurora["mae_seq_equal_weight_mean"]
                    - old["mae_seq_equal_weight"]}
        same_env_gap = {"mse": aurora["mse_seq_equal_weight_mean"]
                        - new["mse_seq_equal_weight"],
                        "mae": aurora["mae_seq_equal_weight_mean"]
                        - new["mae_seq_equal_weight"]}
        ratio = {k: (env_drift[k] / orig_gap[k] if orig_gap[k] != 0 else None)
                 for k in ("mse", "mae")}
        verdict = {
            "aurora_still_mse_lower_vs_vts_new":
                bool(same_env_gap["mse"] < 0),
            "aurora_still_mae_higher_vs_vts_new":
                bool(same_env_gap["mae"] > 0),
            "env_drift_mse_direction":
                "lower" if env_drift["mse"] < 0 else
                ("higher" if env_drift["mse"] > 0 else "zero"),
            "env_drift_mae_direction":
                "lower" if env_drift["mae"] < 0 else
                ("higher" if env_drift["mae"] > 0 else "zero"),
        }

        # per-seed Aurora gaps vs VTS-NEW (sign robustness)
        per_seed_gap = {str(s): {
            "mse": per_seed[s]["mse_seq_equal_weight"]
            - new["mse_seq_equal_weight"],
            "mae": per_seed[s]["mae_seq_equal_weight"]
            - new["mae_seq_equal_weight"]} for s in P.AURORA_SEEDS}

        pdiff = pred_diff_stats(old_rows, new_rows)

        res = {"subset": ds, "pred_len": Pl,
               "vts_old": {k: v for k, v in old.items() if k != "_seqs"},
               "vts_new": {k: v for k, v in new.items() if k != "_seqs"},
               "aurora": aurora,
               "env_drift_new_minus_old": env_drift,
               "orig_gap_aurora_mean_minus_vts_old": orig_gap,
               "same_env_gap_aurora_mean_minus_vts_new": same_env_gap,
               "drift_over_orig_gap_ratio": ratio,
               "per_seed_gap_aurora_minus_vts_new": per_seed_gap,
               "verdict": verdict,
               "pred_diff_vts_new_minus_vts_old": pdiff}

        C.dump_json(res, os.path.join(P.RESULTS_DIR, f"{ds}_metrics.json"))
        C.dump_json({"subset": ds, "vts_new": {k: v for k, v in
                                               new.items() if k != "_seqs"}},
                    os.path.join(P.RESULTS_DIR,
                                 f"vts_new_metrics_{ds}.json"))
        write_sequence_csv(os.path.join(P.RESULTS_DIR,
                                        f"sequence_level_{ds}.csv"),
                           new["_seqs"])
        with open(os.path.join(P.RESULTS_DIR,
                               f"comparison_env_drift_{ds}.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "vts_old", "vts_new", "aurora_mean",
                        "aurora_std", "env_drift", "same_env_gap",
                        "orig_gap", "drift_ratio"])
            for met in ("mse", "mae"):
                k = f"{met}_seq_equal_weight"
                w.writerow([met.upper(),
                            repr(old[k]), repr(new[k]),
                            repr(aurora[f"{k}_mean"]),
                            repr(aurora[f"{k}_std"]),
                            repr(env_drift[met]),
                            repr(same_env_gap[met]),
                            repr(orig_gap[met]),
                            repr(ratio[met])])
        np.savez(os.path.join(P.RESULTS_DIR, f"pred_diff_{ds}.npz"),
                 idx=np.array([r[0] for r in old_rows], dtype=np.int64),
                 date=np.array([r[1] for r in old_rows]),
                 mean_abs_diff=np.array(
                     [float(np.mean(np.abs(rn[2].astype(np.float64)
                                            - ro[2].astype(np.float64))))
                      for ro, rn in zip(old_rows, new_rows)],
                     dtype=np.float64))
        summary["subsets"][ds] = res
        print(f"[eval] {ds}:")
        print(f"       VTS-old  MSE {old['mse_seq_equal_weight']:.6f}  "
              f"MAE {old['mae_seq_equal_weight']:.6f}")
        print(f"       VTS-new  MSE {new['mse_seq_equal_weight']:.6f}  "
              f"MAE {new['mae_seq_equal_weight']:.6f}")
        print(f"       Aurora   MSE {aurora['mse_seq_equal_weight_mean']:.6f}"
              f"+-{aurora['mse_seq_equal_weight_std']:.6f}  "
              f"MAE {aurora['mae_seq_equal_weight_mean']:.6f}"
              f"+-{aurora['mae_seq_equal_weight_std']:.6f}")
        print(f"       env_drift  dMSE {env_drift['mse']:+.6f}  "
              f"dMAE {env_drift['mae']:+.6f}   "
              f"ratio MSE {ratio['mse'] if ratio['mse'] is not None else 'NA'}"
              f"  MAE {ratio['mae'] if ratio['mae'] is not None else 'NA'}")
        print(f"       same_env_gap (Aurora vs VTS-new)  "
              f"MSE {same_env_gap['mse']:+.6f}  "
              f"MAE {same_env_gap['mae']:+.6f}  verdict {verdict}")
        print(f"       pred diff new-old: mean|d| "
              f"{pdiff['mean_abs_diff']:.6g}, max|d| "
              f"{pdiff['max_abs_diff']:.6g} @ (Idx {pdiff['argmax_window_idx']}"
              f", {pdiff['argmax_window_date']}, off "
              f"{pdiff['argmax_point_offset']})")
    C.dump_json(summary, os.path.join(P.RESULTS_DIR, "summary.json"))
    print("[eval] wrote summary.json + per-subset metrics/comparisons")


if __name__ == "__main__":
    main()
