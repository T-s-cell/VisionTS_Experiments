#!/usr/bin/env python3
"""Ladder aggregation (float64, original units), fixed order: prediction
points -> windows -> sequences (equal weight inside each subset). The
headline subset metric is the sequence-equal-weight mean; the direct
point-level mean over all points is reported as reference only. Three
subsets are reported separately; NO cross-subset total in original units.
Deterministic reduction: windows in CSV row order, sequences in ascending
Idx order, numpy float64 means over contiguous arrays."""
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


def load_all_shards(ds):
    rows = []
    for shard in range(P.n_shards(ds)):
        npz, hdr, done = P.shard_paths(ds, shard)
        with np.load(npz) as z:
            idx = z["idx"]
            date = z["date"]
            pred = z["pred"]
            tgt = z["target"]
        for i in range(len(idx)):
            rows.append((int(idx[i]), str(date[i]), pred[i], tgt[i]))
    return rows


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


def main():
    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    summary = {"protocol_name": P.PROTOCOL_NAME,
               "ladder": "point->window->sequence-equal-weight (float64, "
                         "original units; no cross-subset total)",
               "subsets": {}}
    for ds in P.ORDER:
        rows = load_all_shards(ds)
        if len(rows) != P.EXPECTED_WINDOWS[ds]:
            sys.exit(f"{ds}: {len(rows)} windows != "
                     f"{P.EXPECTED_WINDOWS[ds]} (run verify_cache)")
        mse, mae, point, seqs = ladder(rows)
        res = {"subset": ds, "n_windows": len(rows), "n_sequences": len(seqs),
               "mse_seq_equal_weight": mse, "mae_seq_equal_weight": mae,
               "mse_point_level_ref": point[0],
               "mae_point_level_ref": point[1]}
        C.dump_json(res, os.path.join(P.RESULTS_DIR, f"{ds}_metrics.json"))
        with open(os.path.join(P.RESULTS_DIR,
                               f"sequence_level_{ds}.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["Idx", "n_windows", "MSE", "MAE"])
            for ix, n, m, a in seqs:
                w.writerow([ix, n, repr(m), repr(a)])
        np.savez(os.path.join(P.RESULTS_DIR, f"window_level_{ds}.npz"),
                 idx=np.array([r[0] for r in rows], dtype=np.int64),
                 date=np.array([r[1] for r in rows]),
                 mse_w=np.array([float(np.mean((r[2] - r[3]) ** 2))
                                 for r in rows], dtype=np.float64),
                 mae_w=np.array([float(np.mean(np.abs(r[2] - r[3])))
                                 for r in rows], dtype=np.float64))
        summary["subsets"][ds] = res
        print(f"[eval] {ds}: MSE {mse:.6f}  MAE {mae:.6f}  "
              f"({len(rows)} windows, {len(seqs)} sequences)")
    C.dump_json(summary, os.path.join(P.RESULTS_DIR, "summary.json"))
    print("[eval] wrote summary.json + per-subset metrics/CSVs/npz")


if __name__ == "__main__":
    main()
