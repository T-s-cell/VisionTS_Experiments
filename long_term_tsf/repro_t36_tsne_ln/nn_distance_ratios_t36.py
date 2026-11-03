"""EXP-036 audit revision: per-row nearest-neighbour distance evidence.

For every encoder condition and all 900 time-series rows (global indices
1000..1899), records the distance to the nearest ImageNet sample and to the
nearest other time-series sample (self excluded by global row), computed in
float64 from the archived (1900, 54528) float32 features — the same matrix
the 30-NN metric used. Pure numpy; no model, no GPU.

Writes save_t36_tsne/nn_distance_ratios.npz with, per condition:
  '{cond}_rows' (900,) int32   global row index
  '{cond}_d_im' (900,) f64     distance to nearest ImageNet sample
  '{cond}_d_ts' (900,) f64     distance to nearest other time-series sample
plus a 'dist_ratio' section in logs/verify_t36.json (per-cond ratio min/max
over rows, file sha256). The RESULT quote is derived from this file.
"""
import json
import os
import time

import numpy as np

import t36_common as C


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    spans = C.group_spans()
    ts0, ts1 = spans["ETTm1"][0], spans["Electricity"][1]
    assert (ts0, ts1) == (1000, 1900)

    arrays, summary = {}, {}
    for cond in C.COND_LIST:
        x = np.load(C.cond_features(cond)).astype(np.float64)
        assert x.shape == (1900, C.FEATURE_DIM)
        sq = (x ** 2).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
        np.maximum(d2, 0.0, out=d2)
        dist = np.sqrt(d2)
        rows = np.arange(ts0, ts1)
        dist[rows, rows] = np.inf  # exclude self by global row index
        d_im = dist[ts0:ts1, :ts0].min(1)
        d_ts = dist[ts0:ts1, ts0:].min(1)
        assert np.isfinite(d_im).all() and np.isfinite(d_ts).all()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(d_ts > 0.0, d_im / d_ts, np.inf)
        pos = ratio[ratio < np.inf]
        arrays["%s_rows" % cond] = rows.astype(np.int32)
        arrays["%s_d_im" % cond] = d_im
        arrays["%s_d_ts" % cond] = d_ts
        summary[cond] = {"n_rows": int(rows.size),
                         "n_d_im_gt_d_ts": int((d_im > d_ts).sum()),
                         "n_dts_zero": int((d_ts == 0.0).sum()),
                         "ratio_min_finite": float(pos.min()) if pos.size else None,
                         "ratio_median_finite": float(np.median(pos)) if pos.size else None}
        log("%s: d_im>d_ts %d/%d (d_ts==0: %d); finite ratio min=%.3f median=%.3f"
            % (cond, summary[cond]["n_d_im_gt_d_ts"], rows.size,
               summary[cond]["n_dts_zero"],
               summary[cond]["ratio_min_finite"], summary[cond]["ratio_median_finite"]))
        del dist, d2

    npz_path = os.path.join(C.SAVE_BASE, "nn_distance_ratios.npz")
    np.savez_compressed(npz_path, **arrays)

    data = C.load_json(C.VERIFY_JSON) if os.path.isfile(C.VERIFY_JSON) else {}
    data["dist_ratio"] = {"npz": npz_path, "npz_sha256": C.sha256_of(npz_path),
                          "per_cond": summary,
                          "all_rows_d_im_gt_d_ts": all(s["n_d_im_gt_d_ts"] == s["n_rows"]
                                                       for s in summary.values())}
    C.dump_json(C.VERIFY_JSON, data)
    log("saved %s (sha256 %s...)" % (npz_path, data["dist_ratio"]["npz_sha256"][:16]))
    log("DONE")


if __name__ == "__main__":
    main()
