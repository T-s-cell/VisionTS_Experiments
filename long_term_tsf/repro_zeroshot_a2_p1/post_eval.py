"""EXP-030 post evaluation (numpy-only, run after probes OK per group).
Per group: recompute standardized MSE/MAE with the original metric()
(== metrics.npy exact), read the F3-verified baseline metrics, and save
  - arr_<DS>_<H>.npz (in the P=1 run dir): pred, truth, window_index,
    win_start (absolute csv row of first predicted step), win_var_mse,
    win_var_mae (N x 7, averaged over forecast steps), win_mse, win_mae
    (N,), channel_names — kept on the server, NOT shipped in the audit pack
  - metrics_<DS>_<H>.json: overall metrics, baseline metrics, deltas,
    geometry evidence, config, code md5s.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2p1_common as C


def eval_group(ds, h, metric):
    rd = C.p1_dir(ds, h)
    bd = C.base_dir(ds, h)
    ev_path = os.path.join(rd, "probes_evidence.json")
    if not os.path.isfile(ev_path):
        raise SystemExit("[post_eval] %s_%d: probes_evidence.json missing — "
                         "run probes_p1.py first" % (ds, h))
    evidence = C.load_json(ev_path)

    pred = np.load(os.path.join(rd, "results", "_", "pred.npy"))
    true = np.load(os.path.join(rd, "results", "_", "true.npy"))
    mets = np.load(os.path.join(rd, "results", "_", "metrics.npy"))
    base_mets = np.load(os.path.join(bd, "results", "_", "metrics.npy"))

    mae_r, mse_r, rmse_r, mape_r, mspe_r = metric(pred, true)
    recomputed = np.array([mae_r, mse_r, rmse_r, mape_r, mspe_r], dtype=np.float64)
    if not np.array_equal(recomputed, mets[:5].astype(np.float64), equal_nan=True):
        raise SystemExit("[post_eval] %s_%d: recomputed metrics != metrics.npy"
                         % (ds, h))

    # per-window per-variable errors (f32, mean over forecast steps)
    err = pred - true  # (N, H, 7) f32
    win_var_mse = np.mean(np.square(err), axis=1)      # (N, 7)
    win_var_mae = np.mean(np.abs(err), axis=1)         # (N, 7)
    win_mse = np.mean(np.square(err), axis=(1, 2))     # (N,)
    win_mae = np.mean(np.abs(err), axis=(1, 2))        # (N,)
    n = pred.shape[0]
    window_index = np.arange(n, dtype=np.int64)
    win_start = C.first_pred_abs_row0(ds) + window_index
    names = C.channel_names(ds)
    assert len(names) == pred.shape[2], (names, pred.shape)

    npz_path = os.path.join(rd, "arr_%s_%d.npz" % (ds, h))
    np.savez_compressed(
        npz_path, pred=pred, truth=true, window_index=window_index,
        win_start=win_start, win_var_mse=win_var_mse, win_var_mae=win_var_mae,
        win_mse=win_mse, win_mae=win_mae,
        channel_names=np.array(names, dtype="U8"))

    u = evidence["update_config"]
    out = {
        "group": "%s_%d" % (ds, h),
        "periodicity": 1, "seq_len": C.DATASETS[ds]["ctx"], "pred_len": h,
        "n_windows": n,
        "p1": {"mse": float(mse_r), "mae": float(mae_r)},
        "baseline_p_orig": {"period": C.DATASETS[ds]["period"],
                            "mse": float(base_mets[1]), "mae": float(base_mets[0])},
        "delta_p1_minus_orig": {
            "mse": float(mse_r) - float(base_mets[1]),
            "mae": float(mae_r) - float(base_mets[0])},
        "rel_p1_minus_orig": {
            "mse": float(mse_r) / float(base_mets[1]) - 1.0,
            "mae": float(mae_r) / float(base_mets[0]) - 1.0},
        "geometry": {"pad_left": u["pad_left"], "pad_right": u["pad_right"],
                     "num_patch_input": u["num_patch_input"],
                     "scale_x": u["scale_x"],
                     "seg_shape_first_batch": evidence["segmentation"]["first_shape"]},
        "arrays": {"npz": os.path.relpath(npz_path, C.ROOT),
                   "pred_npy": "results/_/pred.npy",
                   "note": "pred/truth kept on server (large)"},
        "code_md5": {f: C.md5_of(os.path.join(C.ROOT, f)) for f in C.FP_FILES},
        "ckpt_sha256": C.CKPT_SHA256,
    }
    C.dump_json(os.path.join(rd, "metrics_%s_%d.json" % (ds, h)), out)
    print("[post_eval] %s_%d  P1 mse=%.6f mae=%.6f | origP mse=%.6f mae=%.6f "
          "(P=%.0f) | dMSE=%+.4f dMAE=%+.4f"
          % (ds, h, mse_r, mae_r, base_mets[1], base_mets[0],
             C.DATASETS[ds]["period"],
             out["delta_p1_minus_orig"]["mse"], out["delta_p1_minus_orig"]["mae"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs=2, metavar=("DS", "H"), action="append")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    metric = C.load_metric()
    groups = C.group_ids() if args.all else [(ds, int(h)) for ds, h in args.group]
    outs = []
    for ds, h in groups:
        outs.append(eval_group(ds, h, metric))
    os.makedirs(os.path.join(C.PKG, "logs"), exist_ok=True)
    C.dump_json(os.path.join(C.PKG, "logs", "post_eval_groups.json"),
                {"groups": [o["group"] for o in outs]})
    print("[post_eval] done: %d groups" % len(outs))


if __name__ == "__main__":
    main()
