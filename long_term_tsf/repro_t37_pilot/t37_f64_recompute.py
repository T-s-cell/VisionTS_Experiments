"""EXP-037 pack fix (user verdict 2026-11-04): recompute ETTh1 MSE/MAE from
the existing pred/true npy with an EXPLICIT float64 cast BEFORE any
arithmetic (the earlier BG3 recompute ran in float32, matching the SUCCESS
computation path bit-for-bit). Plain-numpy f64 metric, independent of
utils/metrics.py. Also recomputes the EXP-003 LN anchor arrays the same way.
Writes logs/recompute_f64.json. CPU-only.
Run: python t37_f64_recompute.py
"""
import os

import numpy as np

import t37_common as C
import t22_common as T


def f64_mae_mse(pred_flat, true_flat, n, h):
    pred = np.asarray(pred_flat, dtype=np.float64)[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
    true = np.asarray(true_flat, dtype=np.float64)[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
    diff = pred - true
    return float(np.abs(diff).mean()), float((diff * diff).mean())


def main():
    n, h = C.ETTH1["n_windows"], C.ETTH1["pred"]
    out = {"note": "float64 cast BEFORE arithmetic; plain numpy; "
                   "SUCCESS values were computed in float32 by utils/metrics.py",
           "shape_restored": [n, h, 7], "arms": {}}
    for arm in C.B_ARMS:
        res_dir = os.path.join(C.b_job_dir(arm), "results", "_")
        pred = np.load(os.path.join(res_dir, "pred.npy"))
        true = np.load(os.path.join(res_dir, "true.npy"))
        mae, mse = f64_mae_mse(pred, true, n, h)
        succ = C.load_json(os.path.join(C.b_job_dir(arm), "SUCCESS.txt"))
        out["arms"][arm] = {
            "dtype_on_disk": str(pred.dtype), "n_elements": int(pred.size),
            "mse_f64": mse, "mae_f64": mae,
            "mse_success_f32": succ["mse"], "mae_success_f32": succ["mae"],
            "d_mse": mse - succ["mse"], "d_mae": mae - succ["mae"],
        }
    ln_dir = C.exp003_job()
    ln_pred = os.path.join(ln_dir, "results", "_", "pred.npy")
    if os.path.isfile(ln_pred):
        mae, mse = f64_mae_mse(np.load(ln_pred),
                               np.load(os.path.join(ln_dir, "results", "_", "true.npy")), n, h)
        out["arms"]["ln_exp003_anchor"] = {
            "mse_f64": mse, "mae_f64": mae,
            "mse_success_f32": C.LN_ANCHOR["mse"], "mae_success_f32": C.LN_ANCHOR["mae"],
            "d_mse": mse - C.LN_ANCHOR["mse"], "d_mae": mae - C.LN_ANCHOR["mae"],
        }
    out["max_abs_d_mse"] = max(abs(v["d_mse"]) for v in out["arms"].values())
    out["max_abs_d_mae"] = max(abs(v["d_mae"]) for v in out["arms"].values())
    C.dump_json(C.RECOMPUTE_F64_JSON, out)
    print("[t37] f64 recompute -> %s (max |d| mse %.3e mae %.3e)"
          % (C.RECOMPUTE_F64_JSON, out["max_abs_d_mse"], out["max_abs_d_mae"]))
    for k, v in out["arms"].items():
        print("[t37]   %-18s mse %.16f mae %.16f (d %.2e / %.2e)"
              % (k, v["mse_f64"], v["mae_f64"], v["d_mse"], v["d_mae"]))


if __name__ == "__main__":
    main()
