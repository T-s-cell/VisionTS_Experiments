"""EXP-043 acceptance check: TRUE cross-path comparison for the 6 smoke
configs (user correction, 2026-11-12).

val_pass.py --crosspath only proved repeat-run determinism (the same
forward_loop twice).  This script instead runs the ORIGINAL inference
entry -- exp.exp_long_term_forecasting.Exp_Long_Term_Forecast.test, the
exact method run.py invokes -- on the validation split, with its
data_provider patched to return the official Dataset_ETT_* (flag='val')
under official test-loader semantics (bs=32, shuffle=False,
drop_last=False, num_workers=10): the same batches the new validation
program consumed.  Predictions and targets are then compared against the
archived val_pass arrays (save_t43_period_val_val/<DS>_H96_P<P>/arr_val.npz,
produced in the S2 smoke): max-abs diffs and sha256 of both sides are
recorded in logs/xcheck_<DS>_P<P>_H96.json; gates live in verify_t43.py
--xcheck.

Zero side effects on frozen assets: model is built with CWD=long_term_tsf
(relative ckpt path), then the process chdirs into a sandbox
(xcheck_wd/<DS>_P<P>/) so exp.test's result files and its append to
result_long_forecast.txt land inside the sandbox, never in the shared
framework tree.  Ledger and all timing records are untouched.

Acceptance-only script: NOT part of any run fingerprint (same hygiene
class as verify_t43.py / collect_t43.py).
"""
import argparse
import hashlib
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t43_common as X

for p in (X.PKG, X.ROOT, X.REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

AP = argparse.ArgumentParser()
AP.add_argument("--ds", required=True, choices=X.ETT4)
AP.add_argument("--P", required=True, type=int)
AP.add_argument("--H", type=int, default=96)
ARGS = AP.parse_args()

DS, P, H = ARGS.ds, ARGS.P, ARGS.H
C = X.DATASETS[DS]

random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
torch.cuda.manual_seed_all(2021)

import exp.exp_long_term_forecasting as elf  # noqa: E402
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute  # noqa
from torch.utils.data import DataLoader  # noqa


def patched_data_provider(args, flag):
    """Same val dataset/loader as val_pass.build_loader, with an absolute
    root_path so the sandbox chdir cannot break CSV reads."""
    assert flag == "test", flag  # exp.test always asks for flag='test'
    root_abs = os.path.abspath(os.path.join(X.ROOT, C["root_path"]))
    cls = Dataset_ETT_minute if C["minute"] else Dataset_ETT_hour
    ds_ = cls(root_path=root_abs, data_path=C["data_path"],
              flag="val", size=[C["ctx"], X.RUN_DEFAULTS["label_len"], H],
              features="M", target="OT", scale=True,
              timeenc=1 if X.RUN_DEFAULTS["embed"] == "timeF" else 0,
              freq=X.RUN_DEFAULTS["freq"],
              seasonal_patterns=X.RUN_DEFAULTS["seasonal_patterns"])
    dl = DataLoader(ds_, batch_size=X.RUN_DEFAULTS["batch_size"],
                    shuffle=False, num_workers=X.RUN_DEFAULTS["num_workers"],
                    drop_last=False)
    return ds_, dl


def sha256_arr(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def main():
    t0 = time.time()
    jp = os.path.join(X.LOG_DIR, "xcheck_%s_P%d_H%d.json" % (DS, P, H))
    if os.path.isfile(jp):
        print("[xcheck] %s P%d H%d: json exists -> skip" % (DS, P, H))
        return
    sandbox = os.path.join(X.PKG, "xcheck_wd", "%s_P%d" % (DS, P))
    setting = "t43_xcheck_%s_P%d_H%d" % (DS, P, H)

    # 1. original inference entry, model built with CWD=long_term_tsf
    cfg = dict(
        task_name="long_term_forecast", model="VisionTS",
        pred_len=H, seq_len=C["ctx"], label_len=X.RUN_DEFAULTS["label_len"],
        vm_arch="mae_base", ft_type=X.RUN_DEFAULTS["ft_type"],
        vm_pretrained=X.RUN_DEFAULTS["vm_pretrained"], vm_ckpt="../ckpt/",
        periodicity=P, interpolation=X.RUN_DEFAULTS["interpolation"],
        norm_const=X.RUN_DEFAULTS["norm_const"],
        align_const=X.RUN_DEFAULTS["align_const"],
        features="M", use_gpu=True, gpu=0, use_multi_gpu=False,
        gpu_type="cuda", devices="0", use_amp=False,
        output_attention=False, inverse=False, save_dir=".",
        batch_size=X.RUN_DEFAULTS["batch_size"],
        num_workers=X.RUN_DEFAULTS["num_workers"], seed=2021,
        train_epochs=0)
    import argparse as _ap
    args = _ap.Namespace(**cfg)
    exp = elf.Exp_Long_Term_Forecast(args)  # builds + loads MAE ckpt here
    elf.data_provider = patched_data_provider

    # 2. run exp.test inside a sandbox (its writes stay contained)
    os.makedirs(sandbox, exist_ok=True)
    owd = os.getcwd()
    os.chdir(sandbox)
    try:
        exp.test(setting)
    finally:
        os.chdir(owd)

    # 3. compare vs the new validation program's archived arrays
    rdir = os.path.join(sandbox, "results", setting)
    pred_off = np.load(os.path.join(rdir, "pred.npy"))
    true_off = np.load(os.path.join(rdir, "true.npy"))
    mets_off = np.load(os.path.join(rdir, "metrics.npy"))
    z = np.load(os.path.join(X.val_dir(DS, H, P), "arr_val.npz"))
    pred_val, true_val = z["pred"], z["true"]

    n = X.expected_n_windows(DS, H)
    pred_d = np.abs(pred_off.astype(np.float64) - pred_val.astype(np.float64))
    true_d = np.abs(true_off.astype(np.float64) - true_val.astype(np.float64))
    out = {
        "ds": DS, "P": P, "H": H,
        "entry": "Exp_Long_Term_Forecast.test (the method run.py invokes)",
        "loader": "official Dataset_ETT_* flag='val' under test-loader "
                  "semantics bs=32 shuffle=False drop_last=False "
                  "num_workers=10 (same batches as val_pass)",
        "compared_to": "save_t43_period_val_val/%s_%d_P%d/arr_val.npz "
                       "(val_pass.py S2 smoke)" % (DS, H, P),
        "n_windows": int(pred_off.shape[0]),
        "n_windows_expected": int(n),
        "shape_official": [int(x) for x in pred_off.shape],
        "shape_valpass": [int(x) for x in pred_val.shape],
        "pred_bitwise": bool(np.array_equal(pred_off, pred_val)),
        "true_bitwise": bool(np.array_equal(true_off, true_val)),
        "pred_max_abs_diff": float(pred_d.max()),
        "true_max_abs_diff": float(true_d.max()),
        "sha256_pred_official": sha256_arr(pred_off),
        "sha256_pred_valpass": sha256_arr(pred_val),
        "sha256_true_official": sha256_arr(true_off),
        "sha256_true_valpass": sha256_arr(true_val),
        "official_metrics_layout": [float(v) for v in mets_off[:2]],
        "official_mse": float(mets_off[1]),
        "official_mae": float(mets_off[0]),
        "sandbox": os.path.relpath(sandbox, X.ROOT),
        "wall_s": round(time.time() - t0, 1),
    }
    vj = X.load_json(X.vjson_path(DS, P, H))
    out["val_mse_from_sse"] = vj["mse_from_sse"]
    out["val_mae_from_sae"] = vj["mae_from_sae"]
    out["abs_diff_mse_official_vs_valjson"] = \
        abs(out["official_mse"] - vj["mse_from_sse"])
    out["abs_diff_mae_official_vs_valjson"] = \
        abs(out["official_mae"] - vj["mae_from_sae"])

    assert out["n_windows"] == n, "window count mismatch"
    if not (out["pred_bitwise"] and out["true_bitwise"]):
        print("[xcheck] %s P%d: NON-BITWISE pred=%s true=%s maxdiff=%.3e/%.3e"
              % (DS, P, out["pred_bitwise"], out["true_bitwise"],
                 out["pred_max_abs_diff"], out["true_max_abs_diff"]))
    X.dump_json(jp, out)
    print("[xcheck] %s P%d H%d done: pred_bitwise=%s true_bitwise=%s "
          "maxdiff=%.3e/%.3e mse_off=%.10f vs val %.10f wall=%.0fs"
          % (DS, P, H, out["pred_bitwise"], out["true_bitwise"],
             out["pred_max_abs_diff"], out["true_max_abs_diff"],
             out["official_mse"], out["val_mse_from_sse"], out["wall_s"]))


if __name__ == "__main__":
    main()
