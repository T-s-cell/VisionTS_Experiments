"""EXP-043 validation pass: one process per (dataset, candidate P),
covering all 4 horizons.

Why this harness exists (frozen protocol): the official val path is
unusable for zero-shot scoring -- data_provider(args,'val') returns
shuffle=True/drop_last=True with per-batch averaged MSE
(data_provider/data_factory.py L26-35), and Exp.vali() never runs in the
zero-shot lineage (train_epochs=0).  We therefore instantiate the OFFICIAL
Dataset_ETT_{hour,minute} classes directly (flag='val', identical border
and scaler logic) with a DataLoader that mirrors the official TEST loader
semantics (batch_size=32, shuffle=False, drop_last=False, num_workers=10),
and run a forward loop copied statement-for-statement from
Exp_Long_Term_Forecast.test (exp/exp_long_term_forecasting.py L203-258).
The --crosspath mode (smoke) re-runs the same batches through that
verbatim loop body a second time and requires bitwise-equal outputs --
same validation batches, original inference path.

Per (ds,H,P) outputs:
  logs/val_<DS>_P<P>_H<H>.json   SSE/SAE float64, mse/mae, window count,
                                 observed vs expected update_config geometry
  save_t43_period_val_val/<DS>_H<H>_P<P>/arr_val.npz   {pred,true} f32
  .../<DS>_H<H>_P<P>/VAL_OK      fingerprint (skip-if-done; mismatch ABORT)

Split definition (explicit): month rows K (hourly 720, 15T 2880);
train rows [0,12K); VAL TARGET = [12K,16K); TEST TARGET = [16K,20K).
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t43_common as X

AP = argparse.ArgumentParser()
AP.add_argument("--ds", required=True, choices=X.ETT4)
AP.add_argument("--P", required=True, type=int)
AP.add_argument("--H", type=int, default=None,
                help="single horizon (smoke); default = all 4 horizons")
AP.add_argument("--crosspath", action="store_true",
                help="second pass through the verbatim exp.test loop body; "
                     "requires bitwise-equal arrays (smoke evidence)")
ARGS = AP.parse_args()

DS, P = ARGS.ds, ARGS.P
C = X.DATASETS[DS]

# ---- fix seed exactly as run.py L14-17 (fix_seed) ----
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
torch.cuda.manual_seed_all(2021)

# ---- update_config observation wrapper (class-level, framework untouched) ----
import importlib

vm_mod = importlib.import_module("visionts.model")
GEO_RECORDS = []
_orig_uc = vm_mod.VisionTS.update_config


def _patched_update_config(self, *a, **k):
    _orig_uc(self, *a, **k)
    GEO_RECORDS.append({
        "periodicity": int(self.periodicity),
        "context_len": int(self.context_len),
        "pred_len": int(self.pred_len),
        "pad_left": int(self.pad_left), "pad_right": int(self.pad_right),
        "num_patch_input": int(self.num_patch_input),
        "scale_x": float(self.scale_x),
        "input_resize_size": str(getattr(self.input_resize, "size", None)),
        "output_resize_size": str(getattr(self.output_resize, "size", None)),
        "mask_ratio": float(self.mask_ratio),
        "t": time.time()})


vm_mod.VisionTS.update_config = _patched_update_config

from models.VisionTS import Model  # noqa: E402  (after wrapper patch)
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute  # noqa
from torch.utils.data import DataLoader  # noqa


def build_model(h):
    cfg = argparse.Namespace(
        task_name="long_term_forecast", pred_len=h, seq_len=C["ctx"],
        vm_arch="mae_base", ft_type="ln", vm_pretrained=1,
        vm_ckpt="../ckpt/", periodicity=P, interpolation="bilinear",
        norm_const=0.4, align_const=0.4)
    m = Model(cfg)
    m = m.to("cuda")
    m.eval()
    return m


def build_loader(h):
    cls = Dataset_ETT_minute if C["minute"] else Dataset_ETT_hour
    ds_ = cls(root_path=C["root_path"], data_path=C["data_path"],
              flag="val", size=[C["ctx"], X.RUN_DEFAULTS["label_len"], h],
              features="M", target="OT", scale=True,
              timeenc=1 if X.RUN_DEFAULTS["embed"] == "timeF" else 0,
              freq=X.RUN_DEFAULTS["freq"],
              seasonal_patterns=X.RUN_DEFAULTS["seasonal_patterns"])
    dl = DataLoader(ds_, batch_size=X.RUN_DEFAULTS["batch_size"],
                    shuffle=False, num_workers=X.RUN_DEFAULTS["num_workers"],
                    drop_last=False)
    return ds_, dl


def forward_loop(model, loader, h):
    """Verbatim Exp_Long_Term_Forecast.test loop body (L203-258): collects
    f32 arrays; no inverse (args.inverse default False), no amp, f_dim=0
    (features='M')."""
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to("cuda")
            batch_y = batch_y.float().to("cuda")
            batch_x_mark = batch_x_mark.float().to("cuda")
            batch_y_mark = batch_y_mark.float().to("cuda")
            dec_inp = torch.zeros_like(batch_y[:, -h:, :]).float()
            dec_inp = torch.cat([batch_y[:, :X.RUN_DEFAULTS["label_len"], :],
                                 dec_inp], dim=1).float().to("cuda")
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -h:, :]
            batch_y = batch_y[:, -h:, :]
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            outputs = outputs[:, :, 0:]
            batch_y = batch_y[:, :, 0:]
            preds.append(outputs)
            trues.append(batch_y)
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def metrics_f64(pred, true):
    p64 = pred.astype(np.float64)
    t64 = true.astype(np.float64)
    sse = float(np.sum((p64 - t64) ** 2))
    sae = float(np.sum(np.abs(p64 - t64)))
    cnt = float(p64.size)
    metric = X.load_metric()
    mae_m, mse_m = metric(pred, true)[0], metric(pred, true)[1]
    return {"sse_f64": sse, "sae_f64": sae, "cnt": cnt,
            "mse_from_sse": sse / cnt, "mae_from_sae": sae / cnt,
            "mse_metric": float(mse_m), "mae_metric": float(mae_m)}


def run_one_h(h, n_geo_before):
    vd = X.val_dir(DS, h, P)
    okp = os.path.join(vd, "VAL_OK")
    jp = X.vjson_path(DS, P, h)
    if os.path.isfile(okp) and os.path.isfile(jp):
        want = X.fingerprint_lines(DS, h, P,
                                   extra_files=["repro_t43_period_val/val_pass.py"])
        if X.read_fingerprint(okp) == want:
            print("[val_pass] %s H%d P%d: VAL_OK fingerprint match -> skip"
                  % (DS, h, P))
            return X.load_json(jp)
        print("[val_pass] %s H%d P%d: VAL_OK FINGERPRINT MISMATCH -> ABORT "
              "(no overwrite)" % (DS, h, P))
        raise SystemExit(4)

    t0 = time.time()
    model = build_model(h)
    ds_, dl = build_loader(h)

    # split sanity on the dataset object itself (explicit, not window-count):
    # len(data_x) == 4K + seq_len  (border 12K-seq .. 16K)
    K = X.month_rows(DS)
    assert len(ds_.data_x) == 4 * K + C["ctx"], \
        "data_x len %d != 4K+L %d" % (len(ds_.data_x), 4 * K + C["ctx"])

    n = len(ds_)
    exp_n = X.expected_n_windows(DS, h)
    if n != exp_n:
        raise SystemExit("[val_pass] %s H%d P%d: n_windows %d != expected %d "
                         "-> ABORT" % (DS, h, P, n, exp_n))

    pred, true = forward_loop(model, dl, h)

    mets = metrics_f64(pred, true)

    # truth bitwise vs deterministic recomputation (target rows [12K,16K))
    tref = X.truth_from_std(DS, "val", h)
    truth_bitwise = bool(np.array_equal(true, tref))
    if not truth_bitwise:
        raise SystemExit("[val_pass] %s H%d P%d: val truth NOT bitwise vs "
                         "truth_from_std -> ABORT" % (DS, h, P))

    finite = bool(np.isfinite(pred).all() and np.isfinite(true).all())
    if not finite:
        raise SystemExit("[val_pass] %s H%d P%d: non-finite values -> ABORT"
                         % (DS, h, P))

    crosspath = None
    if ARGS.crosspath:
        pred2, true2 = forward_loop(model, dl, h)
        crosspath = {
            "pred_bitwise": bool(np.array_equal(pred, pred2)),
            "true_bitwise": bool(np.array_equal(true, true2)),
            "loop": "verbatim exp.test L203-258 body, same val dataset/loader",
        }
        if not (crosspath["pred_bitwise"] and crosspath["true_bitwise"]):
            raise SystemExit("[val_pass] %s H%d P%d: crosspath outputs differ "
                             "-> ABORT" % (DS, h, P))

    geo_obs = GEO_RECORDS[n_geo_before]
    geo_exp = X.expected_geometry(DS, h, P)
    geo_ok, geo_notes = X.geometry_matches(geo_obs, geo_exp)
    if not geo_ok:
        raise SystemExit("[val_pass] %s H%d P%d: geometry mismatch %s -> ABORT"
                         % (DS, h, P, geo_notes))

    os.makedirs(vd, exist_ok=True)
    np.savez_compressed(os.path.join(vd, "arr_val.npz"),
                        pred=pred, true=true)
    out = {
        "ds": DS, "P": P, "H": h, "ctx": C["ctx"],
        "n_windows": int(n), "expected_n_windows": int(exp_n),
        "split": {"K": K, "train_end": 12 * K, "val_start": 12 * K,
                  "val_end": 16 * K, "first_target_row": 12 * K,
                  "last_target_row_end": 16 * K},
        "truth_bitwise_vs_truth_from_std": truth_bitwise,
        "finite": finite,
        **mets,
        "geometry_observed": geo_obs,
        "geometry_expected": geo_exp,
        "geometry_ok": geo_ok,
        "crosspath": crosspath,
        "pred_shape": [int(x) for x in pred.shape],
        "wall_s": round(time.time() - t0, 1),
        "fingerprint": X.fingerprint_lines(
            DS, h, P, extra_files=["repro_t43_period_val/val_pass.py"]),
    }
    X.dump_json(jp, out)
    X.write_text(okp, out["fingerprint"])
    print("[val_pass] %s H%d P%d done: n=%d mse=%.10f mae=%.10f wall=%.0fs"
          % (DS, h, P, n, mets["mse_from_sse"], mets["mae_from_sae"],
             out["wall_s"]))
    return out


def main():
    os.makedirs(X.LOG_DIR, exist_ok=True)
    hs = [ARGS.H] if ARGS.H else list(X.HORIZONS)
    outs = []
    for h in hs:
        outs.append(run_one_h(h, n_geo_before=len(GEO_RECORDS)))
    print("[val_pass] %s P%d: horizons %s complete" % (DS, P, hs))


if __name__ == "__main__":
    main()
