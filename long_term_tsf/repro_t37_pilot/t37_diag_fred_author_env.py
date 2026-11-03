"""EXP-037 G4 follow-up (user verdict 2026-11-04, task 1): FRED-MD series-level
controlled comparison between the current software env and an author-pinned
env (python 3.8.18 / torch 1.7.1 / torchvision 0.8.2 / timm 0.3.2, README.md:95).
Env-agnostic: run once per env with --tag {current,author}.

Fixed for both runs: same frozen ckpt (sha256 asserted), 107 series, last 300
steps input, 12-step truth, P=1, batch 512, fp64 forward, official run.forecast
path. The eval-local dataset.py generate is bound over dataset_recipes for
fred_md in BOTH envs so the recipe code is byte-identical across envs (only
numpy/pandas/gluonts/torch differ); per-series truth bitwise equality across
the two runs is asserted by t37_diag_fred_series_compare.py.

Outputs (in repro_t37_pilot/logs/):
  fred_series_<tag>.npz   preds/truth/abs_err float64 (107, 12)
  fred_series_<tag>.json  versions, asserts, totals, per-series MAE

Run (current env):  $VISIONTS_PY t37_diag_fred_author_env.py --tag current
Run (author env):   HOME=/dev_data/wlt/t37_author_home \
                    /dev_data/wlt/conda/envs/visionts_t37author/bin/python \
                    t37_diag_fred_author_env.py --tag author
"""
import argparse
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t37_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["current", "author"])
    args = ap.parse_args()

    ckpt_path = os.path.abspath(os.path.join(C.ROOT, "..", "ckpt",
                                             "mae_visualize_vit_base.pth"))
    assert os.path.isfile(ckpt_path) and C.sha256_of(ckpt_path) == C.CKPT_SHA256, ckpt_path

    os.chdir(C.EVAL_GLUONTS)
    if C.EVAL_GLUONTS not in sys.path:
        sys.path.insert(0, C.EVAL_GLUONTS)
    import run
    import dataset as eval_dataset_mod
    import gluonts
    import gluonts.dataset.repository.datasets as gds
    import gluonts.dataset.repository._tsf_datasets as tsf_mod
    import pandas as pd
    import torch
    import torchvision
    import timm
    import einops
    from functools import partial
    assert os.path.samefile(run.__file__, os.path.join(C.EVAL_GLUONTS, "run.py")), run.__file__
    assert os.path.samefile(eval_dataset_mod.__file__,
                            os.path.join(C.EVAL_GLUONTS, "dataset.py")), eval_dataset_mod.__file__

    versions = {"tag": args.tag, "python": platform.python_version(),
                "torch": torch.__version__, "torchvision": torchvision.__version__,
                "timm": timm.__version__, "gluonts": gluonts.__version__,
                "pandas": pd.__version__, "numpy": np.__version__,
                "einops": einops.__version__, "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "home": os.path.expanduser("~"),
                "recipe_binding": "eval_local_dataset_py_generate (bound below, identical code both envs)"}

    # pandas>=2.2 M->ME alias shim; inert when the native map hits (pandas<2.2)
    _orig_fallback = eval_dataset_mod.default_prediction_length_from_frequency
    shim = {"fired": False}

    def fallback_compat(freq):
        try:
            return _orig_fallback(freq)
        except ValueError:
            from pandas.tseries.frequencies import to_offset
            name = str(to_offset(freq).name)
            alias = {"ME": 12, "YE": 4}
            if name not in alias:
                raise
            shim["fired"] = True
            shim["offset_name"] = name
            return alias[name]

    eval_dataset_mod.default_prediction_length_from_frequency = fallback_compat

    # bind the eval-local generate for fred_md in BOTH envs -> identical
    # recipe code; its native map gives M->12 (pandas<2.2) or via shim (2.2)
    gds.dataset_recipes["fred_md"] = partial(
        eval_dataset_mod.generate_forecasting_dataset, dataset_name="fred_md")

    rec = {"versions": versions, "shim": shim}
    t0 = time.time()
    test_data, metadata = eval_dataset_mod.get_gluonts_test_dataset("fred_md", regenerate=True)
    data_train = [run.imputation_nan(x["target"]) for x in test_data.input]
    data_test = [x["target"] for x in test_data.label]
    pred_len = len(data_test[0])
    input_lens = sorted({len(x) for x in data_train})
    rec["split"] = {"meta_prediction_length": int(metadata.prediction_length),
                    "meta_freq": str(metadata.freq),
                    "runtime_pred_len": pred_len, "n_series": len(data_test),
                    "input_len_set": input_lens}
    assert pred_len == 12 and len(data_test) == 107 and input_lens == [716], rec["split"]

    model = run.VisionTS("mae_base", ckpt_dir="../ckpt/").to("cuda:0")
    train = [x[-300:] for x in data_train]
    preds32 = run.forecast(model, train, data_test, 512, "cuda:0", 1)

    truth = np.stack(data_test).astype(np.float64)
    preds = preds32.astype(np.float64)
    abs_err = np.abs(truth - preds)
    per_mae = abs_err.mean(axis=1)
    mae = float(abs_err.mean())
    mse = float(((truth - preds) ** 2).mean())
    rec["metrics_f64"] = {"mae": mae, "mse": mse,
                          "per_series_mae_mean_check": float(per_mae.mean())}
    rec["anchor"] = C.AUTHOR_ANCHOR["fred_md"]["mae"]
    rec["rel_diff_vs_anchor"] = abs(mae - C.AUTHOR_ANCHOR["fred_md"]["mae"]) / C.AUTHOR_ANCHOR["fred_md"]["mae"]
    rec["wall_sec"] = time.time() - t0
    if args.tag == "current":
        formal = 1807.417940070353  # formal-run MAE (visionts_monash CSV)
        ok = abs(mae - formal) <= 1e-6
        rec["regression_vs_formal"] = {"formal_mae": formal, "abs_diff": abs(mae - formal),
                                       "ok": bool(ok)}
        assert ok, rec["regression_vs_formal"]

    npz = os.path.join(C.LOG_DIR, "fred_series_%s.npz" % args.tag)
    np.savez_compressed(npz, preds=preds, truth=truth, abs_err=abs_err)
    rec["npz"] = npz
    C.dump_json(os.path.join(C.LOG_DIR, "fred_series_%s.json" % args.tag), rec)
    print("[t37][fred-series][%s] mae=%.9f mse=%.3f anchor_rel=%.4e wall=%.1fs npz=%s"
          % (args.tag, mae, mse, rec["rel_diff_vs_anchor"], rec["wall_sec"], npz))


if __name__ == "__main__":
    main()
