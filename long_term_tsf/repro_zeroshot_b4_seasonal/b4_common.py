"""EXP-031 helpers: VisionTS paper Appendix B.4 Table 12 seasonal baselines
(Seasonal Naive / Seasonal Avg) on the EXP-001 zero-shot protocol.
Baseline = EXP-001 zero-shot runs (save_repro_zeroshot/, framework files
md5-fingerprinted equivalent to commit 71da50a). Data loading and
StandardScaler are reused exactly like data_provider/data_loader.py
(fit on train rows [0,12k), transform all), cast to float32 (what the
model saw). numpy/pandas/sklearn only; the original metric() is imported
from utils.metrics for standardized MSE/MAE.
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # long_term_tsf/
REPO = os.path.dirname(ROOT)
PKG = os.path.dirname(os.path.abspath(__file__))
SAVE_BASE = os.path.join(ROOT, "save_b4_seasonal")
BASE_SAVE_BASE = os.path.join(ROOT, "save_repro_zeroshot")
A2P1_SAVE_BASE = os.path.join(ROOT, "save_a2p1_zeroshot")
REPRO_ZERO_DIR = os.path.join(ROOT, "repro_zeroshot")

HORIZONS = [96, 192, 336, 720]
ETT4 = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
METHODS = ["naive", "avg"]  # seasonal baselines; "visionts" = EXP-001 orig-P

# per-dataset config, identical to repro_zeroshot/run_one.sh EXP-001 table
DATASETS = {
    "ETTh1": dict(ctx=2880, period=24, root_path="./dataset/ETT-small/",
                  data_path="ETTh1.csv", data="ETTh1", minute=False),
    "ETTh2": dict(ctx=1728, period=24, root_path="./dataset/ETT-small/",
                  data_path="ETTh2.csv", data="ETTh2", minute=False),
    "ETTm1": dict(ctx=2304, period=96, root_path="./dataset/ETT-small/",
                  data_path="ETTm1.csv", data="ETTm1", minute=True),
    "ETTm2": dict(ctx=4032, period=96, root_path="./dataset/ETT-small/",
                  data_path="ETTm2.csv", data="ETTm2", minute=True),
}

# paper Appendix B.4 Table 12 (per-H rows and 4H-average "avg" rows,
# printed to 3 decimals; extracted from papers/VisionTS/full.md Table 12)
# {ds: {h_or_avg: {"visionts": (mse, mae), "naive": (...), "avg": (...)}}}
PAPER_T12 = {
    "ETTh1": {
        96:   {"visionts": (0.353, 0.383), "naive": (0.512, 0.433), "avg": (0.589, 0.585)},
        192:  {"visionts": (0.392, 0.410), "naive": (0.581, 0.469), "avg": (0.598, 0.590)},
        336:  {"visionts": (0.407, 0.423), "naive": (0.650, 0.501), "avg": (0.610, 0.597)},
        720:  {"visionts": (0.406, 0.441), "naive": (0.655, 0.514), "avg": (0.656, 0.624)},
        "avg": {"visionts": (0.390, 0.414), "naive": (0.600, 0.479), "avg": (0.613, 0.599)},
    },
    "ETTh2": {
        96:   {"visionts": (0.271, 0.328), "naive": (0.391, 0.380), "avg": (0.457, 0.494)},
        192:  {"visionts": (0.328, 0.367), "naive": (0.482, 0.429), "avg": (0.466, 0.500)},
        336:  {"visionts": (0.345, 0.381), "naive": (0.532, 0.466), "avg": (0.476, 0.509)},
        720:  {"visionts": (0.388, 0.422), "naive": (0.525, 0.474), "avg": (0.542, 0.548)},
        "avg": {"visionts": (0.333, 0.375), "naive": (0.483, 0.437), "avg": (0.485, 0.513)},
    },
    "ETTm1": {
        96:   {"visionts": (0.341, 0.347), "naive": (0.423, 0.387), "avg": (0.369, 0.399)},
        192:  {"visionts": (0.360, 0.360), "naive": (0.463, 0.406), "avg": (0.374, 0.402)},
        336:  {"visionts": (0.377, 0.374), "naive": (0.496, 0.426), "avg": (0.382, 0.407)},
        720:  {"visionts": (0.416, 0.405), "naive": (0.574, 0.464), "avg": (0.394, 0.416)},
        "avg": {"visionts": (0.374, 0.372), "naive": (0.489, 0.421), "avg": (0.380, 0.406)},
    },
    "ETTm2": {
        96:   {"visionts": (0.228, 0.282), "naive": (0.263, 0.301), "avg": (0.365, 0.411)},
        192:  {"visionts": (0.262, 0.305), "naive": (0.321, 0.337), "avg": (0.369, 0.414)},
        336:  {"visionts": (0.293, 0.328), "naive": (0.376, 0.370), "avg": (0.375, 0.418)},
        720:  {"visionts": (0.343, 0.370), "naive": (0.471, 0.422), "avg": (0.380, 0.423)},
        "avg": {"visionts": (0.282, 0.321), "naive": (0.358, 0.357), "avg": (0.372, 0.417)},
    },
}
PAPER_T12_OVERALL = {"visionts": (0.344, 0.370), "naive": (0.482, 0.424),
                     "avg": (0.463, 0.484)}

# baseline fingerprint file list: ONLY the files EXP-001 already recorded in
# its SUCCESS.txt (preflight checks these; EXP-031's own scripts are
# fingerprinted separately into this experiment's SUCCESS.txt)
FP_FILES = [
    "run.py", "exp/exp_long_term_forecasting.py", "exp/exp_basic.py",
    "models/VisionTS.py", "data_provider/data_factory.py",
    "data_provider/data_loader.py",
    "../visionts/model.py", "../visionts/models_mae.py",
]
# EXP-031's own scripts (recorded separately in this experiment's artifacts)
PKG_FP_FILES = [
    "repro_zeroshot_b4_seasonal/b4_common.py",
    "repro_zeroshot_b4_seasonal/run_b4.py",
]

_STD_CACHE = {}


def group_ids():
    return [(ds, h) for ds in ETT4 for h in HORIZONS]


def base_dir(ds, h):
    return os.path.join(BASE_SAVE_BASE, "%s_%d" % (ds, h))


def method_dir(ds, h, method):
    return os.path.join(SAVE_BASE, "%s_%d" % (ds, h), method)


def month_rows(ds):
    # rows per 30-day month block in Dataset_ETT_{hour,minute} borders
    return 30 * 24 * (4 if DATASETS[ds]["minute"] else 1)


def expected_n_windows(ds, h):
    # border1s[2]=12k+4k-seq, border2s[2]=12k+8k -> data_x len = 4k+seq,
    # N = 4k+seq-seq-h+1 = 4k-h+1 (independent of seq_len)
    return 4 * month_rows(ds) - h + 1


def first_pred_abs_row0(ds):
    # absolute csv row of the first predicted step for test window i = this + i
    return 16 * month_rows(ds)


def std_array(ds):
    """Standardized test-relevant array, replicating data_provider
    data_loader.py exactly: columns = csv header[1:] (7 vars), sklearn
    StandardScaler fit on train rows [0,12k), transform all, then float32
    cast (the values the model actually saw through torch)."""
    if ds in _STD_CACHE:
        return _STD_CACHE[ds]
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    c = DATASETS[ds]
    df = pd.read_csv(os.path.join(ROOT, c["root_path"], c["data_path"]))
    vals = df[df.columns[1:]].values.astype(np.float64)  # (rows, 7) f64
    k = month_rows(ds)
    scaler = StandardScaler()
    scaler.fit(vals[0:12 * k])
    data = scaler.transform(vals)  # f64, full range
    st = data.astype(np.float32)
    _STD_CACHE[ds] = st
    return st


def channel_names(ds):
    import pandas as pd
    c = DATASETS[ds]
    df = pd.read_csv(os.path.join(ROOT, c["root_path"], c["data_path"]), nrows=1)
    return [str(x) for x in df.columns[1:]]


def load_metric():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from utils.metrics import metric
    return metric


def seasonal_naive(hist, period, horizon):
    """Repeat the last `period` steps in original order to `horizon` steps.
    hist: (L, 7) float32; returns (H, 7) float32 (pure row indexing)."""
    template = hist[-period:]
    idx = np.arange(horizon) % period
    return template[idx]


def seasonal_avg(hist, period, horizon):
    """Average all complete cycles of the look-back into a (P, 7) template,
    then repeat it along the time axis: template[arange(H) % P]."""
    n_rows = hist.shape[0]
    n_cycles = n_rows // period
    assert n_cycles * period == n_rows, "look-back L not divisible by P"
    template = hist.reshape(n_cycles, period, hist.shape[1]).mean(axis=0)
    idx = np.arange(horizon) % period
    return template[idx]


def truth_recomputed(st, ds, h):
    """Stack ST[t:t+h] over test windows; equals EXP-001 true.npy bitwise."""
    t0 = first_pred_abs_row0(ds)
    n = expected_n_windows(ds, h)
    seg = st[t0:t0 + n + h - 1]
    sw = np.lib.stride_tricks.sliding_window_view(seg, h, axis=0)  # (n, 7, h)
    return np.ascontiguousarray(sw.transpose(0, 2, 1))             # (n, h, 7)


def window_errors(pred, true):
    """Per-window (and per-variable) f32 errors averaged over forecast steps."""
    err = pred - true  # (N, H, 7) f32
    return (np.mean(np.square(err), axis=1),   # win_var_mse (N, 7)
            np.mean(np.abs(err), axis=1),      # win_var_mae (N, 7)
            np.mean(np.square(err), axis=(1, 2)),  # win_mse (N,)
            np.mean(np.abs(err), axis=(1, 2)))     # win_mae (N,)


def win_start(ds, n):
    return first_pred_abs_row0(ds) + np.arange(n, dtype=np.int64)


def md5_of(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
        f.write("\n")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_log_metrics(txt):
    import re
    m = re.search(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", txt)
    if not m:
        return None
    return {"mse": float(m.group(1)), "mae": float(m.group(2))}


def math_ceil_div(a, b):
    return -(-a // b)
