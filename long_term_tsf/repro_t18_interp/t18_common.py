"""EXP-034 helpers: VisionTS paper Table 18 — interpolation-strategy
ablation (bilinear vs bicubic vs nearest), original image orientation,
zero-shot, ETT (4 datasets x H in {96,192,336,720}, Table 8 lookbacks:
ETTh1 2880/24, ETTh2 1728/24, ETTm1 2304/96, ETTm2 4032/96).

Bilinear column = EXP-001 (save_repro_zeroshot/, verified by EXP-031)
reused in full; its overall self-check (user recomputation, also the
Average row of EXP-031's own summary) is
0.34439535439014435 / 0.37042006477713585 (hard gate <=1e-6).
Bicubic/nearest = 32 new frozen-inference groups under save_t18_interp/
{interp}/{DS}_{H} via `--interpolation <mode>` (already wired end-to-end:
run.py:106 -> models/VisionTS.py:22 update_config -> visionts/model.py:84-92
builds BOTH input_resize and output_resize from the ONE value).

Corrected window counts (EXP-031 win_start; 16 groups 109,840 windows):
hourly test N = 2881 - H, first predicted csv row (0-based) = 11520;
minute N = 11521 - H, first row = 46080 — independent of L, so truths are
bitwise comparable across EXP-001 / EXP-003 / this experiment.

Paper Table 18 anchors: transcribed cell-by-cell from
audit_supplement/paper/VisionTS_paper_full.md (Table 18, line 454) by two
independent reads. Per-H cells, per-dataset "avg" rows and the "Average"
row are THREE INDEPENDENT tables exactly as printed — the paper averages
unrounded values, so recomputing its avg from the printed 3-decimal per-H
cells does NOT reproduce the printed row (e.g. ETTh1 nearest MAE:
mean-of-printed = 0.44675 vs printed 0.446). NO code cross-check between
the levels; measured aggregation uses unrounded metrics.

Imaging geometry is interpolation-independent (visionts/model.py:60-92):
MAE-base here is ViT-B/16 -> image 224, patch 16, num_patch 14 (the probe
asserts these against the loaded ckpt). Expected geometry per (ds, H) is
recomputed from the exact update_config formulas and asserted at probe
time; ETTh1_96 yields num_patch_input=5, input_resize (224, 80),
output_resize (24, 336).
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # long_term_tsf/
REPO = os.path.dirname(ROOT)
PKG = os.path.dirname(os.path.abspath(__file__))
SAVE_BASE = os.path.join(ROOT, "save_t18_interp")
EXP001_BASE = os.path.join(ROOT, "save_repro_zeroshot")
EXP003_BASE = os.path.join(ROOT, "save_fullshot")

HORIZONS = [96, 192, 336, 720]
INTERPS = ["bilinear", "bicubic", "nearest"]
VARIANT_INTERPS = ["bicubic", "nearest"]

# paper Table 8 lookbacks (EXP-001 case table)
DATASETS_T8 = {
    "ETTh1": dict(ctx=2880, period=24, root_path="./dataset/ETT-small/",
                  data_path="ETTh1.csv", data="ETTh1", minute=False),
    "ETTh2": dict(ctx=1728, period=24, root_path="./dataset/ETT-small/",
                  data_path="ETTh2.csv", data="ETTh2", minute=False),
    "ETTm1": dict(ctx=2304, period=96, root_path="./dataset/ETT-small/",
                  data_path="ETTm1.csv", data="ETTm1", minute=True),
    "ETTm2": dict(ctx=4032, period=96, root_path="./dataset/ETT-small/",
                  data_path="ETTm2.csv", data="ETTm2", minute=True),
}

CKPT_REL = "../ckpt/mae_visualize_vit_base.pth"
CKPT_DIR_REL = "../ckpt/"  # --vm_ckpt takes the DIR (EXP-001 protocol); VisionTS joins the filename
CKPT_SHA256 = "c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e"
CKPT_BYTES = 447725789

MAE_IMAGE_SIZE = 224
MAE_PATCH_SIZE = 16
MAE_NUM_PATCH = 14

BILINEAR_SELF_CHECK = {"overall_mse": 0.34439535439014435,
                       "overall_mae": 0.37042006477713585}
SELF_TOL = 1e-6

# package-carried snapshot of EXP-031 summary_b4_seasonal.csv (the
# original-period VisionTS column = EXP-001 metrics.npy, full precision)
EXP031_SNAPSHOT = os.path.join(PKG, "exp031_visionts_column_snapshot.csv")
EXP031_SHA256 = "b454fb9a3fc4da15930c9934e117827a2bcdc8f3c8f7f17e2622445aaaedafbf"

# paper Table 18, interpolation sub-table ONLY (image-orientation half is
# out of scope). {ds: {h: {"bilinear": (mse,mae), "bicubic": ..., "nearest": ...}}}
PAPER_T18 = {
    "ETTh1": {
        96:  {"bilinear": (0.353, 0.383), "bicubic": (0.351, 0.383), "nearest": (0.426, 0.424)},
        192: {"bilinear": (0.392, 0.410), "bicubic": (0.392, 0.409), "nearest": (0.450, 0.443)},
        336: {"bilinear": (0.407, 0.423), "bicubic": (0.407, 0.422), "nearest": (0.451, 0.450)},
        720: {"bilinear": (0.406, 0.441), "bicubic": (0.405, 0.440), "nearest": (0.454, 0.470)},
    },
    "ETTh2": {
        96:  {"bilinear": (0.271, 0.328), "bicubic": (0.274, 0.329), "nearest": (0.298, 0.349)},
        192: {"bilinear": (0.328, 0.367), "bicubic": (0.330, 0.367), "nearest": (0.343, 0.380)},
        336: {"bilinear": (0.345, 0.381), "bicubic": (0.345, 0.380), "nearest": (0.373, 0.401)},
        720: {"bilinear": (0.388, 0.422), "bicubic": (0.386, 0.419), "nearest": (0.404, 0.431)},
    },
    "ETTm1": {
        96:  {"bilinear": (0.341, 0.347), "bicubic": (0.366, 0.354), "nearest": (0.399, 0.374)},
        192: {"bilinear": (0.360, 0.360), "bicubic": (0.383, 0.367), "nearest": (0.397, 0.376)},
        336: {"bilinear": (0.377, 0.374), "bicubic": (0.396, 0.381), "nearest": (0.386, 0.380)},
        720: {"bilinear": (0.416, 0.405), "bicubic": (0.429, 0.409), "nearest": (0.417, 0.409)},
    },
    "ETTm2": {
        96:  {"bilinear": (0.228, 0.282), "bicubic": (0.246, 0.296), "nearest": (0.264, 0.326)},
        192: {"bilinear": (0.262, 0.305), "bicubic": (0.273, 0.313), "nearest": (0.273, 0.328)},
        336: {"bilinear": (0.293, 0.328), "bicubic": (0.303, 0.334), "nearest": (0.297, 0.343)},
        720: {"bilinear": (0.343, 0.370), "bicubic": (0.343, 0.370), "nearest": (0.334, 0.369)},
    },
}
# per-dataset "avg" rows exactly as printed (NOT recomputed — see docstring)
PAPER_T18_DS = {
    "ETTh1": {"bilinear": (0.390, 0.414), "bicubic": (0.389, 0.414), "nearest": (0.445, 0.446)},
    "ETTh2": {"bilinear": (0.333, 0.375), "bicubic": (0.334, 0.374), "nearest": (0.354, 0.390)},
    "ETTm1": {"bilinear": (0.374, 0.372), "bicubic": (0.393, 0.378), "nearest": (0.400, 0.384)},
    "ETTm2": {"bilinear": (0.282, 0.321), "bicubic": (0.291, 0.328), "nearest": (0.292, 0.341)},
}
PAPER_T18_OVERALL = {"bilinear": (0.344, 0.370), "bicubic": (0.352, 0.373),
                     "nearest": (0.373, 0.391)}

FP_FILES = [
    "run.py", "exp/exp_long_term_forecasting.py", "exp/exp_basic.py",
    "models/VisionTS.py", "data_provider/data_factory.py",
    "data_provider/data_loader.py",
    "../visionts/model.py", "../visionts/models_mae.py",
]

_ETT4 = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
_STD_CACHE = {}


def group_ids():
    return [(ds, h) for ds in _ETT4 for h in HORIZONS]


def month_rows(ds):
    return 30 * 24 * (4 if DATASETS_T8[ds]["minute"] else 1)


def expected_n_windows(ds, h):
    # hourly: 4*720 - h + 1 = 2881-h; minute: 4*2880 - h + 1 = 11521-h
    return 4 * month_rows(ds) - h + 1


def first_pred_abs_row0(ds):
    # hourly 11520, minute 46080; independent of L
    return 16 * month_rows(ds)


def expected_geometry(ds, h):
    """Exact update_config geometry (visionts/model.py:60-92) for the
    pinned MAE-base (image 224, patch 16, num_patch 14); the probe asserts
    the live model against this. ETTh1_96 -> npi=5, in (224,80), out (24,336)."""
    c = DATASETS_T8[ds]
    ctx, p = c["ctx"], c["period"]
    align = 0.4
    pad_left = 0 if ctx % p == 0 else p - ctx % p
    pad_right = 0 if h % p == 0 else p - h % p
    input_ratio = (pad_left + ctx) / (pad_left + ctx + pad_right + h)
    npi = int(input_ratio * MAE_NUM_PATCH * align)
    if npi == 0:
        npi = 1
    adjust = npi / MAE_NUM_PATCH
    w_in = int(MAE_IMAGE_SIZE * adjust)
    scale_x = ((pad_left + ctx) // p) / w_in
    w_out = int(round(MAE_IMAGE_SIZE * scale_x))
    return {"pad_left": pad_left, "pad_right": pad_right,
            "num_patch_input": npi,
            "num_patch_output": MAE_NUM_PATCH - npi,
            "input_size": [MAE_IMAGE_SIZE, w_in],
            "output_size": [p, w_out]}


def exp001_dir(ds, h):
    return os.path.join(EXP001_BASE, "%s_%d" % (ds, h))


def exp003_dir(ds, h):
    return os.path.join(EXP003_BASE, "%s_%d_seed2021" % (ds, h))


def interp_dir(ds, h, interp):
    return os.path.join(SAVE_BASE, interp, "%s_%d" % (ds, h))


def std_array(ds):
    """Standardized full csv array replicating data_provider/data_loader.py:
    7 columns, sklearn StandardScaler fit on train rows [0, 12 months),
    transform all, float32 cast (what the model saw)."""
    if ds in _STD_CACHE:
        return _STD_CACHE[ds]
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    c = DATASETS_T8[ds]
    df = pd.read_csv(os.path.join(ROOT, c["root_path"], c["data_path"]))
    vals = df[df.columns[1:]].values.astype(np.float64)  # (rows, 7) f64
    k = month_rows(ds)
    scaler = StandardScaler()
    scaler.fit(vals[0:12 * k])
    st = scaler.transform(vals).astype(np.float32)
    _STD_CACHE[ds] = st
    return st


def truth_recomputed(st, ds, h):
    """Stack std[t0+i : t0+i+h] over test windows -> (N, H, 7); bitwise
    equal to EXP-001 true.npy for all 16 groups (EXP-031 verified)."""
    t0 = first_pred_abs_row0(ds)
    n = expected_n_windows(ds, h)
    seg = st[t0:t0 + n + h - 1]
    sw = np.lib.stride_tricks.sliding_window_view(seg, h, axis=0)  # (n, 7, h)
    return np.ascontiguousarray(sw.transpose(0, 2, 1))             # (n, h, 7)


def exp003_truth_restored(ds, h):
    """Load EXP-003 seed2021 true.npy (N*7, H, 1), assert shape, restore to
    (N, H, 7) with the FIXED reshape (flatten index 7*win + var)."""
    p = os.path.join(exp003_dir(ds, h), "results", "_", "true.npy")
    t = np.load(p)
    n = expected_n_windows(ds, h)
    assert t.shape == (n * 7, h, 1), \
        "%s_%d EXP-003 true shape %s != (%d*7, %d, 1)" % (ds, h, t.shape, n, h)
    rec = t[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
    return np.ascontiguousarray(rec.astype(np.float32))


def exp031_visionts_column():
    """Snapshot of EXP-031 summary_b4_seasonal.csv (sha256-gated) ->
    {(ds, h): {"mse","mae","n"}} for its VisionTS rows (per-group), plus
    {"ds_avg": {ds: (mse,mae)}, "overall": (mse,mae)} from the avg/Average
    rows. This IS the original-period VisionTS column EXP-031 verified
    (taken from EXP-001 metrics.npy at full precision)."""
    got = sha256_of(EXP031_SNAPSHOT)
    if got != EXP031_SHA256:
        raise SystemExit("[t18] exp031 snapshot sha256 %s != pinned %s"
                         % (got, EXP031_SHA256))
    per, ds_avg, overall = {}, {}, None
    with open(EXP031_SNAPSHOT) as f:
        header = f.readline().strip().split(",")
        assert header[:3] == ["dataset", "pred_len", "method"], header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9 or parts[2] != "VisionTS":
                continue
            ds, pred = parts[0], parts[1]
            mse, mae = float(parts[7]), float(parts[8])
            if pred == "avg":
                if ds == "Average":
                    overall = (mse, mae)
                else:
                    ds_avg[ds] = (mse, mae)
            else:
                per[(ds, int(pred))] = {"mse": mse, "mae": mae,
                                        "n": int(parts[11]) if parts[11] else None}
    for ds in _ETT4:
        for h in HORIZONS:
            assert (ds, h) in per, "exp031 snapshot missing %s_%d" % (ds, h)
    assert overall is not None and len(ds_avg) == 4
    if (abs(overall[0] - BILINEAR_SELF_CHECK["overall_mse"]) > SELF_TOL
            or abs(overall[1] - BILINEAR_SELF_CHECK["overall_mae"]) > SELF_TOL):
        raise SystemExit("[t18] exp031 snapshot overall %s != pinned self-check"
                         % (overall,))
    return {"per_group": per, "ds_avg": ds_avg, "overall": overall}


def load_metric():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from utils.metrics import metric
    return metric


def metrics_from_arrays(metric, pred, true):
    mae, mse, rmse, mape, mspe = metric(pred, true)
    return np.array([mae, mse, rmse, mape, mspe], dtype=np.float64)


def parse_success(path):
    """SUCCESS.txt -> {"config": {k: v}, "md5": {f: md5}}; config lines are
    space-separated key=value tokens (ckpt_sha256 included)."""
    cfg, md5s = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("md5:") and "=" in line:
                k, v = line[4:].rsplit("=", 1)
                md5s[k] = v
                continue
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    cfg[k] = v
    return {"config": cfg, "md5": md5s}


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of(path):
    h = hashlib.sha256()
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


def merge_json(path, section, obj):
    data = {}
    if os.path.isfile(path):
        data = load_json(path)
    data[section] = obj
    dump_json(path, data)
