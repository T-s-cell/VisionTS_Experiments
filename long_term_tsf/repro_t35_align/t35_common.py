"""EXP-035 helpers: VisionTS paper appendix B.9 Figure 9(c) — alignment
constant c sensitivity on ETTm1, zero-shot, H in {96,192,336,720},
c in {0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}. EXP-001 zero-shot protocol
(Table 8 lookback: ETTm1 L=2304/P=96, r=0.4, bilinear, original image
orientation, bs=32, seed=2021, train_epochs=0, frozen MAE-base).

c=0.4 column reuses EXP-001 (save_repro_zeroshot/ETTm1_{H}/, verified by
EXP-031); its four-H equal-weight average (user recomputation, hard gate
<=1e-6) is 0.373517170548439 / 0.37162137776613235. The other 8 c values
= 32 new frozen-inference groups under save_t35_align/c{C}/ETTm1_{H}
via `--align_const {C}` (run.py:108 float default 0.4 ->
models/VisionTS.py:22 update_config -> visionts/model.py
npi = int(input_ratio * num_patch * c), one value driving the geometry;
bilinear resize unchanged).

Window counts (EXP-031 win_start; minute dataset, independent of L/c):
N = 11521 - H = 11425/11329/11185/10801, first predicted csv row
(0-based) = 46080 -> window-0 history rows [43776, 46080), future truth
[46080, 46080+H).

Paper anchors (trend ONLY — the figure has no per-point numbers):
audit_supplement/paper/VisionTS_paper_full.md — B.9 at L424, "MSE
averaged on four prediction lengths" L426, Figure 9 caption L494, (c)
ETTm1 sub-figure L483-484, n = floor(c*N*L/(L+H)) and c=0.4 default at
L68, qualitative "around 0.4" claim at L134. Original (c) panel image:
audit_supplement/paper/images/7cfc719171b33e2c898c86960e818b2ae407fc121
2bccc20880d4d8c0bad634c.jpg.

Geometry per (H, c) from the exact update_config formulas (patch 16 ->
image 224, num_patch 14; MAE-base asserted at probe time). H=336/720
have a 48-step right padding. Reference values: H=96 c=0.4 -> npi=5,
input 224x80, output 96x67; H=96 c=0.6 -> npi=8, 224x128, 96x42.
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # long_term_tsf/
PKG = os.path.dirname(os.path.abspath(__file__))
SAVE_BASE = os.path.join(ROOT, "save_t35_align")
EXP001_BASE = os.path.join(ROOT, "save_repro_zeroshot")
EXP003_BASE = os.path.join(ROOT, "save_fullshot")

HORIZONS = [96, 192, 336, 720]
ALIGN_C = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BASELINE_C = 0.4
VARIANT_C = [c for c in ALIGN_C if c != BASELINE_C]

ETTM1 = dict(ctx=2304, period=96, root_path="./dataset/ETT-small/",
             data_path="ETTm1.csv", data="ETTm1", minute=True)

CKPT_REL = "../ckpt/mae_visualize_vit_base.pth"
CKPT_DIR_REL = "../ckpt/"  # --vm_ckpt takes the DIR (EXP-001 protocol); VisionTS joins the filename
CKPT_SHA256 = "c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e"
CKPT_BYTES = 447725789

MAE_IMAGE_SIZE = 224
MAE_PATCH_SIZE = 16
MAE_NUM_PATCH = 14

BASELINE_SELF_CHECK = {"avg_mse": 0.373517170548439,
                       "avg_mae": 0.37162137776613235}
SELF_TOL = 1e-6

# package-carried snapshot of EXP-031 summary_b4_seasonal.csv (the
# c=0.4 VisionTS column = EXP-001 metrics.npy, full precision)
EXP031_SNAPSHOT = os.path.join(PKG, "exp031_visionts_column_snapshot.csv")
EXP031_SHA256 = "b454fb9a3fc4da15930c9934e117827a2bcdc8f3c8f7f17e2622445aaaedafbf"

# paper Figure 9(c) original panel (trend comparison only)
FIG9C_IMAGE = os.path.join(ROOT, "..", "audit_supplement", "paper", "images",
                           "7cfc719171b33e2c898c86960e818b2ae407fc1212bccc20880d4d8c0bad634c.jpg")

FP_FILES = [
    "run.py", "exp/exp_long_term_forecasting.py", "exp/exp_basic.py",
    "models/VisionTS.py", "data_provider/data_factory.py",
    "data_provider/data_loader.py",
    "../visionts/model.py", "../visionts/models_mae.py",
]

_ETTM1_DIR_CACHE = {}


def c_str(c):
    return "%.1f" % c


def month_rows():
    return 30 * 24 * 4  # ETTm1: minute


def expected_n_windows(h):
    return 4 * month_rows() - h + 1  # 11521 - h


def first_pred_abs_row0():
    return 16 * month_rows()  # 46080


def expected_geometry(h, c):
    """Exact update_config geometry for ETTm1 (L=2304/P=96) at align=c;
    the probe asserts the live model against this. H=336/720 get a
    48-step right padding."""
    ctx, p = ETTM1["ctx"], ETTM1["period"]
    pad_left = 0 if ctx % p == 0 else p - ctx % p
    pad_right = 0 if h % p == 0 else p - h % p
    input_ratio = (pad_left + ctx) / (pad_left + ctx + pad_right + h)
    npi = int(input_ratio * MAE_NUM_PATCH * c)
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


def exp001_dir(h):
    return os.path.join(EXP001_BASE, "ETTm1_%d" % h)


def group_dir(h, c):
    return os.path.join(SAVE_BASE, "c" + c_str(c), "ETTm1_%d" % h)


def std_array():
    """Standardized full ETTm1 csv replicating data_provider/data_loader.py:
    7 columns, sklearn StandardScaler fit on train rows [0, 12 months),
    transform all, float32 cast (what the model saw)."""
    if "st" in _ETTM1_DIR_CACHE:
        return _ETTM1_DIR_CACHE["st"]
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, ETTM1["root_path"], ETTM1["data_path"]))
    vals = df[df.columns[1:]].values.astype(np.float64)  # (rows, 7) f64
    k = month_rows()
    scaler = StandardScaler()
    scaler.fit(vals[0:12 * k])
    st = scaler.transform(vals).astype(np.float32)
    _ETTM1_DIR_CACHE["st"] = st
    return st


def truth_recomputed(h):
    """Stack std[t0+i : t0+i+h] over test windows -> (N, H, 7); bitwise
    equal to EXP-001 true.npy (EXP-031 verified)."""
    t0 = first_pred_abs_row0()
    n = expected_n_windows(h)
    seg = std_array()[t0:t0 + n + h - 1]
    sw = np.lib.stride_tricks.sliding_window_view(seg, h, axis=0)  # (n, 7, h)
    return np.ascontiguousarray(sw.transpose(0, 2, 1))             # (n, h, 7)


def exp003_truth_restored(h):
    """Load EXP-003 seed2021 true.npy (N*7, H, 1), assert shape, restore to
    (N, H, 7) with the FIXED reshape (flatten index 7*win + var)."""
    p = os.path.join(EXP003_BASE, "ETTm1_%d_seed2021" % h, "results", "_", "true.npy")
    t = np.load(p)
    n = expected_n_windows(h)
    assert t.shape == (n * 7, h, 1), \
        "ETTm1_%d EXP-003 true shape %s != (%d*7, %d, 1)" % (h, t.shape, n, h)
    rec = t[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
    return np.ascontiguousarray(rec.astype(np.float32))


def exp031_ettm1_column():
    """Snapshot of EXP-031 summary_b4_seasonal.csv (sha256-gated) ->
    {h: {"mse","mae","n"}} for its ETTm1 VisionTS rows (per-H, = EXP-001
    metrics.npy at full precision)."""
    got = sha256_of(EXP031_SNAPSHOT)
    if got != EXP031_SHA256:
        raise SystemExit("[t35] exp031 snapshot sha256 %s != pinned %s"
                         % (got, EXP031_SHA256))
    per = {}
    with open(EXP031_SNAPSHOT) as f:
        header = f.readline().strip().split(",")
        assert header[:3] == ["dataset", "pred_len", "method"], header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9 or parts[2] != "VisionTS":
                continue
            ds, pred = parts[0], parts[1]
            if ds != "ETTm1" or pred == "avg":
                continue
            per[int(pred)] = {"mse": float(parts[7]), "mae": float(parts[8]),
                              "n": int(parts[11]) if parts[11] else None}
    for h in HORIZONS:
        assert h in per, "exp031 snapshot missing ETTm1_%d" % h
    return per


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


def sha256_of_bytes(b):
    return hashlib.sha256(b).hexdigest()


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
