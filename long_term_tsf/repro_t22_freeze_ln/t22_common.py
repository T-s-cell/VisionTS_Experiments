"""EXP-033 helpers: VisionTS paper Table 22 — fully frozen vs LayerNorm-only
finetuning on ETT (4 datasets x H in {96,192,336,720}, Table 19 lookbacks).
Freeze column = MAE-base weights, zero training, zero-shot inference.
LN column = EXP-003 (save_fullshot/, commit b40d663), reused via the
aggregate.csv snapshot pinned in this package (3-seed means, seeds
2021/2022/2023); overall LN self-check 0.33235446/0.36986023 (user).

Corrected window counts (user, verified against EXP-031 win_start; 16
groups total 109,840 windows): hourly test N = 2881 - H, first predicted
csv row (0-based) = 11520; minute (ETTm*) test N = 11521 - H, first row =
46080 — both independent of L, so truths are bitwise comparable across
EXP-001 / EXP-003 / this experiment.

EXP-003 truth layout (archived exp_univariate.py: 7 variables saved
sequentially per window, flatten index 7*win + var): shape (N*7, H, 1),
restore with true[..., 0].reshape(N, 7, H).transpose(0, 2, 1) -> (N, H, 7).
Single fixed reshape, no candidates.
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # long_term_tsf/
REPO = os.path.dirname(ROOT)
PKG = os.path.dirname(os.path.abspath(__file__))
SAVE_BASE = os.path.join(ROOT, "save_t22_freeze")
EXP001_BASE = os.path.join(ROOT, "save_repro_zeroshot")
EXP003_BASE = os.path.join(ROOT, "save_fullshot")

HORIZONS = [96, 192, 336, 720]
NEW_DS = ["ETTh1", "ETTh2", "ETTm2"]   # 12 new Freeze groups
CACHE_DS = ["ETTm1"]                   # reuse EXP-001 (same L/P/r/c), per-group

# paper Table 19 lookbacks (EXP-003 args.json confirms seq_len; periodicity
# from the official finetune script)
DATASETS_T19 = {
    "ETTh1": dict(ctx=1152, period=24, root_path="./dataset/ETT-small/",
                  data_path="ETTh1.csv", data="ETTh1", minute=False),
    "ETTh2": dict(ctx=1152, period=24, root_path="./dataset/ETT-small/",
                  data_path="ETTh2.csv", data="ETTh2", minute=False),
    "ETTm1": dict(ctx=2304, period=96, root_path="./dataset/ETT-small/",
                  data_path="ETTm1.csv", data="ETTm1", minute=True),
    "ETTm2": dict(ctx=1152, period=96, root_path="./dataset/ETT-small/",
                  data_path="ETTm2.csv", data="ETTm2", minute=True),
}

CKPT_REL = "../ckpt/mae_visualize_vit_base.pth"
CKPT_SHA256 = "c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e"
CKPT_BYTES = 447725789

# package-carried snapshot of EXP-003 outputs/aggregate.csv (commit b40d663)
AGG_SNAPSHOT = os.path.join(PKG, "aggregate_snapshot.csv")
AGG_SHA256 = "a1622c33b35a54c75e6d4ac5fe1b6649b3797e3234270807c91b87e280f75dcf"
LN_SELF_CHECK = {"overall_mse": 0.33235446, "overall_mae": 0.36986023}
LN_SELF_TOL = 1e-6

# paper Table 22 anchors (3 decimals, user-frozen): {ds: {"freeze": (mse,mae),
# "ln": (mse,mae)}}, plus overall
PAPER_T22 = {
    "ETTh1": {"freeze": (0.419, 0.418), "ln": (0.395, 0.409)},
    "ETTh2": {"freeze": (0.340, 0.376), "ln": (0.336, 0.382)},
    "ETTm1": {"freeze": (0.374, 0.372), "ln": (0.338, 0.367)},
    "ETTm2": {"freeze": (0.305, 0.334), "ln": (0.261, 0.319)},
}
PAPER_T22_OVERALL = {"freeze": (0.360, 0.375), "ln": (0.333, 0.369)}

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
    return 30 * 24 * (4 if DATASETS_T19[ds]["minute"] else 1)


def expected_n_windows(ds, h):
    # hourly: 4*720 - h + 1 = 2881-h; minute: 4*2880 - h + 1 = 11521-h
    return 4 * month_rows(ds) - h + 1


def first_pred_abs_row0(ds):
    # hourly 11520, minute 46080; independent of L
    return 16 * month_rows(ds)


def freeze_dir(ds, h):
    return os.path.join(SAVE_BASE, "%s_%d" % (ds, h))


def exp001_dir(ds, h):
    return os.path.join(EXP001_BASE, "%s_%d" % (ds, h))


def exp003_dir(ds, h):
    return os.path.join(EXP003_BASE, "%s_%d_seed2021" % (ds, h))


def std_array(ds):
    """Standardized full csv array replicating data_provider/data_loader.py:
    7 columns, sklearn StandardScaler fit on train rows [0, 12 months),
    transform all, float32 cast (what the model saw)."""
    if ds in _STD_CACHE:
        return _STD_CACHE[ds]
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    c = DATASETS_T19[ds]
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


def exp003_args_record(ds, h):
    """Best-effort read of EXP-003 args.json: gate seq_len/periodicity to
    Table 19 when the keys are present; return the recorded subset."""
    p = os.path.join(exp003_dir(ds, h), "args.json")
    if not os.path.isfile(p):
        return {"present": False}
    with open(p) as f:
        j = json.load(f)
    rec = {"present": True, "seq_len": j.get("seq_len"),
           "periodicity": j.get("periodicity"),
           "ft_type": j.get("ft_type"), "seed": j.get("seed"),
           "train_epochs": j.get("train_epochs"),
           "batch_size": j.get("batch_size")}
    c = DATASETS_T19[ds]
    if rec["seq_len"] is not None and int(rec["seq_len"]) != c["ctx"]:
        raise SystemExit("[t22] EXP-003 %s_%d seq_len=%s != Table19 %d"
                         % (ds, h, rec["seq_len"], c["ctx"]))
    if rec["periodicity"] is not None and int(rec["periodicity"]) != c["period"]:
        raise SystemExit("[t22] EXP-003 %s_%d periodicity=%s != Table19 %d"
                         % (ds, h, rec["periodicity"], c["period"]))
    return rec


def load_aggregate():
    """Snapshot aggregate.csv -> {(ds, h): (mse_mean, mae_mean) f64} with
    sha256 gate and header check."""
    got = sha256_of(AGG_SNAPSHOT)
    if got != AGG_SHA256:
        raise SystemExit("[t22] aggregate_snapshot.csv sha256 %s != pinned %s"
                         % (got, AGG_SHA256))
    out = {}
    with open(AGG_SNAPSHOT) as f:
        header = f.readline().strip().split(",")
        assert header[:6] == ["dataset", "pred", "mse_mean", "mse_std",
                              "mae_mean", "mae_std"], header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            ds, pred = parts[0], parts[1]
            if ds in _ETT4:
                out[(ds, int(pred))] = (float(parts[2]), float(parts[4]))
    for ds in _ETT4:
        for h in HORIZONS:
            assert (ds, h) in out, "aggregate snapshot missing %s_%d" % (ds, h)
    return out


def ln_aggregates(agg):
    """Per-dataset 4H equal mean + overall 4ds equal mean of the LN column;
    gate overall against the user's self-check values (<=1e-6)."""
    per_ds = {}
    for ds in _ETT4:
        mses = [agg[(ds, h)][0] for h in HORIZONS]
        maes = [agg[(ds, h)][1] for h in HORIZONS]
        per_ds[ds] = (float(np.mean(mses)), float(np.mean(maes)))
    overall = (float(np.mean([per_ds[d][0] for d in _ETT4])),
               float(np.mean([per_ds[d][1] for d in _ETT4])))
    for key, got in zip(("overall_mse", "overall_mae"), overall):
        want = LN_SELF_CHECK[key]
        if abs(got - want) > LN_SELF_TOL:
            raise SystemExit("[t22] LN self-check FAIL: overall %s=%.8f "
                             "!= %.8f (tol 1e-6)" % (key, got, want))
    return per_ds, overall


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
