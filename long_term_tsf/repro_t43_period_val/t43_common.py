"""Shared constants/helpers for EXP-043: VisionTS periodic-P validation-set
selection (paper A.2 claim: P chosen from Table-5 candidates on the
validation split; final P=24/96 per Table 6).

Frozen protocol (user review 2026-11-11, corrections incorporated):
  1. Padding geometry is FORMULA-DRIVEN everywhere: pad_left=(-L)%P,
     pad_right=(-H)%P (Python modulo semantics == the official
     P - x%P-if-nonzero branch in visionts/model.py update_config).
     Hand-written pad tables are explanatory only, never gated.
  2. Split definition, explicit: month rows K = 30*24*(4 for 15T-minutely,
     else 1) -> hourly K=720, minutely K=2880.  Train = rows [0,12K);
     VAL TARGET = [12K,16K); TEST TARGET = [16K,20K).  Window count
     equality alone is NOT treated as proof of the split; target-region
     truth is verified bitwise against deterministic recomputation.
  3. The 48 validation configs are ACCEPTED (windows/truth/finiteness/
     metric recompute) BEFORE the selection ledger is generated; SSE is
     accumulated in float64 and selection uses unrounded values.
  4. No "close to paper" hard gates anywhere: deviations from paper
     values are reported observations, never stop conditions.  Gates
     check frozen archives and self-consistency only.

Reuse (read-only, gated): default-P arm = EXP-001 save_repro_zeroshot/;
P=1 arm = EXP-030 save_a2p1_zeroshot/.  Runs reproduce EXP-001's
invocation verbatim (run.py untouched) with only --periodicity varied.
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # long_term_tsf/
REPO = os.path.dirname(ROOT)
PKG = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PKG, "logs")

SAVE_BASE = os.path.join(ROOT, "save_t43_period_val")          # new test runs (selected P)
VAL_BASE = os.path.join(ROOT, "save_t43_period_val_val")       # per-(ds,H,P) val arrays
SMOKE_BASE = os.path.join(ROOT, "save_t43_period_val_smoke")   # test-path smoke (P=24 ETTh1/H96)

EXP001_BASE = os.path.join(ROOT, "save_repro_zeroshot")
EXP030_BASE = os.path.join(ROOT, "save_a2p1_zeroshot")
EXP030_DIR = os.path.join(ROOT, "repro_zeroshot_a2_p1")

CKPT_FILE = os.path.join(REPO, "ckpt", "mae_visualize_vit_base.pth")
CKPT_SHA256 = "c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e"
CKPT_BYTES = 447725789

HORIZONS = [96, 192, 336, 720]
ETT4 = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

# paper Table 5 candidate ranges; order = {paper P, second candidate, 1}
# (paper_full.md L348-354: Hourly {24,168,1}, 15T {96,672,1})
CANDIDATE_P = {
    "ETTh1": [24, 168, 1],
    "ETTh2": [24, 168, 1],
    "ETTm1": [96, 672, 1],
    "ETTm2": [96, 672, 1],
}
PAPER_P = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96}

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

# run.py defaults that the EXP-001 invocation leaves implicit
RUN_DEFAULTS = dict(label_len=48, embed="timeF", freq="h", num_workers=10,
                    batch_size=32, inverse=False, use_amp=False,
                    output_attention=False, ft_type="ln", vm_pretrained=1,
                    interpolation="bilinear", seasonal_patterns="Monthly",
                    norm_const=0.4, align_const=0.4, seed=2021,
                    train_epochs=0, features="M", vm_arch="mae_base")

# EXP-001 fingerprint file list (relative to ROOT), verbatim from run_one.sh
FP_FILES = [
    "run.py", "exp/exp_long_term_forecasting.py", "exp/exp_basic.py",
    "models/VisionTS.py", "data_provider/data_factory.py",
    "data_provider/data_loader.py",
    "../visionts/model.py", "../visionts/models_mae.py",
]
# t43 files recorded in fingerprints (paths relative to ROOT)
T43_FP_FILES = ["repro_t43_period_val/infer_one_t43.py",
                "repro_t43_period_val/t43_common.py",
                "repro_t43_period_val/val_pass.py"]

# ---- split definition (user correction #2: explicit, not window-count) ----
def month_rows(ds):
    """K: rows per 30-day month block.  hourly K=720, 15T-minutely K=2880."""
    return 30 * 24 * (4 if DATASETS[ds]["minute"] else 1)


def split_rows(ds):
    """Absolute CSV-row boundaries: train [0,12K), val target [12K,16K),
    test target [16K,20K)."""
    K = month_rows(ds)
    return {"train_end": 12 * K, "val_start": 12 * K, "val_end": 16 * K,
            "test_start": 16 * K, "test_end": 20 * K}


def expected_n_windows(ds, h):
    # dataset len per split = 4K + seq_len -> N = 4K - h + 1 (val == test)
    return 4 * month_rows(ds) - h + 1


# ---- padding geometry, formula-driven (user correction #1) ----
def expected_geometry(ds, h, P):
    """Replicates visionts/model.py VisionTS.update_config verbatim.
    pad_left = P - L%P (0 if divides) == (-L)%P in Python semantics; same
    for pad_right.  image_size=224, patch=16, num_patch=14 (mae_base)."""
    c = DATASETS[ds]
    ctx, pred = c["ctx"], h
    pad_left = (-ctx) % P
    pad_right = (-pred) % P
    num_patch = 14
    align_const = 0.4
    input_ratio = (pad_left + ctx) / (pad_left + ctx + pad_right + pred)
    npi = int(input_ratio * num_patch * align_const)
    if npi == 0:
        npi = 1
    npo = num_patch - npi
    adjust = npi / num_patch
    image_size = 224
    scale_x = ((pad_left + ctx) // P) / int(image_size * adjust)
    in_h = int(image_size * adjust)
    out_w = int(round(image_size * scale_x))
    return {
        "periodicity": P, "context_len": ctx, "pred_len": pred,
        "pad_left": pad_left, "pad_right": pad_right,
        "num_patch_input": npi, "num_patch_output": npo,
        "scale_x": scale_x,
        "input_resize_size": (image_size, in_h),
        "output_resize_size": (P, out_w),
        "mask_ratio": 1.0 - npi / num_patch,
    }


def geometry_row(ds, h, P):
    g = expected_geometry(ds, h, P)
    return g


# ---- deterministic truth recomputation (b4/t35 pattern) ----
def std_array(ds):
    """f64 read -> StandardScaler fit on rows [0,12K) -> transform all ->
    f32 cast.  Mirrors Dataset_ETT_{hour,minute}.__read_data__ exactly
    (features='M': columns 1:, scaler on train block)."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    c = DATASETS[ds]
    df_raw = pd.read_csv(os.path.join(ROOT, c["root_path"], c["data_path"]))
    cols_data = df_raw[df_raw.columns[1:]]
    sp = split_rows(ds)
    scaler = StandardScaler()
    scaler.fit(cols_data.values[: sp["train_end"]])
    data = scaler.transform(cols_data.values)
    return data.astype(np.float32)


def truth_from_std(ds, target, h):
    """Targets for 'val' ([12K,16K)) or 'test' ([16K,20K)): sliding H-step
    windows of the standardized data -> (N, H, 7) float32, window i covering
    absolute rows [start+i, start+i+H).  N == 4K - h + 1."""
    std = std_array(ds)
    sp = split_rows(ds)
    start, end = (sp["val_start"], sp["val_end"]) if target == "val" \
        else (sp["test_start"], sp["test_end"])
    n = end - start - h + 1
    out = np.empty((n, h, std.shape[1]), dtype=np.float32)
    for i in range(n):
        out[i] = std[start + i: start + i + h]
    return out


# ---- paths ----
def vjson_path(ds, P, h):
    return os.path.join(LOG_DIR, "val_%s_P%d_H%d.json" % (ds, P, h))


def val_dir(ds, h, P):
    return os.path.join(VAL_BASE, "%s_%d_P%d" % (ds, h, P))


def t43_run_dir(ds, h):
    return os.path.join(SAVE_BASE, "%s_%d" % (ds, h))


def base_dir(ds, h):
    return os.path.join(EXP001_BASE, "%s_%d" % (ds, h))


def p1_dir(ds, h):
    return os.path.join(EXP030_BASE, "%s_%d" % (ds, h))


def group_ids():
    return [(ds, h) for ds in ETT4 for h in HORIZONS]


# ---- ledger ----
LEDGER_CSV = os.path.join(LOG_DIR, "val_table.csv")
LEDGER_JSON = os.path.join(LOG_DIR, "selection.json")
LEDGER_OK = os.path.join(LOG_DIR, "LEDGER_OK")
VALFULL_JSON = os.path.join(LOG_DIR, "verify_t43_valfull.json")
VERIFY_JSON = os.path.join(LOG_DIR, "verify_t43.json")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def framework_md5_dict():
    return {f: md5_of(os.path.join(ROOT, f)) for f in FP_FILES}


# ---- fingerprints ----
def config_lines(ds, h, P):
    c = DATASETS[ds]
    return [
        "dataset=%s pred_len=%d seq_len=%d periodicity=%d arch=mae_base"
        % (ds, h, c["ctx"], P),
        "norm_const=0.4 align_const=0.4 batch_size=32 train_epochs=0 seed=2021",
        "root_path=%s data_path=%s data=%s features=M"
        % (c["root_path"], c["data_path"], c["data"]),
        "ckpt_sha256=" + CKPT_SHA256,
    ]


def fingerprint_lines(ds, h, P, extra_files=(), ledger_sha=None):
    lines = list(config_lines(ds, h, P))
    for f in FP_FILES:
        lines.append("md5:%s=%s" % (f, md5_of(os.path.join(ROOT, f))))
    for f in list(extra_files) + T43_FP_FILES:
        lines.append("md5:%s=%s" % (f, md5_of(os.path.join(ROOT, f))))
    lines.append("ledger_sha256=%s" % (ledger_sha if ledger_sha else "none"))
    return lines


def read_fingerprint(path):
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def write_text(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---- metric (exact utils.metrics import) ----
def load_metric():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from utils.metrics import metric
    return metric


def parse_log_metrics(txt):
    import re
    m = re.search(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", txt)
    if not m:
        return None
    return {"mse": float(m.group(1)), "mae": float(m.group(2))}


# ---- json ----
def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
        f.write("\n")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_resize_ints(s):
    """'(224, 56)' / '[224, 56]' -> (224, 56); robust to torchvision str()."""
    import re
    nums = re.findall(r"\d+", str(s))
    return tuple(int(x) for x in nums)


def geometry_matches(observed, expected):
    """observed: dict as recorded by the update_config wrapper (resize sizes
    as strings); expected: expected_geometry dict.  Returns (ok, notes)."""
    notes = []
    ok = True
    for k in ("periodicity", "context_len", "pred_len", "pad_left",
              "pad_right", "num_patch_input"):
        o, e = observed.get(k), expected[k]
        if o != e:
            ok = False
            notes.append("%s observed=%s expected=%s" % (k, o, e))
    for k in ("scale_x", "mask_ratio"):
        o, e = observed.get(k), expected[k]
        if o is None or abs(float(o) - float(e)) > 1e-6:
            ok = False
            notes.append("%s observed=%s expected=%s" % (k, o, e))
    o_in = parse_resize_ints(observed.get("input_resize_size"))
    if o_in != tuple(expected["input_resize_size"]):
        ok = False
        notes.append("input_resize_size observed=%s expected=%s"
                     % (o_in, expected["input_resize_size"]))
    o_out = parse_resize_ints(observed.get("output_resize_size"))
    if o_out != tuple(expected["output_resize_size"]):
        ok = False
        notes.append("output_resize_size observed=%s expected=%s"
                     % (o_out, expected["output_resize_size"]))
    return ok, notes
