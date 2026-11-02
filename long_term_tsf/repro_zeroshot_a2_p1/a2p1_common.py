"""Shared helpers for the VisionTS paper A.2 Table 7 P=1 reproduction
(EXP-032). Baseline = EXP-001 zero-shot runs (save_repro_zeroshot/,
framework files byte-identical between 71da50a and HEAD, verified by
`git diff 71da50a..HEAD -- long_term_tsf/... visionts/` = empty).
numpy-only; imports the exact metric() used by exp_long_term_forecasting.
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # long_term_tsf/
REPO = os.path.dirname(ROOT)
PKG = os.path.dirname(os.path.abspath(__file__))
SAVE_BASE = os.path.join(ROOT, "save_a2p1_zeroshot")
BASE_SAVE_BASE = os.path.join(ROOT, "save_repro_zeroshot")
REPRO_ZERO_DIR = os.path.join(ROOT, "repro_zeroshot")
CKPT_FILE = os.path.join(REPO, "ckpt", "mae_visualize_vit_base.pth")
CKPT_SHA256 = "c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e"

HORIZONS = [96, 192, 336, 720]
ETT4 = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

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

# paper Appendix A.2 Table 7 (4H averages, printed to 3 decimals)
PAPER_T7 = {
    # ds: {orig: (mse, mae), p1: (mse, mae)}
    "ETTh1": {"orig": (0.390, 0.414), "p1": (0.840, 0.628)},
    "ETTh2": {"orig": (0.333, 0.375), "p1": (0.424, 0.445)},
    "ETTm1": {"orig": (0.374, 0.372), "p1": (0.660, 0.533)},
    "ETTm2": {"orig": (0.282, 0.321), "p1": (0.312, 0.363)},
}
PAPER_T7_AVG = {"orig": (0.344, 0.370), "p1": (0.559, 0.492)}

# EXP-001 fingerprint file list (relative to ROOT), verbatim from run_one.sh
FP_FILES = [
    "run.py", "exp/exp_long_term_forecasting.py", "exp/exp_basic.py",
    "models/VisionTS.py", "data_provider/data_factory.py",
    "data_provider/data_loader.py",
    "../visionts/model.py", "../visionts/models_mae.py",
]


def group_ids():
    return [(ds, h) for ds in ETT4 for h in HORIZONS]


def base_dir(ds, h):
    return os.path.join(BASE_SAVE_BASE, "%s_%d" % (ds, h))


def p1_dir(ds, h):
    return os.path.join(SAVE_BASE, "%s_%d" % (ds, h))


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


def parse_log_metrics(txt):
    m = re.search(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", txt)
    if not m:
        return None
    return {"mse": float(m.group(1)), "mae": float(m.group(2))}


def success_metrics(path):
    with open(path) as f:
        return parse_log_metrics(f.read())


def md5_of(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_lines(period, ds, h):
    c = DATASETS[ds]
    lines = [
        "dataset=%s pred_len=%d seq_len=%d periodicity=%d arch=mae_base"
        % (ds, h, c["ctx"], period),
        "norm_const=0.4 align_const=0.4 batch_size=32 train_epochs=0 seed=2021",
        "root_path=%s data_path=%s data=%s features=M"
        % (c["root_path"], c["data_path"], c["data"]),
        "ckpt_sha256=" + CKPT_SHA256,
    ]
    for f in FP_FILES:
        lines.append("md5:%s=%s" % (f, md5_of(os.path.join(ROOT, f))))
    for f in ["repro_zeroshot_a2_p1/infer_one_p1.py",
              "repro_zeroshot_a2_p1/a2p1_common.py"]:
        lines.append("md5:%s=%s" % (f, md5_of(os.path.join(ROOT, f))))
    return lines


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
        f.write("\n")


def load_json(path):
    with open(path) as f:
        return json.load(f)
