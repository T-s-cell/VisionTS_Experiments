"""EXP-037 helpers: two independent pilots sharing one package.

A. Monash zero-shot (paper Table 15): FRED-MD (P=1, L=300; paper VisionTS
   MAE 1893.67 / Naive 2825.67) and NN5 Daily without missing (P=7, L=1000;
   paper 3.51 / 8.26). Official eval_gluonts entry point (run.py flags from
   run_monash.sh: --periodicity=autotune --context_len=1000
   --no_periodicity_context_len=300 --batch_size 512). Frozen MAE-base,
   bilinear, r=c=0.4. Naive denominator: Woo et al. 2024 (MOIRAI) report the
   Monash baseline "Naive" = last-value repetition (Godahewa et al. 2021
   benchmark set) -> N1 is the primary denominator; seasonal-naive (period P,
   last season tiled) is kept alongside as N2. NN5 Daily prediction length is
   expected to be 56 from the raw .tsf forecast_horizon (30 would be the
   frequency fallback default and must pause the subset).

B. ETTh1 pretrained-vs-random backbone (paper Table 22, single H=96):
   EXP-003 recipe (L=1152, P=24, r=c=0.4, bilinear, FP32, Adam lr=1e-4,
   bs=256, 1 epoch, full-val MSE selection, strict reload, seed=2021).
   arm "all"  = verified ImageNet MAE-base, full finetune (vm_pretrained=1).
   arm "rand" = identical construction, random init (vm_pretrained=0).
   Gate evidence from inside the two FORMAL trainings via observation-only
   hooks: initial weights differ across arms (All bitwise == ckpt tensors;
   Rand recorded and != All), trainable-name lists equal, full-epoch sampler
   index sequences equal; first-batch hashes are auxiliary only.

Stage markers: A_OK / B_OK files in this package; failures write
logs/FAILED.txt and stop that stage. A and B are fully independent.
"""
import hashlib
import json
import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.abspath(__file__))            # .../repro_t37_pilot
ROOT = os.path.dirname(PKG)                                  # long_term_tsf/
REPO_ROOT = os.path.dirname(ROOT)                            # repo root
EVAL_GLUONTS = os.path.join(REPO_ROOT, "eval_gluonts")
LOG_DIR = os.path.join(PKG, "logs")
VERIFY_JSON = os.path.join(LOG_DIR, "verify_t37.json")

SAVE_A_BASE = os.path.join(ROOT, "save_t37_monash")
SAVE_B_BASE = os.path.join(ROOT, "save_t37_full")

A_OK = os.path.join(PKG, "A_OK")
B_OK = os.path.join(PKG, "B_OK")
FAILED_TXT = os.path.join(LOG_DIR, "FAILED.txt")

for _p in (REPO_ROOT, ROOT, os.path.join(ROOT, "repro_t22_freeze_ln"),
           os.path.join(REPO_ROOT, "repro_fullshot")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 2021

# frozen MAE-base backbone (pinned since EXP-033/t35/t36)
CKPT_PATH_ETA = "/dev_data/wlt/VisionTS_Experiments/ckpt/mae_visualize_vit_base.pth"
CKPT_REL = "../ckpt/mae_visualize_vit_base.pth"   # from long_term_tsf cwd
CKPT_SHA256 = "c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e"
CKPT_BYTES = 447725789

# ---------------------------------------------------------------- A: Monash
MONASH_BATCH = 512
MONASH_CONTEXT = 1000
MONASH_NO_P_CONTEXT = 300
ANCHOR_TOL = 1e-3            # rel diff vs author precomputed MAE
N1_ALIGN_TOL = 0.05          # N1 (last-value) vs paper Naive: window-alignment check

# paper Table 15 MAE (2dp, reference only)
PAPER_T15 = {
    "fred_md": {"visionts": 1893.67, "naive": 2825.67},
    "nn5_daily_without_missing": {"visionts": 3.51, "naive": 8.26},
}
# author precomputed full-precision anchors (eval_gluonts/precomputed_results/monash.csv)
AUTHOR_ANCHOR = {
    "fred_md": {"mae": 1893.668241959648, "mse": 228818192.9765722},
    "nn5_daily_without_missing": {"mae": 3.514067169133817, "mse": 29.21466767449964},
}
# raw-benchmark expectation (user correction: NN5 horizon = 56, not the 30
# frequency-fallback default); context = code-computed convert_context_len,
# never hardcoded here
SUBSETS = {
    "fred_md": {"p": 1, "pred_len": 12,
                "series_info": 107},   # Moirai paper: 107 monthly series
    "nn5_daily_without_missing": {"p": 7, "pred_len": 56,
                                  "series_info": 111},  # 111 daily series
}
CSV_COLUMNS = ["dataset", "MSE[mean]", "MAE[0.5]", "sMAPE[0.5]",
               "MASE[0.5]", "ND[0.5]", "NRMSE[mean]"]


def monash_json(ds):
    return os.path.join(LOG_DIR, "monash_%s.json" % ds)


def naive_json(ds):
    return os.path.join(LOG_DIR, "naive_%s.json" % ds)


def vision_csv(ds):
    return os.path.join(SAVE_A_BASE, "visionts_monash_%s.csv" % ds)


def naive_csv(ds):
    return os.path.join(SAVE_A_BASE, "naive_monash_%s.csv" % ds)


def gluonts_cache():
    return os.path.join(os.path.expanduser("~"), ".gluonts")


# ------------------------------------------------------- B: ETTh1 all/rand
# EXP-003 recipe (audited args.json ETTh1_96_seed2021); deltas for this
# experiment are ft_type=full, save_dir/model_id, vm_pretrained per arm.
T19_ETTH1 = dict(ctx=1152, period=24, pred=96, r=0.4, c=0.4, lr=1e-4,
                 epochs=1, bs=256, patience=3, seed=SEED, label_len=48,
                 num_workers=10)

B_ARMS = ["all", "rand"]
B_JOB = {"all": "ETTh1_96_seed2021_all", "rand": "ETTh1_96_seed2021_rand"}

# allowed fingerprint args deltas vs EXP-003 (everything else must match)
FP_ALLOWED_ARG_DELTA = {"model_id", "save_dir", "ft_type", "vm_pretrained"}

PAPER_T22_ETTH1 = {"all": (0.534, 0.470), "rand": (0.534, 0.470),
                   "ln": (0.395, 0.409), "freeze": (0.419, 0.418)}  # 4H means

# anchors for the results table (unrounded where archived)
LN_ANCHOR = {"mse": 0.34744012355804443, "mae": 0.3768464922904968}     # EXP-003 seed2021
FREEZE_ANCHOR = {"mse": 0.3760134279727936, "mae": 0.379026}            # EXP-033

METRIC_TOL = 1e-6        # recomputed metrics vs SUCCESS.txt
ETTH1 = dict(n_windows=2785, flat_rows=2785 * 7, pred=96)

# EXP-003 job dir (eta) with local audited fallback
_B_EXP003_CANDIDATES = [
    os.path.join(ROOT, "save_fullshot", "ETTh1_96_seed2021"),      # eta
    os.path.join(REPO_ROOT, "save_fullshot", "ETTh1_96_seed2021"),  # local audited
]
# EXP-033 freeze job dir (eta) with local fallback
_B_FREEZE_CANDIDATES = [
    os.path.join(ROOT, "save_t22_freeze", "ETTh1_96"),
    os.path.join(REPO_ROOT, "save_t22_freeze", "ETTh1_96"),
]


def first_existing(paths, tail):
    for d in paths:
        if os.path.isfile(os.path.join(d, tail)):
            return d
    return paths[0]


def exp003_job():
    return first_existing(_B_EXP003_CANDIDATES, "fingerprint.json")


def b_job_dir(arm):
    return os.path.join(SAVE_B_BASE, B_JOB[arm])


def train_json(arm):
    return os.path.join(LOG_DIR, "train_%s.json" % arm)


# run_train.py FINGERPRINT_FILES (repo-relative) — md5 gate vs EXP-003
EXP003_FP_FILES = [
    "repro_fullshot/run_train.py",
    "repro_fullshot/exp_univariate.py",
    "long_term_tsf/exp/exp_long_term_forecasting.py",
    "long_term_tsf/exp/exp_basic.py",
    "long_term_tsf/models/VisionTS.py",
    "long_term_tsf/data_provider/data_factory.py",
    "long_term_tsf/data_provider/data_loader.py",
    "long_term_tsf/utils/tools.py",
    "long_term_tsf/utils/metrics.py",
    "visionts/model.py",
    "visionts/models_mae.py",
    "visionts/util.py",
]

ETTH1_CSV_REL = "long_term_tsf/dataset/ETT-small/ETTh1.csv"


# ---------------------------------------------------------------- helpers
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


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def mark_failed(stage, msg):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(FAILED_TXT, "a") as f:
        f.write("[%s] %s: %s\n" % (time_str(), stage, msg))
    print("[t37][FAILED] %s: %s" % (stage, msg), flush=True)


def clear_failed(stage):
    """Remove only this stage's lines from FAILED.txt (a-fail must not
    block stage b)."""
    if not os.path.isfile(FAILED_TXT):
        return
    with open(FAILED_TXT) as f:
        lines = [ln for ln in f.read().splitlines() if not ln.startswith("[%s]" % stage)]
    with open(FAILED_TXT, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def time_str():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_success_mse_mae(path):
    """SUCCESS.txt (json) -> (mse, mae)."""
    d = load_json(path)
    return float(d["mse"]), float(d["mae"])


def parse_csv_metrics(path):
    """Official results CSV -> {metric_name: float} of the single row."""
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, "expected 1 result row, got %d in %s" % (len(rows), path)
    return {k: float(v) for k, v in rows[0].items() if k != "dataset"}, rows[0]["dataset"]


def rel_diff(got, want):
    if want == 0:
        return float("inf") if got != 0 else 0.0
    return abs(got - want) / abs(want)
