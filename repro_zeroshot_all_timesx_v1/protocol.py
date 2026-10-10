#!/usr/bin/env python3
"""Frozen protocol constants for the all-native-windows P=1 zero-shot experiment.

Single source of truth. Everything here is frozen before any inference:
  Experiment: "VisionTS on all TimesX native windows — fixed P=1 zero-shot
  supplementary evaluation" (TimesX-all-native-P1-v1).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
OLD_DIR = os.path.join(REPO_ROOT, "repro_zeroshot_ext")          # read-only
OLD_SPLITS = os.path.join(OLD_DIR, "splits.json")
OLD_CACHE = os.path.join(OLD_DIR, "cache")
LN_MANIFEST = os.path.join(REPO_ROOT, "repro_ln_timesx_v1", "manifests", "split_manifest.json")

PROTOCOL_VERSION = "TimesX-all-native-P1-v1"
PROTOCOL_NAME = "VisionTS on all TimesX native windows — fixed P=1 zero-shot supplementary evaluation"

# ---------------------------------------------------------------- data / ckpt
ZIP_SHA256 = "9adb9540733c128e05e774e81cf0e238a9cc80b37e99da0c883bee9f57988bcc"
CKPT_SHA256 = "c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e"
# sanity prefix of the frozen old splits.json sha256 (full value computed at
# runtime and checked against the old cache manifest binding — never hardcoded)
OLD_SPLITS_SHA256_PREFIX = "d012d683"

# ------------------------------------------------------------------- counts
N_VARS = 190
N_DOMAINS = 19
TOTAL_WINDOWS = 8106
N_TEST = 1677
N_VAL = 4736
N_CROSS_REMOVED = 242
N_TRAIN_REGION = 1451          # calib_remaining_ids (not evaluated in the old protocol)
N_CACHE_P1_EXPECT = 6413       # old cache P=1 coverage (test+val)
N_FILL_EXPECT = 1693           # TOTAL - CACHE_P1_EXPECT

CONTEXT_LEN = 96
PRED_LEN = 12
EPS_STD = 1e-8
ZERO_VAR_EXPECTED = 4          # cobalt_usd_t x2, potatoes x2 (established in stage-1)

# ------------------------------------------------------------------- model
RUN_CONFIG = {
    "arch": "mae_base",
    "finetune_type": "none",
    "context_len": CONTEXT_LEN,
    "pred_len": PRED_LEN,
    "periodicity": 1,
    "norm_const_r": 0.4,
    "align_const_c": 0.4,
    "interpolation": "bilinear",
    "precision": "fp32",
    "tf32": False,
    "image_size": 224,
    "patch_size": 16,
    "seed": 2021,
    "batch_size": 64,
}
EXPECTED_GEOM_P1 = (0, 0, 4)   # (pad_left, pad_right, num_patch_input) with ctx=96, pred=12, c=0.4

# --------------------------------------------------- cache-compatibility gates
# Preflight dual criteria (frozen 2026-10-10): BOTH must hold per checked window,
# predictions finite. Engineering tolerance for cache reuse, NOT a significance
# test of model performance. Failing variable => whole variable re-inferred;
# thresholds are never loosened based on results.
TOL_PRED_REL = 1e-5            # max_i|new-old| / max(1e-8, max_i|old|) <= TOL_PRED_REL
TOL_PRED_OVER_D = 1e-4         # max_i|new-old| / d <= TOL_PRED_OVER_D  (d = formal scoring denom)
PF_VARS_PER_DOMAIN = 2
PF_WINDOWS_PER_VAR = 5         # evenly spaced incl. first & last

# ------------------------------------------------------- reference reproduction
REF_1677_STD_MSE = 3.254715
REF_1677_STD_MAE = 1.042067
REF_TOL = 5e-7                 # 6-decimal match; applies to the PURE-cache recompute only

# --------------------------------------------- independent recompute tolerance
RECOMPUTE_TOL_VAR = 1e-9
RECOMPUTE_TOL_AGG = 1e-12
RECOMPUTE_TOL_D = 1e-12        # per-window d recompute deviation (relative)

EXPECTED_DOMAINS = {
    "CropsAndStaples", "EnergyAndFuels", "LivestockAndFoodProducts",
    "RawMaterialsAndConstruction", "SpecialtyAndAdvancedMaterials",
    "StrategicAndHighValueMaterials", "Currency",
    "arts", "climate", "economy", "electronic_technology", "finance", "pets",
    "public_health", "public_policy", "science", "shopping", "society", "traffic",
}

# npz key sets
OLD_NPZ_KEYS = {"sample_ids", "splits", "pred", "true", "mu", "win_std", "calib_std", "geom"}
NEW_NPZ_KEYS = {"sample_ids", "pred", "true", "mu", "win_std", "calib_std", "geom"}

# code fingerprint groups (md5)
NEW_CODE_ALL = [
    "protocol.py", "common.py", "freeze_full.py", "verify_cache.py",
    "preflight.py", "infer_fill.py", "assemble.py", "evaluate.py",
    "recompute_check.py", "verify_acceptance.py", "make_report.py",
    "run_all.sh", "audit_all.sh", "stage_guard.py",
]
NEW_CODE_FORWARD = ["protocol.py", "common.py", "infer_fill.py", "preflight.py"]
OLD_CODE_REUSED = ["timesx_data.py", "record_fingerprint.py", "metrics_util.py"]
OLD_CODE_INFO = ["infer.py", "select_period.py", "evaluate_test.py", "aggregate.py"]
