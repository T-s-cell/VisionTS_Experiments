"""EXP-042 helpers: paper Table 22 remaining finetune types — Bias / MLP
/ Attention — on the ORIGINAL MAE-base pretrained weights.

144 new trainings = 4 ETT datasets x 4 horizons (H=96/192/336/720) x
3 seeds (2021/2022/2023) x 3 ft types.  LN (EXP-003, 48 per-seed pins),
All / Rand-VM (EXP-038 seed-2021 archive + EXP-039 seeds 2022/2023) and
Freeze (EXP-033 Freeze16) are REUSED from archives, NOT retrained.

Recipe = EXP-003 verbatim (bilinear, r=c=0.4, FP32, Adam 1e-4, bs 256,
1 epoch, full fixed val set element-weighted MSE selection, strict
best-ckpt reload then test), executed ZERO-CHANGE through
repro_fullshot/run_train.py main().  Training entry provenance and the
bias-scope口径差异 (official code broader than the paper's BitFit text)
are recorded in README_t42.md.

Everything seed-independent is inherited from t38_common / t39_common;
EXP-037/038/039 directories are READ-ONLY.
"""
import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_T39_DIR = os.path.join(_PKG_DIR, "..", "repro_t39_t22seeds")
if os.path.isdir(_T39_DIR) and _T39_DIR not in sys.path:
    sys.path.insert(0, _T39_DIR)

import t39_common as X39   # first import: fixes sys.path for t38/t22 pkgs
import t38_common as C

PKG = _PKG_DIR
LOG_DIR = os.path.join(PKG, "logs")
VERIFY_JSON = os.path.join(LOG_DIR, "verify_t42.json")
SUMMARY_CSV = os.path.join(LOG_DIR, "summary_t42.csv")
RESULT_MD = os.path.join(LOG_DIR, "RESULT_t42_fttypes.md")
FIG_PNG = os.path.join(LOG_DIR, "fig_t42_fttypes.png")
RECOMPUTE_JSON = os.path.join(LOG_DIR, "recompute_f64.json")
FAILED_TXT = os.path.join(LOG_DIR, "FAILED.txt")
OK_MARKER = os.path.join(PKG, "T42_OK")
SCOPE_JSON = os.path.join(LOG_DIR, "scope_audit_t42.json")
FROZEN_DIGEST_JSON = os.path.join(LOG_DIR, "frozen_digest.json")
REENTRY_JSON = os.path.join(LOG_DIR, "reentry_check_t42.json")

SAVE_BASE = os.path.join(C.ROOT, "save_t42_fttypes")

FT_TYPES = ["bias", "mlp", "attn"]
SEEDS = [2021, 2022, 2023]
CONFIGS = [(ds, h) for ds in C.ETT4 for h in C.HORIZONS]          # 16
TRIPLES = [(ds, h, s) for (ds, h) in CONFIGS for s in SEEDS]      # 48
TASKS = [(ds, h, s, ft) for (ds, h, s) in TRIPLES for ft in FT_TYPES]
assert len(TASKS) == 144 and len(set(TASKS)) == 144, \
    "TASKS must be 4x4x3x3 = 144 unique"

# smoke = ETTh1_96 x 3 seeds x 3 ft types (9 units, counted in TASKS)
SMOKE = [("ETTh1", 96, s, ft) for s in SEEDS for ft in FT_TYPES]
assert len(SMOKE) == 9 and all(t in TASKS for t in SMOKE)
# queue = everything else, config-major (ETTm1 last, as in EXP-039)
QUEUE_CONFIGS = ([(ds, h) for ds, h in
                  [("ETTh1", h) for h in (192, 336, 720)]
                  + [("ETTh2", h) for h in C.HORIZONS]
                  + [("ETTm2", h) for h in C.HORIZONS]
                  + [("ETTm1", h) for h in C.HORIZONS]])
QUEUE = [(ds, h, s, ft) for (ds, h) in QUEUE_CONFIGS
         for s in SEEDS for ft in FT_TYPES]
assert sorted(QUEUE) == sorted(t for t in TASKS if t not in SMOKE), \
    "QUEUE must be exactly TASKS minus SMOKE (135)"

LN_KEYS_SEEDS = X39.LN_KEYS_SEEDS          # (2021, 2022, 2023)
LN48 = X39.LN48                            # EXP-003 per-seed pins (48)
ln_success_path = X39.ln_success_path
exp003_seed_job = X39.exp003_seed_job
t38_train_json = X39.t38_train_json
t39_train_json = X39.train_json            # EXP-039 All/Rand seeds 22/23
load_freeze16 = C.load_freeze16


def job_name(ds, h, s, ft):
    return "%s_%s_%d_seed%d" % (ft, ds, h, s)


def job_dir(ds, h, s, ft):
    return os.path.join(SAVE_BASE, job_name(ds, h, s, ft))


def train_json(ds, h, s, ft):
    return os.path.join(LOG_DIR, "train_%s_%d_%s_seed%d.json"
                        % (ds, h, ft, s))


# args delta whitelist vs the config's own EXP-003 args.json: vm_pretrained
# stays 1 (= EXP-003 value, original pretrained weights) so it is NOT an
# allowed delta; seed/model_id/save_dir/ft_type/gpu are the only keys that
# may differ.  seed MAY be absent from the observed deltas (seed-2021 tasks
# share EXP-003's base seed) — it must EQUAL the requested seed, not
# necessarily differ.
FT_ALLOWED_ARG_DELTA = {"model_id", "save_dir", "ft_type", "gpu", "seed"}
FT_MUST_DIFFER = {"model_id", "save_dir", "ft_type"}

# paper Table 22 anchors (transcribed verbatim from
# audit_supplement/paper/VisionTS_paper_full.md L549-566; values are the
# 4H mean per dataset, Average row = 4-dataset mean).  Paper anchors are
# compared ONLY at the dataset-4H-mean and overall layers.  ETTm2
# counterexample to re-examine: Bias 0.256/0.318 < LN 0.261/0.319.
PAPER_T42 = {
    "ETTh1": {"bias": (0.401, 0.414), "mlp": (0.534, 0.471),
              "attn": (0.554, 0.479)},
    "ETTh2": {"bias": (0.347, 0.392), "mlp": (0.401, 0.419),
              "attn": (0.392, 0.414)},
    "ETTm1": {"bias": (0.343, 0.368), "mlp": (0.441, 0.415),
              "attn": (0.444, 0.415)},
    "ETTm2": {"bias": (0.256, 0.318), "mlp": (0.292, 0.342),
              "attn": (0.289, 0.339)},
}
PAPER_T42_OVERALL = {"bias": (0.337, 0.373), "mlp": (0.417, 0.412),
                     "attn": (0.420, 0.412)}

# EXP-039 published overall stats (cross-seed mean±std ddof=1 over the
# per-seed 4-dataset means), pinned from
# repro_t39_t22seeds/logs/RESULT_t39_ett.md T3 (L87-89).  The collector's
# ln/all/rand overall rows are cross-checked against these (±5e-5, the
# rounding step of the published 4-decimal values).  The overall std is
# the THREE-SEED spread of per-seed overalls — never the cross-dataset
# spread of the per-dataset means.
EXP039_T3_OVERALL = {
    "ln":   {"mse": (0.3324, 0.0006), "mae": (0.3699, 0.0003)},
    "all":  {"mse": (0.4237, 0.0040), "mae": (0.4163, 0.0023)},
    "rand": {"mse": (0.4733, 0.0030), "mae": (0.4598, 0.0027)},
}
EXP039_T3_TOL = 5e-5
