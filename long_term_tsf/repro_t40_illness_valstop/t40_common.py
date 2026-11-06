"""EXP-040 helpers: Illness validation-rule / early-stop diagnostic.

Question: does the EXP-003 Illness paper gap relate to the validation /
model-selection / early-stopping mechanism?  Three validation rules are
evaluated on the SAME frozen training trajectory per (H, seed):

  A = full fixed val set, element-weighted MSE (EXP-003-consistent;
      mirrors repro_fullshot/exp_univariate.py Exp_Univariate.vali)
  B = the SAME full fixed set, equal-weight per-batch criterion means
      (official base Exp_Long_Term_Forecast.vali semantics; the partial
      last batch gets 1/n_batches weight - distortion probe)
  C = verified official rule: per-epoch random permutation + drop_last
      truncation to floor(N/256) full batches + equal-weight per-batch
      means (data_factory.py:30-43 semantics; a dedicated np.random
      RandomState simulates the official selection rule, NOT its exact
      RNG interleaving - documented)

Everything seed/stream-specific is sha256-derived from task keys
(Python hash() is salted per process - never use it).  EXP-003
directories are READ-ONLY.  Reuses t38_common for paths/sha/json.

NOTATION (EXP-038-unified): the diagnosis is fully inside the EXP-003
recipe (seq_len=104, label_len=48, LN, Adam 1e-4, bs=256, patience=3,
lradj=type1, max 100 epochs); only the validation rule / model selection
differs between arms.  Approaching paper values is NOT an acceptance
criterion - this is a controlled diagnostic.
"""
import hashlib
import os
import sys

_T38_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "repro_t38_t15t22")
if os.path.isdir(_T38_DIR) and _T38_DIR not in sys.path:
    sys.path.insert(0, _T38_DIR)

import t38_common as C   # noqa: E402  (first import fixes sys.path chain)

PKG = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PKG, "logs")
AUDIT_JSON = os.path.join(LOG_DIR, "audit_t40.json")
VERIFY_JSON = os.path.join(LOG_DIR, "verify_t40.json")
RECOMPUTE_JSON = os.path.join(LOG_DIR, "recompute_t40_f64.json")
SUMMARY_CSV = os.path.join(LOG_DIR, "summary_t40.csv")
RESULT_MD = os.path.join(LOG_DIR, "RESULT_t40_illness_valstop.md")
CTRL_JSON = os.path.join(LOG_DIR, "ctrl_compare_t40.json")
FAILED_TXT = os.path.join(LOG_DIR, "FAILED.txt")

ROOT = C.ROOT                       # long_term_tsf/  (EXP-003 CWD contract)
REPO_ROOT = C.REPO_ROOT             # repo root

# delivered round (rules A,B,C) vs control round (A-only, smoke)
SAVE_BASE = os.path.join(ROOT, "save_t40_valstop")
SAVE_CTRL_BASE = os.path.join(ROOT, "save_t40_valstop_ctrl")

DS = "illness"
HS = (24, 36, 48, 60)
SEEDS = (2021, 2022, 2023)
TASKS = [(h, s) for h in HS for s in SEEDS]
assert len(TASKS) == 12 and len(set(TASKS)) == 12
SMOKE = (24, 2021)                  # two rounds on this task
QUEUE = [t for t in TASKS if t != SMOKE]
assert len(QUEUE) == 11

RULES = ("A", "B", "C")
RULE_DESC = {
    "A": "full fixed val, element-weighted MSE (EXP-003 Exp_Univariate.vali)",
    "B": "same full fixed set, equal-weight per-batch criterion means "
         "(official base vali semantics; partial last batch = 1/n_batches)",
    "C": "official counterfactual: per-epoch perm + drop_last floor(N/256) "
         "full batches + equal-weight batch means (dedicated RandomState)",
}

# ---- EXP-003 archive / theta-log resolution (read-only) ------------------
_EXP003_CANDIDATES = [
    os.path.join(REPO_ROOT, "save_fullshot"),
    os.path.join("/dev_data/wlt/VisionTS_Experiments", "save_fullshot"),
]
_THETA_W = ("w0", "w1")


def exp003_base():
    for d in _EXP003_CANDIDATES:
        if os.path.isdir(os.path.join(d, "illness_24_seed2021")):
            return d
    return _EXP003_CANDIDATES[0]


def exp003_job(h, seed):
    return os.path.join(exp003_base(), "illness_%d_seed%d" % (h, seed))


def theta_log(h, seed):
    for w in _THETA_W:
        p = os.path.join(REPO_ROOT, "repro_fullshot", "logs_theta", w,
                         "illness_%d_seed%d.log" % (h, seed))
        if os.path.isfile(p):
            return p
    return None


# ---- data / provenance pins (from the EXP-003 fingerprints) --------------
CSV_REL = os.path.join("dataset", "illness", "national_illness.csv")
CSV_SHA256 = "93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a"
CKPT_REL = os.path.join("ckpt", "mae_visualize_vit_base.pth")
CKPT_SHA256 = C.CKPT_SHA256

# val coverage arithmetic (per-channel illness windows), asserted at
# runtime against the actual Dataset_Custom borders in the audit.
VAL_N_PIN = {24: 518, 36: 434, 48: 350, 60: 266}          # full val samples
C_BATCHES_PIN = {h: VAL_N_PIN[h] // 256 for h in HS}       # 2/1/1/1
VAL_DROPPED_PIN = {h: VAL_N_PIN[h] - 256 * C_BATCHES_PIN[h]
                   for h in HS}                             # 6/178/94/10
assert VAL_DROPPED_PIN == {24: 6, 36: 178, 48: 94, 60: 10}
assert C_BATCHES_PIN == {24: 2, 36: 1, 48: 1, 60: 1}

# paper Table 21 (Appendix C.2) Illness VISIONTS rows (LN finetune)
PAPER_T21 = {
    24: (2.034, 0.937), 36: (1.866, 0.888),
    48: (1.784, 0.870), 60: (1.910, 0.912),
    "avg": (1.899, 0.902),
}

# lr cadence (type1, adjust_learning_rate called with epoch+1 AFTER the
# epoch's training; verified against long_term_tsf/utils/tools.py L13-28):
#   train_lr_used[1] = 1e-4;  train_lr_used[E>=2] = 1e-4 * 0.5**(E-2)
#   lr_after_update[E] = 1e-4 * 0.5**(E-1)
LR_BASE = 1e-4


def expected_train_lr(epoch_1idx):
    if epoch_1idx == 1:
        return LR_BASE
    return LR_BASE * (0.5 ** (epoch_1idx - 2))


def expected_lr_after_update(epoch_1idx):
    return LR_BASE * (0.5 ** (epoch_1idx - 1))


# ---- deterministic dedicated-stream seeds (never hash()) -----------------
def derived_seed(key, n_bytes=8):
    d = hashlib.sha256(key.encode("utf-8")).digest()
    v = int.from_bytes(d[:n_bytes], "big")
    return v & 0x7FFFFFFF          # positive int32: safe for
                                   # Generator.manual_seed and RandomState


def gen_seed(h, seed, rule, round_tag):
    """torch.Generator.manual_seed for the rule's val DataLoader."""
    return derived_seed("t40|illness|%d|seed%d|%s|%s|valgen"
                        % (h, seed, rule, round_tag))


def c_stream_seed(h, seed, round_tag):
    """np.random.RandomState seed for rule C's per-epoch permutations."""
    return derived_seed("t40|illness|%d|seed%d|C|%s|perm"
                        % (h, seed, round_tag))


def rule_state_seed(h, seed, rule, round_tag):
    return derived_seed("t40|illness|%d|seed%d|%s|%s|rulestate"
                        % (h, seed, rule, round_tag))


# ---- paths ----------------------------------------------------------------
def job_name(h, seed, round_tag):
    return "illness_%d_seed%d_%s" % (h, seed, round_tag)


def job_dir(h, seed, round_tag="main"):
    base = SAVE_BASE if round_tag == "main" else SAVE_CTRL_BASE
    return os.path.join(base, "illness_%d_seed%d" % (h, seed))


def train_json(h, seed, round_tag="main"):
    return os.path.join(LOG_DIR, "train_illness_%d_seed%d_%s.json"
                        % (h, seed, round_tag))


def curves_path(h, seed, round_tag="main"):
    return os.path.join(job_dir(h, seed, round_tag), "curves.npz")


def rng_audit_path(h, seed, round_tag="main"):
    return os.path.join(job_dir(h, seed, round_tag), "rng_audit.json")


def lock_events_path(h, seed, round_tag="main"):
    return os.path.join(job_dir(h, seed, round_tag), "lock_events.json")


def locks_dir(h, seed, rule, round_tag="main"):
    return os.path.join(job_dir(h, seed, round_tag), "locks", rule)


def rule_ckpt(h, seed, rule, round_tag="main"):
    return os.path.join(locks_dir(h, seed, rule, round_tag), "checkpoint.pth")


def rule_results_dir(h, seed, rule, round_tag="main"):
    return os.path.join(job_dir(h, seed, round_tag), "results", rule)


def save_base(round_tag="main"):
    return SAVE_BASE if round_tag == "main" else SAVE_CTRL_BASE


def allowed_rules(round_tag):
    return RULES if round_tag == "main" else ("A",)


def round_tag_of(round_label):
    """'main'/'ctrl' driver label -> provenance string kept in records."""
    return round_label


def mark_failed(stage, msg):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(FAILED_TXT, "a") as f:
        f.write("[%s] %s: %s\n" % (C.time_str(), stage, msg))
    print("[t40][FAILED] %s: %s" % (stage, msg), flush=True)
