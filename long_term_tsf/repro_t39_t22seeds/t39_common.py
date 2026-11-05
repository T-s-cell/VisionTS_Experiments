"""EXP-039 helpers: paper Table 22 B-line extended to three seeds.

Adds All / Rand-VM at seed 2022 and 2023 (64 new trainings: 16 configs x
2 seeds x 2 arms) on top of the frozen EXP-038 seed-2021 archive (32
groups, incl. the EXP-037 ETTh1_96 reuse pair). LN column = EXP-003
per-seed archives (seeds 2021/2022/2023, 48 pinned SUCCESS values — the
paper's LN column is a 3-seed mean); Freeze column = EXP-033
summary_t22.csv (single frozen inference, seed-independent).

Everything seed-independent is inherited from t38_common (Table 19
L/P, window/batch formulas, paper anchors, frozen fingerprints,
sha/json/csv helpers). EXP-037/EXP-038 directories are READ-ONLY.
"""
import os
import sys

_T38_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "repro_t38_t15t22")
if os.path.isdir(_T38_DIR) and _T38_DIR not in sys.path:
    sys.path.insert(0, _T38_DIR)

import t38_common as C

PKG = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PKG, "logs")
VERIFY_JSON = os.path.join(LOG_DIR, "verify_t39.json")
SUMMARY_CSV = os.path.join(LOG_DIR, "summary_t39_ett.csv")
RESULT_MD = os.path.join(LOG_DIR, "RESULT_t39_ett.md")
FIG_PNG = os.path.join(LOG_DIR, "fig_t39_seed_effect.png")
RECOMPUTE_JSON = os.path.join(LOG_DIR, "recompute_f64.json")
FAILED_TXT = os.path.join(LOG_DIR, "FAILED.txt")
OK_MARKER = os.path.join(PKG, "T39_OK")

SAVE_B_BASE = os.path.join(C.ROOT, "save_t39_full")

# frozen EXP-038 package (read-only provenance for the seed-2021 arms)
T38_DIR = os.path.join(C.ROOT, "repro_t38_t15t22")
T38_LOGS = os.path.join(T38_DIR, "logs")

SEEDS = [2022, 2023]
CONFIGS = [(ds, h) for ds in C.ETT4 for h in C.HORIZONS]   # 16
# 64 new tasks, built programmatically; uniqueness asserted below
TASKS = [(ds, h, arm, s) for (ds, h) in CONFIGS for s in SEEDS
         for arm in C.B_ARMS]
assert len(TASKS) == 64, "expected 64 new tasks, got %d" % len(TASKS)
assert len(set(TASKS)) == 64, "duplicate tasks in TASKS"
assert all((ds, h) in CONFIGS for ds, h, _, _ in TASKS)

# smoke = ETTh1_96 x {2022,2023} x {all,rand}; counted in TASKS
SMOKE = [("ETTh1", 96, arm, s) for s in SEEDS for arm in C.B_ARMS]
assert len(SMOKE) == 4 and all(t in TASKS for t in SMOKE)
# queue = everything else, config-major: ETTh1 192/336/720 -> ETTh2 ->
# ETTm2 -> ETTm1 (ETTm1_96 included — it is NOT the pilot here)
QUEUE_CONFIGS = ([(ds, h) for ds, h in
                  [("ETTh1", h) for h in (192, 336, 720)]
                  + [("ETTh2", h) for h in C.HORIZONS]
                  + [("ETTm2", h) for h in C.HORIZONS]
                  + [("ETTm1", h) for h in C.HORIZONS]])
QUEUE = [(ds, h, arm, s) for (ds, h) in QUEUE_CONFIGS
         for s in SEEDS for arm in C.B_ARMS]
assert sorted(QUEUE) == sorted(t for t in TASKS if t not in SMOKE), \
    "QUEUE must be exactly TASKS minus SMOKE (60)"

# EXP-003 LN per-seed SUCCESS values, 48 pins (transcribed verbatim from
# the archives; seed2021 entries must equal t38_common.LN16 exactly).
LN48 = {
    "ETTh1_96_seed2021": (0.34744012355804443, 0.3768464922904968),
    "ETTh1_96_seed2022": (0.3464500308036804, 0.3759533166885376),
    "ETTh1_96_seed2023": (0.34608694911003113, 0.37594494223594666),
    "ETTh1_192_seed2021": (0.38549375534057617, 0.4003775715827942),
    "ETTh1_192_seed2022": (0.3832252025604248, 0.39948710799217224),
    "ETTh1_192_seed2023": (0.38290300965309143, 0.39868634939193726),
    "ETTh1_336_seed2021": (0.40655839443206787, 0.4161790907382965),
    "ETTh1_336_seed2022": (0.40941572189331055, 0.41681769490242004),
    "ETTh1_336_seed2023": (0.40815064311027527, 0.41517874598503113),
    "ETTh1_720_seed2021": (0.43681803345680237, 0.44175195693969727),
    "ETTh1_720_seed2022": (0.4405474364757538, 0.4440388083457947),
    "ETTh1_720_seed2023": (0.43083539605140686, 0.43958860635757446),
    "ETTh2_96_seed2021": (0.26963549852371216, 0.32940933108329773),
    "ETTh2_96_seed2022": (0.26857367157936096, 0.3273617923259735),
    "ETTh2_96_seed2023": (0.26854807138442993, 0.3285598158836365),
    "ETTh2_192_seed2021": (0.3324255049228668, 0.3762674927711487),
    "ETTh2_192_seed2022": (0.3356940448284149, 0.37831875681877136),
    "ETTh2_192_seed2023": (0.3397817313671112, 0.3813319206237793),
    "ETTh2_336_seed2021": (0.35398921370506287, 0.3985902965068817),
    "ETTh2_336_seed2022": (0.3530617356300354, 0.39764395356178284),
    "ETTh2_336_seed2023": (0.35578030347824097, 0.3999505043029785),
    "ETTh2_720_seed2021": (0.38139450550079346, 0.42356714606285095),
    "ETTh2_720_seed2022": (0.38393205404281616, 0.4254649579524994),
    "ETTh2_720_seed2023": (0.38525572419166565, 0.4260551333427429),
    "ETTm1_96_seed2021": (0.2832735478878021, 0.3229347765445709),
    "ETTm1_96_seed2022": (0.28413909673690796, 0.32342013716697693),
    "ETTm1_96_seed2023": (0.2880454957485199, 0.324727863073349),
    "ETTm1_192_seed2021": (0.32713618874549866, 0.3552905321121216),
    "ETTm1_192_seed2022": (0.3277602195739746, 0.35416004061698914),
    "ETTm1_192_seed2023": (0.3261748254299164, 0.3540242612361908),
    "ETTm1_336_seed2021": (0.35902950167655945, 0.3817652761936188),
    "ETTm1_336_seed2022": (0.3539494276046753, 0.3790690004825592),
    "ETTm1_336_seed2023": (0.34707075357437134, 0.3780372440814972),
    "ETTm1_720_seed2021": (0.3920411467552185, 0.41577470302581787),
    "ETTm1_720_seed2022": (0.38968411087989807, 0.4146157205104828),
    "ETTm1_720_seed2023": (0.38857924938201904, 0.4136257469654083),
    "ETTm2_96_seed2021": (0.16750384867191315, 0.25540876388549805),
    "ETTm2_96_seed2022": (0.1660427451133728, 0.25381404161453247),
    "ETTm2_96_seed2023": (0.16715869307518005, 0.25591808557510376),
    "ETTm2_192_seed2021": (0.22219505906105042, 0.29210129380226135),
    "ETTm2_192_seed2022": (0.23548467457294464, 0.30303624272346497),
    "ETTm2_192_seed2023": (0.21908120810985565, 0.28984686732292175),
    "ETTm2_336_seed2021": (0.2849220633506775, 0.33980801701545715),
    "ETTm2_336_seed2022": (0.27357974648475647, 0.3322795033454895),
    "ETTm2_336_seed2023": (0.27545788884162903, 0.3336550295352936),
    "ETTm2_720_seed2021": (0.3702132999897003, 0.3943851888179779),
    "ETTm2_720_seed2022": (0.37354162335395813, 0.3950386643409729),
    "ETTm2_720_seed2023": (0.37895286083221436, 0.3971821367740631),
}
assert len(LN48) == 48
for (ds, h), (mse, mae) in C.LN16.items():
    k = "%s_%d_seed2021" % (ds, h)
    assert LN48[k] == (mse, mae), "LN48 seed2021 pin %s != LN16" % k

# LN three-seed overall self-check anchor (per seed: 4H equal mean per
# dataset -> 4 datasets equal mean; then equal mean across the 3 seeds)
LN_OVERALL_ANCHOR = (0.3323544589802623, 0.36986022753020126)


def ln_overall():
    per_seed = []
    per_seed_mae = []
    for s in (2021, 2022, 2023):
        ds_m, ds_a = [], []
        for ds in C.ETT4:
            vals = [LN48["%s_%d_seed%d" % (ds, h, s)] for h in C.HORIZONS]
            ds_m.append(sum(v[0] for v in vals) / 4.0)
            ds_a.append(sum(v[1] for v in vals) / 4.0)
        per_seed.append(sum(ds_m) / 4.0)
        per_seed_mae.append(sum(ds_a) / 4.0)
    return (sum(per_seed) / 3.0, sum(per_seed_mae) / 3.0)


assert abs(ln_overall()[0] - LN_OVERALL_ANCHOR[0]) <= 1e-6, ln_overall()
assert abs(ln_overall()[1] - LN_OVERALL_ANCHOR[1]) <= 1e-6, ln_overall()


# seed-2021 B-line provenance (EXP-038 logs + EXP-037 reuse pair)
def t38_train_json(ds, h, arm):
    if (ds, h) == ("ETTh1", 96):
        return C.t37_train_json(arm)
    return os.path.join(T38_LOGS, "train_%s_%d_%s.json" % (ds, h, arm))


def t38_job_dir(ds, h, arm):
    if (ds, h) == ("ETTh1", 96):
        return C.t37_job_dir(arm)
    return os.path.join(C.SAVE_B_BASE, "%s_%d_seed2021_%s" % (ds, h, arm))


# seed-aware paths for this round
def config_job(ds, h, arm, seed):
    return "%s_%d_seed%d_%s" % (ds, h, seed, arm)


def b_job_dir(ds, h, arm, seed):
    return os.path.join(SAVE_B_BASE, config_job(ds, h, arm, seed))


def train_json(ds, h, arm, seed):
    return os.path.join(LOG_DIR, "train_%s_%d_%s_seed%d.json"
                        % (ds, h, arm, seed))


# EXP-003 LN per-seed job dirs (eta first, local audited fallback)
_EXP003_SEED_CANDIDATES = [
    os.path.join(C.ROOT, "save_fullshot"),       # eta
    os.path.join(C.REPO_ROOT, "save_fullshot"),  # local audited
]


def exp003_seed_base():
    marker = "ETTh1_96_seed2022"
    for d in _EXP003_SEED_CANDIDATES:
        if os.path.isdir(os.path.join(d, marker)):
            return d
    return _EXP003_SEED_CANDIDATES[0]


def exp003_seed_job(ds, h, seed):
    return os.path.join(exp003_seed_base(), "%s_%d_seed%d" % (ds, h, seed))


def ln_success_path(ds, h, seed):
    return os.path.join(exp003_seed_job(ds, h, seed), "SUCCESS.txt")


# driver must allow the seed delta on top of t38's whitelist
FP_ALLOWED_ARG_DELTA = set(C.FP_ALLOWED_ARG_DELTA) | {"seed"}

LN_KEYS_SEEDS = (2021, 2022, 2023)
