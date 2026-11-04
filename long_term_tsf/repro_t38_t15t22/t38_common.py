"""EXP-038 helpers: two independent scale-ups on top of frozen EXP-037 v4.

A. Monash zero-shot, FULL paper Table 15 coverage (29 paper rows / 30 code
   subsets) in the AUTHOR-PINNED env (visionts_t37author: python 3.8.20,
   torch 1.7.1+cu110, gluonts 0.14.4, pandas 1.5.3, numpy 1.23.5;
   HOME=/dev_data/wlt/t37_author_home). Two replay subsets (FRED-MD,
   NN5 Daily) must reproduce the author anchors (rel <= 1e-3) AND the Naive
   denominator must round-match the paper print, before the queue starts.
   Every subset's prediction length is re-derived at runtime from the raw
   .tsf metadata or the official rules (hard overrides for
   australian_electricity_demand=336 / pedestrian_counts=24, else the
   frequency fallback map) and checked against the actual test windows.
   Paper Table 6 P values are expectations only (weather is P=7, the DAILY
   Monash Weather; P=144 belongs to the 10-minute long-term Weather) — the
   run still autotunes P on the validation slice. CIF 2016 is pooled across
   cif_2016_6 (15 series) + cif_2016_12 (57 series) into one paper row,
   weighting the unrounded MAEs by series count separately for VisionTS and
   Naive; the geomean of the 29 ratios is only computed when all 29 rows
   are valid.

B. Paper Table 22, four ETT datasets x H{96,192,336,720} x {All, Rand-VM},
   seed=2021 (EXP-003 recipe; LN column = EXP-003 seed2021 archive — the
   paper's LN column is a 3-seed mean, so a single-seed comparison uses the
   seed-2021 archive; Freeze column = EXP-033 summary_t22.csv freeze values,
   provenance per freeze_source). ETTh1_96 reuses the frozen EXP-037 arms.
   Batch counts are FORMULA-derived (bs=256, drop_last=True): e.g. ETTh1/2
   epochs = 202/199/195/185 batches for H=96/192/336/720.

EXP-037 directories are READ-ONLY for this package.
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

PKG = os.path.dirname(os.path.abspath(__file__))            # .../repro_t38_t15t22
ROOT = os.path.dirname(PKG)                                  # long_term_tsf/
REPO_ROOT = os.path.dirname(ROOT)                            # repo root
EVAL_GLUONTS = os.path.join(REPO_ROOT, "eval_gluonts")
LOG_DIR = os.path.join(PKG, "logs")
VERIFY_JSON = os.path.join(LOG_DIR, "verify_t38.json")
SUMMARY_A_CSV = os.path.join(LOG_DIR, "summary_t38_monash.csv")
RESULT_A_MD = os.path.join(LOG_DIR, "RESULT_t38_monash.md")
SUMMARY_B_CSV = os.path.join(LOG_DIR, "summary_t38_ett.csv")
RESULT_B_MD = os.path.join(LOG_DIR, "RESULT_t38_ett.md")
FIG5_PNG = os.path.join(LOG_DIR, "fig5_monash_ratio.png")

SAVE_A_BASE = os.path.join(ROOT, "save_t38_monash")
SAVE_B_BASE = os.path.join(ROOT, "save_t38_full")

# frozen EXP-037 package (read-only)
T37_DIR = os.path.join(ROOT, "repro_t37_pilot")
T37_LOGS = os.path.join(T37_DIR, "logs")
T37_SAVE = os.path.join(ROOT, "save_t37_full")

A_OK = os.path.join(PKG, "A_OK")
B_OK = os.path.join(PKG, "B_OK")
FAILED_TXT = os.path.join(LOG_DIR, "FAILED.txt")

for _p in (ROOT, os.path.join(ROOT, "repro_t22_freeze_ln"),
           os.path.join(REPO_ROOT, "repro_fullshot")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 2021

# frozen MAE-base backbone (pinned since EXP-033/t35/t36/t37)
CKPT_PATH_ETA = "/dev_data/wlt/VisionTS_Experiments/ckpt/mae_visualize_vit_base.pth"
CKPT_REL = "../ckpt/mae_visualize_vit_base.pth"
CKPT_SHA256 = "c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e"
CKPT_BYTES = 447725789

# ---------------------------------------------------------------- A: Monash
MONASH_BATCH = 512
MONASH_CONTEXT = 1000
MONASH_NO_P_CONTEXT = 300
ANCHOR_TOL = 1e-3            # rel diff vs author precomputed MAE
N1_ALIGN_TOL = 0.05          # N1 vs paper Naive window-alignment evidence
GEOMEAN_WARN_REL = 0.02      # warn (not gate) if geomean drifts >2% from paper
GEOMEAN_PAPER = 0.729        # paper Table 15 "Normalized MAE" of VISIONTS

# official run_monash.sh order (30 code subsets)
MONASH_ORDER = [
    "m1_monthly", "monash_m3_monthly", "monash_m3_other", "m4_monthly",
    "m4_weekly", "m4_daily", "m4_hourly", "tourism_quarterly",
    "tourism_monthly", "cif_2016_6", "cif_2016_12",
    "australian_electricity_demand", "bitcoin", "pedestrian_counts",
    "vehicle_trips_without_missing", "kdd_cup_2018_without_missing",
    "weather", "nn5_daily_without_missing", "nn5_weekly",
    "car_parts_without_missing", "fred_md", "traffic_hourly",
    "traffic_weekly", "rideshare_without_missing", "hospital",
    "covid_deaths", "temperature_rain_without_missing",
    "sunspot_without_missing", "saugeenday", "us_births",
]
REPLAY = ["fred_md", "nn5_daily_without_missing"]
QUEUE = ["tourism_monthly"] + [ds for ds in MONASH_ORDER
                               if ds not in REPLAY and ds != "tourism_monthly"]

# M4 subsets download from GitHub M4-methods (gluonts _m4.py), not zenodo:
# raw-zip probes do not apply; provenance + cache diff instead.
M4_SUBSETS = {"m4_monthly", "m4_weekly", "m4_daily", "m4_hourly"}

# official get_gluonts_test_dataset hard overrides (dataset.py:403-406)
HARD_OVERRIDE = {"australian_electricity_demand": 336, "pedestrian_counts": 24}
# official default_prediction_length_from_frequency map (dataset.py:41-49)
FALLBACK_MAP = {"T": 10080, "H": 168, "D": 30, "W-SUN": 8, "M": 12,
                "Y": 4, "S": 604800}
# pandas>=2.2 renamed month/year END aliases; shim only these keys with the
# OFFICIAL values (inert on the author env, pandas 1.5.3)
FALLBACK_SHIM = {"ME": 12, "YE": 4}

# pinned prediction lengths (hard gates): replay subsets + CIF split halves
PINNED_PRED = {"fred_md": 12, "nn5_daily_without_missing": 56,
               "cif_2016_6": 6, "cif_2016_12": 12}

# paper Table 6 final P (expectation only; autotune still decides).
# User correction 2026-11-04: Monash DAILY Weather is P=7 (Table 6 row
# "D | 7 | Bitcoin Vehicle Trips Weather NN5 Daily ..."); P=144 is the
# long-term TSF 10-minute Weather and must NOT be used here.
EXPECTED_P = {
    "m1_monthly": 12, "monash_m3_monthly": 3, "monash_m3_other": 1,
    "m4_monthly": 3, "m4_weekly": 4, "m4_daily": 1, "m4_hourly": 24,
    "tourism_quarterly": 4, "tourism_monthly": 12, "cif_2016_6": 3,
    "cif_2016_12": 6, "australian_electricity_demand": 336, "bitcoin": 7,
    "pedestrian_counts": 24, "vehicle_trips_without_missing": 7,
    "kdd_cup_2018_without_missing": 24, "weather": 7,
    "nn5_daily_without_missing": 7, "nn5_weekly": 1,
    "car_parts_without_missing": 6, "fred_md": 1, "traffic_hourly": 24,
    "traffic_weekly": 4, "rideshare_without_missing": 24, "hospital": 12,
    "covid_deaths": 1, "temperature_rain_without_missing": 7,
    "sunspot_without_missing": 30, "saugeenday": 7, "us_births": 7,
}

# paper Table 15 prints (VERBATIM strings; bitcoin in scientific notation,
# compared by its significant digits, user correction 2026-11-04). Key
# CIF_2016_POOLED is the pooled paper row for cif_2016_6+cif_2016_12.
PAPER_T15 = {
    "m1_monthly": {"vts": "1987.69", "naive": "2707.75"},
    "monash_m3_monthly": {"vts": "737.93", "naive": "837.14"},
    "monash_m3_other": {"vts": "315.85", "naive": "278.43"},
    "m4_monthly": {"vts": "666.54", "naive": "671.27"},
    "m4_weekly": {"vts": "404.23", "naive": "347.99"},
    "m4_daily": {"vts": "215.63", "naive": "180.83"},
    "m4_hourly": {"vts": "288.37", "naive": "1218.06"},
    "tourism_quarterly": {"vts": "12931.88", "naive": "15845.1"},
    "tourism_monthly": {"vts": "2560.19", "naive": "5636.83"},
    "CIF_2016_POOLED": {"vts": "570907.24", "naive": "578596.5"},
    "australian_electricity_demand": {"vts": "237.44", "naive": "659.6"},
    "bitcoin": {"vts": "2.33E+18", "naive": "7.78E+17"},
    "pedestrian_counts": {"vts": "52.01", "naive": "170.88"},
    "vehicle_trips_without_missing": {"vts": "22.08", "naive": "31.42"},
    "kdd_cup_2018_without_missing": {"vts": "38.16", "naive": "42.13"},
    "weather": {"vts": "2.06", "naive": "2.36"},
    "nn5_daily_without_missing": {"vts": "3.51", "naive": "8.26"},
    "nn5_weekly": {"vts": "14.67", "naive": "16.71"},
    "car_parts_without_missing": {"vts": "0.58", "naive": "0.65"},
    "fred_md": {"vts": "1893.67", "naive": "2825.67"},
    "traffic_hourly": {"vts": "0.01", "naive": "0.03"},
    "traffic_weekly": {"vts": "1.14", "naive": "1.19"},
    "rideshare_without_missing": {"vts": "5.92", "naive": "6.29"},
    "hospital": {"vts": "19.36", "naive": "24.07"},
    "covid_deaths": {"vts": "137.51", "naive": "353.71"},
    "temperature_rain_without_missing": {"vts": "6.37", "naive": "9.39"},
    "sunspot_without_missing": {"vts": "2.81", "naive": "3.93"},
    "saugeenday": {"vts": "30.22", "naive": "21.5"},
    "us_births": {"vts": "519.94", "naive": "1152.67"},
}
# Table 15 paper rows (29) in print order; CIF is the pooled row
PAPER_T15_ORDER = [
    "m1_monthly", "monash_m3_monthly", "monash_m3_other", "m4_monthly",
    "m4_weekly", "m4_daily", "m4_hourly", "tourism_quarterly",
    "tourism_monthly", "CIF_2016_POOLED", "australian_electricity_demand",
    "bitcoin", "pedestrian_counts", "vehicle_trips_without_missing",
    "kdd_cup_2018_without_missing", "weather", "nn5_daily_without_missing",
    "nn5_weekly", "car_parts_without_missing", "fred_md", "traffic_hourly",
    "traffic_weekly", "rideshare_without_missing", "hospital",
    "covid_deaths", "temperature_rain_without_missing",
    "sunspot_without_missing", "saugeenday", "us_births",
]
# author precomputed UNROUNDED anchors (eval_gluonts/precomputed_results/
# monash.csv, transcribed verbatim; 30 code rows)
AUTHOR_ANCHOR = {
    "m1_monthly": {"mae": 1987.688524778489, "mse": 473242806.3574582},
    "monash_m3_monthly": {"mae": 737.9286053731669, "mse": 1816747.784919702},
    "monash_m3_other": {"mae": 315.85190011671443, "mse": 353847.59726091765},
    "m4_monthly": {"mae": 666.542285729571, "mse": 2220099.9432321494},
    "m4_weekly": {"mae": 404.2335016314153, "mse": 556923.9748224751},
    "m4_daily": {"mae": 215.6305384347969, "mse": 463570.92101390415},
    "m4_hourly": {"mae": 288.3720925388093, "mse": 1841385.6171803745},
    "tourism_quarterly": {"mae": 12931.875315263896, "mse": 18784779831.08448},
    "tourism_monthly": {"mae": 2560.1894177780323, "mse": 270591746.82708025},
    "cif_2016_6": {"mae": 2690186.32265504, "mse": 37220248790871.2},
    "cif_2016_12": {"mae": 13202.203990686825, "mse": 7349010031.238033},
    "australian_electricity_demand": {"mae": 237.435282621234,
                                      "mse": 129165.89440201422},
    "bitcoin": {"mae": 2.3263519906884823e+18, "mse": 1.0359595139106075e+38},
    "pedestrian_counts": {"mae": 52.00575514256805, "mse": 36589.98451330238},
    "vehicle_trips_without_missing": {"mae": 22.082236005438084,
                                      "mse": 3805.4200116202046},
    "kdd_cup_2018_without_missing": {"mae": 38.16027990301013,
                                     "mse": 19458.290071308733},
    "weather": {"mae": 2.062313813461159, "mse": 16.235318423548073},
    "nn5_daily_without_missing": {"mae": 3.514067169133817,
                                  "mse": 29.21466767449964},
    "nn5_weekly": {"mae": 14.673289428601764, "mse": 439.43390790646106},
    "car_parts_without_missing": {"mae": 0.5750384417170816,
                                  "mse": 1.585925972781912},
    "fred_md": {"mae": 1893.668241959648, "mse": 228818192.9765722},
    "traffic_hourly": {"mae": 0.0120616098617277, "mse": 0.0005793749814068},
    "traffic_weekly": {"mae": 1.1392348558854624, "mse": 4.090418197431147},
    "rideshare_without_missing": {"mae": 5.924230305868256,
                                  "mse": 135.20118230824698},
    "hospital": {"mae": 19.362289049205128, "mse": 3188.750778909795},
    "covid_deaths": {"mae": 137.50615390907925, "mse": 557226.5882195022},
    "temperature_rain_without_missing": {"mae": 6.372606752077468,
                                         "mse": 156.1453984423366},
    "sunspot_without_missing": {"mae": 2.806585902972492,
                                "mse": 21.258672422380087},
    "saugeenday": {"mae": 30.217300610643388, "mse": 1977.6161744534404},
    "us_births": {"mae": 519.9384686380703, "mse": 473984.9620770046},
}

# CIF 2016 pooling: expected series counts (runtime-verified from splits)
CIF_WEIGHTS = {"cif_2016_6": 15, "cif_2016_12": 57}

CSV_COLUMNS = ["dataset", "MSE[mean]", "MAE[0.5]", "sMAPE[0.5]",
               "MASE[0.5]", "ND[0.5]", "NRMSE[mean]"]

# EXP-037 current-env FRED-MD (kept as G4 FAIL; never re-run here)
T37_FRED_CURRENT_ENV_MAE = 1807.417940
T37_FRED_CURRENT_ENV_REL = 4.555e-2


def paper_round_match(got, printed):
    """True if `got` rounds to the paper's printed value at the print's own
    precision. Handles plain decimals ("21.5" -> quantum 0.1) and
    scientific notation ("2.33E+18" -> quantum 1e16). Returns
    (ok, quantum, printed_float)."""
    s = printed.strip().upper().replace("E+", "E")
    if "E" in s:
        mant, exp = s.split("E")
        exp = int(exp)
        dec = len(mant.split(".")[1]) if "." in mant else 0
        quantum = 10.0 ** (exp - dec)
    else:
        dec = len(s.split(".")[1]) if "." in s else 0
        quantum = 10.0 ** (-dec)
    want = float(printed)
    ok = abs(got - want) <= 0.5 * quantum + 1e-9 * abs(want)
    return ok, quantum, want


def gluonts_cache():
    return os.path.join(os.path.expanduser("~"), ".gluonts")


def monash_json(ds):
    return os.path.join(LOG_DIR, "monash_%s.json" % ds)


def naive_json(ds):
    return os.path.join(LOG_DIR, "naive_%s.json" % ds)


def vision_csv(ds):
    return os.path.join(SAVE_A_BASE, "visionts_monash_%s.csv" % ds)


def naive_n_csv(ds, tag):
    return os.path.join(SAVE_A_BASE, "naive_%s_monash_%s.csv" % (tag, ds))


def forecast_npz(ds):
    return os.path.join(SAVE_A_BASE, "forecast_monash_%s.npz" % ds)


# ------------------------------------------------------- B: four ETT, t22
T19 = {  # Table 19: (context L, periodicity P)
    "ETTh1": (1152, 24), "ETTh2": (1152, 24),
    "ETTm1": (2304, 96), "ETTm2": (1152, 96),
}
HORIZONS = [96, 192, 336, 720]
ETT4 = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
B_BS = 256
B_ARMS = ["all", "rand"]

# queue order: pilot ETTm1_96 first (minute-level budget probe), then the
# remaining 14 configs x 2 arms
B_PILOT = ("ETTm1", 96)
B_QUEUE = [("ETTh1", h) for h in (192, 336, 720)] + \
          [("ETTh2", h) for h in HORIZONS] + \
          [("ETTm2", h) for h in HORIZONS] + \
          [("ETTm1", h) for h in (192, 336, 720)]


def month_rows(ds):
    return 30 * 24 * (4 if ds.startswith("ETTm") else 1)


def expected_n_windows(ds, h):
    return 4 * month_rows(ds) - h + 1


def expected_epoch_len(ds, h):
    # channel-independent: 7 vars x train windows; train rows = 12 months
    L = T19[ds][0]
    return 7 * (12 * month_rows(ds) - L - h + 1)


def expected_train_batches(ds, h):
    return expected_epoch_len(ds, h) // B_BS            # drop_last=True


def expected_eval_batches(ds, h, windows=None):
    n = expected_n_windows(ds, h) if windows is None else windows
    return int(math.ceil(n * 7 / float(B_BS)))          # drop_last=False


def config_job(ds, h, arm):
    return "%s_%d_seed2021_%s" % (ds, h, arm)


def b_job_dir(ds, h, arm):
    return os.path.join(SAVE_B_BASE, config_job(ds, h, arm))


def train_json(ds, h, arm):
    return os.path.join(LOG_DIR, "train_%s_%d_%s.json" % (ds, h, arm))


def t37_train_json(arm):
    return os.path.join(T37_LOGS, "train_%s.json" % arm)


def t37_job_dir(arm, run=None):
    name = "ETTh1_96_seed2021_%s" % arm + ("_run1" if run else "")
    return os.path.join(T37_SAVE, name)


# EXP-003 LN seed2021 SUCCESS values (16 groups, transcribed verbatim from
# the eta archives; BG9 re-reads SUCCESS.txt and gates == these pins).
# The paper's LN column is a 3-seed mean; this round compares at seed 2021
# (user correction 2026-11-04).
LN16 = {
    ("ETTh1", 96): (0.34744012355804443, 0.3768464922904968),
    ("ETTh1", 192): (0.38549375534057617, 0.4003775715827942),
    ("ETTh1", 336): (0.40655839443206787, 0.4161790907382965),
    ("ETTh1", 720): (0.43681803345680237, 0.44175195693969727),
    ("ETTh2", 96): (0.26963549852371216, 0.32940933108329773),
    ("ETTh2", 192): (0.3324255049228668, 0.3762674927711487),
    ("ETTh2", 336): (0.35398921370506287, 0.3985902965068817),
    ("ETTh2", 720): (0.38139450550079346, 0.42356714606285095),
    ("ETTm1", 96): (0.2832735478878021, 0.3229347765445709),
    ("ETTm1", 192): (0.32713618874549866, 0.3552905321121216),
    ("ETTm1", 336): (0.35902950167655945, 0.3817652761936188),
    ("ETTm1", 720): (0.3920411467552185, 0.41577470302581787),
    ("ETTm2", 96): (0.16750384867191315, 0.25540876388549805),
    ("ETTm2", 192): (0.22219505906105042, 0.29210129380226135),
    ("ETTm2", 336): (0.2849220633506775, 0.33980801701545715),
    ("ETTm2", 720): (0.3702132999897003, 0.3943851888179779),
}

# EXP-037 frozen reuse pair (ETTh1_96, seed 2021) — SUCCESS.txt values
REUSE = {
    ("ETTh1", 96, "all"): {"mse": 0.4468631446361542, "mae": 0.4180694818496704},
    ("ETTh1", 96, "rand"): {"mse": 0.696180522441864, "mae": 0.5582699179649353},
}

FREEZE16_SOURCE_NOTE = ("freeze column read from repro_t22_freeze_ln/"
                        "summary_t22.csv (freeze_mse/freeze_mae); "
                        "freeze_source for ETTm1 rows is exp001_cache")

# paper Table 22 (4H means, 3dp): {ds: {arm: (mse, mae)}}; "rand" is the
# Rand-VM column (paper coincidentally prints the same values as All).
PAPER_T22 = {
    "ETTh1": {"all": (0.534, 0.470), "rand": (0.534, 0.470),
              "ln": (0.395, 0.409), "freeze": (0.419, 0.418)},
    "ETTh2": {"all": (0.411, 0.432), "rand": (0.411, 0.432),
              "ln": (0.336, 0.382), "freeze": (0.340, 0.376)},
    "ETTm1": {"all": (0.433, 0.413), "rand": (0.433, 0.413),
              "ln": (0.338, 0.367), "freeze": (0.374, 0.372)},
    "ETTm2": {"all": (0.288, 0.341), "rand": (0.288, 0.341),
              "ln": (0.261, 0.319), "freeze": (0.305, 0.334)},
}
PAPER_T22_OVERALL = {"all": (0.417, 0.414), "rand": (0.417, 0.414),
                     "ln": (0.333, 0.369), "freeze": (0.360, 0.375)}

METRIC_TOL = 1e-6        # recomputed metrics vs SUCCESS.txt
FP_ALLOWED_ARG_DELTA = {"model_id", "save_dir", "ft_type", "vm_pretrained",
                        "gpu"}

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

ETT_CSV_REL = {ds: "long_term_tsf/dataset/ETT-small/%s.csv" % ds
               for ds in ETT4}

# EXP-003 job dirs (eta first, local audited fallback)
_EXP003_CANDIDATES = [
    os.path.join(ROOT, "save_fullshot"),       # eta
    os.path.join(REPO_ROOT, "save_fullshot"),  # local audited
]
_T22_SUMMARY_CANDIDATES = [
    os.path.join(ROOT, "repro_t22_freeze_ln", "summary_t22.csv"),
    os.path.join(REPO_ROOT, "repro_t22_freeze_ln", "summary_t22.csv"),
]


def first_existing_dir(candidates, marker_subpath):
    for d in candidates:
        if os.path.isdir(os.path.join(d, marker_subpath)):
            return d
    return candidates[0]


def exp003_base():
    for d in _EXP003_CANDIDATES:
        if os.path.isdir(os.path.join(d, "ETTh1_96_seed2021")):
            return d
    return _EXP003_CANDIDATES[0]


def exp003_job(ds, h):
    return os.path.join(exp003_base(), "%s_%d_seed2021" % (ds, h))


def t22_summary_csv():
    for p in _T22_SUMMARY_CANDIDATES:
        if os.path.isfile(p):
            return p
    return _T22_SUMMARY_CANDIDATES[0]


def load_freeze16():
    """summary_t22.csv -> {(ds, h): (mse, mae, source)}; 16 group rows."""
    import csv
    out = {}
    with open(t22_summary_csv()) as f:
        for r in csv.DictReader(f):
            if r.get("level") != "group":
                continue
            key = (r["dataset"], int(r["horizon"]))
            out[key] = (float(r["freeze_mse"]), float(r["freeze_mae"]),
                        r.get("freeze_source", "?"))
    return out


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
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
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
    print("[t38][FAILED] %s: %s" % (stage, msg), flush=True)


def time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_success_mse_mae(path):
    d = load_json(path)
    return float(d["mse"]), float(d["mae"])


def parse_csv_metrics(path):
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1, "no result rows in %s" % path
    row = rows[-1]
    return {k: float(v) for k, v in row.items() if k != "dataset"}, row["dataset"]


def rel_diff(got, want):
    if want == 0:
        return float("inf") if got != 0 else 0.0
    return abs(got - want) / abs(want)


def cif_pool(values_by_subset):
    """values = {"cif_2016_6": mae6, "cif_2016_12": mae12} -> pooled MAE
    weighted by series counts (paper row = the two code subsets pooled)."""
    n6 = CIF_WEIGHTS["cif_2016_6"]
    n12 = CIF_WEIGHTS["cif_2016_12"]
    return (n6 * values_by_subset["cif_2016_6"]
            + n12 * values_by_subset["cif_2016_12"]) / float(n6 + n12)


def pooled_anchor(name):
    """Author unrounded anchor for the pooled CIF row."""
    return cif_pool({k: AUTHOR_ANCHOR[k][name] for k in CIF_WEIGHTS})
