"""EXP-041 helpers: complete paper Fig.7 by adding 300 Monash samples to
the frozen EXP-002 1,900-sample t-SNE.

Notation (user correction, frozen):
  H = forecast length (official Monash horizon; from the EXP-038 archived
      pred_rule.expected, cross-checked against gluonts metadata);
  P = selected periodicity (EXP-038 autotune accepted_p);
  L = ACTUAL input history length per sample (L_eff = min(L_config, N-H);
      official short histories are kept, e.g. NN5 Daily 735 < 1001).

Two-env contract (same as EXP-038):
  - sample_monash.py runs in the AUTHOR-pinned env (visionts_t37author,
    python 3.8, gluonts 0.14.4, HOME=/dev_data/wlt/t37_author_home) so the
    series come from the same cache EXP-038 read;
  - everything else runs in the visionts env (torch 2.12, GPU forward).

Frozen EXP-002 assets are reused with ZERO re-sampling/re-extraction:
  - outputs/sampling_lists/*.json (ImageNet/ETTm1/Weather/Electricity)
  - outputs/features_1900x54528.npy (file sha pinned below)
  - outputs/mask_ref.json (ids_keep_order = left 5 columns, row-major)
EXP-041 adds ONLY the Monash rows [1900, 2200).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)                                  # long_term_tsf/
REPO_ROOT = os.path.dirname(ROOT)                            # repo root
LOG_DIR = os.path.join(PKG, "logs")
VERIFY_JSON = os.path.join(LOG_DIR, "verify_t41.json")
SMOKE_JSON = os.path.join(LOG_DIR, "smoke_t41.json")
RESULT_MD = os.path.join(LOG_DIR, "RESULT_t41_fig7_monash.md")

SAVE_BASE = os.path.join(ROOT, "save_t41_monash")
MANIFEST_JSON = os.path.join(SAVE_BASE, "monash_manifest.json")
WINDOWS_NPZ = os.path.join(SAVE_BASE, "monash_windows.npz")
FEATURES_MONASH = os.path.join(SAVE_BASE, "features_monash_300x54528.npy")
GROUPS_MONASH = os.path.join(SAVE_BASE, "groups_monash.npy")
INPUT_HASHES_JSON = os.path.join(SAVE_BASE, "input_hashes_monash.json")
EXTRACT_ENV_JSON = os.path.join(SAVE_BASE, "extract_env.json")

for _p in (REPO_ROOT, ROOT, os.path.join(ROOT, "repro_t38_t15t22")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import t38_common as T38   # noqa: E402  (PAPER_T15_ORDER / EXPECTED_P / helpers)

SEED = 2021
STABILITY_SEEDS = (2022, 2023)
N_IMAGENET = 1000
N_TS_PER_SOURCE = 300
N_MONASH = 300
FEATURE_DIM = (1 + 70) * 768        # 54528 (cls + 70 visible patches)
NUM_VISIBLE = 70
TOTAL_PATCHES = 196                 # 14x14, patch 16, image 224
MASK_RATIO_PIN = 0.6428571343421936  # mask_ref.json mask_ratio (9/14)

# ---- frozen EXP-002 asset pins -------------------------------------------
FROZEN_FEATURES_SHA256 = "5ac9bbd232b80f569eedc0e030eddcfaa60c029b48cc92e36ae0c433ff9fe48a"
CKPT_SHA256 = T38.CKPT_SHA256

_FIG7_BASE_CANDIDATES = [
    os.path.join(REPO_ROOT, "repro_fig7"),                # local layout
    os.path.join("/dev_data/wlt/VisionTS_Experiments", "repro_fig7"),  # eta
]
_T38_LOG_CANDIDATES = [
    os.path.join(ROOT, "repro_t38_t15t22", "logs"),       # eta (t38 ran there)
    os.path.join(REPO_ROOT, "long_term_tsf", "repro_t38_t15t22", "logs"),
]

# group spans in the joint 2,200-row matrix (fixed global order)
GROUPS = ["ImageNet", "ETTm1", "Weather", "Electricity", "Monash"]
SPAN_IMAGENET = (0, 1000)
SPAN_ETTM1 = (1000, 1300)
SPAN_WEATHER = (1300, 1600)
SPAN_ELECTRICITY = (1600, 1900)
SPAN_MONASH = (1900, 2200)
SPANS = {"ImageNet": SPAN_IMAGENET, "ETTm1": SPAN_ETTM1,
         "Weather": SPAN_WEATHER, "Electricity": SPAN_ELECTRICITY,
         "Monash": SPAN_MONASH}
TS_SOURCES = ["ETTm1", "Weather", "Electricity", "Monash"]
N_NEIGHBORS = 30
NN_DENOM_PER_SOURCE = N_TS_PER_SOURCE * N_NEIGHBORS    # 9000

# EXP-002 t-SNE parameters (run_tsne.py:23-27 verbatim; random_state passed
# per run: 2021 main, 2022/2023 stability)
TSNE_PARAMS = dict(
    n_components=2, perplexity=30, learning_rate=200.0, init='pca',
    early_exaggeration=12.0, max_iter=1000, metric='euclidean',
)
SKLEARN_PIN_PREFIX = "1.9."     # EXP-002/036 sklearn; hard gate at preflight

# EXP-038 context rule (run.py convert_context_len; expected_context in
# t38_monash_runner) — L_config per paper row
MONASH_CONTEXT = T38.MONASH_CONTEXT            # 1000
MONASH_NO_P_CONTEXT = T38.MONASH_NO_P_CONTEXT  # 300

QUOTA = 10        # floor(300 / 29)
SURPLUS_ROWS = 10  # remaining 10 samples -> largest-capacity rows
BATCH = 32

# paper Fig.7 caption anchors (audit_supplement/paper/VisionTS_paper_full.md)
PAPER_ANCHOR_LINES = (128, 136)
PAPER_FIG7_TEXT = ("Figure 7. Modality visualization of the images (ImageNet) "
                   "and time serie (Monash, Weather, Electricity, and ETTm1) "
                   "based on the MAE encoder.")
PAPER_FIG7_IMG_REL = "audit_supplement/paper/images/50bcc728de8d071080e1bcc2f6ab7833c7c51c9ac5331575bb08966b11376c61.jpg"
PAPER_FIG7_IMG_SHA256 = "b9dc92e123ff183d70a8dbe2ff1c1a441021f812de4d76663b78279db3aa1646"
PLOT_COLORS = {"ImageNet": "#1f77b4", "Electricity": "#d62728",
               "ETTm1": "#ff7f0e", "Weather": "#2ca02c",
               "Monash": "#9467bd"}


def first_existing(candidates, marker):
    for p in candidates:
        if os.path.exists(os.path.join(p, marker) if marker else p):
            return p
    return candidates[0]


def fig7_base():
    return first_existing(_FIG7_BASE_CANDIDATES, "outputs/mask_ref.json")


def frozen_features_path():
    return os.path.join(fig7_base(), "outputs", "features_1900x54528.npy")


def mask_ref_path():
    return os.path.join(fig7_base(), "outputs", "mask_ref.json")


def sampling_list_path(group):
    return os.path.join(fig7_base(), "outputs", "sampling_lists", "%s.json" % group)


def t38_logs_base():
    for p in _T38_LOG_CANDIDATES:
        if os.path.isdir(p) and os.path.isfile(
                os.path.join(p, "monash_fred_md.json")):
            return p
    return _T38_LOG_CANDIDATES[0]


def monash_json(ds):
    return os.path.join(t38_logs_base(), "monash_%s.json" % ds)


# CIF 2016 pooled paper row -> the two code subsets with a deterministic
# 50/50 split of the row's quota
CIF_SUBSETS = ["cif_2016_6", "cif_2016_12"]


def row_subsets(row):
    """PAPER_T15_ORDER row -> code subset names feeding it."""
    if row == "CIF_2016_POOLED":
        return list(CIF_SUBSETS)
    return [row]


def subset_row(subset):
    if subset in CIF_SUBSETS:
        return "CIF_2016_POOLED"
    return subset


def expected_context(p):
    """L_config for a subset with periodicity p (t38_monash_runner formula)."""
    if p is None:
        return None
    if p == 1:
        return MONASH_NO_P_CONTEXT
    return int(round(float(MONASH_CONTEXT) / p)) * p


def imputation_nan(array):
    """eval_gluonts/run.py:31-45 verbatim (forward fill; all-NaN -> zeros)."""
    not_nan_mask = ~np.isnan(array)
    if not_nan_mask.all():
        return array
    if not not_nan_mask.any():
        return np.zeros_like(array)
    array_imputed = np.copy(array)
    for i in range(len(array)):
        if not not_nan_mask[i]:
            array_imputed[i] = array_imputed[i - 1]
    return array_imputed


def derived_seed(*parts):
    """Deterministic seed (python `hash()` is per-process salted)."""
    blob = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(blob).hexdigest()[:8], 16) & 0x7FFFFFFF


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_bytes(b):
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


def time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def row_quota_plan(capacities):
    """capacities: {row: max non-overlapping window count}.  Returns
    (quota: {row: int}, surplus_log).  29 rows x floor(300/29)=10 = 290;
    remaining 10 -> rows with the largest capacity (ties by
    PAPER_T15_ORDER order)."""
    order = T38.PAPER_T15_ORDER
    assert len(order) == 29
    quota = {r: QUOTA for r in order}
    ranked = sorted(order, key=lambda r: (-capacities[r], order.index(r)))
    for r in ranked[:SURPLUS_ROWS]:
        quota[r] += 1
    surplus_log = [{"row": r, "quota": quota[r], "capacity": capacities[r]}
                   for r in ranked[:SURPLUS_ROWS]]
    assert sum(quota.values()) == N_MONASH
    return quota, surplus_log


def redistribute(quota, capacities):
    """Deficit redistribution chain: rows whose capacity cannot fill their
    quota hand the shortfall to rows with the largest spare capacity
    (ties by PAPER_T15_ORDER order); iterated to a fixed point.
    Returns (final_quota, transfers)."""
    order = T38.PAPER_T15_ORDER
    q = dict(quota)
    transfers = []
    while True:
        donors = [r for r in order if capacities[r] < q[r]]
        if not donors:
            break
        moved_any = False
        for donor in donors:
            need = q[donor] - capacities[donor]
            if need <= 0:
                continue
            recipients = [r for r in order
                          if r != donor and (capacities[r] - q[r]) > 0]
            recipients.sort(key=lambda r: (-(capacities[r] - q[r]),
                                           order.index(r)))
            for rec in recipients:
                if need <= 0:
                    break
                spare = capacities[rec] - q[rec]
                take = min(need, spare)
                if take <= 0:
                    continue
                q[donor] -= take
                q[rec] += take
                need -= take
                transfers.append({"from": donor, "to": rec, "n": take})
                moved_any = True
        if not moved_any:
            break
    unmet = {r: q[r] - capacities[r] for r in order if q[r] > capacities[r]}
    return q, transfers, unmet


def max_windows(length, l_eff):
    """Max pairwise-non-overlapping stride-l_eff windows in [0, length)."""
    if length < l_eff or l_eff <= 0:
        return 0
    return (length - l_eff) // l_eff + 1
