"""EXP-036 helpers: t-SNE of the frozen MAE encoder vs the three EXP-003
LayerNorm-finetuned encoders (ETTm1 / Electricity / Weather, H=96,
seed=2021 best-val checkpoints) over the exact 1,900 fixed samples of
EXP-002 (PPT page-5 right figure, repro_fig7 @ 7419873).

Row order (archived in repro_fig7/outputs): ImageNet(1000) -> ETTm1(300)
-> Weather(300) -> Electricity(300); inputs are rebuilt from the archived
sampling lists (never re-sampled) and are identical across the four
conditions; only the encoder weights change. Encoder-LN splice = the 50
encoder tensors of EXP-003's ft_type='ln' checkpoint (48 block norms +
final vision_model.norm) copied into the frozen backbone; decoder and all
non-LN params stay frozen. t-SNE runs only for the three LN conditions
(EXP-002 params verbatim, same sklearn); the frozen panel reuses the
archived tsne_coords_1900x2.npy. High-dim evidence: 30-NN ImageNet
proportion per (encoder, ts-source), denominator 300*30 = 9000, neighbor
row indices archived.

EXP-003 probe replay runs each dataset under its own training config
(args.json: ETTm1 L=2304/P=96/r=0.4/c=0.4 freq=t; Electricity L=1152/
P=24/r=0.4/c=0.4; Weather L=576/P=144/r=1.0/c=0.7), first test batch vs
saved pred.npy, gate max|diff| <= 1e-4. Frozen anchors pinned from
SUCCESS.txt (unrounded): ETTm1 0.2832735478878021/0.3229347765445709,
Electricity 0.12593606114387512/0.21679048240184784, Weather
0.14431872963905334/0.19301669299602509.
"""
import hashlib
import json
import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.abspath(__file__))            # .../repro_t36_tsne_ln
ROOT = os.path.dirname(PKG)                                  # long_term_tsf/
REPO_ROOT = os.path.dirname(ROOT)                            # repo root
SAVE_BASE = os.path.join(ROOT, "save_t36_tsne")
LOG_DIR = os.path.join(PKG, "logs")
VERIFY_JSON = os.path.join(LOG_DIR, "verify_t36.json")

for _p in (REPO_ROOT, ROOT, os.path.join(REPO_ROOT, "repro_fullshot")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# repro_fig7.common fixes its own sys.path (repo root + long_term_tsf)
# and must only be imported where torch is available.
FIG7_DIR = os.path.join(REPO_ROOT, "repro_fig7")
FIG7_OUT = os.path.join(FIG7_DIR, "outputs")
LIST_DIR = os.path.join(FIG7_OUT, "sampling_lists")

SEED = 2021
FEATURE_DIM = (1 + 70) * 768  # 54528
N_IMAGENET = 1000
N_TS = 300

GROUPS = ["ImageNet", "ETTm1", "Weather", "Electricity"]
GROUP_COUNTS = {"ImageNet": N_IMAGENET, "ETTm1": N_TS, "Weather": N_TS, "Electricity": N_TS}


def group_spans():
    """Row spans in the fixed 1,900 order: {'ImageNet': [0,1000), ...}."""
    spans, pos = {}, 0
    for g in GROUPS:
        spans[g] = [pos, pos + GROUP_COUNTS[g]]
        pos += GROUP_COUNTS[g]
    assert pos == 1900
    return spans


COND_LIST = ["frozen", "ettm1_ln", "ecl_ln", "weather_ln"]
COND_SOURCE = {"ettm1_ln": "ETTm1", "ecl_ln": "Electricity", "weather_ln": "Weather"}
COND_LABEL = {"frozen": "Frozen (original MAE-base)",
              "ettm1_ln": "ETTm1-LN (encoder-LN finetuned on ETTm1)",
              "ecl_ln": "Electricity-LN (encoder-LN finetuned on Electricity)",
              "weather_ln": "Weather-LN (encoder-LN finetuned on Weather)"}

# frozen MAE-base (EXP-001/EXP-002 weights; sha pinned since EXP-033/t35)
CKPT_PATH = os.path.join(REPO_ROOT, "ckpt", "mae_visualize_vit_base.pth")
CKPT_SHA256 = "c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e"
CKPT_BYTES = 447725789

# EXP-003 job dirs: training tree used long_term_tsf as CWD (eta); the
# audited local tree keeps the SUCCESS/args records at repo root.
_EXP003_CANDIDATES = [
    os.path.join(ROOT, "save_fullshot"),          # eta (long_term_tsf/save_fullshot)
    os.path.join(REPO_ROOT, "save_fullshot"),     # local audited tree
]
EXP003_DS = ["ETTm1", "Electricity", "Weather"]


def exp003_job(ds):
    """First candidate job dir whose checkpoint.pth exists."""
    for base in _EXP003_CANDIDATES:
        d = os.path.join(base, "%s_96_seed2021" % ds)
        if os.path.isfile(os.path.join(d, "checkpoints", "_", "checkpoint.pth")):
            return d
    return os.path.join(_EXP003_CANDIDATES[0], "%s_96_seed2021" % ds)


def exp003_pred(ds):
    return os.path.join(exp003_job(ds), "results", "_", "pred.npy")


def exp003_success(ds):
    return os.path.join(exp003_job(ds), "SUCCESS.txt")


# unrounded full-test anchors (EXP-003 SUCCESS.txt), hard cross-check
ANCHORS = {
    "ETTm1": {"mse": 0.2832735478878021, "mae": 0.3229347765445709},
    "Electricity": {"mse": 0.12593606114387512, "mae": 0.21679048240184784},
    "Weather": {"mse": 0.14431872963905334, "mae": 0.19301669299602509},
}

PROBE_TOL = 1e-4      # first-batch replay vs saved pred.npy
ANCHOR_TOL = 1e-6     # full-test replay fallback vs SUCCESS unrounded values

# encoder LayerNorm = 50 tensors (visionts.VisionTS-relative names);
# EXP-003 ft_type='ln' trained 84 (these 50 + 34 decoder norms).
ENC_LN_KEYS = (
    ["vision_model.blocks.%d.norm1.%s" % (i, w) for i in range(12) for w in ("weight", "bias")]
    + ["vision_model.blocks.%d.norm2.%s" % (i, w) for i in range(12) for w in ("weight", "bias")]
    + ["vision_model.norm.weight", "vision_model.norm.bias"]
)
assert len(ENC_LN_KEYS) == 50

# code fingerprint files of EXP-003 training (repro_fullshot/run_train.py
# FINGERPRINT_FILES); md5 re-check proves the deploy tree == training tree
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

# EXP-002 archived artifacts (all tracked in git at 7419873)
ORIG_FEATURES = os.path.join(FIG7_OUT, "features_1900x54528.npy")
ORIG_COORDS = os.path.join(FIG7_OUT, "tsne_coords_1900x2.npy")
ORIG_META = os.path.join(FIG7_OUT, "meta.csv")
ORIG_GROUPS = os.path.join(FIG7_OUT, "groups.npy")
ORIG_MASK_REF = os.path.join(FIG7_OUT, "mask_ref.json")
ORIG_TANE_PARAMS = os.path.join(FIG7_OUT, "tsne_params.json")
ARCHIVE_HEAD = "group,sample_ref,window_idx,channel_idx,abs_row_start,abs_row_end,t_start,t_end"

SKLEARN_PIN = "1.9.1"  # EXP-002 env.json / tsne_params.json; re-checked in preflight

TSNE_PARAMS = {
    "n_components": 2, "perplexity": 30, "learning_rate": 200.0, "init": "pca",
    "early_exaggeration": 12.0, "max_iter": 1000, "random_state": SEED,
    "metric": "euclidean",
}

N_NEIGHBORS = 30
NN_DENOM = N_TS * N_NEIGHBORS  # 9000


def cond_features(cond):
    return os.path.join(SAVE_BASE, "features_%s.npy" % cond)


def cond_features_sha(cond):
    return os.path.join(SAVE_BASE, "features_%s_sha256.txt" % cond)


def cond_inputs_json(cond):
    return os.path.join(SAVE_BASE, "input_hashes_%s.json" % cond)


def cond_coords(cond):
    return os.path.join(SAVE_BASE, "coords_%s_1900x2.npy" % cond)


def load_sampling_list(group):
    with open(os.path.join(LIST_DIR, "%s.json" % group)) as f:
        return json.load(f)


def meta_rows_from_lists():
    """Rebuild the 1,901 meta.csv lines from the archived sampling lists
    (same construction as repro_fig7/extract_features.py L99-105)."""
    rows = [ARCHIVE_HEAD]
    for g in GROUPS:
        entries = load_sampling_list(g)
        entries = entries["filenames"] if g == "ImageNet" else entries["samples"]
        for e in entries:
            if g == "ImageNet":
                rows.append(("%s,%s,,,,," % (g, e)).rstrip(","))
            else:
                rows.append(
                    "%s,%s,%s,%s,%s,%s,%s,%s" % (g, e["variable"], e["window_idx"],
                                                 e["channel_idx"], e["abs_row_start"],
                                                 e["abs_row_end"], e["t_start"], e["t_end"]))
    assert len(rows) == 1901
    return rows


def check_meta_rows():
    got = meta_rows_from_lists()
    with open(ORIG_META) as f:
        want = f.read().splitlines()
    if got != want:
        for i, (a, b) in enumerate(zip(got, want)):
            if a != b:
                raise AssertionError("meta.csv row %d differs: rebuilt %r vs archived %r"
                                     % (i, a, b))
        raise AssertionError("meta.csv line count differs: rebuilt %d vs archived %d"
                             % (len(got), len(want)))


def check_groups_order():
    want = np.array(sum(([g] * GROUP_COUNTS[g] for g in GROUPS), []))
    got = np.load(ORIG_GROUPS, allow_pickle=True)
    assert got.shape == (1900,) and (got == want).all(), "groups.npy order mismatch"


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


def sha256_of_bytes(b):
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


def load_vision_model():
    """Frozen model exactly as EXP-002 built it (repro_fig7.common)."""
    sys.path.insert(0, FIG7_DIR)
    import common as fig7
    return fig7, fig7.load_model()


def load_exp003_visionts_dict(ds):
    """EXP-003 checkpoint -> {'vm.'-stripped} full state dict."""
    import torch
    state = torch.load(os.path.join(exp003_job(ds), "checkpoints", "_", "checkpoint.pth"),
                       map_location="cpu")
    assert isinstance(state, dict) and all(k.startswith("vm.") for k in state), \
        "unexpected checkpoint.pth key layout"
    return {k[len("vm."):]: v for k, v in state.items()}


def splice_encoder_ln(model, src):
    """Copy the 50 encoder-LN tensors from an EXP-003 {'vm.'-stripped} state
    dict into a frozen visionts.VisionTS instance. Returns per-key evidence;
    raises unless every key lands bitwise-equal."""
    import torch
    named = dict(model.named_parameters())
    missing = [k for k in ENC_LN_KEYS if k not in named or k not in src]
    if missing:
        raise AssertionError("encoder-LN keys missing (model or ckpt): %s..." % missing[:5])
    evidence = {}
    with torch.no_grad():
        for k in ENC_LN_KEYS:
            named[k].copy_(src[k])
    for k in ENC_LN_KEYS:
        ok = bool(torch.equal(named[k].detach().cpu(), src[k].detach().cpu()))
        if not ok:
            raise AssertionError("spliced value != source checkpoint: %s" % k)
        evidence[k] = {"equal_to_src": True,
                       "shape": list(named[k].shape)}
    return evidence


def ln_deltas_vs_frozen(src, frozen):
    """Per-key max|src - frozen| for the 50 encoder-LN keys and for every
    other vision_model param (non-LN encoder + decoder), from two
    {'vm.'-stripped} state dicts. frozen = original MAE state (keys without
    the 'vision_model.' prefix are mapped in)."""
    import torch
    mae = frozen["model"] if "model" in frozen else frozen
    out_enc, out_other = {}, {}
    vm_keys = [k for k in src if k.startswith("vision_model.")]
    for k in vm_keys:
        mae_k = k[len("vision_model."):]
        if mae_k not in mae:
            out_other[k] = {"present_in_frozen": False}
            continue
        d = (src[k].float() - mae[mae_k].float()).abs().max().item()
        if k in ENC_LN_KEYS:
            out_enc[k] = d
        else:
            out_other[k] = d
    return out_enc, out_other


def nn_recompute_proportions(neighbors):
    """neighbors: {cond: {group: (300,30) int array}} -> proportions using
    archived row spans (ImageNet rows first 1000). Pure numpy."""
    spans = group_spans()
    im0, im1 = spans["ImageNet"]
    out = {}
    for cond, per in neighbors.items():
        out[cond] = {}
        for g, arr in per.items():
            arr = np.asarray(arr)
            assert arr.shape == (N_TS, N_NEIGHBORS), (cond, g, arr.shape)
            s0, s1 = spans[g]
            rows = np.arange(s0, s1)
            assert not (arr == rows[:, None]).any(), "self in neighbors"
            cnt = int(((arr >= im0) & (arr < im1)).sum())
            out[cond][g] = cnt / float(NN_DENOM)
    return out
