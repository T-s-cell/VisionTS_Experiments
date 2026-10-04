#!/usr/bin/env python3
"""Shared foundation for EXP-009 (per-domain LN before/after t-SNE).

This module MUST be imported under the unique name "exp009_common" (via
importlib), never as "common": repro_ln_timesx_v1/trainer.py does
`from common import ...`, so sys.modules["common"] is reserved for the LN
experiment's own common.py, which we register before importing the trainer.
All reuse is read-only; PYTHONDONTWRITEBYTECODE=1 keeps old trees clean.

torch-dependent imports (featvis_ltfs_style -> featvis -> torch) are lazy via
ltfs_stack(), so pure-python parts work on hosts without torch.
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
LN_DIR = os.path.join(REPO_ROOT, "repro_ln_timesx_v1")
ZERO_DIR = os.path.join(REPO_ROOT, "repro_zeroshot_ext")
FIG7_LIST_JSON = os.path.join(REPO_ROOT, "repro_fig7", "outputs",
                              "sampling_lists", "ImageNet.json")

CONFIG_PATH = os.path.join(HERE, "configs", "exp009.yaml")
with open(CONFIG_PATH) as _f:
    CFG = json.load(_f)

FEATURES = os.path.join(HERE, "features")
COORDS = os.path.join(HERE, "coords")
FIGURES = os.path.join(HERE, "figures")
MANIFESTS = os.path.join(HERE, "manifests")
AUDIT = os.path.join(HERE, "audit")
LOGS = os.path.join(HERE, "logs")
PAUSE_SENTINEL = os.path.join(HERE, "PAUSE")
for _d in (FEATURES, COORDS, FIGURES, MANIFESTS, AUDIT, LOGS):
    os.makedirs(_d, exist_ok=True)

CACHE_INDEX = os.path.join(FEATURES, "_cache_index.json")
BASELINE_JSON = os.path.join(AUDIT, "baseline.json")
CKPT_TABLE = os.path.join(MANIFESTS, "ckpt_table.json")
SELECTION_JSON = os.path.join(LN_DIR, "results", "selection.json")
SPLIT_MANIFEST = os.path.join(LN_DIR, "manifests", "split_manifest.json")
DATA_CACHE = os.path.join(LN_DIR, "manifests", "data_cache.npz")
LN_PREDICTIONS = os.path.join(LN_DIR, "predictions")
DOMAIN_SUMMARY_CSV = os.path.join(LN_DIR, "results", "domain_summary.csv")

IMAGENET_VAL_DIR = os.environ.get(
    "IMAGENET_VAL_DIR", CFG["imagenet"]["val_dir_default"])
IMAGENET_N = int(CFG["imagenet"]["n"])
BATCH = int(CFG["features"]["batch"])
FLAT_DIM = 57 * 768            # [B,57,768] flattened, CLS kept
N_VISIBLE = 56                 # P=1: leftmost 4 columns of the 14x14 grid
STATE_BEFORE = CFG["states"]["before"]["tag"]          # "Z0"
STATE_AFTER = CFG["states"]["after"]["tag"]            # "F0_s2021"
F0_SEED = int(CFG["states"]["after"]["seed"])

# ---- register repro_ln_timesx_v1/common.py as sys.modules["common"] BEFORE
# ---- the trainer import; trainer resolves `from common import ...` to it.
_lnc_spec = importlib.util.spec_from_file_location(
    "common", os.path.join(LN_DIR, "common.py"))
lnc = importlib.util.module_from_spec(_lnc_spec)
sys.modules["common"] = lnc
_lnc_spec.loader.exec_module(lnc)

_tr_spec = importlib.util.spec_from_file_location(
    "ln_trainer", os.path.join(LN_DIR, "trainer.py"))
ln_trainer = importlib.util.module_from_spec(_tr_spec)
sys.modules["ln_trainer"] = ln_trainer
_tr_spec.loader.exec_module(ln_trainer)

DOMAIN_NAMES = list(lnc.DOMAIN_NAMES)
CTX, PRED = int(lnc.CTX), int(lnc.PRED)
sha256_file = lnc.sha256_file
md5_file = lnc.md5_file

_ltfs_stack = None


def ltfs_stack():
    """Lazily load featvis_ltfs_style/common.py (imports torch at module
    level; pulls featvis/common.py mechanics under 'featvis_common')."""
    global _ltfs_stack
    if _ltfs_stack is None:
        spec = importlib.util.spec_from_file_location(
            "featvis_ltfs_common",
            os.path.join(ZERO_DIR, "featvis_ltfs_style", "common.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules["featvis_ltfs_common"] = m
        spec.loader.exec_module(m)
        _ltfs_stack = m
    return _ltfs_stack


# ---------------------------------------------------------------- fingerprints
def code_fingerprint():
    """md5 of every .py/.sh/.yaml under THIS experiment dir (outputs excluded)."""
    out = {}
    for root, _dirs, files in os.walk(HERE):
        rel_root = os.path.relpath(root, HERE)
        if rel_root.split(os.sep)[0] in (
                "features", "coords", "figures", "manifests", "audit",
                "logs", "__pycache__"):
            continue
        for fn in sorted(files):
            if fn.endswith((".py", ".sh", ".yaml")):
                rel = os.path.normpath(os.path.join(rel_root, fn))
                out[rel] = md5_file(os.path.join(root, fn))
    return out


REUSED_SOURCES = [
    ("ln_common", os.path.join(LN_DIR, "common.py")),
    ("ln_trainer", os.path.join(LN_DIR, "trainer.py")),
    ("ln_augmentation", os.path.join(LN_DIR, "augmentation.py")),
    ("timesx_data", os.path.join(ZERO_DIR, "timesx_data.py")),
    ("featvis_ltfs_common", os.path.join(ZERO_DIR, "featvis_ltfs_style",
                                         "common.py")),
    ("featvis_common", os.path.join(ZERO_DIR, "featvis", "common.py")),
]


def reused_sources_sha256():
    out = {}
    for name, p in REUSED_SOURCES:
        out[name] = sha256_file(p)
    vt = os.path.join(REPO_ROOT, "visionts")
    for fn in sorted(os.listdir(vt)):
        if fn.endswith(".py"):
            out[f"visionts/{fn}"] = sha256_file(os.path.join(vt, fn))
    return out


def input_hashes():
    """Hashes of every frozen input this experiment depends on."""
    return {
        "mae_base_ckpt": sha256_file(os.path.join(REPO_ROOT, "ckpt",
                                                  "mae_visualize_vit_base.pth")),
        "data_cache": sha256_file(DATA_CACHE),
        "split_manifest": sha256_file(SPLIT_MANIFEST),
        "ln_selection": sha256_file(SELECTION_JSON),
        "imagenet_list": sha256_file(FIG7_LIST_JSON),
        "exp009_config": md5_file(CONFIG_PATH),
        "exp009_code": code_fingerprint(),
        "reused_sources": reused_sources_sha256(),
    }


def feature_cache_key(kind, model_weight_sha):
    """Deterministic cache key for one feature artifact."""
    import json as _j
    comps = {"kind": kind, "model_weight_sha256": model_weight_sha,
             "dtype": CFG["features"]["dtype"], "batch": BATCH,
             "flat_dim": FLAT_DIM, **input_hashes()}
    blob = _j.dumps(comps, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), comps


# --------------------------------------------------------------- frozen inputs
def imagenet_list():
    """Frozen LTFS 1,000-image list, order preserved (pure json; no torch)."""
    with open(FIG7_LIST_JSON) as f:
        d = json.load(f)
    files = [str(s) for s in d["filenames"]]
    assert len(files) == IMAGENET_N and len(set(files)) == IMAGENET_N, \
        f"fig7 ImageNet list must hold {IMAGENET_N} unique files"
    return files


def f0_selection():
    """{domain: {run_id, best_epoch, selected_ckpt_sha256, fallback_to_zeroshot}}
    for the 19 F0/s2021 runs of EXP-004. We bind the MAIN-TABLE best_ft
    checkpoint (epochs 1-10 rule; min best_epoch over all runs is 1). Some
    domains also have a SEPARATE fallback variant (epoch 0) flagged
    fallback_to_zeroshot in EXP-004 — that variant is never used here; the
    flag is carried only as manifest metadata."""
    with open(SELECTION_JSON) as f:
        sel = json.load(f)
    out = {}
    for d in DOMAIN_NAMES:
        rid = f"F0__{d}__s{F0_SEED}"
        r = sel["per_run"][rid]
        assert r["method"] == "F0" and r["seed"] == F0_SEED
        assert int(r["best_epoch"]) >= 1, \
            f"{rid}: best_ft must be a trained epoch (>=1), got {r['best_epoch']}"
        out[d] = {"run_id": rid, "best_epoch": int(r["best_epoch"]),
                  "selected_ckpt_sha256": r["selected_ckpt_sha256"],
                  "exp004_fallback_to_zeroshot": bool(
                      r["fallback_to_zeroshot"])}
    assert len(out) == len(DOMAIN_NAMES) == 19
    return out


def f0_ckpt_path(domain):
    rid = f0_selection()[domain]["run_id"]
    ep = f0_selection()[domain]["best_epoch"]
    return os.path.join(lnc.CHECKPOINTS, rid, f"ep{ep:02d}.ln.pt")


def test_rows_all_domains():
    """[(domain, var_key, sample_id, x)] for ALL native test windows, canonical
    order (sorted domains; split_rows order within domain = sorted vars, then
    stored native ids). x is the past-96 window; y never enters EXP-009."""
    data = ln_trainer.ExperimentData()
    rows = []
    for d in DOMAIN_NAMES:
        for vk, sid, x, _y in data.split_rows(d, "test"):
            rows.append((d, vk, sid, x))
    n_expected = int(CFG["data"]["expected"]["n_test_windows"])
    assert len(rows) == n_expected, \
        f"expected {n_expected} test windows, got {len(rows)}"
    vars_seen = {vk for _d, vk, _s, _x in rows}
    assert len(vars_seen) == int(CFG["data"]["expected"]["n_vars"])
    return rows


def metrics_from_npz(path, domain=None):
    """Recompute std-MSE/MAE the EXP-004 way from a predictions npz:
    window metric with stored d -> per-var mean -> var-equal within domain.
    domain=None uses all rows (F0 per-domain npz); a domain name subsets the
    rows first (Z0__test.npz holds all 2,474 windows in one file).
    Returns dict {mse, mae, n_windows, n_vars, per_var: {vk: (mse, mae, n)}}."""
    z = np.load(path, allow_pickle=False)
    pred, target, d = z["pred"], z["target"], z["d"]
    vks = [str(v) for v in z["var_keys"]]
    if domain is not None:
        doms = [str(v) for v in z["domains"]]
        keep = [i for i, dd in enumerate(doms) if dd == domain]
        assert keep, f"no rows for domain {domain} in {path}"
        pred, target, d = pred[keep], target[keep], d[keep]
        vks = [vks[i] for i in keep]
    sq = ((pred - target) / d[:, None]) ** 2      # [N,12]
    ab = np.abs(pred - target) / d[:, None]
    w_mse, w_mae = sq.mean(1), ab.mean(1)         # window level
    per_var = {}
    for i, vk in enumerate(vks):
        per_var.setdefault(vk, []).append(i)
    vv = {vk: (float(np.mean(w_mse[ix])), float(np.mean(w_mae[ix])), len(ix))
          for vk, ix in per_var.items()}
    mse = float(np.mean([v[0] for v in vv.values()]))
    mae = float(np.mean([v[1] for v in vv.values()]))
    return {"mse": mse, "mae": mae, "n_windows": len(vks), "n_vars": len(vv),
            "per_var": vv}


Z0_PRED_NPZ = os.path.join(LN_PREDICTIONS, "Z0__test.npz")


def atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gpu_free_mib():
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return min(int(v) for v in out if v.strip())


def pause_if_requested(stage):
    if os.path.isfile(PAUSE_SENTINEL):
        print(f"[{stage}] PAUSE sentinel present; remove {PAUSE_SENTINEL} "
              f"to continue", flush=True)
        sys.exit(3)


def bind_and_load_f0(model, domain):
    """Two-level checkpoint binding for one F0 s2021 domain (mirrors EXP-004
    evaluate_test.py, then goes further):
      L1: whole .ln.pt FILE sha256 == selection.json selected_ckpt_sha256;
      L2: load ln_state, then verify all 84 LN tensors item-by-item
          (name in expected list, shape, bitwise value vs the file).
    Returns an audit record. Any mismatch raises."""
    import torch
    sel = f0_selection()[domain]
    ck = f0_ckpt_path(domain)
    file_sha = sha256_file(ck)
    if file_sha != sel["selected_ckpt_sha256"]:
        raise RuntimeError(
            f"{domain}: ln.pt file sha {file_sha} != selection.json "
            f"{sel['selected_ckpt_sha256']}")
    sd = torch.load(ck, map_location="cpu", weights_only=False)["ln_state"]
    expected = lnc.expected_ln_names()
    assert set(sd) == expected, \
        f"{domain}: ckpt keys != expected 84-name list"
    missing = model.load_state_dict(sd, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    msd = model.state_dict()
    for name, t in sd.items():
        ref = msd[name]
        assert ref.shape == t.shape, f"{domain}: {name} shape {ref.shape} != {t.shape}"
        assert torch.equal(ref.detach().cpu(), t), \
            f"{domain}: {name} value mismatch after load"
    return {"domain": domain, "run_id": sel["run_id"],
            "best_epoch": sel["best_epoch"],
            "ln_pt_sha256": file_sha, "n_tensors": len(sd),
            "n_scalars": int(sum(t.numel() for t in sd.values())),
            "names_shapes_values_verified": True}
