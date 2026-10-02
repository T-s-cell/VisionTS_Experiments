#!/usr/bin/env python3
"""Shared config for the LTFS-style flatten visualization (featvis_ltfs_style).

featvis/common.py mechanics are reused under the UNIQUE module name
"featvis_common", loaded from its absolute path via importlib so it can never
collide with this directory's own common.py. All paths are built from explicit
locations, never from the current working directory.
"""
import importlib.util
import csv
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ZERO_DIR = os.path.dirname(HERE)              # repro_zeroshot_ext/
REPO_ROOT = os.path.dirname(ZERO_DIR)         # VisionTS_Experiments/
FEATVIS_DIR = os.path.join(ZERO_DIR, "featvis")
FIG7_LIST_JSON = os.path.join(REPO_ROOT, "repro_fig7", "outputs",
                              "sampling_lists", "ImageNet.json")
IMAGENET_VAL_DIR = os.environ.get(
    "IMAGENET_VAL_DIR", "/dev_data/wlt/data/ImageNet/val")

_spec = importlib.util.spec_from_file_location(
    "featvis_common", os.path.join(FEATVIS_DIR, "common.py"))
fv = importlib.util.module_from_spec(_spec)
sys.modules["featvis_common"] = fv
_spec.loader.exec_module(fv)

for _p in (ZERO_DIR, FEATVIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 2021
CTX, PRED, PERIOD = fv.CTX, fv.PRED, fv.PERIOD
NORM_CONST = ALIGN_CONST = fv.NORM_CONST
EMBED_DIM = 768
N_TOKENS = 57                      # CLS + 56 visible patches
FLAT_DIM = N_TOKENS * EMBED_DIM    # 43776 (LTFS-style flatten, CLS kept)
N_VISIBLE = fv.N_VISIBLE           # 56
BATCH = 64
PRED_RTOL = 1e-4                   # declared; relative to per-row pred scale
IMAGENET_N = 1000

DIRS = {name: os.path.join(HERE, name)
        for name in ("lists", "features", "tsne", "figures", "logs")}


def ensure_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)


# Re-exports from the featvis mechanics (single source of truth).
load_model = fv.load_model            # TF32 (matmul+cudnn) disabled inside
noise_mask = fv.noise_mask
attach_capture = fv.attach_capture
build_ts_image = fv.build_ts_image
expected_kept_set = fv.expected_kept_set
ids_keep_from_restore = fv.ids_keep_from_restore
pooled = fv.pooled
l2n = fv.l2n
imagenet_transform = fv.imagenet_transform
pred_compare = fv.pred_compare
parse_args = fv.parse_args
device_auto = fv.device_auto
sha256 = fv.sha256


def timesx_windows():
    out = []
    with open(os.path.join(DIRS["lists"], "timesx_windows.csv")) as f:
        for r in csv.DictReader(f):
            out.append({"var_key": r["var_key"], "sel_group": r["sel_group"],
                        "frequency": r["frequency"], "sample_id": r["sample_id"]})
    assert len(out) == 117
    return out


def load_fig7_imagenet_json():
    """The frozen LTFS 1000-image list, order preserved (read-only)."""
    with open(FIG7_LIST_JSON) as f:
        d = json.load(f)
    files = [str(s) for s in d["filenames"]]
    assert len(files) == IMAGENET_N and len(set(files)) == IMAGENET_N, \
        f"fig7 ImageNet list must hold {IMAGENET_N} unique files"
    return files, fv.sha256(FIG7_LIST_JSON), str(d.get("index_rule", ""))


def flat(feat_norm):
    """[B,57,768] -> [B,43776]; CLS KEPT, token order preserved."""
    return feat_norm.detach().float().reshape(feat_norm.shape[0], -1)


def keep_order_from_noise(model, batch, device):
    """Exact ids_keep ORDER the production masking yields for this batch size
    (mirrors random_masking: argsort(noise)[:len_keep]), without calling it."""
    noise = fv.noise_mask(model, batch, device)
    return torch.argsort(noise, dim=1)[:, :N_VISIBLE]


def capture_ids_restore(model):
    """Runtime instrumentation: wrap vision_model.forward_encoder so every call
    (production forward included) exposes its ids_restore. No source change."""
    orig = model.vision_model.forward_encoder
    box = {}

    def wrapped(img, mask_ratio, noise=None, **kw):
        out, mbool, ids_restore = orig(img, mask_ratio, noise=noise, **kw)
        box["ids_restore"] = ids_restore
        return out, mbool, ids_restore

    model.vision_model.forward_encoder = wrapped
    return box
