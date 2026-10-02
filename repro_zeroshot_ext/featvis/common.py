#!/usr/bin/env python3
"""Shared helpers for featvis (TimesX encoder-feature visualization).

Feature protocol (frozen): fixed P=1 zero-shot path (ctx=96, pred=12, r=c=0.4,
mae_base). Features = MAE encoder final-LayerNorm output, CLS excluded, mean
over the 56 visible patch tokens -> 768-d. Extraction uses runtime hooks on the
PRODUCTION forward (no reimplementation of the TS->image path for extraction);
an independent image build exists only for cross-checking in 02_sanity_check.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ZERO_DIR = os.path.dirname(HERE)            # repro_zeroshot_ext/
REPO_ROOT = os.path.dirname(ZERO_DIR)       # VisionTS_Experiments/
for p in (REPO_ROOT, ZERO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
import einops

SEED = 2021
CTX, PRED, PERIOD = 96, 12, 1
NORM_CONST = ALIGN_CONST = 0.4
ARCH, FT = "mae_base", "none"
FEAT_DIM = 768
N_VISIBLE = 56                              # P=1: leftmost 4 of 14 columns, 14 rows
BATCH = 64

# cache-pred comparison tolerance (declared up front; never loosened to pass)
PRED_RTOL = 1e-4                            # relative to per-row pred scale

IMAGENET_VAL_DIR = os.environ.get(
    "IMAGENET_VAL_DIR", "/dev_data/wlt/data/ImageNet/val")
IMAGENET_N = 2000

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DIRS = {name: os.path.join(HERE, name)
        for name in ("selection", "lists", "features", "tsne", "figures",
                     "highdim", "logs")}


def ensure_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)


def device_auto():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def load_model(device):
    # Match infer.py numerics: TF32 disabled (cudnn TF32 would otherwise hit
    # the patch-embed conv on L20 and drift ~1e-3 vs the frozen cache).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from visionts.model import VisionTS
    from record_fingerprint import ckpt_path
    ckpt = ckpt_path(None)
    model = VisionTS(arch=ARCH, finetune_type=FT,
                     ckpt_dir=os.path.dirname(ckpt), load_ckpt=True)
    model.eval().to(device)
    model.update_config(context_len=CTX, pred_len=PRED, periodicity=PERIOD,
                        norm_const=NORM_CONST, align_const=ALIGN_CONST,
                        interpolation="bilinear")
    return model, ckpt


def noise_mask(model, batch, device):
    """Production masking noise: registered buffer [1,196] expanded to [B,196]."""
    return einops.repeat(model.mask, "1 l -> n l", n=batch).to(device)


def noise_arange(batch, device):
    """Deterministic natural-order noise for the no-mask control (196 tokens)."""
    return torch.arange(196, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch, 1)


def attach_capture(model):
    """Capture (a) the image actually fed to the encoder and (b) the final
    encoder-LayerNorm output, during the production forward."""
    cap = {"image": None, "feat_norm": None}
    h1 = model.vision_model.register_forward_pre_hook(
        lambda m, inp: cap.__setitem__("image", inp[0]))
    h2 = model.vision_model.norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("feat_norm", o))
    return cap, (h1, h2)


def pooled(feat_norm):
    """[B,1+56,768] -> [B,768]: drop CLS, mean over visible patch tokens."""
    return feat_norm[:, 1:, :].float().mean(1)


def build_ts_image(model, x):
    """Independent replication of VisionTS.forward L107-128 (cross-check only)."""
    means = x.mean(1, keepdim=True)
    x_enc = x - means
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    stdev = stdev / model.norm_const
    x_enc = x_enc / stdev
    x_enc = einops.rearrange(x_enc, "b s n -> b n s")
    x_pad = F.pad(x_enc, (model.pad_left, 0), mode="replicate")
    x_2d = einops.rearrange(x_pad, "b n (p f) -> (b n) 1 f p", f=model.periodicity)
    x_resize = model.input_resize(x_2d)
    masked = torch.zeros((x_2d.shape[0], 1, model.image_size,
                          model.num_patch_output * model.patch_size),
                         device=x_2d.device, dtype=x_2d.dtype)
    img = torch.cat([x_resize, masked], dim=-1)
    img = einops.repeat(img, "b 1 h w -> b c h w", c=3)
    return img


def expected_kept_set():
    """Leftmost 4 columns of each of the 14 rows (row-major grid)."""
    return {r * 14 + c for r in range(14) for c in range(4)}


def ids_keep_from_restore(ids_restore, len_keep):
    """MAE: ids_restore = argsort(ids_shuffle); ids_shuffle = argsort(ids_restore)."""
    return torch.argsort(ids_restore, dim=1)[:, :len_keep]


def l2n(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def imagenet_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_lists():
    with open(os.path.join(DIRS["lists"], "timesx_windows.csv")) as f:
        import csv
        sel = [r for r in csv.DictReader(f)]
    with open(os.path.join(DIRS["lists"], "all_windows.csv")) as f:
        import csv
        allv = [r for r in csv.DictReader(f)]
    with open(os.path.join(DIRS["lists"], "imagenet_manifest.json")) as f:
        man = json.load(f)
    return sel, allv, man


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pred_compare(pred_new, pred_ref):
    """Return (max_abs, rel) with rel = max_abs / max(|pred_ref_row|)."""
    pred_new = np.asarray(pred_new, dtype=np.float64)
    pred_ref = np.asarray(pred_ref, dtype=np.float64)
    d = np.abs(pred_new - pred_ref)
    scale = np.maximum(np.max(np.abs(pred_ref)), 1e-12)
    return float(d.max()), float(d.max() / scale)


def parse_args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--device", default=None)
    return ap.parse_args()
