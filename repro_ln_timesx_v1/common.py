#!/usr/bin/env python3
"""Shared foundation for the TimesX LN fine-tuning experiment (spec v1).

- All paths are built from THIS file's location; never from the cwd.
- configs/v1.yaml is JSON (a YAML subset) so no yaml dependency is needed.
- repro_zeroshot_ext/timesx_data.py is reused READ-ONLY via importlib under a
  unique module name; native sample_id construction stays identical to the old
  protocol. PYTHONDONTWRITEBYTECODE=1 is enforced so read-only imports never
  create __pycache__ in the old tree.
- torch is imported lazily so prepare_data.py can run without it.
"""
import contextlib
import hashlib
import importlib.util
import json
import os
import random
import sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)                       # VisionTS_Experiments/
ZERO_DIR = os.path.join(REPO_ROOT, "repro_zeroshot_ext")

CONFIG_PATH = os.path.join(HERE, "configs", "v1.yaml")
with open(CONFIG_PATH) as _f:
    CFG = json.load(_f)

MANIFESTS = os.path.join(HERE, "manifests")
CHECKPOINTS = os.path.join(HERE, "checkpoints")
LOGS = os.path.join(HERE, "logs")
PREDICTIONS = os.path.join(HERE, "predictions")
RESULTS = os.path.join(HERE, "results")
AUDIT = os.path.join(HERE, "audit")
DATASET_DIR = os.path.join(HERE, "dataset")
for _d in (MANIFESTS, CHECKPOINTS, LOGS, PREDICTIONS, RESULTS, AUDIT, DATASET_DIR):
    os.makedirs(_d, exist_ok=True)

DATA_CACHE = os.path.join(MANIFESTS, "data_cache.npz")
PROTOCOL_JSON = os.path.join(MANIFESTS, "protocol.json")
SPLIT_MANIFEST = os.path.join(MANIFESTS, "split_manifest.json")
SPLIT_COUNTS = os.path.join(MANIFESTS, "split_counts.csv")
BASELINE_JSON = os.path.join(AUDIT, "baseline.json")
PREFLIGHT_JSON = os.path.join(AUDIT, "preflight_report.json")
SELECTION_JSON = os.path.join(RESULTS, "selection.json")
RUN_MANIFEST = os.path.join(RESULTS, "run_manifest.json")
PAUSE_SENTINEL = os.path.join(HERE, "PAUSE")

SEEDS = list(CFG["budget"]["seeds"])
METHODS_TRAIN = list(CFG["budget"]["methods_train"])
DOMAINS = sorted(CFG["expected_domain_table"], key=lambda r: r["domain"])
DOMAIN_NAMES = [r["domain"] for r in DOMAINS]
DOMAIN_IDX = {d: i for i, d in enumerate(DOMAIN_NAMES)}
CTX = CFG["data"]["ctx"]
PRED = CFG["data"]["pred"]
PERIODICITY = CFG["model"]["periodicity"]
LR = CFG["optim"]["lr"]
CLIP = CFG["optim"]["clip_grad"]
MICRO = CFG["optim"]["microbatch"]
ACCUM = CFG["optim"]["accum"]
EPOCHS = CFG["budget"]["epochs"]
AUG_P = CFG["aug"]["prob"]
MASK_RATE = CFG["aug"]["mask_rate"]
STD_EPS = CFG["data"]["std_eps"]
ANCHOR_COEF = CFG["anchor"]["coef"]
TIE_TOL = CFG["selection"]["tie_tol"]

_sql_spec = importlib.util.spec_from_file_location(
    "timesx_data_ln_reuse", os.path.join(ZERO_DIR, "timesx_data.py"))
timesx_data = importlib.util.module_from_spec(_sql_spec)
sys.modules["timesx_data_ln_reuse"] = timesx_data
_sql_spec.loader.exec_module(timesx_data)


def zip_path():
    cands = [os.environ.get("TIMESX_ZIP"),
             os.path.join(DATASET_DIR, "TimesX_Datasets.zip")]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "TimesX_Datasets.zip not found; put it in repro_ln_timesx_v1/dataset/ "
        "or set TIMESX_ZIP")


def sha256_file(path, buf_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(path, buf_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_fingerprint():
    """md5 of every .py/.sh/.yaml in this experiment dir (isolation + resume)."""
    out = {}
    for root, _dirs, files in os.walk(HERE):
        rel_root = os.path.relpath(root, HERE)
        if rel_root.split(os.sep)[0] in (
                "checkpoints", "logs", "predictions", "results", "audit",
                "manifests", "dataset", "__pycache__"):
            continue
        for fn in sorted(files):
            if fn.endswith((".py", ".sh", ".yaml")):
                rel = os.path.normpath(os.path.join(rel_root, fn))
                out[rel] = md5_file(os.path.join(root, fn))
    return out


def actual_fingerprints():
    """RECOMPUTED hashes of everything the protocol pins (never trust the
    declared values alone). ckpt is None only on hosts without the file."""
    ckpt = os.path.join(REPO_ROOT, CFG["model"]["ckpt_rel"])
    return {"data_cache": sha256_file(DATA_CACHE),
            "split_manifest": sha256_file(SPLIT_MANIFEST),
            "zip": sha256_file(zip_path()),
            "ckpt": sha256_file(ckpt) if os.path.isfile(ckpt) else None,
            "code": code_fingerprint()}


def expected_ln_names():
    """The exact 84 LayerNorm tensors (mirrors repro_fullshot/ln_check.py)."""
    names = set()
    for i in range(12):
        names.add(f"vision_model.blocks.{i}.norm1.weight")
        names.add(f"vision_model.blocks.{i}.norm1.bias")
        names.add(f"vision_model.blocks.{i}.norm2.weight")
        names.add(f"vision_model.blocks.{i}.norm2.bias")
    names.add("vision_model.norm.weight")
    names.add("vision_model.norm.bias")
    for i in range(8):
        names.add(f"vision_model.decoder_blocks.{i}.norm1.weight")
        names.add(f"vision_model.decoder_blocks.{i}.norm1.bias")
        names.add(f"vision_model.decoder_blocks.{i}.norm2.weight")
        names.add(f"vision_model.decoder_blocks.{i}.norm2.bias")
    names.add("vision_model.decoder_norm.weight")
    names.add("vision_model.decoder_norm.bias")
    return names


def load_model(device, finetune_type=None):
    """Fresh VisionTS with frozen numerics (TF32 both off) and P=1 geometry."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if "visionts" not in sys.modules and str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from visionts import VisionTS
    ft = finetune_type or CFG["model"]["finetune_type"]
    ckpt_dir = os.path.join(REPO_ROOT, "ckpt") + os.sep
    model = VisionTS(arch=CFG["model"]["arch"], finetune_type=ft,
                     ckpt_dir=ckpt_dir, load_ckpt=True)
    model.update_config(context_len=CTX, pred_len=PRED,
                        periodicity=PERIODICITY,
                        norm_const=CFG["model"]["norm_const"],
                        align_const=CFG["model"]["align_const"],
                        interpolation=CFG["model"]["interpolation"])
    return model.to(device)


def seed_all(seed):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj, path):
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def environment_snapshot():
    import numpy as np
    import torch
    import sys as _s
    return {"python": _s.version.split()[0], "numpy": np.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}


def run_id(method, domain, seed):
    return f"{method}__{domain}__s{seed}"


def u_d_s_d(dense_count):
    """U_d = ceil(D_d/32) updates per epoch; S_d = 32*U_d presentations."""
    u = -(-dense_count // 32)
    return u, 32 * u


@contextlib.contextmanager
def train_mode(model, training):
    was = model.training
    model.train(training)
    yield
    model.train(was)
