#!/usr/bin/env python3
"""Fingerprint helpers: bind results to data, splits, code, ckpt, config, env.

CLI: python record_fingerprint.py --splits splits.json --zip <zip> --ckpt-dir ../ckpt
Prints the fingerprint dict as JSON.
"""
import argparse
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CODE_FILES = [
    "timesx_data.py", "build_splits.py", "data_audit.py", "metrics_util.py",
    "infer.py", "select_period.py", "evaluate_test.py", "aggregate.py",
    "record_fingerprint.py", "verify_acceptance.py", "run_all.sh", "audit_ext.sh",
]

# files whose change invalidates cached forward predictions
FORWARD_CODE_FILES = ["timesx_data.py", "infer.py", "record_fingerprint.py"]
# files whose change invalidates P_selection.json (selection + scoring rules)
SELECTION_CODE_FILES = ["select_period.py", "metrics_util.py"]

RUN_CONFIG = {
    "arch": "mae_base",
    "finetune_type": "none",
    "context_len": 96,
    "pred_len": 12,
    "periodicity_candidates": {"daily": [1, 5], "weekly": [1, 52]},
    "norm_const_r": 0.4,
    "align_const_c": 0.4,
    "interpolation": "bilinear",
    "precision": "fp32",
    "tf32": False,
    "image_size": 224,
    "patch_size": 16,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_md5s():
    out = {}
    for name in CODE_FILES:
        p = os.path.join(HERE, name)
        if os.path.isfile(p):
            out[name] = md5(p)
    return out


def ckpt_path(ckpt_dir):
    name = "mae_visualize_vit_base.pth"
    cands = [os.path.join(ckpt_dir or "", name),
             os.path.join(os.path.dirname(HERE), "ckpt", name)]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(f"{name} not found in {ckpt_dir} or repo ckpt/")


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    return info


def make_fingerprint(zip_path, splits_path, ckpt_dir):
    all_md5 = code_md5s()
    fp = {
        "run_config": dict(RUN_CONFIG),
        "zip_sha256": sha256(zip_path),
        "splits_sha256": sha256(splits_path) if splits_path and os.path.isfile(splits_path) else None,
        "ckpt_sha256": sha256(ckpt_path(ckpt_dir)),
        "code_md5_forward": {k: all_md5[k] for k in FORWARD_CODE_FILES},
        "code_md5_selection": {k: all_md5[k] for k in SELECTION_CODE_FILES},
        "code_md5": all_md5,  # informational: changes here alone invalidate nothing
        "env": env_info(),
    }
    return fp


def binding_diffs(fp, zip_path, splits_path, ckpt_dir, require_selection=False):
    """Return list of binding mismatches vs current state.

    Binding sets:
      - cache/forward documents: run_config, zip, splits, ckpt, forward code
      - selection documents (P_selection.json): additionally selection code
        (select_period.py + metrics_util.py) when require_selection=True;
        a missing code_md5_selection means a legacy file -> must regenerate.
    """
    cur = make_fingerprint(zip_path, splits_path, ckpt_dir)
    diffs = []
    for key in ("run_config", "zip_sha256", "splits_sha256", "ckpt_sha256"):
        if fp.get(key) != cur.get(key):
            diffs.append(key)
    old_code = fp.get("code_md5_forward") or fp.get("code_md5") or {}
    for k, v in cur["code_md5_forward"].items():
        if old_code.get(k) != v:
            diffs.append(f"code:{k}")
    if require_selection:
        if "code_md5_selection" not in fp:
            diffs.append("code_md5_selection:missing(legacy P_selection — rerun select_period.py)")
        else:
            for k, v in cur["code_md5_selection"].items():
                if fp["code_md5_selection"].get(k) != v:
                    diffs.append(f"selection_code:{k}")
    return diffs, cur


def verify_fingerprint(fp, zip_path, splits_path, ckpt_dir, what="", require_selection=False):
    diffs, cur = binding_diffs(fp, zip_path, splits_path, ckpt_dir, require_selection)
    if diffs:
        raise SystemExit(f"fingerprint mismatch ({what}): changed {diffs}; aborting")
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--splits", default=os.path.join(HERE, "splits.json"))
    ap.add_argument("--ckpt-dir", default=None)
    args = ap.parse_args()
    from timesx_data import find_zip
    fp = make_fingerprint(find_zip(args.zip), args.splits, args.ckpt_dir)
    print(json.dumps(fp, indent=1))


if __name__ == "__main__":
    main()
