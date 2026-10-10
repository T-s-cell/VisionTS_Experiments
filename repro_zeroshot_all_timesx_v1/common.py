#!/usr/bin/env python3
"""Shared plumbing: path setup, read-only imports of the frozen old modules,
hashing, state markers, fingerprints. The old experiment directory is READ-ONLY.
"""
import hashlib
import json
import os
import sys

import protocol as P

HERE = P.HERE
REPO_ROOT = P.REPO_ROOT
OLD_DIR = P.OLD_DIR

for p in (REPO_ROOT, OLD_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

FROZEN_DIR = os.path.join(HERE, "frozen")
CACHE_DIR = os.path.join(HERE, "cache_all")
RESULTS_DIR = os.path.join(HERE, "results")
LOGS_DIR = os.path.join(HERE, "logs")
STATE_DIR = os.path.join(HERE, ".state")

INV_PATH = os.path.join(FROZEN_DIR, "inventory.json")
ARRAYS_PATH = os.path.join(FROZEN_DIR, "frozen_arrays.npz")
VARS_PATH = os.path.join(FROZEN_DIR, "vars.json")
FREEZE_REPORT = os.path.join(FROZEN_DIR, "freeze_report.json")
REUSE_PLAN = os.path.join(HERE, "reuse_plan.json")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_md5_map(names, base=HERE):
    return {n: md5_file(os.path.join(base, n)) for n in names if os.path.isfile(os.path.join(base, n))}


def new_code_md5():
    return code_md5_map(P.NEW_CODE_ALL, HERE)


def old_code_md5():
    return code_md5_map(P.OLD_CODE_REUSED + P.OLD_CODE_INFO, OLD_DIR)


def dump_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def state_path(stage):
    return os.path.join(STATE_DIR, f"{stage}.done")


def mark_done(stage, payload):
    os.makedirs(STATE_DIR, exist_ok=True)
    payload = dict(payload)
    payload["status"] = "done"
    dump_json(payload, state_path(stage))


def state_of(stage):
    p = state_path(stage)
    if os.path.isfile(p):
        return load_json(p)
    return None


def env_info():
    import platform
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "host": os.uname().nodename}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    except ImportError:
        info["torch"] = None
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    return info


def old_cache_guards():
    """sha256 of every old-cache npz referenced by reuse_plan.json (reuse
    candidates). Used as input guards for stages that consume old cache."""
    plan = load_json(REUSE_PLAN)
    return {"old_cache:" + v["file"]: sha256_file(os.path.join(P.OLD_CACHE, v["file"]))
            for v in plan["variables"].values() if v.get("verdict") == "reuse_candidate"}


def stage_input_guards(stage, zip_path=None):
    """Declarative per-stage INPUT fingerprint: frozen artifacts, upstream
    reports, config-critical files and code md5s. Recorded into the stage
    marker at completion; stage_guard recomputes the same declaration and
    invalidates the stage when anything (incl. protocol.py) changed.
    Keys: relpath under HERE -> sha256; 'code_md5:<name>' -> md5;
    friendly keys (zip_sha256/old_splits_sha256/ln_manifest_sha256) -> sha256.
    """
    g = {}

    def add_rel(rel):
        g[rel] = sha256_file(os.path.join(HERE, rel))

    def add_abs(path, key):
        g[key] = sha256_file(path)

    def add_code(names, base=None):
        for n in names:
            g["code_md5:" + n] = md5_file(os.path.join(base or HERE, n))

    frozen = ["frozen/inventory.json", "frozen/frozen_arrays.npz", "frozen/vars.json"]
    old_fwd = ["timesx_data.py", "record_fingerprint.py"]
    if stage == "freeze":
        assert zip_path, "freeze input guards need the zip path"
        add_abs(zip_path, "zip_sha256")
        add_abs(P.OLD_SPLITS, "old_splits_sha256")
        add_code(["protocol.py", "common.py", "freeze_full.py"])
        add_code(["timesx_data.py"], base=P.OLD_DIR)
    elif stage == "verify":
        for r in frozen:
            add_rel(r)
        add_abs(P.OLD_SPLITS, "old_splits_sha256")
        add_code(["protocol.py", "common.py", "verify_cache.py"])
        add_code(old_fwd + ["metrics_util.py"], base=P.OLD_DIR)
    elif stage == "sample":
        for r in frozen:
            add_rel(r)
        add_code(["protocol.py", "common.py", "preflight.py"])
    elif stage == "preflight":
        for r in frozen:
            add_rel(r)
        add_rel("reuse_plan.json")
        add_rel("preflight_sample.json")
        add_code(["protocol.py", "common.py", "preflight.py", "infer_fill.py"])
        add_code(old_fwd, base=P.OLD_DIR)
    elif stage == "fill":
        for r in frozen:
            add_rel(r)
        add_rel("reuse_plan.json")
        add_rel("reuse_final.json")
        add_rel("cache_all/manifest.json")
        add_code(["protocol.py", "common.py", "preflight.py", "infer_fill.py"])
        add_code(old_fwd, base=P.OLD_DIR)
        g.update(old_cache_guards())
    elif stage == "assemble":
        for r in frozen:
            add_rel(r)
        add_rel("reuse_plan.json")
        add_rel("reuse_final.json")
        add_rel("cache_all/manifest.json")
        add_code(["protocol.py", "common.py", "assemble.py"])
        g.update(old_cache_guards())
    elif stage == "evaluate":
        for r in frozen:
            add_rel(r)
        add_rel("results/preds.jsonl.gz")
        add_rel("reuse_plan.json")
        add_rel("reuse_final.json")
        add_abs(P.LN_MANIFEST, "ln_manifest_sha256")
        add_code(["protocol.py", "common.py", "assemble.py", "evaluate.py"])
        add_code(["metrics_util.py"], base=P.OLD_DIR)
    elif stage == "recompute":
        # standalone script: must stay identical to the dict built inside
        # recompute_check.py (which cannot import this module)
        for r in frozen:
            add_rel(r)
        for r in ("results/preds.jsonl.gz", "results/variable_level.csv",
                  "results/domain_level.csv", "results/overall.json"):
            add_rel(r)
        add_code(["recompute_check.py"])
    elif stage == "accept":
        for r in frozen:
            add_rel(r)
        for r in ("results/preds.jsonl.gz", "results/recompute_check.json",
                  "results/subset_1677.json"):
            add_rel(r)
        add_rel("reuse_plan.json")
        add_rel("reuse_final.json")
        add_abs(P.LN_MANIFEST, "ln_manifest_sha256")
        add_code(["protocol.py", "common.py", "verify_acceptance.py"])
    else:
        raise ValueError(f"unknown stage: {stage}")
    return g


def forward_fingerprint(splits_ref=None):
    """Fingerprint binding new-cache forwards to data + frozen window manifest +
    ckpt + code + config. `splits_ref` optionally pins an extra reference file."""
    import record_fingerprint as RF  # read-only use of the old module's helpers
    ckpt = RF.ckpt_path(None)
    ckpt_sha = sha256_file(ckpt)
    assert ckpt_sha == P.CKPT_SHA256, f"ckpt sha256 mismatch: {ckpt_sha}"
    zip_path = _zip_path()
    assert sha256_file(zip_path) == P.ZIP_SHA256, "zip sha256 mismatch"
    fp = {
        "protocol_version": P.PROTOCOL_VERSION,
        "run_config": dict(P.RUN_CONFIG),
        "zip_sha256": P.ZIP_SHA256,
        "inventory_sha256": sha256_file(INV_PATH),
        "arrays_sha256": sha256_file(ARRAYS_PATH),
        "vars_sha256": sha256_file(VARS_PATH),
        "old_splits_sha256": sha256_file(P.OLD_SPLITS),
        "ckpt_sha256": ckpt_sha,
        "code_md5_forward": code_md5_map(P.NEW_CODE_FORWARD, HERE),
        "code_md5_reused_old": code_md5_map(P.OLD_CODE_REUSED, OLD_DIR),
        "code_md5_new_all": new_code_md5(),
        "env": env_info(),
    }
    if splits_ref:
        fp["splits_ref"] = {"path": splits_ref, "sha256": sha256_file(splits_ref)}
    return fp


_ZIP = None


def _zip_path():
    global _ZIP
    if _ZIP is None:
        from timesx_data import find_zip
        _ZIP = find_zip(None)
    return _ZIP
