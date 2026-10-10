#!/usr/bin/env python3
"""Frozen protocol for the VisionTS x CGTSF single-modality zero-shot
experiment (cgtsf_zeroshot_v1).

Freeze flow: `python protocol.py freeze` writes protocol.json (constants +
code fingerprints + data/weights pins). preflight.py / run_infer.py take
--i-approve-frozen-protocol <first 12 hex of sha256(protocol.json)> and
re-verify every pinned fingerprint at start; any mismatch is a hard stop.
Data identity = (dataset, Idx, Date); CSV row order is the fixed window
order everywhere. Policy: protocol/source/weight/data fingerprint mismatch
= stop; a shard that is merely corrupt/unfinished is quarantined and
recomputed (run_infer.py)."""
import hashlib
import json
import os
import subprocess
import sys

PROTOCOL_NAME = "cgtsf_zeroshot_v1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

SEED = 2021
ARCH = "mae_base"
FINETUNE_TYPE = "none"          # fully frozen
NORM_CONST = 0.4
ALIGN_CONST = 0.4
INTERPOLATION = "bilinear"
FP64 = False
TF32 = False
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

HF_REVISION = "f85b3fa3a8b3f1a2adbc6edf49f82b7d9d1758c9"
WEIGHTS_FILE = "mae_visualize_vit_base.pth"
WEIGHTS_SHA256 = ("c551c37b63a221a3e6d7d7f49e96f34078c"
                  "9b183f788bedc0387493521bade0e")
WEIGHTS_BYTES = 447725789

CSV_SHA256 = {
    "PTF": "2f8a90f977b48e38f7ce0dae3016de21a2b20bb8657fc2ac803607d26fdf9d3f",
    "LEU": "9f181bb8527c1518bb86f016acb224a167e71cce1368135f78767be069ad7b1c",
    "MSPG": "7c484a2005351db7d22a1d8477cd51c7b98010defbdfd89b73cdae951489315a",
}
CSV_BYTES = {"PTF": 18549020, "LEU": 27518933, "MSPG": 48389951}

SUBSETS = {
    "PTF":  {"context": 120, "pred": 24, "periodicity": 24, "n_seq": 32},
    "LEU":  {"context": 240, "pred": 48, "periodicity": 48, "n_seq": 16},
    "MSPG": {"context": 480, "pred": 96, "periodicity": 96, "n_seq": 27},
}
ORDER = ("PTF", "LEU", "MSPG")
EXPECTED_WINDOWS = {"PTF": 11520, "LEU": 11616, "MSPG": 10557}
EXPECTED_GEOM = {"pad_left": 0, "pad_right": 0, "num_patch_input": 4}
EXPECTED_PREFLIGHT_WINDOWS = 230   # 75 seqs x first/mid/last + 5 zero-hist extras

SHARD_SIZE = 512
BATCH_CANDIDATE = 32               # preflight probes batch=1 vs this;
                                   # adopted only if bitwise equal, else batch=1

RECOMPUTE_RTOL = 1e-12
RECOMPUTE_ATOL = 1e-12

CODE_FILES = ["protocol.py", "common.py", "load_cgtsf.py", "preflight.py",
              "run_infer.py", "evaluate.py", "verify_cache.py",
              "recompute_check.py", "audit_pack.py"]
VTS_CODE_FILES = ["visionts/model.py", "visionts/models_mae.py",
                  "visionts/pos_embed.py", "visionts/util.py",
                  "visionts/__init__.py"]

DATA_DIR = os.path.join(HERE, "data")
PRED_DIR = os.path.join(HERE, "predictions")
RESULTS_DIR = os.path.join(HERE, "results")
LOGS_DIR = os.path.join(HERE, "logs")
QUARANTINE_DIR = os.path.join(HERE, "quarantine")
PROTO_PATH = os.path.join(HERE, "protocol.json")
COMMIT_PATH = os.path.join(HERE, "COMMIT.txt")


def csv_path(ds):
    return os.path.join(DATA_DIR, f"{ds}.csv")


def weights_path():
    return os.path.join(REPO_ROOT, "ckpt", WEIGHTS_FILE)


def shard_paths(ds, shard):
    base = os.path.join(PRED_DIR, f"{ds}__shard{shard:03d}")
    return base + ".npz", base + ".header.json", base + ".done"


def n_shards(ds):
    n = EXPECTED_WINDOWS[ds]
    return (n + SHARD_SIZE - 1) // SHARD_SIZE


def repo_commit():
    if os.path.isfile(COMMIT_PATH):
        return open(COMMIT_PATH).read().strip()
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_protocol_dict():
    import common
    return {
        "protocol_name": PROTOCOL_NAME,
        "seed": SEED, "arch": ARCH, "finetune_type": FINETUNE_TYPE,
        "norm_const": NORM_CONST, "align_const": ALIGN_CONST,
        "interpolation": INTERPOLATION, "fp64": FP64, "tf32": TF32,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "hf_revision": HF_REVISION,
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": WEIGHTS_SHA256, "weights_bytes": WEIGHTS_BYTES,
        "csv_sha256": dict(CSV_SHA256), "csv_bytes": dict(CSV_BYTES),
        "subsets": {k: dict(v) for k, v in SUBSETS.items()},
        "order": list(ORDER),
        "expected_windows": dict(EXPECTED_WINDOWS),
        "expected_geometry": dict(EXPECTED_GEOM),
        "expected_preflight_windows": EXPECTED_PREFLIGHT_WINDOWS,
        "shard_size": SHARD_SIZE, "batch_candidate": BATCH_CANDIDATE,
        "recompute_rtol": RECOMPUTE_RTOL, "recompute_atol": RECOMPUTE_ATOL,
        "code_md5": {n: common.md5_file(os.path.join(HERE, n))
                     for n in CODE_FILES},
        "visionts_code_md5": {n: common.md5_file(os.path.join(REPO_ROOT, n))
                              for n in VTS_CODE_FILES},
        "repo_commit": repo_commit(),
    }


def freeze():
    import common
    d = build_protocol_dict()
    b = json.dumps(d, indent=1, sort_keys=True).encode()
    tmp = PROTO_PATH + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, PROTO_PATH)
    token = hashlib.sha256(open(PROTO_PATH, "rb").read()).hexdigest()[:12]
    print("frozen protocol.json ->", PROTO_PATH)
    print("approval token:", token)


def check_approval(token):
    """Load protocol.json and verify the approval token (= first 12 hex of
    its sha256). Hard stop on mismatch/absence."""
    if not os.path.isfile(PROTO_PATH):
        sys.exit("protocol.json missing; run: python protocol.py freeze")
    digest = hashlib.sha256(open(PROTO_PATH, "rb").read()).hexdigest()
    if token != digest[:12]:
        sys.exit(f"approval token mismatch: got {token!r}, "
                 f"want {digest[:12]!r} (sha256 of protocol.json)")
    return json.load(open(PROTO_PATH))


def verify_frozen(proto):
    """Hard-stop verification of every pin: experiment code, visionts
    sources, CSV bytes, weights file."""
    import common
    if proto["csv_sha256"] != CSV_SHA256:
        sys.exit("protocol.json/constants drift (csv pins differ)")
    bad = [n for n, m in proto["code_md5"].items()
           if not os.path.isfile(os.path.join(HERE, n))
           or common.md5_file(os.path.join(HERE, n)) != m]
    bad += [n for n, m in proto["visionts_code_md5"].items()
            if not os.path.isfile(os.path.join(REPO_ROOT, n))
            or common.md5_file(os.path.join(REPO_ROOT, n)) != m]
    if bad:
        sys.exit(f"FROZEN CODE MISMATCH (re-freeze required, hard stop): {bad}")
    for ds in ORDER:
        p = csv_path(ds)
        if not os.path.isfile(p) or common.sha256_file(p) != CSV_SHA256[ds]:
            sys.exit(f"CSV sha mismatch for {ds}: {p}")
    w = weights_path()
    if not os.path.isfile(w):
        sys.exit(f"weights missing: {w}")
    got = common.sha256_file(w)
    if got != WEIGHTS_SHA256:
        sys.exit(f"weights sha mismatch: got {got}, want {WEIGHTS_SHA256}")
    if os.path.getsize(w) != WEIGHTS_BYTES:
        sys.exit(f"weights bytes mismatch: {os.path.getsize(w)}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "freeze":
        freeze()
    else:
        print(__doc__)
        print("usage: python protocol.py freeze")
