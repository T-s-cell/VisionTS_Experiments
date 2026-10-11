#!/usr/bin/env python3
"""Frozen protocol for the VisionTS environment-drift diagnostic
(diag_cgtsf_visionts_env_v1).

Question: does re-running the FROZEN VisionTS EXP-016 pipeline in the
Aurora final-run environment (RTX 5060 Ti / WSL2, py3.10, torch 2.13.0+
cu129) explain the "Aurora MSE lower / MAE higher" pattern of EXP-017?

Everything about the MODEL pipeline is inherited unchanged from the frozen
EXP-016 protocol (token a4c01683793a, commit 0932a5ac...): MAE-base fully
frozen, eval(), seed=2021, batch=1 (FIXED here; EXP-016's batch-candidate
probe is deliberately disabled so the ONLY intended variable is the
runtime environment), FP32, fp64=False, TF32/AMP off, norm=align=0.4,
bilinear, history-only input, all-zero windows kept, identity
(dataset, Idx, Date), CSV row order = window order.

Freeze flow: `python protocol.py freeze` writes protocol.json (constants +
fingerprints of EVERY experiment script + the 5 visionts sources, whose
md5 must equal EXP-016's pinned values — a difference means the frozen
model source was touched = hard stop). preflight.py / run_infer.py take
--i-approve-frozen-protocol <first 12 hex of sha256(protocol.json)>.

Two references are pinned and verified BEFORE any GPU work
(verify_refs.py): the EXP-016 VisionTS predictions/metrics (visionts_ref/)
and the Aurora EXP-017 predictions/per-seed metrics (aurora_ref/). All
recompute gates use the double tolerance rtol=atol=1e-12:
    pass  <=>  |got - ref| <= atol + rtol * |ref|.
Policy: fingerprint mismatch = stop; a corrupt/unfinished shard is
quarantined and recomputed; NaN/Inf forward = hard stop."""
import glob
import hashlib
import json
import os
import subprocess
import sys

PROTOCOL_NAME = "diag_cgtsf_visionts_env_v1"
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
BATCH = 1                       # FIXED (user ruling 2026-10-11); no probe

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
EXPECTED_PREFLIGHT_WINDOWS = 230

SHARD_SIZE = 512

RECOMPUTE_RTOL = 1e-12
RECOMPUTE_ATOL = 1e-12

# ---- EXP-016 reference pins (VisionTS "old environment") ----------------
EXP016_ARCHIVE_COMMIT = "0932a5ac841ea399271ce88bde5b607ee0bcb831"
EXP016_TOKEN = "a4c01683793a"   # public sha256[:12] of EXP-016 protocol.json
EXP016_METRICS_SHA256 = {
    "PTF": "b40721154afc82af04ac8916444a3a0c0e826d4cb3d6748d1385d89baf45aac0",
    "LEU": "5ffe1aa3d900b4cc6f0ad03e020cca46b4679fe7d122f2b94b10aede06099ada",
    "MSPG": "41217b9636bd05aafad4635ef25ca67c2517a17c571e7ec88a3b63e158af9aea",
}
EXP016_SEQ_CSV_SHA256 = {
    "PTF": "ca7239786281b55b289f62df2ee967fdf76d5463006d24daf4b5305b5cb46b81",
    "LEU": "56f19be8acf4f01c0b091bc42ee8e3a741cdfb8698cd17cf6107ddda77d33826",
    "MSPG": "3c9dc26146c4e1fa8b429322d5d98ec5e7f3a3fe08fe39978df1360dc98bd390",
}
EXP016_INFER_SUMMARY_SHA256 = \
    "634727dcec774241e84ab1cbccf79d8bd10a1d0c438c941d7c9220c0fc4bb686"
# EXP-016's own pinned md5s of the 5 frozen visionts source files; the
# sources used here must match EXACTLY (frozen model source, not touched).
EXP016_VTS_CODE_MD5 = {
    "visionts/__init__.py": "60249b5b1f0bd22cc6b0bd76e5eee077",
    "visionts/model.py": "621f3f87b299c4bfeb5a64c87eefaa62",
    "visionts/models_mae.py": "20edd0d6da5b21455d6c2ffb35de3d43",
    "visionts/pos_embed.py": "d31375b8ea4a1f1a9803a68b3c292399",
    "visionts/util.py": "d14f84b9bd2650c5bfdab3128457b540",
}
VTS_CODE_FILES = ["visionts/model.py", "visionts/models_mae.py",
                  "visionts/pos_embed.py", "visionts/util.py",
                  "visionts/__init__.py"]

# ---- Aurora EXP-017 reference pins (frozen predictions, not re-run) -----
AURORA_ARCHIVE_COMMIT = "3d20862a35fce5854ab2bd2bc6b0e849bd67da57"
AURORA_CODE_COMMIT = "3526c68673d55c8bcd4ae47aed4b950743d3b70f"
AURORA_TOKEN = "9c164222d4b9"   # public sha256[:12] of EXP-017 protocol.json
AURORA_SEEDS = (2021, 2022, 2023)
AURORA_METRICS_SHA256 = {       # final user-approved metrics (verify_dt pass)
    "PTF": "52df03a130db891d2c799d6d95ab1a44197150db49d6ae9d42e9de6b8659bcab",
    "LEU": "31029fd3f809314ac5a8403f35f8be8c6e9c115774742817ef4289e17744e3d5",
    "MSPG": "c613ddf5e52ee399265ab324c357275225750c675ac012beeddd33af26ef2654",
}


def experiment_code_files():
    """EVERY script of this experiment is fingerprinted (user ruling
    2026-10-11: no fixed count — new scripts must not escape the pin)."""
    names = sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(HERE, "*.py"))
                   + glob.glob(os.path.join(HERE, "*.sh")))
    for extra in ("README.md", ".gitignore", "COMMIT.txt"):
        if os.path.isfile(os.path.join(HERE, extra)):
            names.append(extra)
    return names


DATA_DIR = os.path.join(HERE, "data")
PRED_DIR = os.path.join(HERE, "predictions")
RESULTS_DIR = os.path.join(HERE, "results")
LOGS_DIR = os.path.join(HERE, "logs")
QUARANTINE_DIR = os.path.join(HERE, "quarantine")
VTS_REF_DIR = os.path.join(HERE, "visionts_ref")
AURORA_REF_DIR = os.path.join(HERE, "aurora_ref")
PROTO_PATH = os.path.join(HERE, "protocol.json")
COMMIT_PATH = os.path.join(HERE, "COMMIT.txt")


def csv_path(ds):
    return os.path.join(DATA_DIR, f"{ds}.csv")


def weights_path():
    return os.path.join(REPO_ROOT, "ckpt", WEIGHTS_FILE)


def shard_paths(ds, shard):
    base = os.path.join(PRED_DIR, f"{ds}__shard{shard:03d}")
    return base + ".npz", base + ".header.json", base + ".done"


def aurora_shard_paths(aurora_pred_dir, ds, seed, shard):
    return os.path.join(aurora_pred_dir,
                        f"{ds}__s{seed}__shard{shard:03d}.npz")


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
        "batch": BATCH,
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": WEIGHTS_SHA256, "weights_bytes": WEIGHTS_BYTES,
        "csv_sha256": dict(CSV_SHA256), "csv_bytes": dict(CSV_BYTES),
        "subsets": {k: dict(v) for k, v in SUBSETS.items()},
        "order": list(ORDER),
        "expected_windows": dict(EXPECTED_WINDOWS),
        "expected_geometry": dict(EXPECTED_GEOM),
        "expected_preflight_windows": EXPECTED_PREFLIGHT_WINDOWS,
        "shard_size": SHARD_SIZE,
        "recompute_rtol": RECOMPUTE_RTOL, "recompute_atol": RECOMPUTE_ATOL,
        "exp016": {
            "archive_commit": EXP016_ARCHIVE_COMMIT,
            "token": EXP016_TOKEN,
            "metrics_sha256": dict(EXP016_METRICS_SHA256),
            "seq_csv_sha256": dict(EXP016_SEQ_CSV_SHA256),
            "infer_summary_sha256": EXP016_INFER_SUMMARY_SHA256,
            "visionts_code_md5": dict(EXP016_VTS_CODE_MD5),
        },
        "aurora": {
            "archive_commit": AURORA_ARCHIVE_COMMIT,
            "code_commit": AURORA_CODE_COMMIT,
            "token": AURORA_TOKEN,
            "seeds": list(AURORA_SEEDS),
            "metrics_sha256": dict(AURORA_METRICS_SHA256),
        },
        "code_md5": {n: common.md5_file(os.path.join(HERE, n))
                     for n in experiment_code_files()},
        "visionts_code_md5": {n: common.md5_file(os.path.join(REPO_ROOT, n))
                              for n in VTS_CODE_FILES},
        "repo_commit": repo_commit(),
    }


def freeze():
    import common
    d = build_protocol_dict()
    bad = [n for n, m in d["visionts_code_md5"].items()
           if EXP016_VTS_CODE_MD5.get(n) != m]
    if bad:
        sys.exit(f"FROZEN MODEL SOURCE DRIFT vs EXP-016 pins: {bad} — "
                 f"the frozen visionts/ sources must NOT be touched")
    b = json.dumps(d, indent=1, sort_keys=True).encode()
    tmp = PROTO_PATH + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, PROTO_PATH)
    token = hashlib.sha256(open(PROTO_PATH, "rb").read()).hexdigest()[:12]
    print("frozen protocol.json ->", PROTO_PATH)
    print("approval token:", token)


def check_approval(token):
    if not os.path.isfile(PROTO_PATH):
        sys.exit("protocol.json missing; run: python protocol.py freeze")
    digest = hashlib.sha256(open(PROTO_PATH, "rb").read()).hexdigest()
    if token != digest[:12]:
        sys.exit(f"approval token mismatch: got {token!r}, "
                 f"want {digest[:12]!r} (sha256 of protocol.json)")
    return json.load(open(PROTO_PATH))


def verify_frozen(proto):
    """Hard-stop verification of every pin: ALL experiment scripts, the 5
    visionts sources (vs EXP-016's own pins), CSV bytes, weights file,
    and the protocol constants themselves."""
    import common
    if proto["csv_sha256"] != CSV_SHA256:
        sys.exit("protocol.json/constants drift (csv pins differ)")
    if proto["exp016"]["visionts_code_md5"] != EXP016_VTS_CODE_MD5:
        sys.exit("protocol.json/constants drift (visionts source pins)")
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
