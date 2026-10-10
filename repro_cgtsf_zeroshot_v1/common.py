#!/usr/bin/env python3
"""Shared plumbing: hashing, atomic JSON, in-memory model-state digest,
shard ledger helpers. Protocol constants live in protocol.py; anything the
protocol pins (code/data/weights) failing verification is a hard stop."""
import hashlib
import json
import os

import numpy as np


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


def dump_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def model_state_digest(model):
    """sha256 over sorted (name, raw bytes) of the full state_dict (params +
    buffers). Complements the on-disk ckpt hash: proves the actually-loaded
    in-memory model is unchanged across a subset's inference."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for name in sorted(sd):
        t = sd[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(b"\0")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def all_finite(arr):
    return bool(np.isfinite(arr).all())


def protocol_sha256():
    import protocol as P
    return sha256_file(P.PROTO_PATH) if os.path.isfile(P.PROTO_PATH) else None


def current_guards():
    """Fingerprints every prediction shard is bound to. A stored guard set
    differing from this = protocol/source/weight/data mismatch = hard stop
    (tier-1). Used by run_infer resume checks and recorded in headers."""
    import protocol as P
    g = {
        "protocol_sha256": protocol_sha256(),
        "weights_sha256": P.WEIGHTS_SHA256,
        "csv_sha256": dict(P.CSV_SHA256),
        "code_md5": {n: md5_file(os.path.join(P.HERE, n))
                     for n in P.CODE_FILES},
        "visionts_code_md5": {n: md5_file(os.path.join(P.REPO_ROOT, n))
                              for n in P.VTS_CODE_FILES},
    }
    return g


def guards_equal(a, b):
    return a == b


def env_info():
    import platform
    import sys
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "host": os.uname().nodename,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "")}
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except ImportError:
        info["torchvision"] = None
    import numpy
    info["numpy"] = numpy.__version__
    return info
