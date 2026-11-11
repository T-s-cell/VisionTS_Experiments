"""EXP-043 preflight.  Hard gates:
  P0  GPU free >= 20000 MiB (nvidia-smi, no CUDA init)
  P1  disk free >= 50 GB on the data root (budget: 48 val groups
      ~6.05 GB uncompressed + <=16 new test groups ~2.02 GB, user
      corrected 2026-11-11; 50 GB keeps ample slack)
  P2  frozen MAE ckpt sha256 + byte size
  P3  4 ETT CSVs exist; sha256 recorded (bound downstream by truth-bitwise
      gates, not by cross-env hash pinning)
  P4  EXP-001 inventory: 16 groups x {SUCCESS.txt, results/_/metrics.npy,
      pred.npy, true.npy} + summary.csv  (default-P arm, read-only reuse)
  P5  EXP-030 inventory: 16 groups x {SUCCESS.txt, results/_/metrics.npy,
      pred.npy, true.npy, metrics_<DS>_<H>.json}  (P=1 arm, read-only)
  P6  framework md5s == values recorded in EXP-030's archived SUCCESS.txt
      (the EXP-001/030 lineage fingerprint, verified live 2026-11-11)
Soft: env snapshot -> logs/env_t43.json (warn if torch != 2.12.x).
"""
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t43_common as X

MIN_GPU_FREE_MIB = 20000
MIN_DISK_FREE_GB = 50
EXPECT_TORCH_PREFIX = "2.12."


def gpu_free_mib(gpu_id=0):
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_id),
         "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit("[preflight] nvidia-smi failed: %s" % out.stderr)
    return int(out.stdout.strip().splitlines()[0])


def main():
    res = {"checks": {}, "hard_fail": []}

    # P0 GPU
    free = gpu_free_mib(0)
    res["checks"]["P0_gpu_free_mib"] = free
    if free < MIN_GPU_FREE_MIB:
        res["hard_fail"].append("P0 gpu free %d < %d" % (free, MIN_GPU_FREE_MIB))

    # P1 disk
    du = shutil.disk_usage(X.ROOT)
    free_gb = du.free / (1 << 30)
    res["checks"]["P1_disk_free_gb"] = round(free_gb, 1)
    if free_gb < MIN_DISK_FREE_GB:
        res["hard_fail"].append("P1 disk free %.1f < %d GB"
                                % (free_gb, MIN_DISK_FREE_GB))

    # P2 ckpt
    sha = X.sha256_of(X.CKPT_FILE)
    nbytes = os.path.getsize(X.CKPT_FILE)
    res["checks"]["P2_ckpt_sha256"] = sha
    res["checks"]["P2_ckpt_bytes"] = nbytes
    if sha != X.CKPT_SHA256 or nbytes != X.CKPT_BYTES:
        res["hard_fail"].append("P2 ckpt sha/bytes mismatch")

    # P3 CSVs
    csv_sha = {}
    for ds in X.ETT4:
        c = X.DATASETS[ds]
        p = os.path.join(X.ROOT, c["root_path"], c["data_path"])
        if not os.path.isfile(p):
            res["hard_fail"].append("P3 missing csv %s" % p)
        else:
            csv_sha[c["data_path"]] = X.sha256_of(p)
    res["checks"]["P3_csv_sha256"] = csv_sha

    # P4 EXP-001 inventory
    miss4 = []
    for ds, h in X.group_ids():
        d = X.base_dir(ds, h)
        for rel in ("SUCCESS.txt", "results/_/metrics.npy",
                    "results/_/pred.npy", "results/_/true.npy"):
            if not os.path.isfile(os.path.join(d, rel)):
                miss4.append("%s_%d/%s" % (ds, h, rel))
    if not os.path.isfile(os.path.join(X.ROOT, "repro_zeroshot", "summary.csv")):
        miss4.append("repro_zeroshot/summary.csv")
    res["checks"]["P4_exp001_missing"] = miss4
    if miss4:
        res["hard_fail"].append("P4 EXP-001 inventory missing %d items"
                                % len(miss4))

    # P5 EXP-030 inventory
    miss5 = []
    for ds, h in X.group_ids():
        d = X.p1_dir(ds, h)
        for rel in ("SUCCESS.txt", "results/_/metrics.npy",
                    "results/_/pred.npy", "results/_/true.npy",
                    "metrics_%s_%d.json" % (ds, h)):
            if not os.path.isfile(os.path.join(d, rel)):
                miss5.append("%s_%d/%s" % (ds, h, rel))
    res["checks"]["P5_exp030_missing"] = miss5
    if miss5:
        res["hard_fail"].append("P5 EXP-030 inventory missing %d items"
                                % len(miss5))

    # P6 framework md5s vs EXP-030 archived SUCCESS.txt
    ref = os.path.join(X.p1_dir("ETTh1", 96), "SUCCESS.txt")
    cur = X.framework_md5_dict()
    if not os.path.isfile(ref):
        res["hard_fail"].append("P6 EXP-030 SUCCESS.txt missing")
    else:
        recorded = {}
        for ln in X.read_fingerprint(ref):
            m = re.match(r"md5:(.+)=([0-9a-f]{32})$", ln)
            if m and m.group(1) in cur:
                recorded[m.group(1)] = m.group(2)
        bad = ["%s recorded=%s current=%s" % (f, recorded.get(f), cur[f])
               for f in X.FP_FILES if recorded.get(f) != cur[f]]
        res["checks"]["P6_framework_md5_mismatch"] = bad
        if bad:
            res["hard_fail"].append("P6 framework md5 drift: %s" % bad)

    # env snapshot (soft)
    env = {"python": sys.version.split()[0]}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["gpu_name"] = torch.cuda.get_device_name(0) \
            if torch.cuda.is_available() else "n/a"
    except Exception as e:  # pragma: no cover
        env["torch_error"] = str(e)
    try:
        import sklearn
        env["sklearn"] = sklearn.__version__
        import numpy
        env["numpy"] = numpy.__version__
        import pandas
        env["pandas"] = pandas.__version__
    except Exception as e:
        env["pds_error"] = str(e)
    env["warn_torch_not_%s" % EXPECT_TORCH_PREFIX] = \
        not str(env.get("torch", "")).startswith(EXPECT_TORCH_PREFIX)
    res["env"] = env
    X.dump_json(os.path.join(X.LOG_DIR, "env_t43.json"), env)

    res["all_preflight_pass"] = not res["hard_fail"]
    X.dump_json(os.path.join(X.LOG_DIR, "preflight_t43.json"), res)
    print("[preflight] gpu_free=%dMiB disk_free=%.1fGB hard_fail=%s"
          % (free, free_gb, res["hard_fail"]))
    if res["hard_fail"]:
        raise SystemExit(2)
    print("[preflight] ALL PASS")


if __name__ == "__main__":
    main()
