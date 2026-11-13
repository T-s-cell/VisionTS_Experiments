"""EXP-045 preflight.  Hard gates:
  P0  GPU free >= 20000 MiB (nvidia-smi, no CUDA init)
  P1  disk free >= 50 GB on the data root
  P2  three ckpts (base/large/huge) exist with sha256+bytes == CKPTS
      constants (base == the EXP-001 archived value; large/huge == values
      measured on the CA-verified off-box download, scp'd to eta;
      correction #4: normal HTTPS only, no insecure fallback -- the
      download provenance chain is documented in README_t45.md)
  P2b strict state-dict load for every arch (the official entry path:
      torch.load(key='model') + load_state_dict(strict=True), exactly as
      visionts/model.py does) + parameter count vs the paper Table 17
      header (112M/330M/657M, +-2%).  ANY missing key / shape mismatch ->
      hard stop, no substitute weights (user rule).
  P3  4 ETT CSVs exist; sha256 recorded (bound downstream by
      truth-bitwise gates, not by cross-env hash pinning)
  P4  EXP-001 inventory: 16 groups x {SUCCESS.txt, results/_/metrics.npy,
      pred.npy, true.npy} + summary.csv  (Base reuse arm, read-only)
  P5  framework md5s == values recorded in EXP-030's archived SUCCESS.txt
      (the EXP-001/030 lineage fingerprint, verified live in t43/t44)
  P6  geometry_table_t45.csv written (formula expectations for every
      (ds,H) x arch, 48 rows); static asserts from t45_common import:
      npi_raw >= 1 everywhere, ctx <= 16K, pad_left == 0, visible width
      == patch_size*npi and patch-aligned (correction #3)
Soft: env snapshot -> logs/env_t45.json (warn if torch != 2.12.x).
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t45_common as X  # import itself runs the static geometry asserts

MIN_GPU_FREE_MIB = 20000
MIN_DISK_FREE_GB = 50
EXPECT_TORCH_PREFIX = "2.12."
PARAM_TOL_REL = 0.02


def gpu_free_mib(gpu_id=0):
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_id),
         "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit("[preflight] nvidia-smi failed: %s" % out.stderr)
    return int(out.stdout.strip().splitlines()[0])


def strict_load_and_params(arch):
    """Official entry path: build the arch's MaskedAutoencoderViT via the
    VisionTS constructor (load_ckpt=True -> torch.load + strict state-dict
    load, CPU only), then count parameters.  Raises on any load error --
    the constructor itself is the official gate (model.py try/except)."""
    if X.REPO not in sys.path:
        sys.path.insert(0, X.REPO)
    import torch  # noqa: guaranteed by the visionts env
    from visionts.model import VisionTS
    ckpt_dir = os.path.join(X.REPO, "ckpt")
    model = VisionTS(arch=arch, finetune_type="none",
                     ckpt_dir=ckpt_dir, load_ckpt=True)
    n_params = int(sum(p.numel() for p in model.parameters()))
    n_buffers = int(sum(b.numel() for b in model.buffers()))
    del model
    return n_params, n_buffers, torch.__version__


def main():
    res = {"checks": {}, "hard_fail": [],
           "static_geometry_asserts": "pass (import-time: npi_raw>=1, "
                                      "ctx<=16K, pad_left==0, visible "
                                      "width patch-aligned, correction #3)"}

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

    # P2 ckpts: sha256 + bytes
    p2 = {}
    for arch in X.ARCHS:
        p = X.CKPT_FILE[arch]
        if not os.path.isfile(p):
            res["hard_fail"].append("P2 missing ckpt %s" % p)
            continue
        sha = X.sha256_of(p)
        nbytes = os.path.getsize(p)
        p2[arch] = {"sha256": sha, "bytes": nbytes,
                    "url": X.MAE_URL + X.ARCH_FACTS[arch]["ckpt"]}
        if sha != X.CKPTS[arch]["sha256"] or nbytes != X.CKPTS[arch]["bytes"]:
            res["hard_fail"].append(
                "P2 %s ckpt sha/bytes mismatch: got %s/%d want %s/%d"
                % (arch, sha[:16], nbytes, X.CKPTS[arch]["sha256"][:16],
                   X.CKPTS[arch]["bytes"]))
    res["checks"]["P2_ckpts"] = p2

    # P2b strict load + parameter count vs paper (official entry path)
    p2b = {}
    if not res["hard_fail"]:  # don't waste a full 3-arch load if sha failed
        for arch in X.ARCHS:
            try:
                n_params, n_buffers, torch_v = strict_load_and_params(arch)
            except Exception as e:
                res["hard_fail"].append(
                    "P2b %s strict load FAILED: %s (no substitute weights "
                    "allowed -- stop)" % (arch, repr(e)[:300]))
                continue
            want_m = X.ARCH_FACTS[arch]["paper_params_m"]
            rel = abs(n_params - want_m * 1e6) / (want_m * 1e6)
            p2b[arch] = {"params": n_params, "buffers": n_buffers,
                         "paper_params_m": want_m, "rel_diff": rel,
                         "torch": torch_v}
            if rel > PARAM_TOL_REL:
                res["hard_fail"].append(
                    "P2b %s param count %d vs paper %dM (rel %.4f > %.2f)"
                    % (arch, n_params, want_m, rel, PARAM_TOL_REL))
    res["checks"]["P2b_strict_load_params"] = p2b

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

    # P4 EXP-001 inventory (Base reuse arm)
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

    # P5 framework md5s vs EXP-030 archived SUCCESS.txt (lineage reference)
    ref = os.path.join(X.p1_dir("ETTh1", 96), "SUCCESS.txt")
    cur = X.framework_md5_dict()
    if not os.path.isfile(ref):
        res["hard_fail"].append("P5 EXP-030 SUCCESS.txt missing")
    else:
        recorded = {}
        for ln in X.read_fingerprint(ref):
            m = re.match(r"md5:(.+)=([0-9a-f]{32})$", ln)
            if m and m.group(1) in cur:
                recorded[m.group(1)] = m.group(2)
        bad = ["%s recorded=%s current=%s" % (f, recorded.get(f), cur[f])
               for f in X.FP_FILES if recorded.get(f) != cur[f]]
        res["checks"]["P5_framework_md5_mismatch"] = bad
        if bad:
            res["hard_fail"].append("P5 framework md5 drift: %s" % bad)

    # P6 geometry_table_t45.csv (static asserts already ran at import)
    os.makedirs(X.LOG_DIR, exist_ok=True)
    with open(X.GEOMETRY_TABLE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "patch_size", "num_patch", "ds", "H", "P", "ctx",
                    "pad_left", "pad_right", "npi_raw", "num_patch_input",
                    "num_patch_output", "mask_ratio", "input_resize_size",
                    "input_width_vis", "scale_x", "output_resize_size"])
        for g in X.geometry_table():
            w.writerow([g["arch"], g["patch_size"], g["num_patch"], g["ds"],
                        g["H"], g["periodicity"], g["context_len"],
                        g["pad_left"], g["pad_right"], g["npi_raw"],
                        g["num_patch_input"], g["num_patch_output"],
                        "%.10g" % g["mask_ratio"],
                        str(tuple(g["input_resize_size"])),
                        g["input_width_vis"],
                        "%.10g" % g["scale_x"],
                        str(tuple(g["output_resize_size"]))])
    res["checks"]["P6_geometry_table_csv"] = X.GEOMETRY_TABLE_CSV
    res["checks"]["P6_geometry_table_rows"] = len(X.geometry_table())

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
    X.dump_json(os.path.join(X.LOG_DIR, "env_t45.json"), env)

    res["all_preflight_pass"] = not res["hard_fail"]
    X.dump_json(os.path.join(X.LOG_DIR, "preflight_t45.json"), res)
    print("[preflight] gpu_free=%dMiB disk_free=%.1fGB hard_fail=%s"
          % (free, free_gb, res["hard_fail"]))
    for a, v in p2b.items():
        print("[preflight] %s params=%d (%.2f%% vs paper %dM)"
              % (a, v["params"], 100 * v["rel_diff"], v["paper_params_m"]))
    if res["hard_fail"]:
        raise SystemExit(2)
    print("[preflight] ALL PASS")


if __name__ == "__main__":
    main()
