#!/usr/bin/env python3
"""Serial 285-run training queue (F0-F4 x 19 domains x 3 seeds) on one L20.

Order: domains ascending by dense count (arts first, Currency last), inner
seed -> F0,F1,F2,F3,F4 — every completed (domain, seed) is fully comparable
across methods. Skips runs with SUCCESS.json; resumes epoch-granular from
resume.pt. Honors the PAUSE sentinel and the free-VRAM gate (preflight peak
+ 3 GiB) before EVERY run. Any failure stops the queue with the scene kept.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (AUDIT, CHECKPOINTS, CFG, DOMAIN_NAMES, LOGS, METHODS_TRAIN,
                    PAUSE_SENTINEL, PREFLIGHT_JSON, RUN_MANIFEST, SEEDS,
                    atomic_write_json)
from trainer import ExperimentData, run_training

VRAM_MARGIN_MB = 3072


def gpu_free_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    idx = int(os.environ.get("GPU", "0"))
    return int(out[idx])


def wait_for_gpu(required_mb, rid):
    while True:
        if os.path.isfile(PAUSE_SENTINEL):
            print(f"[queue] PAUSE sentinel present; waiting ({rid})", flush=True)
            time.sleep(60)
            continue
        free = gpu_free_mb()
        if free >= required_mb:
            return free
        print(f"[queue] free VRAM {free}MiB < required {required_mb}MiB; "
              f"waiting for {rid}", flush=True)
        time.sleep(60)


def load_manifest():
    if os.path.isfile(RUN_MANIFEST):
        with open(RUN_MANIFEST) as f:
            return json.load(f)
    return {"runs": {}}


def save_manifest(m):
    atomic_write_json(m, RUN_MANIFEST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(PREFLIGHT_JSON):
        print("[queue] preflight_report.json missing — run preflight first")
        sys.exit(1)
    pre = json.load(open(PREFLIGHT_JSON))
    if not pre.get("all_pass"):
        print("[queue] preflight did not pass — refusing to start")
        sys.exit(1)
    # the preflight report must be bound to the SAME code/data/protocol that
    # is on disk right now (recomputed here, not trusted from the report)
    from common import actual_fingerprints
    act = actual_fingerprints()
    bound = pre.get("fingerprints", {})
    checks = [("code", act["code"], bound.get("code")),
              ("data_cache", act["data_cache"], bound.get("data_cache")),
              ("split_manifest", act["split_manifest"],
               bound.get("split_manifest"))]
    proto_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "manifests", "protocol.json")
    if os.path.isfile(proto_path):
        proto = json.load(open(proto_path))
        checks += [("proto_code", act["code"], proto.get("code_fingerprint")),
                   ("proto_data_cache", act["data_cache"],
                    proto.get("data_cache_sha256"))]
    bad = [n for n, a, b in checks if a != b]
    if bad:
        print(f"[queue] fingerprint binding broken ({bad}) — rerun prepare "
              f"+ preflight; refusing to start")
        sys.exit(1)
    required_mb = int(pre["peak_vram_mb"]) + VRAM_MARGIN_MB
    print(f"[queue] gate: free VRAM >= {required_mb} MiB "
          f"(preflight peak {pre['peak_vram_mb']:.0f} + 3072); "
          f"fingerprints bound OK")

    data = ExperimentData()
    dom_order = sorted(DOMAIN_NAMES,
                       key=lambda d: data.domain_dense_total(d))
    plan = [(d, s, m) for d in dom_order for s in SEEDS
            for m in METHODS_TRAIN]
    print(f"[queue] {len(plan)} runs; domain order: {dom_order}")
    if args.dry_run:
        for d, s, m in plan:
            print(f"  {m}__{d}__s{s}")
        return

    man = load_manifest()
    n_done = 0
    for d, s, m in plan:
        rid = f"{m}__{d}__s{s}"
        entry = man["runs"].get(rid, {"run_id": rid, "method": m,
                                      "domain": d, "seed": s,
                                      "status": "pending"})
        if entry.get("status") == "success" and os.path.isfile(
                os.path.join(CHECKPOINTS, rid, "SUCCESS.json")):
            n_done += 1
            continue
        free = wait_for_gpu(required_mb, rid)
        print(f"[queue] START {rid} (free {free}MiB)", flush=True)
        entry["status"] = "running"
        entry["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        man["runs"][rid] = entry
        save_manifest(man)
        try:
            info = run_training(m, d, s, data, device="cuda")
        except Exception as e:  # noqa: BLE001 — stop queue, keep the scene
            import traceback
            tb = traceback.format_exc()
            with open(os.path.join(LOGS, f"{rid}.error.log"), "w") as f:
                f.write(tb)
            entry["status"] = "failed"
            entry["error"] = str(e)[:500]
            save_manifest(man)
            print(f"[queue] FAILED {rid}: {e}\n{tb}", flush=True)
            print("[queue] queue stopped — scene preserved", flush=True)
            sys.exit(3)
        entry.update({"status": "success",
                      "wall_s": round(info.get("wall_s", 0.0), 1),
                      "peak_vram_mb": round(info.get("peak_vram_mb", 0.0), 1),
                      "local_best_epoch": info.get("local_best_epoch"),
                      "updates_total": info.get("updates_total"),
                      "presented_total": info.get("presented_total"),
                      "U_d": info.get("U_d"), "S_d": info.get("S_d"),
                      "finish_attempt": info.get("finish_attempt"),
                      "val_mse_by_epoch": info.get("val_mse_by_epoch"),
                      "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        man["runs"][rid] = entry
        save_manifest(man)
        n_done += 1
        print(f"[queue] DONE {rid} ({n_done}/{len(plan)}) "
              f"wall={entry['wall_s']}s peak={entry['peak_vram_mb']}MB",
              flush=True)
    print(f"[queue] ALL {n_done}/{len(plan)} runs complete", flush=True)


if __name__ == "__main__":
    main()
