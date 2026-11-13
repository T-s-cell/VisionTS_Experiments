"""EXP-045 cost microbenchmark (layer b).  For every (ds, H) config, the
THREE archs run the SAME real test batch (first batch of the official test
loader, bs=32, identical tensor) on the same eta GPU, sequentially:

  per arch: load (timed separately) -> update_config -> model.eval() ->
    warmup 3 forwards under torch.inference_mode() ->
    torch.cuda.synchronize() -> reset peak-memory stats ->
    5 timed repeats (perf_counter around [synchronize; forward;
    synchronize]) -> record mean/min/std + max_memory_allocated ->
    del model + torch.cuda.empty_cache()  (review correction #6)

The timed region contains ONLY the forward pass on a batch already
resident on the GPU: no dataloader IO, no model load, no metric/save
work.  No autograd (inference_mode), no cross-model memory residue
(sequential load/release).  fp32 everywhere (the run semantics).
Layer (a) -- full-run wall_time_s incl. loader+save -- comes from the
per-group SUCCESS.txt files and is reported by collect_t45.py.

Output: logs/timing_t45.json (meta + 48 rows) + logs/timing_t45.csv.
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t45_common as X

AP = argparse.ArgumentParser()
AP.add_argument("--gpu", type=int, default=0)
A = AP.parse_args()

import torch  # noqa: E402

WARMUP = 3
REPEATS = 5


def ns(ds, h):
    c = X.DATASETS[ds]
    return SimpleNamespace(
        task_name="long_term_forecast", data=c["data"],
        root_path=c["root_path"], data_path=c["data_path"],
        features="M", target="OT", scale=True,
        seq_len=c["ctx"], label_len=48, pred_len=h,
        embed="timeF", freq="h", seasonal_patterns="Monthly",
        batch_size=32, num_workers=10)


def main():
    if X.ROOT not in sys.path:
        sys.path.insert(0, X.ROOT)
    if X.REPO not in sys.path:
        sys.path.insert(0, X.REPO)  # visionts/ package lives one level up
    from data_provider.data_factory import data_provider
    from visionts.model import VisionTS
    ckpt_dir = os.path.join(X.REPO, "ckpt")

    rows = []
    meta = {"gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "warmup": WARMUP, "repeats": REPEATS,
            "cuda_sync": True, "inference_mode": True, "eval": True,
            "released_between_archs": True, "batch_size": 32,
            "precision": "fp32", "timed_region":
                "forward on GPU-resident batch only (no loader/load/save)",
            "archs": X.ARCHS}

    for ds, h in X.group_ids():
        c = X.DATASETS[ds]
        _, loader = data_provider(ns(ds, h), "test")
        batch = next(iter(loader))
        x = batch[0]  # (inputs, batch_y, x_mark, y_mark) -- inputs only
        x = x.float().cuda(A.gpu)
        assert x.shape == (32, c["ctx"], 7), x.shape
        print("[timing] %s/H%d batch %s (loader excluded from timed region)"
              % (ds, h, tuple(x.shape)))
        for arch in X.ARCHS:
            short = X.ARCH_SHORT[arch]
            t_load0 = time.perf_counter()
            model = VisionTS(arch=arch, finetune_type="none",
                             ckpt_dir=ckpt_dir, load_ckpt=True).cuda(A.gpu)
            model.eval()
            model.update_config(c["ctx"], h, c["period"], 0.4, 0.4,
                                "bilinear")
            load_s = time.perf_counter() - t_load0
            with torch.inference_mode():
                for _ in range(WARMUP):
                    model(x)
            torch.cuda.synchronize(A.gpu)
            torch.cuda.reset_peak_memory_stats(A.gpu)
            ts = []
            for _ in range(REPEATS):
                torch.cuda.synchronize(A.gpu)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    model(x)
                torch.cuda.synchronize(A.gpu)
                ts.append(time.perf_counter() - t0)
            peak_gb = torch.cuda.max_memory_allocated(A.gpu) / (1 << 30)
            row = {"ds": ds, "h": h, "arch": arch, "load_s": round(load_s, 3),
                   "mean_s": float(np.mean(ts)), "min_s": float(np.min(ts)),
                   "std_s": float(np.std(ts)),
                   "repeats_s": [round(t, 4) for t in ts],
                   "peak_gb": peak_gb}
            rows.append(row)
            print("[timing]   %-5s load=%.1fs mean=%.3fs min=%.3fs peak=%.2fGB"
                  % (short, load_s, row["mean_s"], row["min_s"], peak_gb))
            del model
            torch.cuda.empty_cache()
        del x
        torch.cuda.empty_cache()

    X.dump_json(os.path.join(X.LOG_DIR, "timing_t45.json"),
                {"meta": meta, "rows": rows})

    import csv
    with open(os.path.join(X.LOG_DIR, "timing_t45.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ds", "H", "arch", "load_s", "warmup", "repeats",
                    "cuda_sync", "inference_mode", "eval",
                    "released_between_archs", "mean_s", "min_s", "std_s",
                    "peak_gb", "precision"])
        for r in rows:
            w.writerow([r["ds"], r["h"], r["arch"], r["load_s"], WARMUP,
                        REPEATS, "True", "True", "True", "True",
                        "%.6f" % r["mean_s"], "%.6f" % r["min_s"],
                        "%.6f" % r["std_s"], "%.4f" % r["peak_gb"], "fp32"])

    # completeness self-check (final MG11 gate re-checks against the json)
    assert len(rows) == 48, rows
    for r in rows:
        assert np.isfinite(r["mean_s"]) and r["peak_gb"] > 0
    print("[timing] wrote 48 rows -> %s" % os.path.join(X.LOG_DIR,
                                                        "timing_t45.json"))


if __name__ == "__main__":
    main()
