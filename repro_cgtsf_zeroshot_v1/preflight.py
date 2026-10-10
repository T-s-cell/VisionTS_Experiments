#!/usr/bin/env python3
"""Stage P: 230-window preflight on the executing GPU host.

Hard gates (any failure = nonzero exit, no full run):
  1. protocol/code/data/weights fingerprints (protocol.verify_frozen)
  2. per-subset update_config geometry == expected (pad 0/0, npi=4)
  3. weights file sha256 unchanged before/after all forwards
  4. in-memory model state digest (params+buffers) unchanged per subset
  5. output shape [1,P,1] and finiteness on all 230 windows (batch=1)
  6. same-env/same-batch/same-order repeat inference bitwise identical
  7. shard roundtrip (save npz -> reload) bitwise identical
Soft probe (NOT a gate): candidate batch vs batch=1 — adopted only if
bitwise equal, else the full run falls back to batch=1 (recorded, not a
failure). PyTorch does not guarantee bitwise results across batch sizes.
Also records per-window timing and peak VRAM to estimate the full run."""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (HERE, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

import common as C
import protocol as P
from load_cgtsf import load_windows, select_preflight


def setup_determinism():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == P.CUBLAS_WORKSPACE_CONFIG, \
        "CUBLAS_WORKSPACE_CONFIG not set (use wrapper_zeta.sh)"
    torch.backends.cuda.matmul.allow_tf32 = P.TF32
    torch.backends.cudnn.allow_tf32 = P.TF32
    torch.manual_seed(P.SEED)
    np.random.seed(P.SEED)


def build_model(ckpt_dir, device):
    from visionts.model import VisionTS
    model = VisionTS(arch=P.ARCH, finetune_type=P.FINETUNE_TYPE,
                     ckpt_dir=ckpt_dir, load_ckpt=True)
    model.eval().to(device)
    return model


def forward_batch(model, hists, device):
    x = torch.from_numpy(np.stack(hists)).float().unsqueeze(-1).to(device)
    with torch.no_grad():
        y = model(x)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-approve-frozen-protocol", required=True)
    ap.add_argument("--ckpt-dir", default=os.path.join(REPO_ROOT, "ckpt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-candidate", type=int, default=P.BATCH_CANDIDATE)
    args = ap.parse_args()

    proto = P.check_approval(args.i_approve_frozen_protocol)
    P.verify_frozen(proto)
    setup_determinism()
    device = args.device
    assert device == "cuda" and torch.cuda.is_available()

    w_sha_0 = C.sha256_file(P.weights_path())
    print(f"[preflight] weights sha256 ok: {w_sha_0[:16]}...")

    windows_by_ds = {ds: load_windows(ds) for ds in P.ORDER}
    sel = select_preflight(windows_by_ds)
    per_ds = {ds: picks for ds, picks in sel}
    print("[preflight] selected windows:",
          {ds: len(p) for ds, p in per_ds.items()}, "total",
          sum(len(p) for p in per_ds.values()))

    model = build_model(args.ckpt_dir, device)

    report = {"stage": "preflight", "env": C.env_info(),
              "per_subset": {}, "checks": {}}
    ok = True
    preds_b1 = {}
    timings_b1 = {}

    for ds in P.ORDER:
        cfg = P.SUBSETS[ds]
        model.update_config(context_len=cfg["context"],
                            pred_len=cfg["pred"],
                            periodicity=cfg["periodicity"],
                            norm_const=P.NORM_CONST,
                            align_const=P.ALIGN_CONST,
                            interpolation=P.INTERPOLATION)
        geom = {"pad_left": model.pad_left, "pad_right": model.pad_right,
                "num_patch_input": model.num_patch_input,
                "num_patch": model.num_patch,
                "mask_ratio": model.mask_ratio}
        geom_ok = (model.pad_left == P.EXPECTED_GEOM["pad_left"]
                   and model.pad_right == P.EXPECTED_GEOM["pad_right"]
                   and model.num_patch_input == P.EXPECTED_GEOM["num_patch_input"])
        print(f"[preflight] {ds}: geometry {geom} ok={geom_ok}")
        ok &= geom_ok
        digest_a = C.model_state_digest(model)

        picks = per_ds[ds]
        hists = [windows_by_ds[ds][i]["hist"] for i in picks]
        Pl = cfg["pred"]

        # --- run 1: batch=1, timed, VRAM peak ---
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out1, t1 = [], []
        for h in hists:
            t0 = time.perf_counter()
            y = forward_batch(model, [h], device)
            torch.cuda.synchronize()
            t1.append(time.perf_counter() - t0)
            a = y.squeeze(-1).double().cpu().numpy()
            assert a.shape == (1, Pl), f"{ds}: pred shape {a.shape} != (1,{Pl})"
            if not C.all_finite(a):
                print(f"[preflight] FAIL {ds}: non-finite output")
                ok = False
            out1.append(a[0])
        vram = torch.cuda.max_memory_allocated() / (1 << 20)
        timings_b1[ds] = t1
        preds_b1[ds] = np.stack(out1)

        # --- run 2: identical repeat, bitwise ---
        out2 = []
        for h in hists:
            y = forward_batch(model, [h], device)
            out2.append(y.squeeze(-1).double().cpu().numpy()[0])
        rep_ok = np.array_equal(preds_b1[ds], np.stack(out2))
        print(f"[preflight] {ds}: repeat bitwise identical = {rep_ok}")
        ok &= rep_ok

        digest_b = C.model_state_digest(model)
        state_ok = digest_a == digest_b
        print(f"[preflight] {ds}: model state unchanged = {state_ok}")
        ok &= state_ok

        report["per_subset"][ds] = {
            "n_windows": len(picks),
            "geometry": geom, "geometry_ok": bool(geom_ok),
            "repeat_bitwise_ok": bool(rep_ok),
            "state_digest": digest_a, "state_unchanged": bool(state_ok),
            "ms_per_window_b1_mean": float(np.mean(t1)),
            "ms_per_window_b1_std": float(np.std(t1)),
            "vram_peak_mib": float(vram),
        }

    # --- roundtrip: simulate shard save/reload for first 8 preds of PTF ---
    import tempfile
    arr = preds_b1["PTF"][:8]
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "rt.npz")
        np.savez(pth + ".tmp.npz", pred=arr)
        os.replace(pth + ".tmp.npz", pth)
        with np.load(pth) as z:
            rt_ok = np.array_equal(z["pred"], arr)
    print(f"[preflight] roundtrip save/reload bitwise identical = {rt_ok}")
    ok &= rt_ok
    report["checks"]["roundtrip_bitwise_ok"] = bool(rt_ok)

    # --- batch probe (soft): candidate batch vs batch=1, same order ---
    chosen_batch = 1
    for ds in P.ORDER:
        cfg = P.SUBSETS[ds]
        model.update_config(context_len=cfg["context"],
                            pred_len=cfg["pred"],
                            periodicity=cfg["periodicity"],
                            norm_const=P.NORM_CONST,
                            align_const=P.ALIGN_CONST,
                            interpolation=P.INTERPOLATION)
        picks = per_ds[ds]
        hists = [windows_by_ds[ds][i]["hist"] for i in picks]
        outs = []
        for i in range(0, len(hists), args.batch_candidate):
            y = forward_batch(model, hists[i:i + args.batch_candidate], device)
            outs.append(y.squeeze(-1).double().cpu().numpy())
        outs = np.concatenate(outs, 0)
        same = np.array_equal(outs, preds_b1[ds])
        report["per_subset"][ds]["batch_probe_bitwise_ok"] = bool(same)
        print(f"[preflight] {ds}: batch={args.batch_candidate} bitwise "
              f"== batch=1 : {same}")
        if not same:
            chosen_batch = 1
            print(f"[preflight] -> batch mismatch: full run will use batch=1 "
                  f"(not a failure)")
    if all(report["per_subset"][ds].get("batch_probe_bitwise_ok")
           for ds in P.ORDER):
        chosen_batch = args.batch_candidate
        print(f"[preflight] -> batch={args.batch_candidate} adopted")
    report["chosen_batch"] = chosen_batch

    w_sha_1 = C.sha256_file(P.weights_path())
    w_ok = w_sha_0 == w_sha_1 == P.WEIGHTS_SHA256
    print(f"[preflight] weights file unchanged = {w_ok}")
    ok &= w_ok
    report["checks"]["weights_unchanged"] = bool(w_ok)
    report["ok"] = bool(ok)

    eta = {}
    for ds in P.ORDER:
        ms = report["per_subset"][ds]["ms_per_window_b1_mean"]
        eta[ds] = {
            "batch1_hours": P.EXPECTED_WINDOWS[ds] * ms / 1000 / 3600,
            "chosen_batch_hours": P.EXPECTED_WINDOWS[ds] * ms
            / (chosen_batch if chosen_batch > 1 else 1) / 1000 / 3600
            if chosen_batch > 1 else None,
        }
    report["full_run_eta_hours"] = eta
    print("[preflight] ETA hours:", {d: round(v["batch1_hours"], 2)
                                     for d, v in eta.items()},
          "chosen_batch:", chosen_batch)

    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    C.dump_json(report, os.path.join(P.RESULTS_DIR, "preflight_report.json"))
    print(f"[preflight] RESULT: {'PASS' if ok else 'FAIL'} "
          f"-> results/preflight_report.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
