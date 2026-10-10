#!/usr/bin/env python3
"""Stage P: 230-window preflight on the executing GPU host.

Hard gates (any failure = nonzero exit, no full run):
  1. protocol/code/data/weights fingerprints (protocol.verify_frozen)
  1b. weights file at the ACTUAL --ckpt-dir load path (sha256, before/after)
  2. per-subset update_config geometry == expected (pad 0/0, npi=4)
  3. weights file sha256 unchanged before/after all forwards
  4. in-memory model state digest (params+buffers) unchanged per subset
  5. output shape [1,P,1] and finiteness on all 230 windows (batch=1)
  6. same-env/same-batch/same-order repeat inference bitwise identical
  7. shard roundtrip (save npz -> reload) bitwise identical
  8. real resume drill on a 32-window PTF chunk through the production
     run_infer code paths (sandboxed predictions/quarantine dirs):
     continuous run vs fresh-process recompute bitwise identical;
     tampered stored guards -> HARD STOP (tier-1); corrupt npz ->
     quarantine + recompute bitwise identical (tier-2).
Soft probe (NOT a gate): candidate batch vs batch=1 — adopted only if
bitwise equal, else the full run falls back to batch=1 (recorded, not a
failure). PyTorch does not guarantee bitwise results across batch sizes.
Also records per-window timing and peak VRAM to estimate the full run."""
import argparse
import json
import os
import subprocess
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


def check_loaded_ckpt(ckpt_dir):
    """Verify the weights file VisionTS will actually load (under the
    given --ckpt-dir), not just the repo-default pinned path."""
    w = os.path.join(ckpt_dir, P.WEIGHTS_FILE)
    if not os.path.isfile(w):
        sys.exit(f"HARD STOP: weights file missing at load path: {w}")
    got = C.sha256_file(w)
    if got != P.WEIGHTS_SHA256:
        sys.exit(f"HARD STOP: loaded-weights sha mismatch at {w}: "
                 f"got {got}, want {P.WEIGHTS_SHA256}")
    return w


RESUME_RUNNER = r'''
import os, sys
HERE = r"{here}"
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)
import numpy as np
import common as C
import protocol as P
import run_infer as R
from load_cgtsf import load_windows

P.PRED_DIR = r"{pred_dir}"
P.QUARANTINE_DIR = r"{quarantine_dir}"
npz_check, _, _ = P.shard_paths("PTF", 0)
assert npz_check.startswith(r"{pred_dir}"), \
    f"sandbox leak: shard would land at {{npz_check}}"

R.setup_determinism()
from visionts.model import VisionTS
model = VisionTS(arch=P.ARCH, finetune_type=P.FINETUNE_TYPE,
                 ckpt_dir=r"{ckpt}", load_ckpt=True)
model.eval().to("cuda")
cfg = P.SUBSETS["PTF"]
model.update_config(context_len=cfg["context"], pred_len=cfg["pred"],
                    periodicity=cfg["periodicity"], norm_const=P.NORM_CONST,
                    align_const=P.ALIGN_CONST, interpolation=P.INTERPOLATION)
windows = load_windows("PTF")[:32]
preds, ms = R.compute_shard(model, windows, "cuda", 1, cfg["pred"])
guards = C.current_guards(batch=1, preflight_report_sha256=None)
R.write_shard("PTF", 0, windows, preds, ms, guards)
print("RESUME_RUNNER_OK")
'''


def resume_test(model, ckpt_dir):
    """Real interrupt->restart->resume drill on 32 PTF windows through the
    production run_infer code paths, in a sandboxed predictions dir.
    Nothing here touches the real predictions/ or quarantine/ trees."""
    import run_infer as R
    import tempfile
    res = {"n_windows": 32}
    ok = True
    saved = (P.PRED_DIR, P.QUARANTINE_DIR)
    try:
        with tempfile.TemporaryDirectory(prefix="cgtsf_resume_") as td:
            P.PRED_DIR = os.path.join(td, "pred")
            P.QUARANTINE_DIR = os.path.join(td, "quarantine")
            cfg = P.SUBSETS["PTF"]
            Pl = cfg["pred"]
            model.update_config(context_len=cfg["context"],
                                pred_len=cfg["pred"],
                                periodicity=cfg["periodicity"],
                                norm_const=P.NORM_CONST,
                                align_const=P.ALIGN_CONST,
                                interpolation=P.INTERPOLATION)
            dg_a = C.model_state_digest(model)
            windows = load_windows("PTF")[:32]

            # 1) continuous run: in-process model computes and writes shard 0
            preds_a, _ = R.compute_shard(model, windows, "cuda", 1, Pl)
            guards = C.current_guards(batch=1, preflight_report_sha256=None)
            R.write_shard("PTF", 0, windows, preds_a,
                          np.zeros(len(windows)), guards)
            v, _ = R.shard_valid("PTF", 0, guards)
            res["baseline_shard_valid"] = bool(v)
            ok &= v

            # 2) simulate a crash: wipe the shard, resume in a fresh process
            for p in P.shard_paths("PTF", 0):
                if os.path.isfile(p):
                    os.remove(p)
            v2, _ = R.shard_valid("PTF", 0, guards)
            res["after_crash_invalid"] = not v2
            ok &= not v2
            runner = os.path.join(td, "resume_runner.py")
            with open(runner, "w") as f:
                f.write(RESUME_RUNNER.format(
                    here=HERE, ckpt=ckpt_dir,
                    pred_dir=P.PRED_DIR,
                    quarantine_dir=P.QUARANTINE_DIR))
            rp = subprocess.run([sys.executable, runner], cwd=HERE,
                                capture_output=True, text=True, timeout=600)
            res["resume_proc_returncode"] = rp.returncode
            if rp.returncode != 0:
                res["resume_proc_tail"] = (rp.stderr or rp.stdout)[-2000:]
                ok = False
            else:
                npz, _, _ = P.shard_paths("PTF", 0)
                with np.load(npz) as z:
                    preds_b = z["pred"]
                same = (preds_b.shape == preds_a.shape
                        and np.array_equal(preds_b, preds_a))
                res["resume_bitwise_ok"] = bool(same)
                ok &= same
                v3, _ = R.shard_valid("PTF", 0, guards)
                res["cross_process_guards_ok"] = bool(v3)
                ok &= bool(v3)

            # 3) tier-1: tampered stored guards must HARD STOP
            npz, hdr, done = P.shard_paths("PTF", 0)
            good_done = open(done).read()
            d = json.loads(good_done)
            d["guards"]["core"]["weights_sha256"] = "0" * 64
            C.dump_json(d, done)
            stopped = False
            try:
                R.shard_valid("PTF", 0, guards)
            except SystemExit as e:
                stopped = "HARD STOP" in str(e)
            res["tier1_hardstop_ok"] = bool(stopped)
            ok &= stopped
            with open(done, "w") as f:
                f.write(good_done)

            # 4) tier-2: corrupt npz -> detect -> quarantine -> recompute
            with open(npz, "r+b") as f:
                f.seek(100)
                f.write(b"\x00\x01\x02\x03")
            v4, why4 = R.shard_valid("PTF", 0, guards)
            res["tier2_detected"] = bool(not v4 and "payload" in why4)
            ok &= res["tier2_detected"]
            if res["tier2_detected"]:
                R.quarantine("PTF", 0, why4)
                qfiles = sorted(os.listdir(P.QUARANTINE_DIR))
                res["tier2_quarantined"] = qfiles
                q_ok = (any(x.startswith("PTF__shard000.npz.")
                            for x in qfiles)
                        and any(x.endswith(".reason.json") for x in qfiles)
                        and not os.path.isfile(npz))
                res["tier2_quarantine_ok"] = bool(q_ok)
                ok &= q_ok
                preds_c, _ = R.compute_shard(model, windows, "cuda", 1, Pl)
                R.write_shard("PTF", 0, windows, preds_c,
                              np.zeros(len(windows)), guards)
                v5, _ = R.shard_valid("PTF", 0, guards)
                with np.load(npz) as z:
                    same_c = np.array_equal(z["pred"], preds_a)
                res["tier2_recompute_bitwise_ok"] = bool(v5 and same_c)
                ok &= res["tier2_recompute_bitwise_ok"]

            dg_b = C.model_state_digest(model)
            res["model_state_unchanged"] = bool(dg_a == dg_b)
            ok &= res["model_state_unchanged"]
    finally:
        P.PRED_DIR, P.QUARANTINE_DIR = saved
    res["ok"] = bool(ok)
    return res


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

    w_path = check_loaded_ckpt(args.ckpt_dir)
    w_sha_0 = C.sha256_file(w_path)
    print(f"[preflight] loaded-weights sha256 ok ({w_path}): "
          f"{w_sha_0[:16]}...")

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
            "ms_per_window_b1_mean": float(np.mean(t1)) * 1000.0,
            "ms_per_window_b1_std": float(np.std(t1)) * 1000.0,
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

    # --- real resume drill (production run_infer paths, sandboxed dirs) ---
    print("[preflight] resume drill: 32 PTF windows, fresh-process recompute "
          "+ tier-1/tier-2 shard anomaly paths ...", flush=True)
    resume = resume_test(model, args.ckpt_dir)
    report["checks"]["resume"] = resume
    ok &= resume["ok"]
    print(f"[preflight] resume drill ok = {resume['ok']} "
          f"(bitwise={resume.get('resume_bitwise_ok')} "
          f"tier1={resume.get('tier1_hardstop_ok')} "
          f"tier2={resume.get('tier2_recompute_bitwise_ok')})", flush=True)

    w_sha_1 = C.sha256_file(w_path)
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
