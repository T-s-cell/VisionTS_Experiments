#!/usr/bin/env python3
"""Full inference PTF -> LEU -> MSPG (33,693 windows), sharded + resumable.

Resume policy (tiered, per user ruling 2026-10-10):
  - stored guard set (protocol sha / code md5 / weights sha / csv sha)
    differing from current  ->  HARD STOP (tier-1: fingerprint mismatch);
  - guards match but the shard npz is missing/corrupt/unfinished
    ->  quarantine the old files (reason + timestamp), recompute the shard,
        log the event (tier-2: shard damage only, not a stop).
A .done marker alone never skips a shard: payload sha256 is re-verified.
Any NaN/Inf in a forward output = immediate hard stop (nothing is written).
Per subset: update_config -> geometry gate -> in-memory model state digest
taken before and after the subset's forwards must be identical.
Preflight binding: results/preflight_report.json must exist with ok=true
before anything runs; the run's batch is pinned to the report's
chosen_batch (a different --batch is a hard stop); every shard's guards
bind batch + env + sha256 of the preflight report; the weights file under
the ACTUAL --ckpt-dir load path is hash-verified before model build and
again after the run."""
import argparse
import datetime
import os
import shutil
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
from load_cgtsf import load_windows


def setup_determinism():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == P.CUBLAS_WORKSPACE_CONFIG, \
        "CUBLAS_WORKSPACE_CONFIG not set (use wrapper_zeta.sh)"
    torch.backends.cuda.matmul.allow_tf32 = P.TF32
    torch.backends.cudnn.allow_tf32 = P.TF32
    torch.manual_seed(P.SEED)
    np.random.seed(P.SEED)


def quarantine(ds, shard, reason):
    os.makedirs(P.QUARANTINE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    npz, hdr, done = P.shard_paths(ds, shard)
    tag = f"{ds}__shard{shard:03d}.{ts}"
    for p in (npz, hdr, done):
        if os.path.isfile(p):
            shutil.move(p, os.path.join(P.QUARANTINE_DIR,
                                        os.path.basename(p) + "." + tag))
    entry = {"ds": ds, "shard": shard, "reason": reason, "timestamp": ts}
    C.dump_json(entry, os.path.join(
        P.QUARANTINE_DIR, f"{tag}.reason.json"))
    print(f"[infer] QUARANTINED {ds} shard {shard}: {reason}", flush=True)
    return entry


def shard_valid(ds, shard, guards):
    npz, hdr, done = P.shard_paths(ds, shard)
    if not os.path.isfile(done):
        return False, "no .done"
    d = C.load_json(done)
    if d.get("guards") != guards:
        sys.exit(f"HARD STOP {ds} shard {shard}: stored guards differ from "
                 f"current (protocol/source/weight/data fingerprint "
                 f"mismatch). Re-freeze and rerun; do NOT overwrite.")
    if not os.path.isfile(npz):
        return False, ".done present but npz missing"
    if C.sha256_file(npz) != d.get("payload_sha256"):
        return False, "payload sha256 mismatch (corrupt/unfinished)"
    return True, "ok"


def compute_shard(model, windows, device, batch, Pl):
    preds = np.zeros((len(windows), Pl), dtype=np.float64)
    ms = np.zeros(len(windows), dtype=np.float64)
    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch]
        t0 = time.perf_counter()
        x = torch.from_numpy(np.stack([w["hist"] for w in chunk])) \
            .float().unsqueeze(-1).to(device)
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        a = y.squeeze(-1).double().cpu().numpy()
        if not C.all_finite(a):
            sys.exit(f"HARD STOP: non-finite (NaN/Inf) prediction at "
                     f"windows [{i},{i + len(chunk)}) — nothing written")
        preds[i:i + len(chunk)] = a
        per = dt / len(chunk) * 1000.0
        ms[i:i + len(chunk)] = per
    return preds, ms


def write_shard(ds, shard, windows, preds, ms, guards):
    npz, hdr, done = P.shard_paths(ds, shard)
    os.makedirs(P.PRED_DIR, exist_ok=True)
    idx = np.array([w["idx"] for w in windows], dtype=np.int64)
    dates = np.array([w["date"] for w in windows])
    target = np.stack([w["target"] for w in windows]).astype(np.float64)
    tmp = npz + ".tmp.npz"
    np.savez(tmp, idx=idx, date=dates, pred=preds, target=target,
             ms=ms, ds=np.array(ds))
    os.replace(tmp, npz)
    payload = {"payload_sha256": C.sha256_file(npz),
               "guards": guards, "ds": ds, "shard": shard,
               "n": len(windows), "written": time.strftime("%F %T")}
    C.dump_json(payload, done)
    C.dump_json(payload, hdr)
    return payload


def check_loaded_ckpt(ckpt_dir):
    w = os.path.join(ckpt_dir, P.WEIGHTS_FILE)
    if not os.path.isfile(w):
        sys.exit(f"HARD STOP: weights file missing at load path: {w}")
    got = C.sha256_file(w)
    if got != P.WEIGHTS_SHA256:
        sys.exit(f"HARD STOP: loaded-weights sha mismatch at {w}: "
                 f"got {got}, want {P.WEIGHTS_SHA256}")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-approve-frozen-protocol", required=True)
    ap.add_argument("--ckpt-dir", default=os.path.join(REPO_ROOT, "ckpt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=None,
                    help="must equal the preflight report's chosen_batch")
    ap.add_argument("--only-ds", choices=P.ORDER, default=None)
    args = ap.parse_args()

    proto = P.check_approval(args.i_approve_frozen_protocol)
    P.verify_frozen(proto)
    setup_determinism()
    device = args.device
    assert device == "cuda" and torch.cuda.is_available()

    # --- preflight gate: no report, or ok!=true -> no full run ---
    rep_path = os.path.join(P.RESULTS_DIR, "preflight_report.json")
    if not os.path.isfile(rep_path):
        sys.exit("HARD STOP: results/preflight_report.json missing — "
                 "run preflight first")
    pre = C.load_json(rep_path)
    if not pre.get("ok"):
        sys.exit("HARD STOP: preflight report ok!=true — full run blocked")
    chosen = pre.get("chosen_batch")
    batch = args.batch if args.batch is not None else chosen
    if batch != chosen:
        sys.exit(f"HARD STOP: --batch {batch} != preflight chosen_batch "
                 f"{chosen} (execution config must match the preflight "
                 f"evidence; rerun preflight to change batch)")
    rep_sha = C.sha256_file(rep_path)
    print(f"[infer] preflight ok; batch={batch} (chosen_batch={chosen}), "
          f"preflight_report sha256 {rep_sha[:16]}...")

    # --- verify the weights file VisionTS will ACTUALLY load ---
    w_path = check_loaded_ckpt(args.ckpt_dir)

    from visionts.model import VisionTS
    model = VisionTS(arch=P.ARCH, finetune_type=P.FINETUNE_TYPE,
                     ckpt_dir=args.ckpt_dir, load_ckpt=True)
    model.eval().to(device)

    guards = C.current_guards(batch=batch,
                              preflight_report_sha256=rep_sha)
    quarantined = []
    t_start = time.perf_counter()

    for ds in P.ORDER:
        if args.only_ds and ds != args.only_ds:
            continue
        cfg = P.SUBSETS[ds]
        model.update_config(context_len=cfg["context"],
                            pred_len=cfg["pred"],
                            periodicity=cfg["periodicity"],
                            norm_const=P.NORM_CONST,
                            align_const=P.ALIGN_CONST,
                            interpolation=P.INTERPOLATION)
        assert (model.pad_left, model.pad_right, model.num_patch_input) == \
            (P.EXPECTED_GEOM["pad_left"], P.EXPECTED_GEOM["pad_right"],
             P.EXPECTED_GEOM["num_patch_input"]), f"{ds}: geometry drift"
        digest_a = C.model_state_digest(model)
        windows = load_windows(ds)
        Pl = cfg["pred"]
        nsh = P.n_shards(ds)
        print(f"[infer] === {ds}: {len(windows)} windows, {nsh} shards, "
              f"batch={batch}", flush=True)

        done_cnt = 0
        for shard in range(nsh):
            lo = shard * P.SHARD_SIZE
            hi = min(lo + P.SHARD_SIZE, len(windows))
            chunk = windows[lo:hi]
            valid, why = shard_valid(ds, shard, guards)
            if valid:
                done_cnt += 1
                continue
            if os.path.isfile(os.path.join(P.PRED_DIR,
                                           f"{ds}__shard{shard:03d}.done")) or \
               os.path.isfile(os.path.join(P.PRED_DIR,
                                           f"{ds}__shard{shard:03d}.npz")):
                quarantined.append(quarantine(ds, shard, why))
            preds, ms = compute_shard(model, chunk, device, batch, Pl)
            write_shard(ds, shard, chunk, preds, ms, guards)
            done_cnt += 1
            el = time.perf_counter() - t_start
            done_w = done_cnt * P.SHARD_SIZE
            eta_s = el / done_w * (P.EXPECTED_WINDOWS[ds] - done_w)
            print(f"[infer] {ds} shard {shard + 1}/{nsh} done "
                  f"({len(chunk)} windows, {ms.mean():.1f} ms/win) "
                  f"elapsed {el:.0f}s, ETA(this ds) {eta_s / 60:.1f} min",
                  flush=True)

        digest_b = C.model_state_digest(model)
        if digest_a != digest_b:
            sys.exit(f"HARD STOP {ds}: in-memory model state changed "
                     f"during inference")
        print(f"[infer] {ds}: model state unchanged; complete "
              f"({done_cnt}/{nsh} shards)", flush=True)

    w = C.sha256_file(w_path)
    if w != P.WEIGHTS_SHA256:
        sys.exit("HARD STOP: weights file changed during run")
    summary = {"finished": time.strftime("%F %T"),
               "wall_seconds": time.perf_counter() - t_start,
               "batch": batch, "chosen_batch": chosen,
               "preflight_report_sha256": rep_sha,
               "ckpt_load_path": w_path,
               "quarantined": quarantined,
               "weights_sha256": w, "env": C.env_info()}
    C.dump_json(summary, os.path.join(P.LOGS_DIR, "infer_summary.json"))
    print(f"[infer] ALL DONE in {summary['wall_seconds'] / 60:.1f} min; "
          f"quarantined shards: {len(quarantined)}", flush=True)


if __name__ == "__main__":
    main()
