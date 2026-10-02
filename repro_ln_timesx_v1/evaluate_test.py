#!/usr/bin/env python3
"""Stage test (runs ONLY after freeze_selection): compute Z0 / N0 on the NEW
2,474-window test list and evaluate every frozen best_ft on its own domain's
test windows. Saves per-window predictions; metrics live in aggregate.py.

- Z0: fresh pretrained model, finetune_type='none' (exactly the old zero-shot
  conditions: TF32 both off, fp32, bs=64, P=1). Deterministic single pass.
- N0: last history value repeated 12 steps (pure numpy).
- F-runs: pristine 'ln' model + selected epoch's 84 LN tensors (strict name
  check). The fallback variant REUSES Z0 predictions (no third model).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CHECKPOINTS, LOGS, PREDICTIONS, RESULTS, append_jsonl,
                    sha256_file,
                    atomic_write_json, expected_ln_names, load_model, run_id)
from trainer import BATCH, ExperimentData, window_d


def predict_windows(model, data, domain, split, device):
    """Returns sample_ids, var_keys, pred[N,12] float64 — native order, bs=64."""
    rows = data.split_rows(domain, split)
    preds, sids, vks = [], [], []
    import torch
    was = model.training
    model.eval()
    with torch.no_grad():
        for i0 in range(0, len(rows), BATCH):
            chunk = rows[i0:i0 + BATCH]
            xt = torch.tensor(np.stack([r[2] for r in chunk]),
                              dtype=torch.float32,
                              device=device).unsqueeze(-1)
            out = model(xt)[..., 0].detach().cpu().numpy().astype(np.float64)
            preds.append(out)
            sids.extend(r[1] for r in chunk)
            vks.extend(r[0] for r in chunk)
    if was:
        model.train()
    return sids, vks, np.concatenate(preds)


def save_npz(path, sids, vks, domains, pred, target, d, extra):
    np.savez_compressed(path, **{
        "sample_ids": np.array(sids), "var_keys": np.array(vks),
        "domains": np.array(domains), "pred": pred, "target": target,
        "d": d, **extra})


def main():
    sel_path = os.path.join(RESULTS, "selection.json")
    if not os.path.isfile(sel_path):
        print("[test] selection.json missing — run freeze_selection first")
        sys.exit(1)
    sel = json.load(open(sel_path))
    if sel["n_runs"] != 285:
        print(f"[test] selection.json has {sel['n_runs']} runs, expected 285")
        sys.exit(1)
    # environment must still match the frozen protocol before any test loads
    from common import actual_fingerprints
    proto = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "manifests",
        "protocol.json")))
    act = actual_fingerprints()
    if (act["data_cache"] != proto["data_cache_sha256"]
            or act["code"] != proto["code_fingerprint"]
            or act["ckpt"] != proto["ckpt_sha256"]):
        print("[test] environment drift vs protocol.json — rerun prepare; "
              "refusing to evaluate")
        sys.exit(1)
    data = ExperimentData()
    device = "cuda"

    log_path = os.path.join(LOGS, "stage_test.jsonl")

    # ---- Z0 (all 19 domains, one model) ----
    z0_path = os.path.join(PREDICTIONS, "Z0__test.npz")
    if not os.path.isfile(z0_path):
        model = load_model(device, finetune_type="none")
        sids, vks, preds, tgts, ds, dms = [], [], [], [], [], []
        for d in sorted(data.domain_vars):
            s, v, p = predict_windows(model, data, d, "test", device)
            for j, vk in enumerate(v):
                x, y = data.split_window_by_id(vk, s[j])
                tgts.append(y)
                ds.append(window_d(x, data.fb_std[vk]))
                dms.append(d)
            sids.extend(s)
            vks.extend(v)
            preds.append(p)
        del model
        import torch
        torch.cuda.empty_cache()
        save_npz(z0_path, sids, vks, dms, np.concatenate(preds),
                 np.array(tgts), np.array(ds), {"method": "Z0", "seed": -1,
                 "epoch": 0})
        print(f"[test] Z0 done: {len(sids)} windows", flush=True)
    else:
        print("[test] Z0 npz exists; reuse")

    # ---- N0 ----
    n0_path = os.path.join(PREDICTIONS, "N0__test.npz")
    if not os.path.isfile(n0_path):
        sids, vks, preds, tgts, ds, dms = [], [], [], [], [], []
        for d in sorted(data.domain_vars):
            rows = data.split_rows(d, "test")
            for vk, sid, x, y in rows:
                preds.append(np.repeat(x[-1], 12))
                sids.append(sid)
                vks.append(vk)
                tgts.append(y)
                ds.append(window_d(x, data.fb_std[vk]))
                dms.append(d)
        save_npz(n0_path, sids, vks, dms, np.array(preds), np.array(tgts),
                 np.array(ds), {"method": "N0", "seed": -1, "epoch": 0})
        print(f"[test] N0 done: {len(sids)} windows", flush=True)
    else:
        print("[test] N0 npz exists; reuse")

    # ---- 285 frozen runs ----
    todo = {rid: r for rid, r in sel["per_run"].items()
            if not os.path.isfile(os.path.join(
                PREDICTIONS, f"{rid}__test.npz"))}
    print(f"[test] {len(sel['per_run'])} runs, {len(todo)} to evaluate")
    model = None
    model_ln = expected_ln_names()
    done = 0
    for rid, r in sorted(todo.items()):
        if model is None:
            model = load_model(device, finetune_type="ln")
        ck = os.path.join(CHECKPOINTS, rid, f"ep{r['best_epoch']:02d}.ln.pt")
        if sha256_file(ck) != r["selected_ckpt_sha256"]:
            print(f"[test] {rid}: selected ckpt hash drift vs "
                  f"selection.json — refusing")
            sys.exit(1)
        sd = torch_load(ck)["ln_state"]
        assert set(sd) == model_ln, "selected checkpoint key mismatch"
        missing = model.load_state_dict(sd, strict=False)
        assert not missing.unexpected_keys
        d = r["domain"]
        s, v, p = predict_windows(model, data, d, "test", device)
        tgts, ds = [], []
        for j, vk in enumerate(v):
            x, y = data.split_window_by_id(vk, s[j])
            tgts.append(y)
            ds.append(window_d(x, data.fb_std[vk]))
        save_npz(os.path.join(PREDICTIONS, f"{rid}__test.npz"),
                 s, v, [d] * len(s), p, np.array(tgts), np.array(ds),
                 {"method": r["method"], "seed": r["seed"],
                  "epoch": r["best_epoch"]})
        done += 1
        if done % 15 == 0 or done == len(todo):
            append_jsonl(log_path, {"done": done, "of": len(todo),
                                    "last": rid})
            print(f"[test] {done}/{len(todo)} ({rid})", flush=True)
    print("[test] complete")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    main()
