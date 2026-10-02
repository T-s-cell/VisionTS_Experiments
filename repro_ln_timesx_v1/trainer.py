#!/usr/bin/env python3
"""LN fine-tuning trainer (spec secs 5-7) + shared data access.

Core invariants (verified in preflight, enforced here):
- RNG is RE-DERIVED from constant seeds at every epoch start; nothing depends on
  pickled RNG streams, so resume after a crash replays the interrupted epoch
  identically and F1-F4 (same seed) share base window sequences by construction.
- Gradient accumulation: each microbatch backprops (pred_loss_mb + reg_full)/4
  -> effective update loss = pred_mean_32 + reg_full (4 * reg/4 = reg).
- Frozen backbone is never wrapped in no_grad and never replaced by caches;
  only the 84 LayerNorm tensors carry gradients.
- jsonl rows carry attempt_id; an interrupted epoch's rows are voided by
  re-running it under a new attempt (consumers dedupe on max attempt).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ACCUM, ANCHOR_COEF, AUG_P, CHECKPOINTS, CFG, CLIP, CTX,
                    DATA_CACHE, DOMAIN_IDX, DOMAIN_NAMES, EPOCHS, LOGS, LR,
                    METHODS_TRAIN, MICRO, PRED, REPO_ROOT, SEEDS, SPLIT_MANIFEST,
                    STD_EPS, actual_fingerprints, append_jsonl,
                    atomic_torch_save, atomic_write_json, code_fingerprint,
                    environment_snapshot, expected_ln_names, load_model,
                    md5_file, run_id, seed_all, sha256_file, u_d_s_d)
from augmentation import FreqMask

BATCH = 64  # inference batches (matches old infer.py composition)


class ExperimentData:
    def __init__(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "manifests", "split_manifest.json")) as f:
            self.manifest = f.read()
        self.manifest = json.loads(self.manifest)
        self.cache = np.load(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "manifests",
            "data_cache.npz"), allow_pickle=False)
        self._keys = [str(k) for k in self.cache["var_keys"]]
        self._series_off = self.cache["series_off"]
        self._series_len = self.cache["series_len"]
        self._series = self.cache["series"]
        self.domain_vars = {d: sorted(vk for vk in self._keys
                                      if self.manifest["variables"][vk]["domain"] == d)
                            for d in DOMAIN_NAMES}
        self.fb_std = {vk: self.manifest["variables"][vk]["fallback_std"]
                       for vk in self._keys}

    def series(self, vk):
        i = self._keys.index(vk)
        o, n = int(self._series_off[i]), int(self._series_len[i])
        return self._series[o:o + n]

    def window(self, vk, s):
        ser = self.series(vk)
        return ser[s:s + CTX], ser[s + CTX:s + CTX + PRED]

    def native_ids(self, vk, split):
        return self.manifest["variables"][vk]["native"][split]

    def start_of(self, vk, sample_id):
        return self.manifest["variables"][vk]["sample_start_idx"][sample_id]

    def split_window_by_id(self, vk, sample_id):
        return self.window(vk, self.start_of(vk, sample_id))

    def dense_count(self, vk):
        return self.manifest["variables"][vk]["dense_count"]

    def domain_dense_total(self, domain):
        return sum(self.dense_count(vk) for vk in self.domain_vars[domain])

    def split_rows(self, domain, split):
        """[(var_key, sample_id, x, y)] in stored (future-start) order."""
        rows = []
        for vk in self.domain_vars[domain]:
            for sid in self.native_ids(vk, split):
                s = self.start_of(vk, sid)
                x, y = self.window(vk, s)
                rows.append((vk, sid, x, y))
        return rows


def window_d(x, fb_std):
    s = float(np.std(x))
    return s if s >= STD_EPS else fb_std


def build_slots(data, method, domain, seed, epoch):
    """Variable-balanced slots, deterministic per (method-family, seed, epoch).

    Base window indices + slot order are method-INDEPENDENT (identical across
    F1-F4 for one seed; F0 indexes its native pool with the same slot counts).
    """
    assert method in METHODS_TRAIN
    dom_idx = DOMAIN_IDX[domain]
    vars_ = data.domain_vars[domain]
    n_vars = len(vars_)
    U, S = u_d_s_d(data.domain_dense_total(domain))
    per, rem = S // n_vars, S % n_vars
    extra = set(np.random.default_rng(
        [seed, 3000 + dom_idx, epoch]).permutation(n_vars)[:rem].tolist())
    rng_base = np.random.default_rng([seed, 9000 + dom_idx, epoch])
    slots = []
    for i, vk in enumerate(vars_):
        k = per + (1 if i in extra else 0)
        if method == "F0":
            pool_n = len(data.native_ids(vk, "train"))
            ids = data.native_ids(vk, "train")
            pick = rng_base.integers(0, pool_n, size=k)
            slots.extend((vk, ids[j]) for j in pick)
        else:
            pool_n = data.dense_count(vk)
            pick = rng_base.integers(0, pool_n, size=k)
            slots.extend((vk, int(j)) for j in pick)
    rng_shuf = np.random.default_rng([seed, 7000 + dom_idx, epoch])
    order = rng_shuf.permutation(len(slots))
    return [slots[j] for j in order], U, S


def _tensor_batch(xs, device):
    import torch
    return torch.tensor(np.stack(xs), dtype=torch.float32,
                        device=device).unsqueeze(-1)


def evaluate_split(model, data, domain, split, device):
    """Full native windows, eval+no_grad, shuffle=False, drop_last=False.

    Returns dict with var-equal-weight std MSE/MAE + per-var detail."""
    import torch
    rows = data.split_rows(domain, split)
    per_var = {}
    n_done = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i0 in range(0, len(rows), BATCH):
            chunk = rows[i0:i0 + BATCH]
            xt = _tensor_batch([r[2] for r in chunk], device)
            pred = model(xt)[..., 0].detach().cpu().numpy().astype(np.float64)
            for j, (vk, sid, x, y) in enumerate(chunk):
                d = window_d(x, data.fb_std[vk])
                e = (pred[j] - y) / d
                mse = float(np.mean(e ** 2))
                mae = float(np.mean(np.abs(e)))
                pv = per_var.setdefault(vk, {"mse": 0.0, "mae": 0.0, "n": 0})
                pv["mse"] += mse
                pv["mae"] += mae
                pv["n"] += 1
                n_done += 1
    if was_training:
        model.train()
    for pv in per_var.values():
        pv["mse"] /= pv["n"]
        pv["mae"] /= pv["n"]
    return {"split": split, "domain": domain, "n_windows": n_done,
            "val_mse": float(np.mean([pv["mse"] for pv in per_var.values()])),
            "val_mae": float(np.mean([pv["mae"] for pv in per_var.values()])),
            "per_var": per_var}


def _trainable(model):
    names = expected_ln_names()
    named = dict(model.named_parameters())
    trainable = {n for n, p in named.items() if p.requires_grad}
    assert trainable == names, \
        f"trainable set mismatch: extra={sorted(trainable - names)[:3]} " \
        f"missing={sorted(names - trainable)[:3]}"
    return names, named


def _max_attempt(log_path):
    mx = -1
    if os.path.isfile(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                a = r.get("attempt")
                if isinstance(a, int):
                    mx = max(mx, a)
    return mx


def run_training(method, domain, seed, data, device):
    """One run; returns dict status. Refuses to resume on any hash mismatch."""
    import torch
    rid = run_id(method, domain, seed)
    ckpt_dir = os.path.join(CHECKPOINTS, rid)
    os.makedirs(ckpt_dir, exist_ok=True)
    success_path = os.path.join(ckpt_dir, "SUCCESS.json")
    if os.path.isfile(success_path):
        with open(success_path) as f:
            succ = json.load(f)
        # carry the FULL verified metadata so the queue manifest stays complete
        return {**succ, "run_id": rid, "status": "skipped", "skipped": True}
    t0 = time.time()
    seed_all(seed)
    model = load_model(device, finetune_type="ln")
    ln_names, named = _trainable(model)
    theta0 = {n: named[n].detach().clone() for n in ln_names}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR, betas=tuple(CFG["optim"]["betas"]),
                           eps=CFG["optim"]["eps"],
                           weight_decay=CFG["optim"]["weight_decay"])

    # budgets fixed BEFORE any epoch (also used by the resume-finalize path)
    U_d, S_d = u_d_s_d(data.domain_dense_total(domain))

    proto = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "manifests",
        "protocol.json")))
    declared = {"data_cache": proto["data_cache_sha256"],
                "ckpt": proto["ckpt_sha256"],
                "protocol_version": proto["protocol_version"],
                "code": proto["code_fingerprint"]}
    actual = actual_fingerprints()
    if (actual["data_cache"] != declared["data_cache"]
            or actual["code"] != declared["code"]
            or actual["ckpt"] != declared["ckpt"]):
        raise RuntimeError(
            f"{rid}: current code/data/ckpt hashes differ from protocol.json "
            f"({[(k, actual[k] == declared[k]) for k in ('data_cache', 'code', 'ckpt')]}) "
            f"— re-run prepare; refusing to train")
    want = {**declared,
            "split_manifest": actual["split_manifest"],
            "zip": actual["zip"]}

    log_path = os.path.join(LOGS, f"{rid}.jsonl")

    val0 = evaluate_split(model, data, domain, "val", device)
    atomic_torch_save({"ln_state": {n: named[n].detach().cpu()
                                    for n in ln_names}},
                      os.path.join(ckpt_dir, "ep00.ln.pt"))
    atomic_write_json({"epoch": 0, "attempt": 0,
                       **{k: v for k, v in val0.items()}},
                      os.path.join(ckpt_dir, "ep00.val.json"))

    start_epoch = 1
    jsonl_max = _max_attempt(log_path)
    resume_path = os.path.join(ckpt_dir, "resume.pt")
    if os.path.isfile(resume_path):
        rs = torch.load(resume_path, map_location="cpu", weights_only=False)
        got = rs["hashes"]
        if (got != want or rs.get("versions") != environment_snapshot()):
            diff = [k for k in want if got.get(k) != want[k]]
            raise RuntimeError(
                f"{rid}: resume refused (hash/version mismatch in {diff}) — "
                f"manual inspection required, not silently restarted")
        missing = model.load_state_dict(rs["ln_state"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        assert set(missing.missing_keys) <= {n for n in model.state_dict()
                                             if n not in ln_names}
        opt.load_state_dict(rs["opt_state"])
        start_epoch = rs["epoch_completed"] + 1
        attempt = max(rs["attempt"] + 1, jsonl_max + 1)
        print(f"[{rid}] resume at epoch {start_epoch} "
              f"(prev attempt {rs['attempt']}, new attempt {attempt})",
              flush=True)
    else:
        # unique attempt per process start: stale partial-epoch rows from any
        # earlier crash carry a lower attempt and are voided by max-attempt
        # dedupe in consumers
        attempt = jsonl_max + 1

    aug_on = CFG["methods"][method].get("aug", False)
    anchor_on = CFG["methods"][method].get("anchor", False)
    fm = FreqMask(AUG_P, CFG["aug"]["mask_rate"], CFG["aug"]["max_retry"],
                  STD_EPS)
    dom_idx = DOMAIN_IDX[domain]
    torch.cuda.reset_peak_memory_stats(device)

    if start_epoch > EPOCHS:
        print(f"[{rid}] all {EPOCHS} epochs complete in checkpoints; "
              f"finalizing (attempt {attempt})", flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        slots, U, S = build_slots(data, method, domain, seed, epoch)
        assert (U, S) == (U_d, S_d), f"{rid}: budget {(U, S)} != {(U_d, S_d)}"
        fm.reset_stats()
        model.train()
        aug_rng = np.random.default_rng([seed, 5000 + dom_idx, epoch])
        ep_pred, ep_reg, ep_gn = [], [], []
        for u in range(U):
            group = slots[u * ACCUM * MICRO:(u + 1) * ACCUM * MICRO]
            opt.zero_grad(set_to_none=True)
            upd_pred = 0.0
            for mb in range(ACCUM):
                chunk = group[mb * MICRO:(mb + 1) * MICRO]
                xs, ys, ds, tags = [], [], [], []
                for vk, idx in chunk:
                    if method == "F0":
                        s = data.start_of(vk, idx)
                    else:
                        s = idx
                    x, y = data.window(vk, s)
                    tag = f"{vk}__{idx}"
                    if aug_on:
                        x, y, _a = fm(x, y, aug_rng, tag=tag)
                    xs.append(x)
                    ys.append(y)
                    ds.append(window_d(x, data.fb_std[vk]))
                    tags.append(tag)
                xt = _tensor_batch(xs, device)
                yt = torch.tensor(np.stack(ys), dtype=torch.float32,
                                  device=device)
                dt = torch.tensor(np.stack(ds), dtype=torch.float32,
                                  device=device).unsqueeze(1)
                pred = model(xt)[..., 0]
                if not bool(torch.isfinite(pred).all()):
                    raise RuntimeError(
                        f"{rid} epoch {epoch} update {u}: non-finite prediction")
                pred_loss = torch.mean(((pred - yt) / dt) ** 2)
                loss = pred_loss
                reg_val = 0.0
                if anchor_on:
                    reg = ANCHOR_COEF * sum(
                        ((named[n] - theta0[n]) ** 2).sum() for n in ln_names)
                    loss = loss + reg
                    reg_val = float(reg.detach())
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(
                        f"{rid} epoch {epoch} update {u}: non-finite loss")
                (loss / ACCUM).backward()
                upd_pred += float(pred_loss.detach())
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                [named[n] for n in ln_names], CLIP))
            post = float(torch.sqrt(sum(
                (named[n].grad ** 2).sum() for n in ln_names
                if named[n].grad is not None)))
            if not np.isfinite(gnorm) or gnorm == 0.0:
                raise RuntimeError(
                    f"{rid} epoch {epoch} update {u}: abnormal grad norm {gnorm}")
            opt.step()
            ep_pred.append(upd_pred / ACCUM)
            ep_reg.append(reg_val)
            ep_gn.append(gnorm)
            append_jsonl(log_path, {
                "type": "update", "attempt": attempt, "epoch": epoch,
                "update": u, "pred_loss_32": upd_pred / ACCUM,
                "reg": reg_val, "grad_norm_pre_clip": gnorm,
                "grad_norm_post_clip": post,
                "aug_applied": fm.stats["applied"]})
        total_sq = sum(float(((named[n] - theta0[n]) ** 2).sum().detach())
                       for n in ln_names)
        total_n = sum(named[n].numel() for n in ln_names)
        ln_rms = float(np.sqrt(total_sq / total_n))
        val = evaluate_split(model, data, domain, "val", device)
        atomic_torch_save({"ln_state": {n: named[n].detach().cpu()
                                        for n in ln_names}},
                          os.path.join(ckpt_dir, f"ep{epoch:02d}.ln.pt"))
        atomic_write_json({"epoch": epoch, "attempt": attempt,
                           "pred_loss_mean": float(
            np.mean(ep_pred)), "reg_mean": float(np.mean(ep_reg)),
            "grad_norm_mean": float(np.mean(ep_gn)),
            "ln_rms_drift": ln_rms, "presented": len(slots),
            "updates": U, "aug": dict(fm.stats),
            "aug_masked_rate": fm.masked_rate(), **{
                k: v for k, v in val.items()}},
            os.path.join(ckpt_dir, f"ep{epoch:02d}.val.json"))
        append_jsonl(log_path, {
            "type": "epoch_end", "attempt": attempt, "epoch": epoch,
            "pred_loss_mean": float(np.mean(ep_pred)),
            "reg_mean": float(np.mean(ep_reg)),
            "grad_norm_mean": float(np.mean(ep_gn)), "ln_rms_drift": ln_rms,
            "presented": len(slots), "updates": U,
            "val_mse": val["val_mse"], "val_mae": val["val_mae"],
            "aug": dict(fm.stats), "aug_masked_rate": fm.masked_rate()})
        import random as _r
        rs_state = {"ln_state": {n: named[n].detach().cpu() for n in ln_names},
                    "opt_state": _to_cpu(opt.state_dict()),
                    "epoch_completed": epoch, "attempt": attempt,
                    "hashes": want, "versions": environment_snapshot(),
                    "rng_audit_only": {"python": _r.getstate().__repr__()[:256],
                                       "torch": str(
                                           torch.get_rng_state().sum().item())}}
        atomic_torch_save(rs_state, resume_path)
        print(f"[{rid}] epoch {epoch}: val_mse={val['val_mse']:.6f} "
              f"pred_loss={float(np.mean(ep_pred)):.6f} reg={float(np.mean(ep_reg)):.3e} "
              f"({time.time() - t0:.0f}s)", flush=True)

    val_by_epoch = {}
    for e in range(EPOCHS + 1):
        p = os.path.join(ckpt_dir, f"ep{e:02d}.val.json")
        if not os.path.isfile(p):
            raise RuntimeError(f"{rid}: missing {os.path.basename(p)} at "
                               f"finalization")
        with open(p) as f:
            val_by_epoch[e] = json.load(f)["val_mse"]
    best = select_epochs(val_by_epoch, fallback=False)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    info = {"run_id": rid, "status": "success", "method": method,
            "domain": domain, "seed": seed, "wall_s": time.time() - t0,
            "peak_vram_mb": peak_mb, "updates_total": U_d * EPOCHS,
            "presented_total": S_d * EPOCHS, "val_mse_by_epoch": val_by_epoch,
            "U_d": U_d, "S_d": S_d, "local_best_epoch": best["epoch"],
            "finish_attempt": attempt}
    atomic_write_json({"epoch": best["epoch"], "val_mse": best["value"],
                       **info}, success_path)
    return info


def _to_cpu(state):
    out = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv.detach().cpu() if hasattr(vv, "detach") else vv)
                      for kk, vv in v.items()}
        elif hasattr(v, "detach"):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def select_epochs(val_by_epoch, fallback, tie_tol=None):
    """Earliest-epoch argmin with tie tolerance; epoch 0 only when fallback."""
    if tie_tol is None:
        tie_tol = CFG["selection"]["tie_tol"]
    epochs = sorted(val_by_epoch)
    candidates = epochs if fallback else [e for e in epochs if e >= 1]
    best_e = candidates[0]
    best_v = val_by_epoch[best_e]
    for e in candidates[1:]:
        v = val_by_epoch[e]
        if v < best_v - tie_tol * max(1.0, best_v):
            best_e, best_v = e, v
    return {"epoch": best_e, "value": best_v,
            "fallback_used": bool(fallback and best_e == 0)}
