"""LN full-shot fine-tune core (EXP-028): fingerprints, logical-batch
gradient accumulation, training, freeze forensics, checkpoint save/restore.

Loss identity: each microbatch contributes sse_mb/(n_logical*H); summed
over microbatches this is exactly the gradient of full-element MSE over
the logical batch (constant divisor, linear additivity). One optimizer
step per logical batch. Tail logical batch uses its actual sample count.
"""
import hashlib
import json
import time

import numpy as np


def npz_key(name):
    return name.replace(".", "__")


def packed_sha(names, arrays):
    """Content hash over sorted (name, raw bytes) — container-independent."""
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode())
        h.update(arrays[n].tobytes())
    return h.hexdigest()


def train_config(microbatch_final):
    return {
        "optimizer": "adam",
        "loss": "full_element_mse_standardized",
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "epochs": 1,
        "seed": 2021,
        "shuffle_seed": 2021,
        "microbatch": int(microbatch_final),
        "effective_batch": 256,
    }


def group_fingerprint(pinfo, dom, h, tcfg):
    """Everything that must not drift for a cached group to be reusable."""
    from repro_timemmd_common_ln_v1 import protocol as P
    wc = pinfo["splits"][dom]["window_counts"][str(h)]["train"]
    payload = {
        "protocol_name": P.PROTOCOL_NAME,
        "pins": [P.PROTOCOL_JSON_SHA256, P.SPLITS_JSON_SHA256,
                 P.SCALER_JSON_SHA256],
        "weights_sha256": pinfo["weights_sha256"],
        "code_md5": P.package_code_md5s(),
        "repo_commit": P.repo_commit(),
        "domain": dom, "h": int(h), "p": int(P.FIXED_P[dom]),
        "L": int(P.TABLE[dom]["L"]), "n_train": int(wc),
        "expected_steps": P.expected_steps(wc),
        "train": tcfg,
    }
    import torch
    payload["torch"] = torch.__version__
    j = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(j.encode()).hexdigest()


def full_guards(pinfo, sel_sha, microbatch_final):
    """Tier-1 guard set shared by run_train and run_infer (must be
    byte-identical construction on both sides)."""
    from repro_timemmd_common_ln_v1 import protocol as P
    from repro_timemmd_common_ln_v1.common import core_guards
    g = core_guards(pinfo, P.BATCH, sel_sha, None)
    g["ln"] = {
        "microbatch_final": int(microbatch_final),
        "lr": P.LN_LR, "betas": list(P.LN_BETAS), "eps": P.LN_EPS,
        "weight_decay": P.LN_WD, "epochs": P.EPOCHS,
        "seed": P.SEED, "shuffle_seed": P.SEED,
        "expected_ln_tensors": P.EXPECTED_LN_TENSORS,
        "expected_ln_scalars": P.EXPECTED_LN_SCALARS,
        "exp027_selected_p_sha256": P.EXP027_SELECTED_P_SHA256,
        "exp027_overall_mse": P.EXP027_OVERALL_MSE,
        "exp027_overall_mae": P.EXP027_OVERALL_MAE,
        "fixed_p": P.FIXED_P,
        "train_windows_total": P.TRAIN_WINDOWS_TOTAL,
        "train_steps_total": P.TRAIN_STEPS_TOTAL,
    }
    return g


def logical_batch_step(model, X, Y, idx, device, optimizer, microbatch,
                       apply_step=True):
    """One logical batch = zero_grad once -> microbatch backwards -> one
    step. With apply_step=False the accumulated gradients are left in
    place (preflight equivalence checks only). Returns (batch_loss, info).
    Raises SystemExit on missing or non-finite gradients on ANY trainable
    tensor."""
    import torch
    n_logical = len(idx)
    denom = float(n_logical * model.pred_len)
    optimizer.zero_grad(set_to_none=True)
    batch_sse = 0.0
    n_mb = 0
    for lo in range(0, n_logical, microbatch):
        ids = idx[lo:lo + microbatch]
        xb = torch.from_numpy(X[ids]).to(device).unsqueeze(-1)
        yb = torch.from_numpy(Y[ids]).to(device)
        out = model(xb, export_image=False, fp64=False)
        assert out.shape == yb.shape and out.dtype == torch.float32
        se = out - yb
        sse = (se * se).sum()
        (sse / denom).backward()
        batch_sse += float(sse.detach().item())
        n_mb += 1
    g_none, g_bad = [], []
    for n, prm in model.named_parameters():
        if prm.requires_grad:
            if prm.grad is None:
                g_none.append(n)
            elif not bool(torch.isfinite(prm.grad).all()):
                g_bad.append(n)
    if g_none or g_bad:
        raise SystemExit("[TIER-1] grad gate: none=%d nonfinite=%d e.g. %s"
                         % (len(g_none), len(g_bad),
                            (g_none + g_bad)[:4]))
    if apply_step:
        optimizer.step()
    return batch_sse / denom, {"n_microbatches": n_mb,
                               "grad_nonfinite": g_bad, "grad_none": g_none}


def oneshot_logical_batch_grads(model, X, Y, idx, device):
    """Reference path for B3: one full-batch forward/backward, loss =
    full-element MSE. Returns (loss, {name: grad clone}). Zeroes any
    existing gradients first (self-contained)."""
    import torch
    model.zero_grad(set_to_none=True)
    xb = torch.from_numpy(X[idx]).to(device).unsqueeze(-1)
    yb = torch.from_numpy(Y[idx]).to(device)
    out = model(xb, export_image=False, fp64=False)
    loss = ((out - yb) ** 2).mean()
    loss.backward()
    grads = {n: prm.grad.detach().clone() if prm.grad is not None else None
             for n, prm in model.named_parameters() if prm.requires_grad}
    return float(loss.item()), grads


def freeze_check(model, before_shas, before_buffer):
    """Post-update forensics. Hard requirements: no non-LN parameter moved,
    buffer digest unchanged, encoder group AND decoder group each moved at
    least one tensor. Unchanged LN tensors are recorded (not fatal)."""
    from repro_timemmd_common_ln_v1.adapter import ln_param_names
    from repro_timemmd_common_ln_v1.common import buffer_digest, \
        named_tensor_sha
    after_shas = named_tensor_sha(model)
    after_buffer = buffer_digest(model)
    moved = sorted(n for n in after_shas if after_shas[n] != before_shas[n])
    ln_names = set(ln_param_names(model))
    moved_nonln = sorted(set(moved) - ln_names)
    enc = sorted(n for n in moved
                 if n.startswith("vision_model.blocks.")
                 or n.startswith("vision_model.norm."))
    dec = sorted(n for n in moved
                 if n.startswith("vision_model.decoder_blocks.")
                 or n.startswith("vision_model.decoder_norm."))
    buffer_ok = (after_buffer == before_buffer)
    ok = (not moved_nonln) and buffer_ok and bool(enc) and bool(dec)
    return {"ok": ok, "buffer_unchanged": buffer_ok,
            "non_ln_moved": moved_nonln,
            "encoder_moved": enc, "decoder_moved": dec,
            "unchanged_ln": sorted(ln_names - set(moved)),
            "n_moved": len(moved)}


def save_ln_ckpt(model, names, npz_path, meta_path, meta_base):
    """npz keys are name with dots -> '__'; meta json binds gid/dom/h/p/
    weights/protocol pins + npz file sha + packed content sha."""
    from repro_timemmd_common_ln_v1.common import dump_json, sha256_of
    ps = dict(model.named_parameters())
    arrays = {npz_key(n): ps[n].detach().cpu().numpy().copy() for n in names}
    np.savez(str(npz_path), ln_names=np.asarray(names),
             **{k: arrays[k] for k in sorted(arrays)})
    meta = dict(meta_base)
    meta.update({
        "ln_names": list(names),
        "n_tensors": len(names),
        "n_scalars": int(sum(arrays[npz_key(n)].size for n in names)),
        "packed_sha256": packed_sha(names,
                                    {n: arrays[npz_key(n)]
                                     for n in names}),
        "npz_sha256": sha256_of(npz_path),
    })
    dump_json(meta_path, meta)
    return meta


def restore_ln_ckpt_strict(model, npz_path, meta_path, pinfo, dom, h):
    """Independent-process restore. Order: npz FILE sha256 -> meta binding
    checks (domain/H/P/L/protocol pins/weights sha/model geometry/name set
    exactly equal to this model's LN inventory) -> packed content hash from
    npz bytes -> per-name shape/dtype + copy_ -> recompute packed sha from
    the LIVE model. Any mismatch is SystemExit."""
    import torch
    from repro_timemmd_common_ln_v1 import protocol as P
    from repro_timemmd_common_ln_v1.adapter import ln_param_names
    from repro_timemmd_common_ln_v1.common import die, load_json, \
        sha256_of
    meta = load_json(meta_path)
    got = sha256_of(npz_path)
    if got != meta.get("npz_sha256"):
        die("[TIER-1] ckpt npz sha mismatch for %s: file %s != meta %s"
            % (meta.get("gid"), got, meta.get("npz_sha256")))
    p = int(P.FIXED_P[dom])
    L = int(P.TABLE[dom]["L"])
    if (meta.get("domain") != dom or int(meta.get("h", -1)) != int(h)
            or int(meta.get("p", -1)) != p or int(meta.get("L", -1)) != L):
        die("[TIER-1] ckpt meta binding mismatch %s: dom=%s h=%s p=%s L=%s"
            % (meta.get("gid"), meta.get("domain"), meta.get("h"),
               meta.get("p"), meta.get("L")))
    if meta.get("weights_sha256") != pinfo["weights_sha256"]:
        die("[TIER-1] ckpt base weights mismatch for %s" % meta.get("gid"))
    for mk, pk in (("protocol_name", "PROTOCOL_NAME"),
                   ("protocol_json_sha256", "PROTOCOL_JSON_SHA256"),
                   ("splits_json_sha256", "SPLITS_JSON_SHA256"),
                   ("scaler_json_sha256", "SCALER_JSON_SHA256")):
        if meta.get(mk) != getattr(P, pk):
            die("[TIER-1] ckpt protocol pin mismatch (%s) for %s"
                % (mk, meta.get("gid")))
    if (model.context_len != L or model.pred_len != int(h)
            or model.periodicity != p):
        die("[TIER-1] model geometry mismatch at restore for %s"
            % meta.get("gid"))
    names = ln_param_names(model)
    if sorted(meta.get("ln_names") or []) != names:
        die("[TIER-1] ckpt ln_names != model LN inventory for %s"
            % meta.get("gid"))
    if (meta.get("n_tensors") != P.EXPECTED_LN_TENSORS
            or meta.get("n_scalars") != P.EXPECTED_LN_SCALARS):
        die("[TIER-1] ckpt LN counts != 84/55,808 for %s" % meta.get("gid"))
    z = np.load(str(npz_path))
    hsh = hashlib.sha256()
    arrays = {}
    for n in names:
        a = z[npz_key(n)]
        hsh.update(n.encode())
        hsh.update(a.tobytes())
        arrays[n] = a
    if hsh.hexdigest() != meta.get("packed_sha256"):
        die("[TIER-1] ckpt packed content sha mismatch for %s"
            % meta.get("gid"))
    ps = dict(model.named_parameters())
    for n in names:
        t = arrays[n]
        prm = ps[n]
        if tuple(prm.shape) != tuple(t.shape):
            die("[TIER-1] shape mismatch %s for %s" % (n, meta.get("gid")))
        if prm.detach().cpu().numpy().dtype != t.dtype:
            die("[TIER-1] dtype mismatch %s for %s" % (n, meta.get("gid")))
        with torch.no_grad():
            prm.copy_(torch.from_numpy(t).to(prm.device))
    live = packed_sha(names, {n: ps[n].detach().cpu().numpy()
                              for n in names})
    if live != meta.get("packed_sha256"):
        die("[TIER-1] live packed sha mismatch after restore for %s"
            % meta.get("gid"))
    return meta


def train_one_group(pinfo, dom, h, out_dir, microbatch_final=None,
                    device="cuda:0"):
    """Full 1-epoch LN training for one group. Returns result dict (also
    suitable as train_log content). Validates freeze forensics and writes
    ckpt npz + meta + val npz."""
    import torch
    from repro_timemmd_common_ln_v1 import protocol as P
    from repro_timemmd_common_ln_v1.adapter import build_visionts, \
        configure, freeze_to_ln, predict_split
    from repro_timemmd_common_ln_v1.common import buffer_digest, die, \
        named_tensor_sha, seed_everything, setup_deterministic_env
    from repro_timemmd_common_ln_v1.data_timemmd import DomainData

    setup_deterministic_env()
    seed_everything(P.SEED)
    mb = int(microbatch_final or P.MICROBATCH)
    p = int(P.FIXED_P[dom])
    L = int(P.TABLE[dom]["L"])
    tcfg = train_config(mb)
    dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], dom)
    sp = dd.load_split(h, "train")
    X, Y = sp["X"], sp["Y"]
    n_train = X.shape[0]
    want_steps = P.expected_steps(n_train)

    model = build_visionts().to(device)
    configure(model, L, h, p)
    names = freeze_to_ln(model)
    optimizer = torch.optim.Adam(
        [prm for prm in model.parameters() if prm.requires_grad],
        lr=tcfg["lr"], betas=tuple(tcfg["betas"]), eps=tcfg["eps"],
        weight_decay=tcfg["weight_decay"])
    perm = np.random.RandomState(P.SEED).permutation(n_train)
    perm_sha = hashlib.sha256(
        perm.astype(np.int64).tobytes()).hexdigest()

    before_shas = named_tensor_sha(model)
    before_buf = buffer_digest(model)
    digest_before = None
    from repro_timemmd_common_ln_v1.common import model_state_digest
    digest_before = model_state_digest(model)

    steps = 0
    losses = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for lo in range(0, n_train, P.BATCH):
        idx = perm[lo:min(lo + P.BATCH, n_train)]
        bl, info = logical_batch_step(model, X, Y, idx, device, optimizer,
                                      mb)
        steps += 1
        losses.append(bl)
    elapsed = time.time() - t0
    peak_mem_mb = int(torch.cuda.max_memory_allocated() >> 20)
    if steps != want_steps:
        die("[TIER-1] %s steps %d != expected %d"
            % (P.group_id(dom, h), steps, want_steps))

    chk = freeze_check(model, before_shas, before_buf)
    if not chk["ok"]:
        die("[TIER-1] %s freeze forensics: non_ln_moved=%s "
            "buffer_ok=%s enc=%d dec=%d"
            % (P.group_id(dom, h), chk["non_ln_moved"][:3],
               chk["buffer_unchanged"], len(chk["encoder_moved"]),
               len(chk["decoder_moved"])))
    digest_after = model_state_digest(model)
    if digest_after == digest_before:
        die("[TIER-1] %s LN training produced no parameter change"
            % P.group_id(dom, h))

    gid = P.group_id(dom, h)
    npz_path = out_dir / P.ckpt_npz_name(dom, h)
    meta_path = out_dir / P.ckpt_meta_name(dom, h)
    meta_base = {
        "gid": gid, "domain": dom, "h": int(h), "p": p, "L": L,
        "protocol_name": P.PROTOCOL_NAME,
        "protocol_json_sha256": P.PROTOCOL_JSON_SHA256,
        "splits_json_sha256": P.SPLITS_JSON_SHA256,
        "scaler_json_sha256": P.SCALER_JSON_SHA256,
        "weights_sha256": pinfo["weights_sha256"],
        "train_config": tcfg,
        "steps": steps, "n_train_windows": int(n_train),
        "perm_sha256": perm_sha,
        "final_train_loss": losses[-1] if losses else None,
        "mean_train_loss": float(np.mean(losses)) if losses else None,
        "first_train_loss": losses[0] if losses else None,
        "freeze_check": chk,
        "digest_before": digest_before, "digest_after": digest_after,
        "elapsed_s": round(elapsed, 3), "peak_mem_mb": peak_mem_mb,
    }
    meta = save_ln_ckpt(model, names, npz_path, meta_path, meta_base)

    spv = dd.load_split(h, "val")
    pred = predict_split(model, spv["X"], device, batch=P.BATCH)
    d64 = pred.astype(np.float64) - spv["Y"].astype(np.float64)
    val_mse = float(np.mean(d64 * d64))
    val_mae = float(np.mean(np.abs(d64)))
    np.savez(str(out_dir / P.val_npz_name(dom, h)),
             pred=pred.astype(np.float32),
             truth=spv["Y"].astype(np.float32),
             window_index=spv["window_index"])
    del model, optimizer
    import torch as _t
    _t.cuda.empty_cache()

    return {"gid": gid, "domain": dom, "h": int(h), "p": p,
            "train_config": tcfg, "steps": steps,
            "n_train_windows": int(n_train), "perm_sha256": perm_sha,
            "losses": [float(x) for x in losses],
            "final_train_loss": losses[-1] if losses else None,
            "mean_train_loss": float(np.mean(losses)) if losses else None,
            "val_mse": val_mse, "val_mae": val_mae,
            "freeze_check": chk,
            "digest_before": digest_before, "digest_after": digest_after,
            "ckpt_npz": str(npz_path),
            "ckpt_npz_sha256": meta["npz_sha256"],
            "packed_sha256": meta["packed_sha256"],
            "elapsed_s": round(elapsed, 3), "peak_mem_mb": peak_mem_mb}
