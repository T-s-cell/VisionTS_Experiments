#!/usr/bin/env python3
"""Stage preflight (spec secs 9.2, 10) on REAL Currency (largest) + arts
(smallest) samples. All checks must pass before the full queue starts.

Sections:
  A data/manifest integrity (CPU)      B LN trainable set + gradients (GPU)
  C initial-prediction equivalence     D old-cache prediction consistency
  E FreqMask behaviour                 F accumulation equivalence
  G budgets & F1-F4 shared sequences   H full val/test coverage
  I timing + VRAM + total estimate     J interrupt->resume consistency
Float equality uses DECLARED allclose tolerances with recorded max error
(config preflight_tolerances); frozen-parameter invariance stays bitwise.
"""
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import (ACCUM, AUDIT, CFG, DOMAIN_NAMES, MICRO, PREFLIGHT_JSON,
                    atomic_write_json, environment_snapshot,
                    expected_ln_names, load_model, u_d_s_d)
import trainer
from trainer import (ExperimentData, build_slots, evaluate_split, run_training,
                     select_epochs, window_d)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"name": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


class _PreflightInterrupt(Exception):
    pass


def section_a(data):
    """Index-based split-boundary checks, mirroring prepare.assign_split:
    train: future_end < T_train; test: future_start >= T_test;
    val: future_start >= T_train and future_end < T_test (window may start
    before T_train). future = last 12 of the 108-step window."""
    m = data.manifest
    ok = check("A1 totals == spec", m["totals"] == {
        "native_train": 4041, "dense": 36763, "val": 895, "test": 2474,
        "excluded": 696}, m["totals"])
    ok &= check("A2 190 vars / 19 domains",
                len(m["variables"]) == 190
                and sorted(m["domains"]) == sorted(DOMAIN_NAMES))
    bad_order, bad_pools, bad_fb, bad_starts = [], [], [], []
    WIN = CFG["data"]["ctx"] + CFG["data"]["pred"]
    for vk, v in m["variables"].items():
        b_tr = v["n_train_obs"]
        n = v["n"]
        starts = v["sample_start_idx"]
        # index of first ts >= T_test: test starts are contiguous, the earliest
        # test window begins exactly 96 steps before the boundary
        b_te = min(starts[sid] for sid in v["native"]["test"]) \
            + CFG["data"]["ctx"]
        for sid in v["native"]["train"]:
            s = starts[sid]
            if not (s >= 0 and s + WIN <= b_tr):
                bad_order.append((vk, sid, "train window crosses T_train"))
        for sid in v["native"]["val"]:
            s = starts[sid]
            if not (s + CFG["data"]["ctx"] >= b_tr and s + WIN <= b_te):
                bad_order.append((vk, sid, "val window out of [T_train,T_test)"))
        for sid in v["native"]["test"]:
            s = starts[sid]
            if not (s + CFG["data"]["ctx"] >= b_te and s + WIN <= n):
                bad_order.append((vk, sid, "test window crosses T_test"))
        for sp in ("train", "val", "test"):
            if not v["native"][sp]:
                bad_pools.append((vk, sp))
        if v["fallback_std"] < CFG["data"]["std_eps"]:
            bad_fb.append(vk)
        for sid, s in starts.items():
            x, y = data.window(vk, s)
            if len(x) != 96 or len(y) != 12 or not np.isfinite(
                    np.concatenate([x, y])).all():
                bad_starts.append((vk, sid))
    ok &= check("A3 domain boundary strictness", not bad_order, bad_order[:3])
    ok &= check("A4 all vars non-empty train/val/test", not bad_pools,
                bad_pools[:3])
    ok &= check("A5 fallback_std non-degenerate", not bad_fb, bad_fb[:3])
    ok &= check("A6 windows finite + located", not bad_starts, bad_starts[:3])
    return ok


def section_b(device):
    import torch
    model = load_model(device, finetune_type="ln")
    names = expected_ln_names()
    named = dict(model.named_parameters())
    trainable = {n for n, p in named.items() if p.requires_grad}
    n_scal = sum(named[n].numel() for n in trainable)
    ok = check("B1 trainable == 84 LN tensors", trainable == names,
               f"{len(trainable)} tensors")
    ok &= check("B2 trainable scalars == 55808", n_scal == 55808, n_scal)
    ok &= check("B3 no pos/patch/attn/mlp trainable", not any(
        ("embed" in n or "attn" in n or "mlp" in n or "cls" in n or "proj" in n)
        for n in trainable))

    data = ExperimentData()
    slots, U, S = build_slots(data, "F1", "Currency", 2021, 1)
    xs, ys, ds = [], [], []
    for vk, idx in slots[:8]:
        x, y = data.window(vk, idx)
        xs.append(x)
        ys.append(y)
        ds.append(window_d(x, data.fb_std[vk]))
    xt = torch.tensor(np.stack(xs), dtype=torch.float32,
                      device=device).unsqueeze(-1)
    yt = torch.tensor(np.stack(ys), dtype=torch.float32, device=device)
    dt = torch.tensor(np.array(ds), dtype=torch.float32,
                      device=device).unsqueeze(1)
    # train() vs eval() identical forward (no dropout)
    model.eval()
    with torch.no_grad():
        p_eval = model(xt)
    model.train()
    with torch.no_grad():
        p_train = model(xt)
    ok &= check("B4 train()==eval() forward bitwise",
                torch.equal(p_eval, p_train),
                f"max|d|={float((p_eval - p_train).abs().max()):.2e}")
    # LN model vs frozen model initial predictions (declared tol)
    model_frozen = load_model(device, finetune_type="none")
    model_frozen.eval()
    with torch.no_grad():
        p_none = model_frozen(xt)
    tol = CFG["preflight_tolerances"]
    md = float((p_train - p_none).abs().max())
    ok &= check("C1 ln-model == zero-shot model predictions (declared tol)",
                np.allclose(p_train.cpu(), p_none.cpu(), rtol=tol["init_pred_rtol"],
                            atol=tol["init_pred_atol"]),
                f"max|d|={md:.3e}")

    # real backward under training-identical loss denominators (window_d)
    model.zero_grad(set_to_none=True)
    pred = model(xt)[..., 0]
    loss = torch.mean(((pred - yt) / dt) ** 2)
    loss.backward()
    bad_train = [n for n in names if named[n].grad is None
                 or not bool(named[n].grad.any())
                 or not bool(torch.isfinite(named[n].grad).all())]
    bad_frozen = [n for n, p in named.items()
                  if n not in names and p.grad is not None]
    ok &= check("B5 all 84 LN grads finite & nonzero", not bad_train,
                bad_train[:3])
    ok &= check("B6 frozen grads None", not bad_frozen, bad_frozen[:3])

    watched = ["vision_model.blocks.0.attn.qkv.weight",
               "vision_model.blocks.0.attn.proj.weight",
               "vision_model.blocks.0.mlp.fc1.weight",
               "vision_model.cls_token", "vision_model.pos_embed",
               "vision_model.decoder_pos_embed",
               "vision_model.patch_embed.proj.weight"]
    before = {n: named[n].detach().clone() for n in watched}
    ln_before = {n: named[n].detach().clone() for n in names}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=CFG["optim"]["lr"],
                           betas=tuple(CFG["optim"]["betas"]),
                           eps=CFG["optim"]["eps"], weight_decay=0.0)
    gnorm = torch.nn.utils.clip_grad_norm_(
        [named[n] for n in names], CFG["optim"]["clip_grad"])
    post = float(torch.sqrt(sum((named[n].grad ** 2).sum() for n in names)))
    ok &= check("B7 clip: post<=1.0", post <= 1.0 + 1e-6,
                f"pre={float(gnorm):.3f} post={post:.6f}")
    opt.step()
    moved = max(float((ln_before[n] - named[n].detach()).abs().max())
                for n in names)
    frozen_same = all(bool(torch.equal(before[n], named[n].detach()))
                      for n in watched)
    ok &= check("B8 step changes LN", moved > 0, f"max|delta|={moved:.3e}")
    ok &= check("B9 frozen params bitwise unchanged", frozen_same)
    del model, model_frozen
    torch.cuda.empty_cache()
    return ok


def section_d(data, device):
    """Old-cache consistency over ALL cached ids: predictions under old infer
    conditions (TF32 off, bs=64, stored order) + targets bitwise. This also
    guards the data cache: any series misalignment shows up as target
    mismatches."""
    import torch
    tol = CFG["preflight_tolerances"]
    model = load_model(device, finetune_type="none")
    worst_rel, worst_abs, n_cmp, true_mism = 0.0, 0.0, 0, 0
    n_ids = 0
    cache_dir = os.path.join(common.ZERO_DIR, "cache")
    for domain in DOMAIN_NAMES:
        for vk in data.domain_vars[domain]:
            npz = os.path.join(cache_dir, f"{vk}__P1.npz")
            if not os.path.isfile(npz):
                continue
            with np.load(npz, allow_pickle=False) as z:
                ids = [str(x) for x in z["sample_ids"]]
                c_pred = z["pred"]
                c_true = z["true"]
            n_ids += len(ids)
            for i0 in range(0, len(ids), 64):
                chunk = ids[i0:i0 + 64]
                xs = [data.window(vk, data.start_of(vk, sid))[0]
                      for sid in chunk]
                xt = torch.tensor(np.stack(xs), dtype=torch.float32,
                                  device=device).unsqueeze(-1)
                with torch.no_grad():
                    p = model(xt)[..., 0].cpu().numpy().astype(np.float64)
                ref = c_pred[i0:i0 + len(chunk)]
                scale = np.maximum(np.abs(ref), 1e-8)
                worst_rel = max(worst_rel, float(
                    np.max(np.abs(p - ref) / scale)))
                worst_abs = max(worst_abs, float(np.max(np.abs(p - ref))))
                if not np.array_equal(c_true[i0:i0 + len(chunk)],
                                      np.stack([data.window(
                                          vk, data.start_of(vk, sid))[1]
                                          for sid in chunk])):
                    true_mism += 1
                n_cmp += len(chunk)
    del model
    torch.cuda.empty_cache()
    ok = check("D1 old-cache predictions within rtol/atol",
               worst_rel <= tol["old_cache_rtol"]
               and worst_abs <= tol["old_cache_atol"],
               f"n={n_cmp} worst_rel={worst_rel:.3e} "
               f"worst_abs={worst_abs:.3e}")
    ok &= check("D2 old-cache targets bitwise equal", true_mism == 0,
                f"mismatched chunks={true_mism}")
    ok &= check("D3 cache coverage == 6413 ids", n_ids == 6413, n_ids)
    return ok


def section_e(data):
    from augmentation import FreqMask, self_test
    ok = check("E1 FreqMask self-test", all(p for _n, p in self_test()))
    fm = FreqMask(1.0, 0.10, 3, CFG["data"]["std_eps"])
    rng = np.random.default_rng(123)
    n_aug, n_d, bad_shape = 0, [], []
    for vk in data.domain_vars["Currency"][:3]:
        for s in range(0, 40, 7):
            x, y = data.window(vk, s)
            xa, ya, aug = fm(x, y, rng)
            if xa.shape != x.shape or ya.shape != y.shape:
                bad_shape.append(vk)
            if aug:
                n_aug += 1
                n_d.append(window_d(xa, data.fb_std[vk]))
    ok &= check("E2 augmented shapes preserved", not bad_shape, bad_shape[:3])
    ok &= check("E3 augmented d from augmented input (finite)",
                n_aug > 0 and all(np.isfinite(n_d)), f"applied={n_aug}")
    return ok


def section_f(data, device):
    """Accumulation equivalence under the REAL training semantics: 4
    microbatches of 8 windows each backprop (loss_mb + reg_full)/4 vs a
    single 32-window pass (mean32 + reg_full). LN params are PERTURBED off
    theta0 so the anchor term contributes a nonzero gradient (audit fix)."""
    import torch
    tol = CFG["preflight_tolerances"]
    assert (ACCUM, MICRO) == (4, 8), (ACCUM, MICRO)
    model = load_model(device, finetune_type="ln")
    names = expected_ln_names()
    named = dict(model.named_parameters())
    theta0 = {n: named[n].detach().clone() for n in names}
    gen = torch.Generator(device="cpu").manual_seed(42)
    with torch.no_grad():
        for n in names:
            named[n].add_(torch.randn(
                named[n].shape, generator=gen).to(device) * 0.01)
    offset = max(float((named[n].detach() - theta0[n]).abs().max())
                 for n in names)
    slots, _U, _S = build_slots(data, "F1", "Currency", 2021, 1)
    groups = [slots[i * MICRO:(i + 1) * MICRO] for i in range(ACCUM)]

    def batch_loss(chunk):
        xs, ys, ds = [], [], []
        for vk, idx in chunk:
            x, y = data.window(vk, idx)
            xs.append(x)
            ys.append(y)
            ds.append(window_d(x, data.fb_std[vk]))
        xt = torch.tensor(np.stack(xs), dtype=torch.float32,
                          device=device).unsqueeze(-1)
        yt = torch.tensor(np.stack(ys), dtype=torch.float32, device=device)
        dt = torch.tensor(np.array(ds), dtype=torch.float32,
                          device=device).unsqueeze(1)
        pred = model(xt)[..., 0]
        return torch.mean(((pred - yt) / dt) ** 2)

    def reg():
        return CFG["anchor"]["coef"] * sum(
            ((named[n] - theta0[n]) ** 2).sum() for n in names)

    # anchor gradient alone must be nonzero once off init
    model.zero_grad(set_to_none=True)
    reg().backward()
    reg_gnorm = float(torch.sqrt(sum(
        (named[n].grad ** 2).sum() for n in names)))
    ok = check("F1 anchor gradient nonzero off-init", reg_gnorm > 0.0,
               f"|grad_reg|={reg_gnorm:.3e} max|theta-theta0|={offset:.3e}")

    # path A: 4 microbatches, each (loss_mb + reg_full)/4
    model.zero_grad(set_to_none=True)
    for g in groups:
        (batch_loss(g) + reg()).div(ACCUM).backward()
    ga = {n: named[n].grad.detach().clone() for n in names}

    # path B: one 32-window pass, (mean32 + reg_full)
    model.zero_grad(set_to_none=True)
    ((batch_loss([it for g in groups for it in g])) + reg()).backward()
    max_err = max(float((ga[n] - named[n].grad).abs().max()) for n in names)
    ok &= check("F2 4x8 (mb+reg)/4 == 32 (mean+reg) grads (declared tol)",
                all(np.allclose(ga[n].cpu(), named[n].grad.cpu(),
                                rtol=tol["accum_grad_rtol"],
                                atol=tol["accum_grad_atol"])
                    for n in names),
                f"max|dgrad|={max_err:.3e}")

    # no loss inflation: sum of mb/4 == mean32 (prediction-only grads)
    model.zero_grad(set_to_none=True)
    for g in groups:
        (batch_loss(g) / ACCUM).backward()
    ga2 = {n: named[n].grad.detach().clone() for n in names}
    model.zero_grad(set_to_none=True)
    (batch_loss([it for g in groups for it in g])).backward()
    gb = {n: named[n].grad.detach().clone() for n in names}
    max_err2 = max(float((ga2[n] - gb[n]).abs().max()) for n in names)
    ok &= check("F3 no loss inflation (sum mb/4 == mean32 grads, tol)",
                np.allclose(
                    np.concatenate([ga2[n].cpu().ravel() for n in names]),
                    np.concatenate([gb[n].cpu().ravel() for n in names]),
                    rtol=tol["accum_grad_rtol"],
                    atol=tol["accum_grad_atol"]),
                f"max|dgrad|={max_err2:.3e}")
    model.zero_grad(set_to_none=True)
    del model
    torch.cuda.empty_cache()
    return ok


def section_g(data):
    ok = True
    for domain in ("Currency", "arts"):
        U0, S0 = None, None
        base_by_seed = {}
        for seed in (2021, 2022, 2023):
            ref_slots = None
            for m in ("F0", "F1", "F2", "F3", "F4"):
                slots, U, S = build_slots(data, m, domain, seed, 3)
                if U0 is None:
                    U0, S0 = U, S
                if (U, S) != (U0, S0):
                    ok &= check(f"G1 {domain} {m} budget differs", False,
                                f"{(U, S)} != {(U0, S0)}")
                if m != "F0" and ref_slots is None:
                    ref_slots = slots
                if m != "F0" and slots != ref_slots:
                    ok &= check(f"G2 {domain} seed{seed} {m} base seq differs",
                                False, "F1-F4 must share base sequences")
            base_by_seed[seed] = ref_slots
        # remainder rotation varies by epoch but is deterministic
        s1, _U, _S = build_slots(data, "F1", domain, 2021, 3)
        s1b, _U, _S = build_slots(data, "F1", domain, 2021, 3)
        s2, _U, _S = build_slots(data, "F1", domain, 2021, 4)
        ok &= check(f"G3 {domain} slots deterministic on re-derivation",
                    s1 == s1b)
        if s1 == s2:
            ok &= check(f"G4 {domain} epoch rotation actually rotates", False,
                        "epoch 3 and 4 slots identical (unexpected)")
    # F0 slots valid native sample ids
    for domain in ("Currency", "arts"):
        slots, _U, _S = build_slots(data, "F0", domain, 2021, 1)
        bad = [sid for vk, sid in slots
               if sid not in data.native_ids(vk, "train")]
        ok &= check(f"G5 {domain} F0 slots from native train pool",
                    not bad, bad[:3])
    return ok


def section_h(data):
    n_val = sum(len(v["native"]["val"]) for v in data.manifest["variables"]
                .values())
    n_test = sum(len(v["native"]["test"]) for v in
                 data.manifest["variables"].values())
    ok = check("H1 val windows == 895", n_val == 895, n_val)
    ok &= check("H2 test windows == 2474", n_test == 2474, n_test)
    rows_test = sum(len(data.split_rows(d, "test")) for d in DOMAIN_NAMES)
    ok &= check("H3 split_rows covers all test windows", rows_test == 2474,
                rows_test)
    return ok


def section_i(data, device):
    import torch
    model = load_model(device, finetune_type="ln")
    names = expected_ln_names()
    named = dict(model.named_parameters())
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=CFG["optim"]["lr"],
                           betas=tuple(CFG["optim"]["betas"]),
                           eps=CFG["optim"]["eps"], weight_decay=0.0)

    def one_update(domain, method="F1"):
        slots, _U, _S = build_slots(data, method, domain, 2021, 1)
        opt.zero_grad(set_to_none=True)
        for mb in range(ACCUM):
            chunk = slots[mb * MICRO:(mb + 1) * MICRO]
            xs, ys, ds = [], [], []
            for vk, idx in chunk:
                x, y = data.window(vk, idx)
                xs.append(x)
                ys.append(y)
                ds.append(window_d(x, data.fb_std[vk]))
            xt = torch.tensor(np.stack(xs), dtype=torch.float32,
                              device=device).unsqueeze(-1)
            yt = torch.tensor(np.stack(ys), dtype=torch.float32,
                              device=device)
            dt = torch.tensor(np.array(ds), dtype=torch.float32,
                              device=device).unsqueeze(1)
            pred = model(xt)[..., 0]
            loss = torch.mean(((pred - yt) / dt) ** 2)
            (loss / ACCUM).backward()
        torch.nn.utils.clip_grad_norm_([named[n] for n in names],
                                       CFG["optim"]["clip_grad"])
        opt.step()

    timing = {}
    torch.cuda.reset_peak_memory_stats(device)
    for domain in ("arts", "Currency"):
        for _ in range(3):
            one_update(domain)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_meas = 10
        for _ in range(n_meas):
            one_update(domain)
        torch.cuda.synchronize()
        per_update = (time.perf_counter() - t0) / n_meas
        # full val epoch timing
        t0 = time.perf_counter()
        evaluate_split(model, data, domain, "val", device)
        val_s = time.perf_counter() - t0
        n_val = len(data.split_rows(domain, "val"))
        timing[domain] = {"s_per_update": per_update,
                          "val_s": val_s, "n_val_windows": n_val,
                          "s_per_val_window": val_s / n_val}
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    total_updates = 173550
    s_per_update = max(t["s_per_update"] for t in timing.values())
    s_per_valwin = float(np.mean([t["s_per_val_window"]
                                  for t in timing.values()]))
    val_windows_total = 895 * 15 * 11  # per-domain val x 15 runs x 11 epochs
    train_s = total_updates * s_per_update
    val_s = val_windows_total * s_per_valwin
    test_windows = 285 * 130 + 2474 * 2
    test_s = test_windows * s_per_valwin
    est_h = (train_s + val_s + test_s) / 3600
    free_mb = torch.cuda.mem_get_info(device)[0] / 1024 / 1024
    gate_needed = peak_mb + 3072
    ok = check("I1 VRAM gate satisfiable now", free_mb >= gate_needed,
               f"peak={peak_mb:.0f}MiB need<={gate_needed:.0f} "
               f"free={free_mb}MiB")
    est = {"s_per_update": s_per_update, "peak_vram_mb": peak_mb,
           "train_h": train_s / 3600, "val_h": val_s / 3600,
           "test_h": test_s / 3600, "total_h_est": est_h,
           "timing_by_domain": timing, "free_mb_now": free_mb,
           "gate_required_mb": gate_needed}
    print(f"[preflight] estimate: {est_h:.1f} h total "
          f"(train {train_s/3600:.1f} + val {val_s/3600:.1f} + "
          f"test {test_s/3600:.1f}); {s_per_update:.3f} s/update, "
          f"peak {peak_mb:.0f} MiB", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return ok, est


def section_j(data, device):
    """Real interrupt -> resume consistency on arts F1 (2 epochs, monkeypatched
    checkpoint/log dirs; nothing touches the real artifacts)."""
    import torch
    tol = CFG["preflight_tolerances"]
    base_ckpt, base_logs = trainer.CHECKPOINTS, trainer.LOGS
    old_epochs = trainer.EPOCHS
    trainer.EPOCHS = 2
    tmp_root = os.path.join(AUDIT, "preflight_resume")
    shutil.rmtree(tmp_root, ignore_errors=True)
    paths = {tag: (os.path.join(tmp_root, tag, "ckpt"),
                   os.path.join(tmp_root, tag, "logs"))
             for tag in ("cont", "intr")}
    try:
        # arts is the smallest domain: interrupt mid-epoch-2 must land inside
        # the epoch, so derive updates/epoch from the actual budget.
        u_arts, _s_arts = u_d_s_d(data.domain_dense_total("arts"))
        stop_at = u_arts + max(1, u_arts // 2)  # fires within epoch 2
        results = {}
        for tag, stop in (("cont", None), ("intr", stop_at)):
            ck, lg = paths[tag]
            os.makedirs(ck, exist_ok=True)
            os.makedirs(lg, exist_ok=True)
            trainer.CHECKPOINTS, trainer.LOGS = ck, lg
            if stop is None:
                run_training("F1", "arts", 2021, data, device)
            else:
                # interrupt mid-epoch-2: real step first, then raise after
                # `stop` total updates (otherwise nothing would train)
                real_step = torch.optim.Adam.step

                def counting_step(self, _c=[0], _stop=stop):
                    real_step(self)
                    _c[0] += 1
                    if _c[0] >= _stop:
                        raise _PreflightInterrupt()
                torch.optim.Adam.step = counting_step
                try:
                    run_training("F1", "arts", 2021, data, device)
                    raise AssertionError("interrupt did not fire")
                except _PreflightInterrupt:
                    pass
                finally:
                    torch.optim.Adam.step = real_step
                # resume: replay epoch 2 under attempt 1
                run_training("F1", "arts", 2021, data, device)

        def rows(tag, epoch, attempt=None):
            p = os.path.join(paths[tag][1], "F1__arts__s2021.jsonl")
            out = {}
            for line in open(p):
                r = json.loads(line)
                if r.get("type") == "update" and r["epoch"] == epoch:
                    if attempt is None or r["attempt"] == attempt:
                        out[r["update"]] = r["pred_loss_32"]
            return out

        c1, c2 = rows("cont", 1), rows("cont", 2)
        i1, i2 = rows("intr", 1), rows("intr", 2, attempt=1)
        stale = rows("intr", 2, attempt=0)
        ok = check("J1 epoch-1 replay identical (interrupted path)",
                   set(c1) == set(i1) and all(
                       np.isclose(c1[k], i1[k], rtol=tol["resume_replay_rtol"],
                                  atol=tol["resume_replay_atol"])
                       for k in c1),
                   f"{len(c1)} updates")
        ok &= check("J2 epoch-2 replay after resume == continuous",
                    set(c2) == set(i2) and all(
                        np.isclose(c2[k], i2[k],
                                   rtol=tol["resume_replay_rtol"],
                                   atol=tol["resume_replay_atol"])
                        for k in c2),
                    f"{len(c2)} updates")
        ok &= check("J3 stale interrupted-epoch rows voided by attempt",
                    len(stale) < len(c2) and set(stale) <= set(c2),
                    f"stale={len(stale)} complete={len(c2)}")
        v_cont = json.load(open(os.path.join(
            paths["cont"][0], "F1__arts__s2021", "ep02.val.json")))["val_mse"]
        v_intr = json.load(open(os.path.join(
            paths["intr"][0], "F1__arts__s2021", "ep02.val.json")))["val_mse"]
        ok &= check("J4 resumed epoch-2 val MSE == continuous",
                    np.isclose(v_cont, v_intr, rtol=1e-6, atol=1e-8),
                    f"{v_cont:.8f} vs {v_intr:.8f}")
    finally:
        trainer.CHECKPOINTS, trainer.LOGS = base_ckpt, base_logs
        trainer.EPOCHS = old_epochs
        shutil.rmtree(tmp_root, ignore_errors=True)
    return ok


def section_k():
    """Aggregate acceptance simulation (post-audit): 19 real domains x 10
    synthetic vars x 3 seeds x 5 methods, domain i has pred=i / target=0 /
    d=1 -> closed-form overall MSE=mean(i^2, i=1..19)=130, MAE=10, seed
    std=0; Z0 pred=1 -> F0-Z0 delta=129. Guards must raise."""
    import aggregate as agg
    i_by_dom = {d: common.DOMAIN_IDX[d] + 1 for d in common.DOMAIN_NAMES}

    def fake_var_table(d, pred):
        return {f"{d}__v{j}": {"mse": float(pred ** 2), "mae": float(pred),
                               "raw_mse": float(pred ** 2),
                               "raw_mae": float(pred), "n": 3}
                for j in range(10)}

    f_tables = {}  # (m, seed) -> 19 domain tables
    for m in ("F0", "F1", "F2", "F3", "F4"):
        for s in (2021, 2022, 2023):
            f_tables[(m, s)] = [fake_var_table(d, i_by_dom[d])
                                for d in common.DOMAIN_NAMES]
    z0_table = {vk: {"mse": 1.0, "mae": 1.0, "raw_mse": 1.0, "raw_mae": 1.0,
                     "n": 3}
                for d in common.DOMAIN_NAMES for vk in
                (f"{d}__v{j}" for j in range(10))}

    seed_ov, seed_merged = {}, {}
    for key, tables in f_tables.items():
        seed_merged[key] = agg.merge_var_tables(tables)
        _dt, ov = agg.dom_overall(seed_merged[key])
        seed_ov[key] = ov
    _zd, z0_ov = agg.dom_overall(z0_table)

    ok = check("K1 merge == 190 vars no dups", len(seed_merged[("F0", 2021)])
               == 190, len(seed_merged[("F0", 2021)]))
    mse = float(np.mean([seed_ov[("F0", s)]["mse"] for s in (2021, 2022, 2023)]))
    mae = float(np.mean([seed_ov[("F0", s)]["mae"] for s in (2021, 2022, 2023)]))
    ok &= check("K2 overall MSE == 130 / MAE == 10",
                abs(mse - 130.0) < 1e-9 and abs(mae - 10.0) < 1e-9,
                f"MSE={mse} MAE={mae}")
    mean, std = agg.seed_stats([seed_ov[("F0", s)]["mse"]
                                for s in (2021, 2022, 2023)])
    ok &= check("K3 seed std == 0 for identical seeds",
                abs(mean - 130.0) < 1e-9 and std == 0.0, f"std={std}")
    delta = mse - z0_ov["mse"]
    ok &= check("K4 F0-Z0 delta == 129", abs(delta - 129.0) < 1e-9, delta)

    def raises(fn):
        try:
            fn()
        except agg.AggregateError:
            return True
        except Exception:
            return False
        return False

    ok &= check("K5 duplicate var raises",
                raises(lambda: agg.merge_var_tables(
                    [fake_var_table("arts", 1), fake_var_table("arts", 2)])),
                "")
    one_dom = agg.merge_var_tables([fake_var_table(d, i_by_dom[d])
                                    for d in common.DOMAIN_NAMES[:5]],
                                   strict=False)
    ok &= check("K6 missing domains raise (strict)",
                raises(lambda: agg.dom_overall(one_dom)), "")
    bad = {vk: dict(v) for vk, v in seed_merged[("F0", 2021)].items()}
    first = next(iter(bad))
    bad[first] = {**bad[first], "mse": float("nan")}
    ok &= check("K7 non-finite raises",
                raises(lambda: agg.dom_overall(bad)), "")
    return ok


def section_l(data, device):
    """Four resume-boundary scenarios (post-audit acceptance) on arts F1,
    EPOCHS=2, monkeypatched dirs; nothing touches real artifacts.
    1 first-epoch interrupt  2 same-epoch double interrupt
    3 crash after final checkpoint, before SUCCESS  4 SUCCESS-skip metadata."""
    import torch
    tol = CFG["preflight_tolerances"]
    base = (trainer.CHECKPOINTS, trainer.LOGS, trainer.EPOCHS)
    trainer.EPOCHS = 2
    tmp_root = os.path.join(AUDIT, "preflight_resume_l")
    shutil.rmtree(tmp_root, ignore_errors=True)
    real_step = torch.optim.Adam.step
    try:
        u_arts, _s_arts = u_d_s_d(data.domain_dense_total("arts"))

        def attempt(tag):
            """(run_training under patches) -> 'done'|'interrupted'."""
            ck = os.path.join(tmp_root, tag, "ckpt")
            lg = os.path.join(tmp_root, tag, "logs")
            os.makedirs(ck, exist_ok=True)
            os.makedirs(lg, exist_ok=True)
            trainer.CHECKPOINTS, trainer.LOGS = ck, lg

            def run(stop_after=None, crash_on_success=False):
                awj = trainer.atomic_write_json
                state = {"n": 0}

                def counting_step(self):
                    real_step(self)
                    state["n"] += 1
                    if stop_after is not None and state["n"] >= stop_after:
                        raise _PreflightInterrupt()
                torch.optim.Adam.step = counting_step
                if crash_on_success:
                    def crashing(obj, path):
                        if path.endswith("SUCCESS.json"):
                            raise _PreflightInterrupt()
                        return awj(obj, path)
                    trainer.atomic_write_json = crashing
                try:
                    run_training("F1", "arts", 2021, data, device)
                    return "done"
                except _PreflightInterrupt:
                    return "interrupted"
                finally:
                    torch.optim.Adam.step = real_step
                    trainer.atomic_write_json = awj
            return run

        def rows(tag, epoch, attempt=None):
            p = os.path.join(tmp_root, tag, "logs", "F1__arts__s2021.jsonl")
            out = {}
            if not os.path.isfile(p):
                return out
            for line in open(p):
                r = json.loads(line)
                if r.get("type") == "update" and r["epoch"] == epoch:
                    if attempt is None or r["attempt"] == attempt:
                        out[r["update"]] = r["pred_loss_32"]
            return out

        def best_of(tag):
            p = os.path.join(tmp_root, tag, "ckpt", "F1__arts__s2021",
                             "SUCCESS.json")
            with open(p) as f:
                return json.load(f)

        def close(a, b):
            return set(a) == set(b) and all(
                np.isclose(a[k], b[k], rtol=tol["resume_replay_rtol"],
                           atol=tol["resume_replay_atol"]) for k in a)

        # continuous reference
        cont = attempt("cont")
        cont()
        succ_c = best_of("cont")

        # scenario 1: interrupt inside epoch 1, then resume to completion
        s1 = attempt("s1")
        r1 = s1(stop_after=max(1, u_arts // 2))
        ok = check("L1a first-epoch interrupt fired", r1 == "interrupted", r1)
        r1b = s1()
        ok &= check("L1b resume after first-epoch interrupt completes",
                    r1b == "done", r1b)
        ok &= check("L1c epoch-1 replay == continuous (max-attempt dedupe)",
                    close(rows("s1", 1), rows("cont", 1)),
                    f"{len(rows('s1', 1))} vs {len(rows('cont', 1))}")
        ok &= check("L1d epoch-2 == continuous",
                    close(rows("s1", 2), rows("cont", 2)), "")

        # scenario 2: two interrupts inside epoch 2, then resume
        s2 = attempt("s2")
        s2(stop_after=u_arts + max(1, u_arts // 3))       # attempt 0, epoch 2
        s2(stop_after=max(1, u_arts // 3))                # attempt 1, epoch 2
        r2 = s2()                                         # attempt 2 completes
        ok &= check("L2a double-interrupt resume completes", r2 == "done", r2)
        e2_att = sorted({json.loads(line)["attempt"] for line in
                         open(os.path.join(tmp_root, "s2", "logs",
                                           "F1__arts__s2021.jsonl"))
                         if json.loads(line).get("epoch") == 2
                         and json.loads(line).get("type") == "update"})
        ok &= check("L2b epoch-2 attempts unique & increasing",
                    e2_att == [0, 1, 2], e2_att)
        ok &= check("L2c epoch-2 max-attempt replay == continuous",
                    close(rows("s2", 2), rows("cont", 2)),
                    f"{len(rows('s2', 2))} updates")

        # scenario 3: crash AFTER final resume.pt, BEFORE SUCCESS.json
        s3 = attempt("s3")
        r3 = s3(crash_on_success=True)
        ok &= check("L3a crash-before-SUCCESS fired", r3 == "interrupted", r3)
        succ_missing = not os.path.isfile(os.path.join(
            tmp_root, "s3", "ckpt", "F1__arts__s2021", "SUCCESS.json"))
        r3b = s3()  # finalize path: start_epoch > EPOCHS
        ok &= check("L3b finalize-after-final-checkpoint completes",
                    r3b == "done" and succ_missing, f"{r3b}")
        succ_3 = best_of("s3")
        ok &= check("L3c finalized best epoch == continuous",
                    succ_3["local_best_epoch"] == succ_c["local_best_epoch"],
                    f"{succ_3['local_best_epoch']} vs "
                    f"{succ_c['local_best_epoch']}")

        # scenario 4: SUCCESS present -> skip returns full metadata
        trainer.CHECKPOINTS, trainer.LOGS = (
            os.path.join(tmp_root, "s3", "ckpt"),
            os.path.join(tmp_root, "s3", "logs"))
        info = run_training("F1", "arts", 2021, data, device)
        ok &= check("L4 SUCCESS-skip returns full metadata",
                    info.get("skipped") is True
                    and info.get("status") == "skipped"
                    and info.get("local_best_epoch")
                    == succ_c["local_best_epoch"]
                    and info.get("U_d") == u_arts and "S_d" in info
                    and "val_mse_by_epoch" in info,
                    {k: info.get(k) for k in
                     ("status", "skipped", "local_best_epoch", "U_d")})

        # stale partial rows must never win over the completed attempt
        stale = rows("s1", 1, attempt=0)
        ok &= check("L5 stale partial-epoch rows voided",
                    len(stale) < len(rows("cont", 1))
                    and set(stale) <= set(rows("cont", 1)),
                    f"stale={len(stale)}")
    finally:
        trainer.CHECKPOINTS, trainer.LOGS, trainer.EPOCHS = base
        torch.optim.Adam.step = real_step
        shutil.rmtree(tmp_root, ignore_errors=True)
    return ok


def section_m():
    """Fingerprint binding: recomputed hashes must equal protocol.json, and
    the values embedded in THIS preflight report are what queue/audit pin."""
    from common import PROTOCOL_JSON, actual_fingerprints
    act = actual_fingerprints()
    proto = json.load(open(PROTOCOL_JSON))
    ok = check("M1 data_cache == protocol",
               act["data_cache"] == proto["data_cache_sha256"])
    ok &= check("M2 ckpt == protocol",
                act["ckpt"] == proto.get("ckpt_sha256"),
                "ckpt file present" if act["ckpt"] else "ckpt MISSING")
    ok &= check("M3 split_manifest == protocol",
                act["split_manifest"] == proto.get("split_manifest_sha256"))
    ok &= check("M4 code fingerprint == protocol",
                act["code"] == proto["code_fingerprint"],
                f"{len(act['code'])} files")
    ok &= check("M5 zip == protocol",
                act["zip"] == proto.get("zip_sha256"))
    return ok, act


def main():
    device = "cuda"
    data = ExperimentData()
    ok_m, act_fp = section_m()
    ok_k = section_k()
    ok_a = section_a(data)
    ok_b = section_b(device)
    ok_d = section_d(data, device)
    ok_e = section_e(data)
    ok_f = section_f(data, device)
    ok_g = section_g(data)
    ok_h = section_h(data)
    ok_i, est = section_i(data, device)
    ok_j = section_j(data, device)
    ok_l = section_l(data, device)
    all_pass = all([ok_a, ok_b, ok_d, ok_e, ok_f, ok_g, ok_h, ok_i, ok_j,
                    ok_k, ok_l, ok_m])
    report = {"environment": environment_snapshot(),
              "fingerprints": {k: act_fp[k]
                               for k in ("code", "data_cache",
                                         "split_manifest", "zip", "ckpt")},
              "checks": CHECKS, "all_pass": all_pass, "peak_vram_mb":
              est["peak_vram_mb"], "timing": est["timing_by_domain"],
              "estimate": {k: v for k, v in est.items()
                           if k != "timing_by_domain"},
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    atomic_write_json(report, PREFLIGHT_JSON)
    n_fail = sum(1 for c in CHECKS if not c["pass"])
    print(f"[preflight] {'ALL PASS' if all_pass else 'FAILED'} "
          f"({len(CHECKS) - n_fail}/{len(CHECKS)} checks)")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
