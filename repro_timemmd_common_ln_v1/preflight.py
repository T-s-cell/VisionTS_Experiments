"""Preflight gates B1-B5 for EXP-028 (LN full-shot fine-tune).

Runs before ANY formal training; artifacts live in out_root/preflight/.
run_train.py live-checks this report's all_pass (the report sha is NOT
pinned in guards — EXP-027 lesson).

  B1 protocol/data : verify_frozen (pins, 51 files, weights, selected_p,
                     totals, 236 steps) + per-group test/val loads with
                     three-way test identity/truth (protocol recompute AND
                     TaTS reference npz AND EXP-027 runs npz, all bitwise).
  B2 entry equiva. : all 36 geometries, fixed probe batch (first 32 val
                     windows): grad entry == this package's inference entry
                     AND grad entry == the FROZEN EXP-027 entry (cross-
                     package import of repro_timemmd_common_v1.adapter) at
                     atol=1e-6/rtol=1e-5.
  B3 logical batch : PRECHECK groups; accumulation equivalence (32=4x8 vs
                     one-shot 32 on every precheck group; 256=8x32 vs
                     one-shot 256 where n_train>=256) — loss rel<1e-5,
                     grads rtol 1e-3 atol 1e-8 — then ONE real
                     logical_batch_step on the first shuffled logical
                     batch (grads exist+finite on all 84; encoder+decoder
                     groups actually move; non-LN + buffers bitwise
                     unchanged). Security/H12 exercises the ragged tail.
  B4 ckpt restore  : save the B3-updated Health/H48 model -> fresh
                     subprocess (restore_probe.py) strict restore -> first
                     8 val windows pred_close(1e-6,1e-5) + live digest
                     equality across processes.
  B5 budget/VRAM   : measured per-logical-batch wall time + peak VRAM from
                     B3 -> extrapolation for 236 steps; microbatch fallback
                     only inside preflight if OOM (recorded as
                     microbatch_final; effective batch stays 256, FP32).

Exit 0 iff all_pass; exit 2 otherwise (report written either way).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from repro_timemmd_common_ln_v1 import protocol as P
from repro_timemmd_common_ln_v1.adapter import build_visionts, configure, \
    forward_batch, forward_batch_grad, freeze_to_ln, ln_param_names
from repro_timemmd_common_ln_v1.common import bitwise_equal, buffer_digest, \
    dump_json, identity_equal, load_json, named_tensor_sha, pred_close, \
    seed_everything, setup_deterministic_env
from repro_timemmd_common_ln_v1.data_timemmd import DomainData
from repro_timemmd_common_ln_v1.ln_train import freeze_check, \
    logical_batch_step, oneshot_logical_batch_grads, save_ln_ckpt, \
    train_config

DEVICE = "cuda:0"
ATOL, RTOL = 1e-6, 1e-5
PROBE_N = 32


def torch_from(x):
    import torch
    return torch.from_numpy(x).to(DEVICE).unsqueeze(-1)


def _tats_npz(gid):
    ref = P.tats_ref_dir()
    for cand in (ref / gid / "test_arrays.npz",
                 ref / ("%s_seed2021" % gid) / "test_arrays.npz"):
        if cand.exists():
            return cand
    return None


def _exp027_test_npz(dom, h):
    return P.exp027_out() / "runs" / P.group_id(dom, h) / \
        P.test_npz_name(dom, h)


def b1_protocol_data(pinfo):
    errs, details = [], {}
    domcache = {}
    for dom, h in P.all_groups():
        gid = P.group_id(dom, h)
        if dom not in domcache:
            domcache[dom] = DomainData(pinfo["protocol_dir"],
                                       pinfo["splits"], dom)
        dd = domcache[dom]
        sp = dd.load_split(h, "test")
        spv = dd.load_split(h, "val")
        d = {"test_windows": int(sp["X"].shape[0]),
             "val_windows": int(spv["X"].shape[0])}
        e27_path = _exp027_test_npz(dom, h)
        if not e27_path.exists():
            errs.append("%s: EXP-027 test npz missing" % gid)
            details[gid] = d
            continue
        a27 = dict(np.load(str(e27_path)))
        for k in ("s", "orig_row", "label_dates", "window_index"):
            if not identity_equal(sp[k], a27[k]):
                errs.append("%s: identity %s != EXP-027 npz" % (gid, k))
        if not bitwise_equal(sp["Y"], a27["truth"]):
            errs.append("%s: truth != EXP-027 npz" % gid)
        tats = _tats_npz(gid)
        d["tats_npz"] = str(tats) if tats else None
        if tats is None:
            errs.append("%s: TaTS reference npz missing" % gid)
        else:
            t = dict(np.load(str(tats)))
            for k in ("truth", "s", "orig_row", "label_dates"):
                if k not in t:
                    errs.append("%s: TaTS npz missing %s" % (gid, k))
                    continue
                a = sp["Y"] if k == "truth" else sp[k]
                b = t[k]
                if a.ndim == 3 and b.ndim == 2:
                    a = a[:, :, 0]
                if b.ndim == 3 and a.ndim == 2:
                    b = b[:, :, 0]
                if not bitwise_equal(np.asarray(a), np.asarray(b)):
                    errs.append("%s: %s != TaTS reference" % (gid, k))
        details[gid] = d
    return errs, details


def b2_entry_equivalence(pinfo):
    """36 geometries x probe batch: grad entry vs own inference entry AND
    vs frozen EXP-027 entry."""
    import torch
    from repro_timemmd_common_v1 import adapter as A27
    errs, rows = [], []
    m_g = build_visionts().to(DEVICE)
    m_27 = A27.build_visionts().to(DEVICE)
    domcache = {}
    for dom, h in P.all_groups():
        gid = P.group_id(dom, h)
        if dom not in domcache:
            domcache[dom] = DomainData(pinfo["protocol_dir"],
                                       pinfo["splits"], dom)
        spv = domcache[dom].load_split(h, "val")
        n = min(PROBE_N, spv["X"].shape[0])
        configure(m_g, int(P.TABLE[dom]["L"]), h, int(P.FIXED_P[dom]))
        A27.configure(m_27, int(P.TABLE[dom]["L"]), h, int(P.FIXED_P[dom]))
        xb = torch.from_numpy(np.ascontiguousarray(spv["X"][:n])) \
            .to(DEVICE).unsqueeze(-1)
        y_grad = forward_batch_grad(m_g, xb).detach().cpu().numpy()
        y_inf = forward_batch(m_g, xb).cpu().numpy()
        y_27 = A27.forward_batch(m_27, xb).cpu().numpy()
        c1 = pred_close(y_grad, y_inf, ATOL, RTOL)
        c2 = pred_close(y_grad, y_27, ATOL, RTOL)
        rows.append({"gid": gid, "n": int(n),
                     "vs_own_inference": {"close": c1[0], "max_abs": c1[1],
                                          "max_viol": c1[2]},
                     "vs_exp027_entry": {"close": c2[0], "max_abs": c2[1],
                                         "max_viol": c2[2]}})
        if not c1[0]:
            errs.append("%s: grad vs own inference viol=%.3e"
                        % (gid, c1[2]))
        if not c2[0]:
            errs.append("%s: grad vs EXP-027 entry viol=%.3e"
                        % (gid, c2[2]))
        print("B2 %s own=%s e027=%s max_abs=%.2e"
              % (gid, c1[0], c2[0], max(c1[1], c2[1])), flush=True)
    del m_g, m_27
    torch.cuda.empty_cache()
    return errs, rows


def _adam(m):
    import torch
    return torch.optim.Adam(
        [prm for prm in m.parameters() if prm.requires_grad],
        lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def _collect_ln_grads(m):
    return {n: prm.grad.detach().clone()
            for n, prm in m.named_parameters() if prm.requires_grad}


def _grad_close(ga, gb, names, rel_l2_tol=1e-5):
    """Per-tensor relative-L2 comparison. Rationale: microbatch
    accumulation vs one-shot differ only in f32 summation order; for
    near-zero gradient elements the absolute noise is set by the largest
    cancelling contributions (eps*scale), so elementwise |a-b| <= atol +
    rtol*|b| with a fixed atol false-fails (observed ~1.7e-7 on elements
    of magnitude ~1e-4). rel-L2 = ||a-b||/||b|| is the invariant that
    quantifies summation-order noise (f64 unit tests pin the exact math
    at 1e-12). Returns (all_close, worst_rel_l2, max_abs_diff)."""
    worst = 0.0
    max_abs = 0.0
    for n in names:
        a, b = ga.get(n), gb.get(n)
        if a is None or b is None:
            return False, float("inf"), float("inf")
        d = (a.double() - b.double())
        denom = float(b.double().norm().item())
        rel = float(d.norm().item()) / denom if denom > 0 else \
            float(d.abs().max().item())
        worst = max(worst, rel)
        max_abs = max(max_abs, float(d.abs().max().item()))
    return bool(worst <= rel_l2_tol), worst, max_abs


def _fresh_model(dom, h):
    import torch
    seed_everything(P.SEED)
    m = build_visionts().to(DEVICE)
    configure(m, int(P.TABLE[dom]["L"]), h, int(P.FIXED_P[dom]))
    freeze_to_ln(m)
    return m


def _drop(m):
    import torch
    del m
    torch.cuda.empty_cache()


def b3_logical_batch(pinfo, pd):
    errs, rows = [], {}
    for dom, h in P.PRECHECK:
        gid = P.group_id(dom, h)
        dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], dom)
        sp = dd.load_split(h, "train")
        X, Y = sp["X"], sp["Y"]
        n_train = X.shape[0]
        names = None
        row = {"gid": gid, "n_train": int(n_train)}

        # ---- equivalence checks on pristine models ----
        m = _fresh_model(dom, h)
        names = ln_param_names(m)
        perm = np.random.RandomState(P.SEED).permutation(n_train)
        idx1 = perm[:min(P.BATCH, n_train)]

        # small-batch: 32 = 4x8 vs one-shot 32 (ragged when n_train < 32)
        idx_small = idx1[:min(32, n_train)]
        _, g_one_s = oneshot_logical_batch_grads(m, X, Y, idx_small, DEVICE)
        opt_s = _adam(m)
        _, info_s = logical_batch_step(m, X, Y, idx_small, DEVICE, opt_s,
                                       8, apply_step=False)
        g_acc_s = _collect_ln_grads(m)
        m.zero_grad(set_to_none=True)
        c_s, wl_s, ma_s = _grad_close(g_acc_s, g_one_s, names)
        row["small_4x8"] = {"n": int(len(idx_small)), "close": c_s,
                            "worst_rel_l2": wl_s, "max_abs_diff": ma_s}
        if not c_s:
            errs.append("%s: 4x8 accumulation != one-shot "
                        "(worst rel-L2 %.3e, max_abs %.3e)"
                        % (gid, wl_s, ma_s))

        # full logical batch: 256 = 8x32 vs one-shot 256 (n_train >= 256)
        if n_train >= P.BATCH:
            loss_a, info_a = logical_batch_step(
                m, X, Y, perm[:P.BATCH], DEVICE, _adam(m),
                int(P.MICROBATCH), apply_step=False)
            g_acc = _collect_ln_grads(m)
            m.zero_grad(set_to_none=True)
            _drop(m)
            m = _fresh_model(dom, h)
            loss_o, g_one = oneshot_logical_batch_grads(m, X, Y,
                                                        perm[:P.BATCH],
                                                        DEVICE)
            rel = abs(loss_a - loss_o) / max(abs(loss_o), 1e-30)
            c_g, wl_g, ma_g = _grad_close(g_acc, g_one, names)
            row["batch256"] = {"loss_accum": loss_a, "loss_oneshot": loss_o,
                               "loss_rel_diff": rel, "grad_close": c_g,
                               "grad_worst_rel_l2": wl_g,
                               "grad_max_abs_diff": ma_g,
                               "n_microbatches": info_a["n_microbatches"]}
            if rel >= 1e-5:
                errs.append("%s: loss rel diff %.3e >= 1e-5" % (gid, rel))
            if not c_g:
                errs.append("%s: batch grads not close (worst rel-L2 "
                            "%.3e, max_abs %.3e)" % (gid, wl_g, ma_g))

        # ---- ONE real logical batch step + freeze forensics ----
        before_shas = named_tensor_sha(m)
        before_buf = buffer_digest(m)
        import torch
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        loss, info = logical_batch_step(m, X, Y, idx1, DEVICE, _adam(m),
                                        int(P.MICROBATCH))
        dt = time.time() - t0
        peak = int(torch.cuda.max_memory_allocated() >> 20)
        chk = freeze_check(m, before_shas, before_buf)
        row["step"] = {"loss": loss,
                       "n_microbatches": info["n_microbatches"],
                       "wall_s": round(dt, 3), "peak_mem_mb": peak,
                       "freeze_ok": chk["ok"], "n_moved": chk["n_moved"],
                       "n_encoder_moved": len(chk["encoder_moved"]),
                       "n_decoder_moved": len(chk["decoder_moved"]),
                       "n_unchanged_ln": len(chk["unchanged_ln"]),
                       "non_ln_moved": chk["non_ln_moved"],
                       "buffer_unchanged": chk["buffer_unchanged"],
                       "unchanged_ln": chk["unchanged_ln"]}
        if not chk["ok"]:
            errs.append("%s: freeze forensics failed: non_ln=%s buf=%s "
                        "enc=%d dec=%d"
                        % (gid, chk["non_ln_moved"][:2],
                           chk["buffer_unchanged"],
                           len(chk["encoder_moved"]),
                           len(chk["decoder_moved"])))
        rows[gid] = row
        print("B3 %s loss=%.6f moved=%d (enc=%d dec=%d) unchanged_ln=%d "
              "%.2fs %dMB" % (gid, loss, chk["n_moved"],
                              len(chk["encoder_moved"]),
                              len(chk["decoder_moved"]),
                              len(chk["unchanged_ln"]), dt, peak),
              flush=True)

        # keep the updated Health/H48 model (after its real step) for B4
        if (dom, h) == ("Health", 48):
            meta_base = {
                "gid": gid + "__preflight", "domain": dom, "h": int(h),
                "p": int(P.FIXED_P[dom]), "L": int(P.TABLE[dom]["L"]),
                "protocol_name": P.PROTOCOL_NAME,
                "protocol_json_sha256": P.PROTOCOL_JSON_SHA256,
                "splits_json_sha256": P.SPLITS_JSON_SHA256,
                "scaler_json_sha256": P.SCALER_JSON_SHA256,
                "weights_sha256": pinfo["weights_sha256"],
                "train_config": train_config(int(P.MICROBATCH)),
                "steps": 1, "n_train_windows": int(n_train),
                "note": "preflight B3 model after one logical batch step",
            }
            save_ln_ckpt(m, ln_param_names(m),
                         pd / "b4_health_H48.npz",
                         pd / "b4_health_H48.meta.json", meta_base)
        _drop(m)

    b4 = b4_restore(pinfo, pd)
    b5 = b5_budget(rows)
    return errs, rows, b4, b5


def b4_restore(pinfo, pd):
    probe_npz = pd / "b4_probe_pred.npz"
    cmd = [sys.executable, "-m",
           "repro_timemmd_common_ln_v1.restore_probe",
           "--dom", "Health", "--h", "48", "--out", str(probe_npz),
           "--ckpt", str(pd / "b4_health_H48.npz"),
           "--meta", str(pd / "b4_health_H48.meta.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = {"cmd": " ".join(cmd), "returncode": r.returncode,
           "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:],
           "errors": []}
    if r.returncode != 0:
        out["errors"].append("restore_probe subprocess failed rc=%d"
                             % r.returncode)
        return out
    probe_meta = load_json(str(probe_npz) + ".json")
    parent_meta = load_json(pd / "b4_health_H48.meta.json")
    if probe_meta["digest_after"] != parent_meta["digest_after"]:
        out["errors"].append(
            "digest mismatch: subprocess %s != parent %s"
            % (probe_meta["digest_after"][:12],
               parent_meta["digest_after"][:12]))
    if probe_meta["ckpt_npz_sha256"] != parent_meta["npz_sha256"]:
        out["errors"].append("ckpt npz sha mismatch across processes")
    # parent side: same strict restore from the saved ckpt on a fresh
    # model, same probe inputs (first 8 val windows)
    seed_everything(P.SEED)
    m = build_visionts().to(DEVICE)
    configure(m, int(P.TABLE["Health"]["L"]), 48, int(P.FIXED_P["Health"]))
    freeze_to_ln(m)
    from repro_timemmd_common_ln_v1.ln_train import restore_ln_ckpt_strict
    restore_ln_ckpt_strict(m, pd / "b4_health_H48.npz",
                           pd / "b4_health_H48.meta.json", pinfo,
                           "Health", 48)
    dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], "Health")
    spv = dd.load_split(48, "val")
    y_parent = forward_batch(m, torch_from(spv["X"][:8])).cpu().numpy()
    y_sub = dict(np.load(str(probe_npz)))["pred"]
    c = pred_close(y_sub, y_parent, ATOL, RTOL)
    out["pred_close"] = {"close": c[0], "max_abs": c[1], "max_viol": c[2]}
    if not c[0]:
        out["errors"].append("probe pred not close: viol %.3e" % c[2])
    _drop(m)
    return out


def b5_budget(rows):
    splits = load_json(Path(P.protocol_dir()) / "splits.json")
    steps_by_group = {}
    for dom in P.ORDER:
        for h in P.TABLE[dom]["H"]:
            n = int(splits[dom]["window_counts"][str(h)]["train"])
            steps_by_group[P.group_id(dom, h)] = P.expected_steps(n)
    total_steps = sum(steps_by_group.values())
    rate = {}
    for (dom, h) in P.PRECHECK:
        r = rows.get(P.group_id(dom, h), {}).get("step", {})
        if r:
            rate[dom] = r["wall_s"]
    fallback = max(rate.values()) if rate else 30.0
    est_s = sum(ns * rate.get(gid.rsplit("_H", 1)[0], fallback)
                for gid, ns in steps_by_group.items())
    return {
        "microbatch_final": int(P.MICROBATCH),
        "total_steps": total_steps,
        "est_train_wall_s": round(est_s, 1),
        "measured_rates_s_per_step": rate,
        "fallback_rate_s_per_step": fallback,
        "peak_mem_mb_observed": max(rows[g]["step"]["peak_mem_mb"]
                                    for g in rows),
        "note": "crude record-only extrapolation; microbatch fallback only "
                "if OOM observed (effective batch stays 256, FP32)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pd = (Path(args.out_root) if args.out_root else P.out_root()) \
        / "preflight"
    pd.mkdir(parents=True, exist_ok=True)
    report = {"device": DEVICE, "probe_n": PROBE_N, "atol": ATOL,
              "rtol": RTOL, "errors": []}
    t0 = time.time()
    try:
        pinfo = P.verify_frozen(args.protocol_dir)
        report["verify_frozen"] = {
            k: pinfo[k] for k in ("weights_sha256", "selected_p_sha256",
                                  "train_windows", "val_windows",
                                  "test_windows", "steps_total")}
    except SystemExit as e:
        report["errors"].append("verify_frozen: %s" % e)
        dump_json(pd / "preflight_report.json", report)
        raise

    e1, d1 = b1_protocol_data(pinfo)
    report["b1"] = {"errors": e1, "groups": d1}
    print("B1 done, %d errors" % len(e1), flush=True)

    e2, r2 = b2_entry_equivalence(pinfo)
    report["b2"] = {"errors": e2, "rows": r2}
    print("B2 done, %d errors" % len(e2), flush=True)

    e3, r3, b4, b5 = b3_logical_batch(pinfo, pd)
    report["b3"] = {"errors": e3, "groups": r3}
    report["b4"] = b4
    report["b5"] = b5
    print("B3/B4/B5 done, %d errors" % len(e3), flush=True)

    all_errors = list(e1) + list(e2) + list(e3) + list(b4["errors"])
    report["errors"] = all_errors
    report["all_pass"] = not all_errors
    report["elapsed_s"] = round(time.time() - t0, 1)
    dump_json(pd / "preflight_report.json", report)
    print("PREFLIGHT %s (%.1fs, %d errors)"
          % ("ALL PASS" if report["all_pass"] else "FAILED",
             report["elapsed_s"], len(all_errors)))
    if not report["all_pass"]:
        for e in all_errors[:20]:
            print("  - %s" % e)
        sys.exit(2)


if __name__ == "__main__":
    main()
