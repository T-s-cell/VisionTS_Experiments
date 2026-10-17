"""Preflight gates A1-A5. All pass -> full campaign proceeds automatically.

A1 protocol/data: frozen verification + per-group counts + first/last
   identity + label boundaries + truth/identity BITWISE vs TaTS 36 npz
   (data-side gate, before any inference).
A2 val precheck: 4 combos (Energy/H48/P52, Environment/H336/P365,
   Health/H48/P52, Security/H12/P12) — largest P, longest L, non-divisible
   periods, tail batch; shapes/finiteness/geometry vs hand formula.
A3 adapter crosscheck: adapter.forward_batch == direct model(x) on the
   same input, atol=1e-6 rtol=1e-5 (current env only, per directive).
A4 weights frozen + state isolation: state digest unchanged across all
   configurations; geometry fields match hand formula after each switch.
A5 budget: throughput/VRAM measured on A2 -> extrapolated wall-clock
   (report-only); batch probe 256 vs 1 with atol/rtol (beyond tolerance =
   hard stop to investigate; NO auto batch reduction).
Writes preflight_report.json; exit 2 on any gate failure.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_visionts, configure, \
    forward_batch, predict_split
from repro_timemmd_common_v1.common import bitwise_equal, die, dump_json, \
    env_info, model_state_digest, pred_close, seed_everything, \
    setup_deterministic_env, sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData
from repro_timemmd_common_v1.p_search import load_status, run_combo, \
    save_status


def geometry(L, H, Pp):
    """Hand-computed VisionTS update_config geometry (image 224, patch 16,
    align_const 0.4)."""
    image_size, num_patch, align = 224, 14, 0.4
    pad_left = (Pp - L % Pp) % Pp
    pad_right = (Pp - H % Pp) % Pp
    input_ratio = (pad_left + L) / (pad_left + L + pad_right + H)
    npi = int(input_ratio * num_patch * align)
    if npi == 0:
        npi = 1
    adj = npi / num_patch
    scale_x = ((pad_left + L) // Pp) / int(image_size * adj)
    return {"pad_left": pad_left, "pad_right": pad_right,
            "num_patch_input": npi,
            "input_resize_h": image_size,
            "input_resize_w": int(image_size * adj),
            "output_resize_h": Pp,
            "output_resize_w": int(round(image_size * scale_x))}


def check_geometry(model, L, H, Pp, errs, tag):
    g = geometry(L, H, Pp)
    if model.pad_left != g["pad_left"] or model.pad_right != g["pad_right"]:
        errs.append("%s pad %d/%d != %d/%d"
                    % (tag, model.pad_left, model.pad_right,
                       g["pad_left"], g["pad_right"]))
    if model.num_patch_input != g["num_patch_input"]:
        errs.append("%s num_patch_input %d != %d"
                    % (tag, model.num_patch_input, g["num_patch_input"]))
    wh = model.input_resize.size
    if wh != (g["input_resize_h"], g["input_resize_w"]):
        errs.append("%s input_resize %s != %s"
                    % (tag, wh, (g["input_resize_h"], g["input_resize_w"])))
    wo = model.output_resize.size
    if wo != (g["output_resize_h"], g["output_resize_w"]):
        errs.append("%s output_resize %s != %s"
                    % (tag, wo, (g["output_resize_h"],
                                 g["output_resize_w"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--tats_ref", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())
    (out_root / "preflight").mkdir(parents=True, exist_ok=True)
    errs = []
    report = {"gates": {}}

    tats_ref = args.tats_ref
    if tats_ref is None:
        tats_ref = str(P.tats_ref_dir())
    use_tats = bool(tats_ref) and Path(tats_ref).exists()

    # ---------- A1: protocol + data + TaTS truth ----------
    a1 = {"use_tats_ref": use_tats, "groups": {}}
    domcache = {}
    n_win = 0
    for dom, h in P.all_groups():
        gid = P.group_id(dom, h)
        if dom not in domcache:
            domcache[dom] = DomainData(pinfo["protocol_dir"],
                                       pinfo["splits"], dom)
        dd = domcache[dom]
        gerr = []
        for split in ("val", "test"):
            sp = dd.load_split(h, split)
            n_win += len(sp["X"]) if split == "test" else 0
            a1.setdefault("edge_identity", {})[gid] = {
                "first_s": int(sp["s"][0]), "last_s": int(sp["s"][-1]),
                "n": int(len(sp["s"]))}
        if use_tats:
            tnpz = (Path(tats_ref) / gid / "test_arrays.npz")
            if not tnpz.exists():
                tnpz = (Path(tats_ref) / ("%s_seed2021" % gid) /
                        "test_arrays.npz")
            t = dict(np.load(tnpz))
            sp = dd.load_split(h, "test")
            tt = t["truth"][:, :, 0] if t["truth"].ndim == 3 else t["truth"]
            if not bitwise_equal(tt, sp["Y"][:, :, 0]):
                gerr.append("truth != TaTS bitwise")
            for k, ours in (("s", sp["s"]), ("orig_row", sp["orig_row"]),
                            ("label_dates", sp["label_dates"])):
                if not bitwise_equal(np.asarray(t[k]), np.asarray(ours)):
                    gerr.append("%s != TaTS" % k)
        a1["groups"][gid] = {"errors": gerr}
        if gerr:
            errs.extend("%s: %s" % (gid, e) for e in gerr)
    a1["test_windows"] = n_win
    if n_win != P.TEST_WINDOWS_TOTAL:
        errs.append("A1 test windows %d != %d" % (n_win,
                                                  P.TEST_WINDOWS_TOTAL))
    report["gates"]["A1"] = a1
    print("A1 %s (%d test windows, TaTS truth bitwise check)"
          % ("PASS" if not errs else "FAIL", n_win), flush=True)

    # ---------- build model (no inference before A1 passes) ----------
    if errs:
        report["errors"] = errs
        dump_json(out_root / "preflight_report.json", report)
        die("[PREFLIGHT FAIL] A1 errors; aborting", 2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_visionts()
    model.to(dev)
    digest0 = model_state_digest(model)
    a4 = {"digest_before": digest0}
    torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None

    # ---------- A2: val precheck combos ----------
    a2 = {"combos": {}}
    ps_status = load_status(out_root)
    t0 = time.time()
    for dom, h, p in P.PRECHECK:
        cid = P.combo_id(dom, p, h)
        res = run_combo(model, dev, domcache[dom], dom, p, h,
                        out_root)
        ps_status["combos"][cid] = res
        a2["combos"][cid] = res
        check_geometry(model, P.TABLE[dom]["L"], h, p, errs, cid)
        if res["windows"] != int(pinfo["splits"][dom]["window_counts"]
                                 [str(h)]["val"]):
            errs.append("%s val coverage %d != expected"
                        % (cid, res["windows"]))
        if not (np.isfinite(res["mse"]) and np.isfinite(res["mae"])):
            errs.append("%s non-finite metrics" % cid)
    a2["seconds"] = round(time.time() - t0, 1)
    save_status(out_root, ps_status)
    report["gates"]["A2"] = a2
    print("A2 %s (4 combos, %.1fs)"
          % ("PASS" if not errs else "FAIL", a2["seconds"]), flush=True)

    # ---------- A3: adapter vs direct forward ----------
    a3 = {}
    dom, h, p = P.PRECHECK[0]
    dd = domcache[dom]
    configure(model, dd.L, h, p)
    sp = dd.load_split(h, "val")
    xb = torch.from_numpy(sp["X"][:8]).to(dev).unsqueeze(-1)
    seed_everything(P.SEED)
    y_adapter = forward_batch(model, xb)
    seed_everything(P.SEED)
    with torch.inference_mode():
        y_direct = model(xb, export_image=False, fp64=P.FP64)
    close, max_abs, max_viol = pred_close(
        y_adapter.cpu().numpy(), y_direct.cpu().numpy(), P.ATOL, P.RTOL)
    a3 = {"max_abs_diff": max_abs, "max_tol_violation": max_viol,
          "pass": close}
    report["gates"]["A3"] = a3
    if not close:
        errs.append("A3 adapter vs direct beyond tolerance (max|d|=%.2e)"
                    % max_abs)
    print("A3 %s max|d|=%.2e" % ("PASS" if close else "FAIL", max_abs),
          flush=True)

    # ---------- A4: state isolation + weights digest ----------
    seq = [("Energy", 12, 1), ("Security", 12, 12), ("Environment", 336, 1),
           ("Health", 48, 52), ("Energy", 48, 52)]
    for dom, h, p in seq:
        dd2 = domcache[dom]
        configure(model, dd2.L, h, p)
        check_geometry(model, dd2.L, h, p, errs, "A4_%s_H%d_P%d"
                       % (dom, h, p))
    a4["digest_after"] = model_state_digest(model)
    a4["pass"] = a4["digest_before"] == a4["digest_after"]
    report["gates"]["A4"] = a4
    if not a4["pass"]:
        errs.append("A4 weights digest changed across configurations")
    print("A4 %s" % ("PASS" if a4["pass"] else "FAIL"), flush=True)

    # ---------- A5: batch probe + throughput/budget ----------
    dom, h, p = P.PRECHECK[0]
    dd = domcache[dom]
    configure(model, dd.L, h, p)
    sp = dd.load_split(h, "val")
    nsub = 32
    xb = torch.from_numpy(sp["X"][:nsub]).to(dev).unsqueeze(-1)
    y_b256 = forward_batch(model, xb)
    rows256 = []
    for i in range(nsub):
        y1 = forward_batch(model, xb[i:i + 1])
        rows256.append(y1[0].cpu().numpy())
    y_b1 = torch.from_numpy(np.stack(rows256)).numpy()
    close, max_abs, max_viol = pred_close(y_b256.cpu().numpy(), y_b1,
                                          P.ATOL, P.RTOL)
    a5 = {"batch_probe_max_abs_diff": max_abs,
          "batch_probe_max_tol_violation": max_viol,
          "batch_probe_pass": close,
          "batch_policy": "atol=%.0e rtol=%.0e; out-of-tolerance => "
                          "investigate, no auto batch change"
                          % (P.ATOL, P.RTOL)}
    if not close:
        errs.append("A5 batch 256 vs 1 beyond tolerance (max|d|=%.2e, "
                    "max violation=%.2e) — investigate, do not reduce "
                    "batch" % (max_abs, max_viol))
    sec_per_win = a2["seconds"] / sum(
        a2["combos"][P.combo_id(*c)]["windows"] for c in P.PRECHECK)
    a5["val_seconds_extrapolated"] = round(P.VAL_WINDOWS_TOTAL *
                                           sec_per_win, 1)
    a5["test_seconds_extrapolated"] = round(P.TEST_WINDOWS_TOTAL *
                                            sec_per_win, 1)
    a5["replay_seconds_extrapolated"] = a5["test_seconds_extrapolated"]
    a5["sec_per_window_measured"] = sec_per_win
    if dev == "cuda":
        a5["peak_mem_gib"] = round(
            torch.cuda.max_memory_allocated() / 2**30, 3)
    report["gates"]["A5"] = a5
    print("A5 %s batch max|d|=%.2e; ETA val %.0fs test %.0fs replay %.0fs"
          % ("PASS" if close else "FAIL", max_abs,
             a5["val_seconds_extrapolated"],
             a5["test_seconds_extrapolated"],
             a5["replay_seconds_extrapolated"]), flush=True)

    report["env"] = env_info()
    report["errors"] = errs
    report["all_pass"] = not errs
    report["weights_digest"] = digest0
    dump_json(out_root / "preflight_report.json", report)
    print("PREFLIGHT %s (%d errors)" % ("ALL PASS" if not errs else "FAIL",
                                        len(errs)))
    if errs:
        raise SystemExit(2)
    print("Preflight report sha256 = %s"
          % sha256_of(out_root / "preflight_report.json"))


if __name__ == "__main__":
    main()
