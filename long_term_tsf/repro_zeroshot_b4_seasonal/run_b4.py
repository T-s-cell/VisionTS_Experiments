"""EXP-031 runner: Seasonal Naive / Seasonal Avg baselines on the EXP-001
zero-shot protocol (CPU numpy only, no model inference).

Per group:
  gates  - truth recomputed from the standardized array (sklearn scaler,
           identical to data_provider/data_loader.py) is bitwise equal to
           EXP-001's true.npy; window count == closed form; channel names
           match the csv header (and EXP-030's saved npz when present);
           EXP-001 VisionTS metrics recompute == metrics.npy exactly.
  baselines
         - naive:  template = last P steps of the L-step history,
                   pred = template[arange(H) % P]
           avg:    template = mean of the L/P complete cycles (f32),
                   pred = template[arange(H) % P]
           history is the f32 standardized rows the model saw.
  outputs - per method: results/_/{pred,true,metrics}.npy, SUCCESS.txt
            (own-script fingerprints, kept separate from EXP-001's),
            metrics_<DS>_<H>.json, win_<DS>_<H>.npz (per-window errors);
            group root: win_visionts_<DS>_<H>.npz + probes_evidence.json.

  hand check
         - --hand_check DS H IDX: naive scalar loop must match the
           vectorized result AND the saved pred bitwise (pure indexing,
           no rounding); avg uses an independent float64 scalar
           accumulation as reference, compared at rtol=1e-5/atol=1e-6
           with the max error recorded; prints a v=0 pred-vs-truth table
           for the first 2P steps; writes HANDCHECK_OK.txt on pass.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b4_common as C


def build_naive(st, ds, h):
    t0 = C.first_pred_abs_row0(ds)
    n = C.expected_n_windows(ds, h)
    p = C.DATASETS[ds]["period"]
    t = t0 + np.arange(n, dtype=np.int64)
    rows = t[:, None] - p + (np.arange(h) % p)[None, :]  # (N, H)
    return st[rows]  # (N, H, 7) f32, pure row indexing


def build_avg(st, ds, h):
    t0 = C.first_pred_abs_row0(ds)
    n = C.expected_n_windows(ds, h)
    p = C.DATASETS[ds]["period"]
    lctx = C.DATASETS[ds]["ctx"]
    n_var = st.shape[1]
    out = np.empty((n, h, n_var), dtype=np.float32)
    for i in range(n):
        hist = st[t0 + i - lctx: t0 + i]
        out[i] = C.seasonal_avg(hist, p, h)
    return out


def save_method_npy(md, pred, true, metric_arr):
    rd = os.path.join(md, "results", "_")
    os.makedirs(rd, exist_ok=True)
    np.save(os.path.join(rd, "pred.npy"), pred)
    np.save(os.path.join(rd, "true.npy"), true)
    np.save(os.path.join(rd, "metrics.npy"), metric_arr)


def success_txt(ds, h, method, metrics, notes):
    c = C.DATASETS[ds]
    lines = [
        "dataset=%s pred_len=%d seq_len=%d periodicity=%d baseline=seasonal_%s"
        % (ds, h, c["ctx"], c["period"], method),
        "scaler=sklearn.StandardScaler fit_rows=[0,%d) dtype=float32 "
        "history_rows=%d" % (12 * C.month_rows(ds), c["ctx"]),
        "root_path=%s data_path=%s data=%s features=M"
        % (c["root_path"], c["data_path"], c["data"]),
        "log_metrics=mse:%r, mae:%r" % (float(metrics[1]), float(metrics[0])),
        "index_rule=template[np.arange(%d) %% %d]" % (h, c["period"]),
    ]
    for f in C.PKG_FP_FILES:
        lines.append("md5:%s=%s" % (f, C.md5_of(os.path.join(C.ROOT, f))))
    lines.append("baseline_fingerprint=EXP-001 FP_FILES checked by "
                 "preflight_b4.py (kept separate from this experiment's md5s)")
    if notes:
        lines.append("notes=%s" % notes)
    return "\n".join(lines) + "\n"


def eval_group(ds, h, metric):
    bd = C.base_dir(ds, h)
    n_exp = C.expected_n_windows(ds, h)
    p = C.DATASETS[ds]["period"]
    lctx = C.DATASETS[ds]["ctx"]

    st = C.std_array(ds)
    true_npy = np.load(os.path.join(bd, "results", "_", "true.npy"))
    problems = []
    if true_npy.shape != (n_exp, h, 7):
        problems.append("true.npy shape %s != (%d, %d, 7)"
                        % (true_npy.shape, n_exp, h))
    true_rec = C.truth_recomputed(st, ds, h)
    truth_bitwise = (true_npy.shape == true_rec.shape
                     and np.array_equal(true_rec, true_npy))
    if not truth_bitwise:
        if true_npy.shape == true_rec.shape:
            d = np.abs(true_rec.astype(np.float64) - true_npy.astype(np.float64))
            problems.append("truth NOT bitwise: max|d|=%g at %s"
                            % (d.max(), np.unravel_index(d.argmax(), d.shape)))
        else:
            problems.append("truth shape mismatch vs recomputed")
    names = C.channel_names(ds)
    a2p1_npz = os.path.join(C.A2P1_SAVE_BASE, "%s_%d" % (ds, h),
                            "win_%s_%d.npz" % (ds, h))
    exp030_channels_ok = None
    if os.path.isfile(a2p1_npz):
        ref = [str(x) for x in np.load(a2p1_npz)["channel_names"]]
        exp030_channels_ok = (ref == names)
        if not exp030_channels_ok:
            problems.append("channel names differ from EXP-030 npz: %s vs %s"
                            % (names, ref))

    # VisionTS orig-P gate: metric() recompute == metrics.npy exactly
    base_pred = np.load(os.path.join(bd, "results", "_", "pred.npy"))
    base_mets = np.load(os.path.join(bd, "results", "_", "metrics.npy"))
    mae_r, mse_r, rmse_r, mape_r, mspe_r = metric(base_pred, true_npy)
    visionts_ok = np.array_equal(
        np.array([mae_r, mse_r, rmse_r, mape_r, mspe_r], dtype=np.float64),
        base_mets[:5].astype(np.float64), equal_nan=True)
    if not visionts_ok:
        problems.append("EXP-001 metrics recompute != metrics.npy")

    if problems:
        for x in problems:
            print("[run_b4] %s_%d GATE FAIL: %s" % (ds, h, x))
        raise SystemExit("[run_b4] %s_%d: gates failed, aborting" % (ds, h))

    true_f32 = true_npy.astype(np.float32, copy=False)
    out = {"group": "%s_%d" % (ds, h), "n_windows": int(n_exp),
           "truth_bitwise": True, "exp030_channels_ok": exp030_channels_ok,
           "visionts_metrics_ok": True,
           "visionts_origp": {"mse": float(mse_r), "mae": float(mae_r)},
           "methods": {}}
    per_method_evidence = {}

    for method in C.METHODS:
        if method == "naive":
            pred = build_naive(st, ds, h)
        else:
            pred = build_avg(st, ds, h)
        assert pred.shape == (n_exp, h, 7) and pred.dtype == np.float32
        mae, mse, rmse, mape, mspe = metric(pred, true_f32)
        assert np.isfinite(mse) and np.isfinite(mae)
        marr = np.array([mae, mse, rmse, mape, mspe, -1, -1])
        md = C.method_dir(ds, h, method)
        save_method_npy(md, pred, true_f32, marr)

        wvmse, wvmae, wmse, wmae = C.window_errors(pred, true_f32)
        npz_path = os.path.join(md, "win_%s_%d.npz" % (ds, h))
        np.savez_compressed(
            npz_path, win_var_mse=wvmse, win_var_mae=wvmae,
            win_mse=wmse, win_mae=wmae,
            window_index=np.arange(n_exp, dtype=np.int64),
            win_start=C.win_start(ds, n_exp),
            channel_names=np.array(names, dtype="U8"))

        with open(os.path.join(md, "SUCCESS.txt"), "w") as f:
            f.write(success_txt(ds, h, method, (mae, mse),
                                "n_windows=%d" % n_exp))
        rec = {"mse": float(mse), "mae": float(mae),
               "rmse": float(rmse), "mape": float(mape), "mspe": float(mspe)}
        C.dump_json(os.path.join(md, "metrics_%s_%d.json" % (ds, h)), {
            "group": "%s_%d" % (ds, h), "method": "seasonal_" + method,
            "periodicity": p, "seq_len": lctx, "pred_len": h,
            "n_windows": int(n_exp), **rec,
            "geometry": {
                "lookback_rows": lctx, "cycles": lctx // p,
                "template_shape": [p, 7],
                "index_rule": "template[np.arange(%d) %% %d]" % (h, p),
                "template_dtype": "float32 (mean of f32 history)"},
            "visionts_origp": out["visionts_origp"],
            "arrays": {"pred_npy": os.path.relpath(
                           os.path.join(md, "results", "_", "pred.npy"), C.ROOT),
                       "win_npz": os.path.relpath(npz_path, C.ROOT)},
            "code_md5": {f: C.md5_of(os.path.join(C.ROOT, f))
                         for f in C.PKG_FP_FILES},
            "t": time.time()})
        out["methods"][method] = rec
        per_method_evidence[method] = {
            "pred_shape": [int(x) for x in pred.shape],
            "win_npz": os.path.relpath(npz_path, C.ROOT),
            "metrics_npy": os.path.relpath(
                os.path.join(md, "results", "_", "metrics.npy"), C.ROOT)}

    # group-root artifacts: visionts per-window npz + evidence
    wvmse, wvmae, wmse, wmae = C.window_errors(base_pred, true_f32)
    vt_npz = os.path.join(C.SAVE_BASE, "%s_%d" % (ds, h),
                          "win_visionts_%s_%d.npz" % (ds, h))
    os.makedirs(os.path.dirname(vt_npz), exist_ok=True)
    np.savez_compressed(
        vt_npz, win_var_mse=wvmse, win_var_mae=wvmae,
        win_mse=wmse, win_mae=wmae,
        window_index=np.arange(n_exp, dtype=np.int64),
        win_start=C.win_start(ds, n_exp),
        channel_names=np.array(names, dtype="U8"))
    evidence = {
        "group": out["group"], "periodicity": p, "seq_len": lctx,
        "pred_len": h, "n_windows": int(n_exp),
        "truth_bitwise_vs_exp001": True,
        "exp030_channels_ok": exp030_channels_ok,
        "visionts_metrics_recompute_exact": True,
        "visionts_origp": out["visionts_origp"],
        "methods": per_method_evidence,
        "visionts_win_npz": os.path.relpath(vt_npz, C.ROOT),
        "code_md5": {f: C.md5_of(os.path.join(C.ROOT, f))
                     for f in C.PKG_FP_FILES},
        "t": time.time()}
    ev_path = os.path.join(C.SAVE_BASE, "%s_%d" % (ds, h),
                           "probes_evidence.json")
    os.makedirs(os.path.dirname(ev_path), exist_ok=True)
    C.dump_json(ev_path, evidence)

    print("[run_b4] %s_%d  naive mse=%.6f mae=%.6f | avg mse=%.6f mae=%.6f "
          "| visionts(P=%d) mse=%.6f mae=%.6f | N=%d"
          % (ds, h, out["methods"]["naive"]["mse"],
             out["methods"]["naive"]["mae"],
             out["methods"]["avg"]["mse"], out["methods"]["avg"]["mae"],
             p, mse_r, mae_r, n_exp))
    return out


def hand_check(ds, h, idx):
    """Independent scalar recomputation of window idx vs the saved arrays."""
    gd = os.path.join(C.SAVE_BASE, "%s_%d" % (ds, h))
    p = C.DATASETS[ds]["period"]
    lctx = C.DATASETS[ds]["ctx"]
    n_exp = C.expected_n_windows(ds, h)
    assert 0 <= idx < n_exp, "window index out of range"
    t0 = C.first_pred_abs_row0(ds)
    t = t0 + idx
    st = C.std_array(ds)
    hist = st[t - lctx: t]  # (L, 7) f32
    n_cycles = lctx // p
    h_idx = idx

    problems = []

    # ---- naive: scalar loop, must be bitwise ----
    n_var = hist.shape[1]
    pred_s = np.empty((h, n_var), dtype=np.float32)
    for hh in range(h):
        for v in range(n_var):
            pred_s[hh, v] = hist[lctx - p + (hh % p), v]
    vec_n = C.seasonal_naive(hist, p, h)
    saved_n = np.load(os.path.join(gd, "naive", "results", "_", "pred.npy"))[h_idx]
    ok_vec = np.array_equal(vec_n, pred_s)
    ok_saved = np.array_equal(saved_n, pred_s)
    if not ok_vec:
        problems.append("naive scalar != vectorized (max|d|=%g)"
                        % np.abs(vec_n.astype(np.float64)
                                 - pred_s.astype(np.float64)).max())
    if not ok_saved:
        problems.append("naive scalar != saved pred.npy window")

    # ---- avg: independent float64 scalar accumulation reference ----
    tmpl_s = np.empty((p, n_var), dtype=np.float64)
    for ph in range(p):
        for v in range(n_var):
            s = 0.0
            for cyc in range(n_cycles):
                s += float(hist[cyc * p + ph, v])
            tmpl_s[ph, v] = s / n_cycles
    ref = tmpl_s[np.arange(h) % p].astype(np.float32)
    vec_a = C.seasonal_avg(hist, p, h)
    saved_a = np.load(os.path.join(gd, "avg", "results", "_", "pred.npy"))[h_idx]
    ok_saved_a = np.array_equal(saved_a, vec_a)
    d = np.abs(vec_a.astype(np.float64) - ref.astype(np.float64))
    max_err = float(d.max())
    ok_tol = np.allclose(vec_a, ref, rtol=1e-5, atol=1e-6)
    if not ok_saved_a:
        problems.append("avg saved pred != vectorized window")
    if not ok_tol:
        problems.append("avg vectorized vs float64 reference out of tolerance "
                        "(max|d|=%g)" % max_err)

    # ---- truth for this window ----
    true_saved = np.load(os.path.join(gd, "naive", "results", "_", "true.npy"))[h_idx]
    ok_truth = np.array_equal(true_saved, st[t: t + h])
    if not ok_truth:
        problems.append("truth window != standardized rows")

    # ---- printed v=0 table (first 2P steps) ----
    show = min(2 * p, h)
    true_win = true_saved
    print("[hand_check] %s_%d window=%d t_abs=%d (L=%d, P=%d, cycles=%d, "
          "H%%P=%d)" % (ds, h, idx, t, lctx, p, n_cycles, h % p))
    print("  h | naive(v0) | avg(v0) | truth(v0)")
    for hh in range(show):
        print("  %3d | %+.6f | %+.6f | %+.6f"
              % (hh, float(saved_n[hh, 0]), float(saved_a[hh, 0]),
                 float(true_win[hh, 0])))
    print("  naive bitwise: vec=%s saved=%s | avg: saved_bitwise=%s "
          "f64ref_max|d|=%.3e tol_ok=%s | truth_bitwise=%s"
          % (ok_vec, ok_saved, ok_saved_a, max_err, ok_tol, ok_truth))

    if problems:
        for x in problems:
            print("[hand_check] FAIL: %s" % x)
        raise SystemExit("[hand_check] %s_%d failed" % (ds, h))
    ev = {"group": "%s_%d" % (ds, h), "window_index": idx, "t_abs": int(t),
          "lookback": lctx, "period": p, "cycles": n_cycles, "h_mod_p": h % p,
          "naive_bitwise_vec": True, "naive_bitwise_saved": True,
          "avg_saved_bitwise": True,
          "avg_f64_reference_max_abs_err": max_err,
          "avg_tolerance": "rtol=1e-5 atol=1e-6", "avg_tol_ok": True,
          "truth_bitwise": True, "t": time.time()}
    with open(os.path.join(gd, "HANDCHECK_OK.txt"), "w") as f:
        f.write("HANDCHECK OK window=%d naive_bitwise avg_f64ref(max=%.3e, "
                "rtol=1e-5 atol=1e-6) truth_bitwise\n" % (idx, max_err))
    C.dump_json(os.path.join(gd, "handcheck_evidence.json"), ev)
    print("[hand_check] OK -> %s" % os.path.join(gd, "HANDCHECK_OK.txt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs=2, metavar=("DS", "H"), action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--hand_check", nargs=3, metavar=("DS", "H", "IDX"),
                    action="append")
    args = ap.parse_args()
    metric = C.load_metric()
    groups = C.group_ids() if args.all else [
        (ds, int(h)) for ds, h in (args.group or [])]
    outs = []
    for ds, h in groups:
        outs.append(eval_group(ds, h, metric))
    for ds, h, idx in ([tuple([x[0], int(x[1]), int(x[2])])
                        for x in args.hand_check] if args.hand_check else []):
        hand_check(ds, h, idx)
    if outs:
        os.makedirs(os.path.join(C.PKG, "logs"), exist_ok=True)
        C.dump_json(os.path.join(C.PKG, "logs", "run_b4_groups.json"),
                    {"groups": [o["group"] for o in outs]})
        print("[run_b4] done: %d groups" % len(outs))


if __name__ == "__main__":
    main()
