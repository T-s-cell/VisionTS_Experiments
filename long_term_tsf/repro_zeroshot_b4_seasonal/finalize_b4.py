"""EXP-031 archive finalization (post-results; the 16 groups are NOT re-run).

 0. fingerprint continuity: b4_common.py/run_b4.py md5s must still match the
    values recorded in the run-time SUCCESS.txt files (evidence <-> code).
 1. std-input verification: instantiate the EXP-001 original loader
    (data_provider.data_loader.Dataset_ETT_{hour,minute}, flag='test') once
    per dataset and compare its full test-segment data_x -- which INCLUDES
    the L history rows before the first prediction start -- against
    std_array() over the same rows; both sides float32, bitwise equal.
    Record: logs/std_input_verification.json (4/4 required).
 2. hand-check float64 retention: recompute the two archived spot checks
    (ETTh1_96 win 1000, ETTm1_336 win 2000) carrying the independent f64
    reference THROUGH the error computation: f64 template -> f64 cycled
    prediction -> f32 cast compared to the saved prediction (rtol=1e-5,
    atol=1e-6, max err) -> the f64 reference's own MSE/MAE vs truth recorded;
    reference arrays saved as handcheck_f64ref.npz next to a
    handcheck_f64ref.json with all numbers.
 3. rebuild logs/run_b4_groups.json: run_b4.py overwrote it per invocation
    (only the last group survived); rebuild the full 16-group list from the
    existing SUCCESS.txt + metrics json credentials.
 4. naive template description fix: the 16 naive metrics json inherited the
    shared geometry string "template_dtype": "float32 (mean of f32 history)",
    which describes seasonal_avg only; naive takes the last P rows of the
    history DIRECTLY (no averaging). Corrected in place with an explicit
    archive_note.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b4_common as C

HANDCHECK_SPOTS = [("ETTh1", 96, 1000), ("ETTm1", 336, 2000)]


def fingerprint_continuity():
    ref = os.path.join(C.method_dir("ETTh1", 96, "naive"), "SUCCESS.txt")
    rec = {}
    with open(ref) as f:
        for line in f:
            line = line.strip()
            if line.startswith("md5:") and "=" in line:
                k, v = line[4:].rsplit("=", 1)
                rec[k] = v
    problems = []
    for f in C.PKG_FP_FILES:
        want = rec.get(f)
        cur = C.md5_of(os.path.join(C.ROOT, f))
        if want != cur:
            problems.append("fingerprint drift %s: success=%s current=%s"
                            % (f, want, cur))
    return problems


def verify_std_input():
    if C.ROOT not in sys.path:
        sys.path.insert(0, C.ROOT)
    from data_provider import data_loader as DL
    rec = {}
    problems = []
    for ds in C.ETT4:
        c = C.DATASETS[ds]
        cls = DL.Dataset_ETT_minute if c["minute"] else DL.Dataset_ETT_hour
        # timeenc=1: EXP-001's pandas-2-era path (the timeenc=0 branch uses
        # df.drop(..., 1), removed in pandas >= 2); data_x is unaffected
        d = cls(root_path=c["root_path"], data_path=c["data_path"],
                flag="test", size=[c["ctx"], 48, 96], features="M",
                target="OT", scale=True, timeenc=1,
                freq="t" if c["minute"] else "h")
        data_x = np.asarray(d.data_x)          # f64, scaled test segment
        k = C.month_rows(ds)
        b1, b2 = 16 * k - c["ctx"], 16 * k + 4 * k
        st = C.std_array(ds)
        mine = st[b1:b2]
        r = {"loader": cls.__name__, "seq_len": c["ctx"], "border1": b1,
             "border2": b2, "data_x_shape": list(data_x.shape)}
        if data_x.shape != mine.shape:
            problems.append("%s: shape %s != %s" % (ds, data_x.shape,
                                                    mine.shape))
            rec[ds] = r
            continue
        theirs32 = data_x.astype(np.float32)
        bitwise = np.array_equal(theirs32, mine)
        d64 = np.abs(data_x - mine.astype(np.float64))
        r.update({"f32_bitwise_equal": bool(bitwise),
                  "max_abs_diff_f64": float(d64.max()),
                  "covers_history_before_first_window": bool(
                      data_x.shape[0] == 4 * k + c["ctx"]),
                  "n_windows_at_H96": int(C.expected_n_windows(ds, 96))})
        if not bitwise:
            problems.append("%s: data_x != std_array (max|d| f64=%g)"
                            % (ds, d64.max()))
        rec[ds] = r
        print("[finalize] std-input %s: loader=%s shape=%s f32_bitwise=%s "
              "max|d|=%.3g" % (ds, cls.__name__, data_x.shape, bitwise,
                               d64.max()))
    rec["n_pass"] = sum(1 for ds in C.ETT4
                        if rec.get(ds, {}).get("f32_bitwise_equal"))
    rec["all_pass"] = not problems
    rec["code_md5"] = {f: C.md5_of(os.path.join(C.ROOT, f))
                       for f in C.PKG_FP_FILES}
    rec["t"] = time.time()
    C.dump_json(os.path.join(C.PKG, "logs", "std_input_verification.json"),
                rec)
    return problems


def _win_errors_f64(pred64, true32):
    err = pred64 - true32.astype(np.float64)
    return (float(np.mean(err * err)), float(np.mean(np.abs(err))))


def handcheck_f64_retention():
    problems = []
    for ds, h, idx in HANDCHECK_SPOTS:
        gd = os.path.join(C.SAVE_BASE, "%s_%d" % (ds, h))
        c = C.DATASETS[ds]
        p, lctx = c["period"], c["ctx"]
        n_var = 7
        t = C.first_pred_abs_row0(ds) + idx
        st = C.std_array(ds)
        hist = st[t - lctx: t]
        n_cycles = lctx // p
        true_win = st[t: t + h]

        # naive: scalar loop, bitwise vs vectorized AND vs saved pred
        pred_s = np.empty((h, n_var), dtype=np.float32)
        for hh in range(h):
            for v in range(n_var):
                pred_s[hh, v] = hist[lctx - p + (hh % p), v]
        vec_n = C.seasonal_naive(hist, p, h)
        saved_n = np.load(os.path.join(gd, "naive", "results", "_",
                                       "pred.npy"))[idx]
        ok_vec = np.array_equal(vec_n, pred_s)
        ok_saved = np.array_equal(saved_n, pred_s)
        if not (ok_vec and ok_saved):
            problems.append("%s_%d naive scalar mismatch (vec=%s saved=%s)"
                            % (ds, h, ok_vec, ok_saved))

        # avg: independent f64 scalar accumulation, CARRIED THROUGH the
        # error computation (template -> cycled f64 prediction -> errors)
        tmpl = np.empty((p, n_var), dtype=np.float64)
        for ph in range(p):
            for v in range(n_var):
                s = 0.0
                for cyc in range(n_cycles):
                    s += float(hist[cyc * p + ph, v])
                tmpl[ph, v] = s / n_cycles
        pred_f64 = tmpl[np.arange(h) % p]           # (H,7) f64 reference
        ref32 = pred_f64.astype(np.float32)
        vec_a = C.seasonal_avg(hist, p, h)
        saved_a = np.load(os.path.join(gd, "avg", "results", "_",
                                       "pred.npy"))[idx]
        ok_saved_a = np.array_equal(saved_a, vec_a)
        d = np.abs(vec_a.astype(np.float64) - ref32.astype(np.float64))
        max_err = float(d.max())
        ok_tol = np.allclose(vec_a, ref32, rtol=1e-5, atol=1e-6)
        if not (ok_saved_a and ok_tol):
            problems.append("%s_%d avg f64-reference check failed "
                            "(saved_bitwise=%s tol=%s max|d|=%g)"
                            % (ds, h, ok_saved_a, ok_tol, max_err))

        ref_mse, ref_mae = _win_errors_f64(pred_f64, true_win)
        ref32_mse, ref32_mae = _win_errors_f64(ref32.astype(np.float64),
                                               true_win)
        saved_mse, saved_mae = _win_errors_f64(saved_a.astype(np.float64),
                                               true_win)
        n_mse, n_mae = _win_errors_f64(saved_n.astype(np.float64), true_win)
        rec = {"group": "%s_%d" % (ds, h), "window_index": idx,
               "t_abs": int(t), "lookback": lctx, "period": p,
               "cycles": n_cycles, "h_mod_p": h % p,
               "naive_bitwise_vec": bool(ok_vec),
               "naive_bitwise_saved": bool(ok_saved),
               "naive_win_mse": n_mse, "naive_win_mae": n_mae,
               "avg_saved_bitwise_vs_vectorized": bool(ok_saved_a),
               "avg_f64_vs_vectorized_max_abs_err": max_err,
               "avg_tolerance": "rtol=1e-5 atol=1e-6", "avg_tol_ok": bool(ok_tol),
               "avg_f64ref_win_mse": ref_mse, "avg_f64ref_win_mae": ref_mae,
               "avg_f64ref_f32cast_win_mse": ref32_mse,
               "avg_f64ref_f32cast_win_mae": ref32_mae,
               "avg_saved_win_mse": saved_mse, "avg_saved_win_mae": saved_mae,
               "note": "f64 scalar-accumulation reference carried through "
                       "template -> cycled prediction -> error computation; "
                       "arrays in handcheck_f64ref.npz",
               "t": time.time()}
        C.dump_json(os.path.join(gd, "handcheck_f64ref.json"), rec)
        np.savez_compressed(
            os.path.join(gd, "handcheck_f64ref.npz"),
            ref_pred_f64=pred_f64, ref_pred_f32=ref32,
            saved_pred_avg=saved_a, saved_pred_naive=saved_n,
            true_win=true_win,
            window_index=np.int64(idx), t_abs=np.int64(t))
        print("[finalize] f64-retention %s_%d win=%d H%%P=%d: naive bitwise "
              "vec=%s saved=%s | avg saved_bitwise=%s f64ref max|d|=%.3e "
              "tol=%s | f64ref mse=%.8f mae=%.8f (saved %.8f/%.8f)"
              % (ds, h, idx, h % p, ok_vec, ok_saved, ok_saved_a, max_err,
                 ok_tol, ref_mse, ref_mae, saved_mse, saved_mae))
    return problems


def rebuild_groups():
    groups, per, problems = [], {}, []
    for ds, h in C.group_ids():
        gid = "%s_%d" % (ds, h)
        entry = {"group": gid}
        for m in C.METHODS:
            sp = os.path.join(C.method_dir(ds, h, m), "SUCCESS.txt")
            jp = os.path.join(C.method_dir(ds, h, m),
                              "metrics_%s_%d.json" % (ds, h))
            if not (os.path.isfile(sp) and os.path.isfile(jp)):
                problems.append("%s %s: missing %s" % (gid, m, sp if not
                                os.path.isfile(sp) else jp))
                continue
            j = C.load_json(jp)
            entry[m] = {"success_txt": os.path.relpath(sp, C.ROOT),
                        "metrics_json": os.path.relpath(jp, C.ROOT),
                        "mse": j["mse"], "mae": j["mae"],
                        "n_windows": j["n_windows"]}
        groups.append(gid)
        per[gid] = entry
    rec = {"groups": groups,
           "rebuilt_from": "SUCCESS.txt + metrics_<DS>_<H>.json per method "
                           "(the original run_b4.py file was overwritten per "
                           "invocation and only kept the last group)",
           "rebuilt_by": "finalize_b4.py", "per_group": per,
           "n_groups": len(groups), "t": time.time()}
    C.dump_json(os.path.join(C.PKG, "logs", "run_b4_groups.json"), rec)
    print("[finalize] run_b4_groups.json rebuilt: %d groups %s"
          % (len(groups), groups))
    return problems


def fix_naive_template_desc():
    fixed, problems = 0, []
    for ds, h in C.group_ids():
        jp = os.path.join(C.method_dir(ds, h, "naive"),
                          "metrics_%s_%d.json" % (ds, h))
        j = C.load_json(jp)
        td = j["geometry"].get("template_dtype")
        if td == "float32 (direct last-P rows of f32 history, no averaging)":
            continue
        if td != "float32 (mean of f32 history)":
            problems.append("%s naive: unexpected template_dtype %r" % (jp, td))
            continue
        j["geometry"]["template_dtype"] = ("float32 (direct last-P rows of "
                                           "f32 history, no averaging)")
        j["geometry"]["archive_note"] = (
            "2026-11-02 finalize_b4.py: template_dtype corrected -- the "
            "naive template is the DIRECT last-P rows of the f32 history; "
            "the former 'mean of f32 history' described seasonal_avg")
        C.dump_json(jp, j)
        fixed += 1
    print("[finalize] naive template_dtype corrected in %d/16 metrics json"
          % fixed)
    return problems


def main():
    os.chdir(C.ROOT)
    problems = []
    p0 = fingerprint_continuity()
    problems += p0
    print("[finalize] step0 fingerprint continuity: %s"
          % ("OK" if not p0 else p0))
    problems += verify_std_input()
    problems += handcheck_f64_retention()
    problems += rebuild_groups()
    problems += fix_naive_template_desc()
    if problems:
        for x in problems:
            print("[finalize] FAIL:", x)
        raise SystemExit("[finalize] %d problem(s)" % len(problems))
    print("[finalize] ALL PASS: std-input 4/4 f32 bitwise vs EXP-001 loader,")
    print("               f64 references retained for both hand checks,")
    print("               16-group list rebuilt, naive template records fixed.")


if __name__ == "__main__":
    main()
