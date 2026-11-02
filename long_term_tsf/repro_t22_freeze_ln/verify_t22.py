"""EXP-033 verification. Three modes (each merges into logs/verify_t22.json):

  --group DS H          verify one NEW Freeze group under save_t22_freeze/
  --cache ETTm1 H       verify one EXP-001 ETTm1 cache group per-user-
                        correction 3 (no global preflight requirement);
                        returns status cache_ok / rerun_needed(code) /
                        stops on weights-or-protocol mismatch
  --first_window DS H   smoke extra: window 0 history (L steps) and
                        variable order from the real test loader vs
                        std_array, bitwise

Gates per group (any failure -> SystemExit, no fallback):
  * SUCCESS.txt present; every recorded md5 re-hashed against the current
    tree; ckpt sha256 == pinned EXP-001 weights
  * config line: Table-19 L/P, r=c=0.4, bs=32, train_epochs=0, seed=2021
  * pred/true shape (N, H, 7), N == corrected window count (2881-H hourly,
    11521-H minute)
  * true bitwise == truth_recomputed(std_array)   (first predicted row
    11520 hourly / 46080 minute is implied by this match)
  * true bitwise == EXP-003 seed2021 truth via the FIXED reshape
    (N*7, H, 1) -> reshape(N, 7, H).transpose(0, 2, 1)
  * metric(pred, true) == metrics.npy[:5] exactly
  * no checkpoint artifacts in the run dir (inference only)
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t22_common as C

VERIFY_JSON = os.path.join(C.PKG, "logs", "verify_t22.json")


def fail(msg):
    raise SystemExit("[verify_t22] FAIL: %s" % msg)


def check_run_dir_clean(run_dir):
    arts = (glob.glob(os.path.join(run_dir, "**", "*.pth"), recursive=True)
            + glob.glob(os.path.join(run_dir, "**", "checkpoint*"),
                        recursive=True))
    return [os.path.relpath(a, run_dir) for a in arts]


def verify_freeze_arrays(dir_, ds, h, metric, label):
    """Shared array-level gates for a results/_ dir; returns evidence dict."""
    rd = os.path.join(dir_, "results", "_")
    for f in ("pred.npy", "true.npy", "metrics.npy"):
        if not os.path.isfile(os.path.join(rd, f)):
            fail("%s %s_%d missing %s" % (label, ds, h, f))
    pred = np.load(os.path.join(rd, "pred.npy"))
    true = np.load(os.path.join(rd, "true.npy"))
    mets = np.load(os.path.join(rd, "metrics.npy"))
    n_exp = C.expected_n_windows(ds, h)

    problems = []
    if true.shape != (n_exp, h, 7):
        problems.append("true shape %s != (%d, %d, 7)" % (true.shape, n_exp, h))
    if pred.shape != true.shape:
        problems.append("pred shape %s != true %s" % (pred.shape, true.shape))
    if problems:
        fail("%s %s_%d: %s" % (label, ds, h, "; ".join(problems)))

    tr = C.truth_recomputed(C.std_array(ds), ds, h)
    eq_std = bool(np.array_equal(true, tr))
    t3 = C.exp003_truth_restored(ds, h)
    eq_ln = bool(np.array_equal(true, t3))
    if not eq_std:
        fail("%s %s_%d: true.npy NOT bitwise == std_array recomputation"
             % (label, ds, h))
    if not eq_ln:
        d = int(np.argmax(np.abs(true.astype(np.float64)
                                 - t3.astype(np.float64)).sum(axis=(1, 2))))
        fail("%s %s_%d: true.npy NOT bitwise == EXP-003 restored truth "
             "(worst window %d)" % (label, ds, h, d))

    rec = C.metrics_from_arrays(metric, pred, true)
    eq_m = bool(np.array_equal(rec, mets[:5].astype(np.float64),
                               equal_nan=True))
    if not eq_m:
        fail("%s %s_%d: metric(pred,true) != metrics.npy "
             "(recomputed=%s stored=%s)"
             % (label, ds, h, rec.tolist(), mets[:5].tolist()))
    return {"dir": os.path.relpath(dir_, C.ROOT), "n": int(true.shape[0]),
            "true_bitwise_vs_std_array": eq_std,
            "true_bitwise_vs_exp003_restored": eq_ln,
            "metrics_recompute_equals_stored": eq_m,
            "mse": float(rec[1]), "mae": float(rec[0]),
            "first_pred_abs_row0": C.first_pred_abs_row0(ds)}


def check_fingerprints(run_dir, ds, h):
    rec = C.parse_success(os.path.join(run_dir, "SUCCESS.txt"))
    cfg = rec["config"]
    c = C.DATASETS_T19[ds]
    want_cfg = {
        "seq_len": str(c["ctx"]), "periodicity": str(c["period"]),
        "pred_len": str(h), "norm_const": "0.4", "align_const": "0.4",
        "batch_size": "32", "train_epochs": "0", "seed": "2021",
    }
    for k, want in want_cfg.items():
        got = cfg.get(k)
        if got != want:
            fail("%s_%d SUCCESS.txt %s=%s != %s" % (ds, h, k, got, want))
    ckpt = os.path.join(C.ROOT, C.CKPT_REL)
    got_sha = C.sha256_of(ckpt)
    if rec["config"].get("ckpt_sha256") != got_sha:
        fail("%s_%d ckpt sha256 %s != current file %s"
             % (ds, h, rec["config"].get("ckpt_sha256"), got_sha))
    if got_sha != C.CKPT_SHA256:
        fail("%s_%d ckpt sha256 %s != pinned EXP-001 weights %s — locate "
             "the cause before proceeding" % (ds, h, got_sha, C.CKPT_SHA256))
    bad = []
    for f in C.FP_FILES:
        want = rec["md5"].get(f)
        cur = C.md5_of(os.path.join(C.ROOT, f))
        if want != cur:
            bad.append("%s: recorded=%s current=%s" % (f, want, cur))
    if bad:
        fail("%s_%d fingerprint md5 drift: %s" % (ds, h, "; ".join(bad)))
    arts = check_run_dir_clean(run_dir)
    if arts:
        fail("%s_%d unexpected checkpoint artifacts: %s" % (ds, h, arts))
    return {"config": cfg, "ckpt_sha256": got_sha,
            "md5_all_match": True, "checkpoint_artifacts": []}


def mode_group(ds, h):
    metric = C.load_metric()
    run_dir = C.freeze_dir(ds, h)
    if not os.path.isfile(os.path.join(run_dir, "SUCCESS.txt")):
        fail("%s_%d SUCCESS.txt missing under %s" % (ds, h, run_dir))
    fp = check_fingerprints(run_dir, ds, h)
    ev = verify_freeze_arrays(run_dir, ds, h, metric, "t22-freeze")
    ev.update({"kind": "t22_freeze_group", "dataset": ds, "horizon": h,
               "fingerprints": fp})
    C.merge_json(VERIFY_JSON, "group_%s_%d" % (ds, h), ev)
    print("[verify_t22] group %s_%d OK: n=%d true bitwise (std & EXP-003), "
          "metrics==stored, mse=%.6f mae=%.6f"
          % (ds, h, ev["n"], ev["mse"], ev["mae"]))


def mode_cache(ds, h):
    """User correction 3: per-group ETTm1 cache verification with a real
    rerun path — never a global abort."""
    metric = C.load_metric()
    d = C.exp001_dir(ds, h)
    sp = os.path.join(d, "SUCCESS.txt")
    if not os.path.isfile(sp):
        status, reason = "rerun_needed", "EXP-001 SUCCESS.txt missing"
    else:
        rec = C.parse_success(sp)
        ckpt = os.path.join(C.ROOT, C.CKPT_REL)
        got_sha = C.sha256_of(ckpt)
        if got_sha != C.CKPT_SHA256:
            fail("ckpt sha256 %s != pinned %s — weights changed, locate the "
                 "cause first (rerun cannot legitimize foreign weights)"
                 % (got_sha, C.CKPT_SHA256))
        if rec["config"].get("ckpt_sha256") != got_sha:
            status, reason = ("rerun_needed",
                              "EXP-001 recorded ckpt %s != current %s"
                              % (rec["config"].get("ckpt_sha256"), got_sha))
        else:
            drift = []
            for f in C.FP_FILES:
                want = rec["md5"].get(f)
                cur = C.md5_of(os.path.join(C.ROOT, f))
                if want != cur:
                    drift.append("%s: recorded=%s current=%s" % (f, want, cur))
            if drift:
                status = "rerun_needed"
                reason = "code fingerprint drift vs EXP-001: " + "; ".join(drift)
            else:
                status, reason = "cache_ok", ""
    if status == "rerun_needed" and "weights" in reason.lower():
        fail("ETTm1_%d %s: %s" % (h, status, reason))

    ev = {"kind": "exp001_cache_check", "dataset": ds, "horizon": h,
          "status": status, "reason": reason}
    if status == "cache_ok":
        arr = verify_freeze_arrays(d, ds, h, metric, "exp001-cache")
        ev["arrays"] = arr
        C.merge_json(VERIFY_JSON, "cache_%s_%d" % (ds, h), ev)
        print("[verify_t22] cache %s_%d OK (EXP-001 reuse): n=%d "
              "true bitwise (std & EXP-003), metrics==stored, mse=%.6f "
              "mae=%.6f" % (ds, h, arr["n"], arr["mse"], arr["mae"]))
    else:
        C.merge_json(VERIFY_JSON, "cache_%s_%d" % (ds, h), ev)
        print("[verify_t22] cache %s_%d %s: %s -> rerun this group"
              % (ds, h, status, reason))


def mode_first_window(ds, h):
    """Smoke extra (user correction 2): window 0 of the real test loader —
    L-step history and variable order vs std_array, bitwise."""
    import argparse
    import random
    import torch

    c = C.DATASETS_T19[ds]
    args = argparse.Namespace(
        task_name="long_term_forecast", is_training=0, model="VisionTS",
        data=c["data"], root_path=c["root_path"], data_path=c["data_path"],
        features="M", target="OT", scale=True, embed="timeF", freq="h",
        seq_len=c["ctx"], label_len=48, pred_len=h, seasonal_patterns=None,
        batch_size=32, num_workers=0)
    os.chdir(C.ROOT)
    sys.path.insert(0, C.ROOT)
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    from data_provider.data_factory import data_provider
    _, loader = data_provider(args, "test")
    n = len(loader.dataset)
    n_exp = C.expected_n_windows(ds, h)
    if n != n_exp:
        fail("%s_%d loader N=%d != expected %d" % (ds, h, n, n_exp))
    sample = loader.dataset[0]
    seq_x = sample[0].numpy().astype(np.float32)  # (L, 7)
    if seq_x.shape != (c["ctx"], 7):
        fail("%s_%d window0 history shape %s != (%d, 7)"
             % (ds, h, seq_x.shape, c["ctx"]))
    t0 = C.first_pred_abs_row0(ds)
    want = C.std_array(ds)[t0 - c["ctx"]:t0]
    if not np.array_equal(seq_x, want):
        d = np.abs(seq_x.astype(np.float64) - want.astype(np.float64))
        wv = int(np.unravel_index(int(np.argmax(d)), d.shape)[1])
        fail("%s_%d window0 history NOT bitwise == std[%d:%d] (worst var "
             "col %d — check border1/L or variable order)"
             % (ds, h, t0 - c["ctx"], t0, wv))
    ev = {"kind": "first_window_check", "dataset": ds, "horizon": h,
          "loader_n": n, "history_len": int(seq_x.shape[0]),
          "history_bitwise_vs_std_array": True,
          "history_rows": [t0 - c["ctx"], t0],
          "variable_order_check": "all 7 columns bitwise match std_array "
                                  "column order (HUFL, HULL, MUFL, MULL, "
                                  "LUFL, LULL, OT)"}
    C.merge_json(VERIFY_JSON, "first_window_%s_%d" % (ds, h), ev)
    print("[verify_t22] first_window %s_%d OK: loader N=%d, win0 history "
          "rows [%d,%d) bitwise == std_array (L=%d, 7 vars, order ok)"
          % (ds, h, n, t0 - c["ctx"], t0, c["ctx"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs=2, metavar=("DS", "H"))
    ap.add_argument("--cache", nargs=2, metavar=("DS", "H"))
    ap.add_argument("--first_window", nargs=2, metavar=("DS", "H"))
    a = ap.parse_args()
    if a.group:
        mode_group(a.group[0], int(a.group[1]))
    elif a.cache:
        mode_cache(a.cache[0], int(a.cache[1]))
    elif a.first_window:
        mode_first_window(a.first_window[0], int(a.first_window[1]))
    else:
        ap.error("one of --group/--cache/--first_window required")


if __name__ == "__main__":
    main()
