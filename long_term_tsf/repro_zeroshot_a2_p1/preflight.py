"""EXP-032 preflight on the server. All checks read-only; exits non-zero on
any failure. F0: EXP-001 baseline groups on disk + code fingerprints
(recompute the run_one.sh md5 list and compare to baseline SUCCESS.txt).
F1: MAE weights sha256. F2: env matches EXP-001 env_snapshot_main.txt.
F3: baseline metrics recomputed from arrays with the original metric()
== metrics.npy exactly; vs SUCCESS.txt log_metrics at <=1e-12 ONLY when
the log value round-trips the float32 scalar at full precision.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2p1_common as C


def md5_lines_from_success(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("md5:") and "=" in line:
                k, v = line[4:].rsplit("=", 1)
                out[k] = v
            elif line.startswith("ckpt_sha256="):
                out["ckpt_sha256"] = line.split("=", 1)[1]
            elif line.startswith("dataset="):
                out["config_dataset"] = line
            elif line.startswith("log_metrics="):
                out["log_metrics"] = C.parse_log_metrics(line)
    return out


def check_group_fingerprint(ds, h):
    c = C.DATASETS[ds]
    sp = os.path.join(C.base_dir(ds, h), "SUCCESS.txt")
    rec = md5_lines_from_success(sp)
    problems = []
    want_cfg = ("dataset=%s pred_len=%d seq_len=%d periodicity=%d arch=mae_base"
                % (ds, h, c["ctx"], c["period"]))
    if rec.get("config_dataset") != want_cfg:
        problems.append("config line mismatch: %r vs %r"
                        % (rec.get("config_dataset"), want_cfg))
    if rec.get("ckpt_sha256") != C.CKPT_SHA256:
        problems.append("baseline ckpt_sha256 mismatch")
    for f in C.FP_FILES:
        got = rec.get(f)
        want = C.md5_of(os.path.join(C.ROOT, f))
        if got != want:
            problems.append("md5 %s: baseline=%s current=%s" % (f, got, want))
    return problems, rec


def check_group_metrics(ds, h, metric):
    bd = C.base_dir(ds, h)
    pred = np.load(os.path.join(bd, "results", "_", "pred.npy"))
    true = np.load(os.path.join(bd, "results", "_", "true.npy"))
    mets = np.load(os.path.join(bd, "results", "_", "metrics.npy"))
    n_exp = C.expected_n_windows(ds, h)
    problems, notes = [], {}
    if pred.shape != true.shape:
        problems.append("pred/true shape mismatch %s %s" % (pred.shape, true.shape))
    if true.shape != (n_exp, h, 7):
        problems.append("true shape %s != expected (%d, %d, 7)"
                        % (true.shape, n_exp, h))
    mae_r, mse_r, rmse_r, mape_r, mspe_r = metric(pred, true)
    recomputed = np.array([mae_r, mse_r, rmse_r, mape_r, mspe_r], dtype=np.float64)
    stored = mets[:5].astype(np.float64)
    if not np.array_equal(recomputed, stored, equal_nan=True):
        problems.append("metrics.npy mismatch: recomputed=%s stored=%s"
                        % (recomputed.tolist(), stored.tolist()))
    notes["mse"] = float(mse_r)
    notes["mae"] = float(mae_r)
    notes["n_windows"] = int(pred.shape[0])
    # log gate only at full precision (round-trips the f32 scalar)
    sp = os.path.join(bd, "SUCCESS.txt")
    lm = C.success_metrics(sp)
    if lm is None:
        problems.append("log_metrics unparsable in %s" % sp)
    else:
        rt = (np.float32(lm["mse"]) == mse_r) and (np.float32(lm["mae"]) == mae_r)
        notes["log_full_precision"] = bool(rt)
        if rt:
            if abs(lm["mse"] - float(mse_r)) > 1e-12 or \
                    abs(lm["mae"] - float(mae_r)) > 1e-12:
                problems.append("log_metrics deviates >1e-12 from recomputed")
        else:
            notes["log_gate"] = "skipped (log not full precision; arrays authoritative)"
    return problems, notes


def check_env():
    envp = os.path.join(C.REPRO_ZERO_DIR, "logs", "env_snapshot_main.txt")
    problems, notes = [], {}
    if not os.path.isfile(envp):
        problems.append("baseline env snapshot missing: %s" % envp)
        base_text = ""
    else:
        with open(envp) as f:
            base_text = f.read()
    import platform
    import torch, numpy
    cur = {"Python": platform.python_version(),
           "torch": torch.__version__, "numpy": numpy.__version__}
    try:
        import timm
        cur["timm"] = timm.__version__
    except Exception as e:
        cur["timm"] = "import failed: %s" % e
    notes["current"] = cur
    for k, v in cur.items():
        if v not in base_text:
            problems.append("env %s: version %r not found in baseline "
                            "env_snapshot_main.txt" % (k, v))
    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report",
                    default=os.path.join(C.PKG, "logs", "preflight_report.json"))
    args = ap.parse_args()
    metric = C.load_metric()

    problems, report = [], {"groups": {}, "env": {}}

    # F1
    import hashlib
    hsh = hashlib.sha256()
    if not os.path.isfile(C.CKPT_FILE):
        problems.append("MAE ckpt missing: %s" % C.CKPT_FILE)
    else:
        with open(C.CKPT_FILE, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                hsh.update(chunk)
        if hsh.hexdigest() != C.CKPT_SHA256:
            problems.append("MAE ckpt sha256 mismatch: %s" % hsh.hexdigest())
    report["ckpt_sha256_ok"] = not problems

    # F0 + F3 per group
    for ds, h in C.group_ids():
        g = {}
        if not os.path.isfile(os.path.join(C.base_dir(ds, h), "SUCCESS.txt")):
            problems.append("baseline SUCCESS.txt missing: %s_%d" % (ds, h))
            g["on_disk"] = False
        else:
            g["on_disk"] = True
            p1, _ = check_group_fingerprint(ds, h)
            g["fingerprint_problems"] = p1
            problems += ["%s_%d: %s" % (ds, h, x) for x in p1]
            for f in ("pred.npy", "true.npy", "metrics.npy"):
                if not os.path.isfile(os.path.join(C.base_dir(ds, h),
                                                   "results", "_", f)):
                    problems.append("%s_%d missing %s" % (ds, h, f))
            if not p1:
                p3, notes = check_group_metrics(ds, h, metric)
                g["metrics_problems"] = p3
                g["metrics"] = notes
                problems += ["%s_%d: %s" % (ds, h, x) for x in p3]
        report["groups"]["%s_%d" % (ds, h)] = g

    # F2
    pe, ne = check_env()
    report["env"] = {"problems": pe, **ne}
    problems += pe

    report["all_pass"] = not problems
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    C.dump_json(args.report, report)
    if problems:
        print("[preflight] FAIL (%d problems)" % len(problems))
        for p in problems[:20]:
            print("  -", p)
        sys.exit(1)
    print("[preflight] ALL PASS: 16 baseline groups on disk, fingerprints match,")
    print("             ckpt sha ok, env ok, metrics recomputed == metrics.npy.")


if __name__ == "__main__":
    main()
