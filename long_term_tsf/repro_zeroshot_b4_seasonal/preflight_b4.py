"""EXP-031 preflight on the server. All checks read-only; exits non-zero on
any failure. F0: EXP-001 baseline groups on disk + code fingerprints — ONLY
the files EXP-001 itself recorded in SUCCESS.txt (FP_FILES), which is the
code-equivalence argument on the git-less eta tree; EXP-031's own scripts
are fingerprinted separately into this experiment's SUCCESS.txt files.
F3: baseline metrics recomputed from arrays with the original metric()
== metrics.npy exactly. F2: env snapshot written and compared (substring)
against EXP-001's env_snapshot_main.txt.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b4_common as C


def md5_lines_from_success(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("md5:") and "=" in line:
                k, v = line[4:].rsplit("=", 1)
                out[k] = v
    return out


def check_baseline_group(ds, h, metric):
    bd = C.base_dir(ds, h)
    sp = os.path.join(bd, "SUCCESS.txt")
    problems = []
    if not os.path.isfile(sp):
        return ["baseline SUCCESS.txt missing: %s" % sp]
    rec = md5_lines_from_success(sp)
    for f in C.FP_FILES:
        got = rec.get(f)
        want = C.md5_of(os.path.join(C.ROOT, f))
        if got != want:
            problems.append("md5 %s: baseline=%s current=%s" % (f, got, want))
    pred = np.load(os.path.join(bd, "results", "_", "pred.npy"))
    true = np.load(os.path.join(bd, "results", "_", "true.npy"))
    mets = np.load(os.path.join(bd, "results", "_", "metrics.npy"))
    n_exp = C.expected_n_windows(ds, h)
    if pred.shape != true.shape:
        problems.append("pred/true shape mismatch %s %s"
                        % (pred.shape, true.shape))
    if true.shape != (n_exp, h, 7):
        problems.append("true shape %s != expected (%d, %d, 7)"
                        % (true.shape, n_exp, h))
    mae_r, mse_r, rmse_r, mape_r, mspe_r = metric(pred, true)
    recomputed = np.array([mae_r, mse_r, rmse_r, mape_r, mspe_r], dtype=np.float64)
    if not np.array_equal(recomputed, mets[:5].astype(np.float64), equal_nan=True):
        problems.append("metrics.npy mismatch: recomputed=%s stored=%s"
                        % (recomputed.tolist(), mets[:5].tolist()))
    return problems


def check_env():
    envp = os.path.join(C.REPRO_ZERO_DIR, "logs", "env_snapshot_main.txt")
    problems, notes = [], {}
    base_text = ""
    if os.path.isfile(envp):
        with open(envp) as f:
            base_text = f.read()
    else:
        problems.append("EXP-001 env snapshot missing: %s" % envp)
    import platform
    import numpy, pandas, sklearn
    cur = {"Python": platform.python_version(),
           "numpy": numpy.__version__, "pandas": pandas.__version__,
           "sklearn": sklearn.__version__}
    notes["current"] = cur
    for k, v in cur.items():
        if base_text and v not in base_text:
            problems.append("env %s: version %r not found in EXP-001 "
                            "env_snapshot_main.txt" % (k, v))
    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report",
                    default=os.path.join(C.PKG, "logs", "preflight_report.json"))
    args = ap.parse_args()
    metric = C.load_metric()
    problems, report = [], {"groups": {}, "env": {}}

    for ds, h in C.group_ids():
        p = check_baseline_group(ds, h, metric)
        report["groups"]["%s_%d" % (ds, h)] = {"problems": p}
        problems += ["%s_%d: %s" % (ds, h, x) for x in p]

    pe, ne = check_env()
    report["env"] = {"problems": pe, **ne}
    problems += pe

    report["all_pass"] = not problems
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    C.dump_json(args.report, report)
    if problems:
        print("[preflight_b4] FAIL (%d problems)" % len(problems))
        for x in problems[:20]:
            print("  -", x)
        sys.exit(1)
    print("[preflight_b4] ALL PASS: 16 EXP-001 baseline groups on disk,")
    print("               recorded-file fingerprints match, env ok,")
    print("               baseline metrics recompute == metrics.npy.")


if __name__ == "__main__":
    main()
