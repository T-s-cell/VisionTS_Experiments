"""EXP-045 gates MG1-MG11.  Modes:
  --reuse    16 EXP-001 archived Base groups (read-only reuse): config
             fingerprint, ckpt sha, framework md5, truth bitwise, f64
             cast-then-recompute, PLUS the pinned Base overall anchor
             (0.34439535439014435/0.37042006477713585, <=1e-6)  [MG7]
  --smoke    2 default-config Base replays vs EXP-001 archived arrays
             (BITWISE pred/true) + full per-group checks on the 4 smoke
             new-arch configs (Large/Huge x ETTh1/H96, ETTm2/H720 -- these
             ARE queue groups, never re-run) + arch effectivity (Large pred
             != Base pred, Huge pred != Large pred, bitwise)  [MG9]
  --new      all 32 new groups (Large 16 + Huge 16): fingerprint == live,
             shape, truth bitwise, finite, recompute_f64 <=1e-6,
             log <=1e-12, per-arch geometry + norm_const recorded
             [MG3/4/5/6/8]
  --reentry  negative skip-if-done test: mutated SUCCESS.txt -> ABORT
             exit 4, file untouched (sandbox dir, real saves untouched)
             [MG10]
  --group ARCH DS H   single-group acceptance for the run queue (ARCH is
             the short form base|large|huge); appends to
             logs/group_accept.jsonl, exits non-zero on any failure
  --all      MG1 (framework md5 lineage + 3 live ckpt sha/bytes) + MG2
             (preflight json + geometry_table_t45.csv) + every mode above
             + MG11a (group_accept coverage of all 34 runs: 32 new + 2
             replays) -> logs/verify_t45.json
  --mg11     FINAL gate, run AFTER timing + collect (review correction #1:
             MG11 consumes their outputs): (a) means_t45.json vs
             metrics.npy recompute <=1e-12 for all 48 rows, (b) Base
             overall anchor recheck <=1e-6 from the means json, (c) timing
             json completeness: 16 (ds,H) x 3 arch rows with warmup/repeat/
             sync/inference_mode/eval/released fields and finite timings,
             (d) group_accept coverage again -> logs/verify_t45_mg11.json

NO paper-value comparisons are gates anywhere: deviations from paper
Table 17 anchors are report-only observations.  verify must run on eta
(EXP-001 arrays and the new arrays live there).

Tolerance layers: recompute_f64 vs saved metrics <=1e-6; log-parsed vs
saved metrics <=1e-12 (official print uses str(float) full repr, so
parsing recovers the exact same float).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t45_common as X

AP = argparse.ArgumentParser()
AP.add_argument("--reuse", action="store_true")
AP.add_argument("--smoke", action="store_true")
AP.add_argument("--new", action="store_true")
AP.add_argument("--reentry", action="store_true")
AP.add_argument("--group", nargs=3, metavar=("ARCH", "DS", "H"))
AP.add_argument("--mg11", action="store_true")
AP.add_argument("--all", action="store_true")
A = AP.parse_args()
if A.all:
    A.reuse = A.smoke = A.new = A.reentry = True

CHECKS = []
_TRUTH_CACHE = {}
_SMK_REPLAY = [("ETTh1", 96), ("ETTm1", 96)]
SMOKE_NEW = [("mae_large", "ETTh1", 96), ("mae_large", "ETTm2", 720),
             ("mae_huge", "ETTh1", 96), ("mae_huge", "ETTm2", 720)]


def truth(ds, h):
    if (ds, h) not in _TRUTH_CACHE:
        _TRUTH_CACHE[(ds, h)] = X.truth_from_std(ds, h)
    return _TRUTH_CACHE[(ds, h)]


def chk(gate, ok, detail=""):
    CHECKS.append({"gate": gate, "ok": bool(ok), "detail": str(detail)})
    return ok


def parse_fp(path):
    fp = {}
    for ln in X.read_fingerprint(path):
        if ln.startswith("md5:"):
            k, v = ln[4:].rsplit("=", 1)
            fp["md5:" + k] = v
            continue
        for tok in ln.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                fp[k] = v
    return fp


# ---- core per-group acceptance (new-arch runs + base replays) ----
def check_run_group(arch, ds, h):
    """Full per-group acceptance for a t45 run (arch in ARCHS; mae_base
    entries are the 2 smoke replays in save_t45_base_replay/)."""
    n0 = len(CHECKS)
    tag = "%s:%s/H%d" % (X.ARCH_SHORT[arch], ds, h)
    d = X.run_dir(ds, h, arch)
    sp = os.path.join(d, "SUCCESS.txt")
    if not chk(tag + ":success", os.path.isfile(sp), sp):
        return False
    want = X.fingerprint_lines(ds, h, arch)
    got = [ln for ln in X.read_fingerprint(sp)
           if not re.match(r"^(wall_time_s=|Peak GPU|nvidia_smi_"
                           r"device_peak_mib|log_metrics=|shape_record="
                           r"|finished=)", ln)]
    chk(tag + ":fingerprint", got == want,
        "config(arch=%s)+8 framework md5+2 t45 md5 == live" % arch)
    recs = []
    srp = os.path.join(d, "shape_record.jsonl")
    if os.path.isfile(srp):
        recs = [json.loads(ln) for ln in open(srp)
                if '"update_config"' in ln]
    inits = [json.loads(ln) for ln in (open(srp) if os.path.isfile(srp)
                                       else []) if '"init"' in ln]
    seg_n = sum(1 for ln in (open(srp) if os.path.isfile(srp) else [])
                if '"segmentation"' in ln)
    chk(tag + ":shape_record", len(recs) >= 1 and seg_n > 0 and len(inits) >= 1,
        "init=%d update_config=%d segmentation=%d"
        % (len(inits), len(recs), seg_n))
    if inits:
        chk(tag + ":init_arch_effective",
            inits[0].get("arch") == arch,
            "constructor received arch=%s" % inits[0].get("arch"))
    if recs:
        rec = recs[-1]
        geo_ok, geo_notes = X.geometry_matches(
            rec, X.expected_geometry(ds, h, X.PAPER_P[ds], X.PAPER_L[ds],
                                     arch))
        chk(tag + ":geometry", geo_ok, geo_notes)
        chk(tag + ":norm_const_effective",
            abs(float(rec.get("norm_const", -1)) - 0.4) <= 1e-12
            and abs(float(rec.get("align_const", -1)) - 0.4) <= 1e-12,
            "model stored norm_const=%s align_const=%s"
            % (rec.get("norm_const"), rec.get("align_const")))
    rdir = os.path.join(d, "results", "_")
    if not chk(tag + ":outputs",
               all(os.path.isfile(os.path.join(rdir, f))
                   for f in ("metrics.npy", "pred.npy", "true.npy")), rdir):
        return False
    n = X.expected_n_windows(ds, h)
    pred = np.load(os.path.join(rdir, "pred.npy"))
    tr = np.load(os.path.join(rdir, "true.npy"))
    chk(tag + ":shape",
        pred.shape == (n, h, 7) and tr.shape == (n, h, 7),
        "%s vs (%d,%d,7); N=4K-H+1 (arch-independent)" % (pred.shape, n, h))
    chk(tag + ":truth_bitwise", np.array_equal(tr, truth(ds, h)),
        "targets vs truth_from_std [16K,20K) (identical across arch)")
    chk(tag + ":finite",
        bool(np.isfinite(pred).all() and np.isfinite(tr).all()))
    mets = np.load(os.path.join(rdir, "metrics.npy"))
    chk(tag + ":metrics_layout",
        mets.shape == (7,) and mets[5] == -1 and mets[6] == -1
        and np.isfinite(mets[:5]).all(), "[mae,mse,rmse,mape,mspe,-1,-1]")
    p64, t64 = pred.astype(np.float64), tr.astype(np.float64)
    del pred, tr
    mse_r = float(np.mean((p64 - t64) ** 2))
    mae_r = float(np.mean(np.abs(p64 - t64)))
    del p64, t64
    chk(tag + ":recompute_f64",
        abs(mse_r - float(mets[1])) <= 1e-6
        and abs(mae_r - float(mets[0])) <= 1e-6,
        "f64 cast-then-recompute vs saved: mse diff %.2e, mae diff %.2e"
        % (abs(mse_r - float(mets[1])), abs(mae_r - float(mets[0]))))
    lp = X.log_path(ds, h, arch)
    lm = X.parse_log_metrics(open(lp).read()) if os.path.isfile(lp) else None
    chk(tag + ":log_metrics_1e-12",
        lm is not None
        and abs(lm["mse"] - float(mets[1])) <= 1e-12
        and abs(lm["mae"] - float(mets[0])) <= 1e-12,
        "log=%s vs metrics.npy (full-repr print, exact float)"
        % os.path.basename(lp))
    return all(c["ok"] for c in CHECKS[n0:])


# ---- MG7: Base reuse ----------------------------------------------------------
def check_reuse_group(ds, h):
    tag = "reuse_exp001:%s/H%d" % (ds, h)
    d = X.base_dir(ds, h)
    rdir = os.path.join(d, "results", "_")
    if not chk(tag + ":dir", os.path.isdir(rdir), rdir):
        return
    fp = parse_fp(os.path.join(d, "SUCCESS.txt"))
    chk(tag + ":cfg",
        fp.get("dataset") == ds and fp.get("pred_len") == str(h)
        and fp.get("seq_len") == str(X.PAPER_L[ds])
        and fp.get("periodicity") == str(X.PAPER_P[ds])
        and fp.get("norm_const") == "0.4" and fp.get("align_const") == "0.4"
        and fp.get("batch_size") == "32" and fp.get("train_epochs") == "0"
        and fp.get("seed") == "2021" and fp.get("arch") == "mae_base"
        and fp.get("root_path") == X.DATASETS[ds]["root_path"]
        and fp.get("data_path") == X.DATASETS[ds]["data_path"]
        and fp.get("data") == X.DATASETS[ds]["data"]
        and fp.get("features") == "M",
        "config identity (arch=mae_base, r=c=0.4, L=table8, P=paper)")
    chk(tag + ":ckpt", fp.get("ckpt_sha256") == X.CKPTS["mae_base"]["sha256"])
    cur = X.framework_md5_dict()
    bad = [f for f in X.FP_FILES if fp.get("md5:" + f) != cur[f]]
    chk(tag + ":framework_md5", not bad, str(bad))
    mets = np.load(os.path.join(rdir, "metrics.npy"))
    chk(tag + ":metrics_layout",
        mets.shape == (7,) and mets[5] == -1 and mets[6] == -1
        and np.isfinite(mets[:5]).all())
    pred = np.load(os.path.join(rdir, "pred.npy"))
    tr = np.load(os.path.join(rdir, "true.npy"))
    n = X.expected_n_windows(ds, h)
    chk(tag + ":shape", pred.shape == (n, h, 7) and tr.shape == (n, h, 7))
    chk(tag + ":truth_bitwise", np.array_equal(tr, truth(ds, h)),
        "EXP-001 targets cross-validate the recomputation pipeline itself")
    chk(tag + ":finite", bool(np.isfinite(pred).all()))
    p64, t64 = pred.astype(np.float64), tr.astype(np.float64)
    del pred, tr
    mse_r = float(np.mean((p64 - t64) ** 2))
    mae_r = float(np.mean(np.abs(p64 - t64)))
    del p64, t64
    chk(tag + ":recompute_f64",
        abs(mse_r - float(mets[1])) <= 1e-6
        and abs(mae_r - float(mets[0])) <= 1e-6,
        "f64 cast-then-recompute vs archived: mse diff %.2e, mae diff %.2e"
        % (abs(mse_r - float(mets[1])), abs(mae_r - float(mets[0]))))


def base_overall_from_metrics():
    """16 archived Base groups -> per-ds 4H equal mean -> 4-ds equal mean.
    Same formula as EXP-018 collect (collect_t18.py L174-185)."""
    ds_avg = {}
    for ds in X.ETT4:
        ms, ma = [], []
        for h in X.HORIZONS:
            mets = np.load(os.path.join(X.base_dir(ds, h), "results", "_",
                                        "metrics.npy"))
            ms.append(float(mets[1]))
            ma.append(float(mets[0]))
        ds_avg[ds] = (float(np.mean(ms)), float(np.mean(ma)))
    overall = (float(np.mean([v[0] for v in ds_avg.values()])),
               float(np.mean([v[1] for v in ds_avg.values()])))
    return overall, ds_avg


def mode_reuse():
    for ds, h in X.group_ids():
        check_reuse_group(ds, h)
    overall, _ = base_overall_from_metrics()
    chk("reuse:base_overall_anchor",
        abs(overall[0] - X.BASE_OVERALL["overall_mse"]) <= X.SELF_TOL
        and abs(overall[1] - X.BASE_OVERALL["overall_mae"]) <= X.SELF_TOL,
        "16-group Base overall %.17g/%.17g vs pinned %.17g/%.17g "
        "(tol %.0e)" % (overall[0], overall[1],
                        X.BASE_OVERALL["overall_mse"],
                        X.BASE_OVERALL["overall_mae"], X.SELF_TOL))


# ---- MG9: smoke ---------------------------------------------------------------
def mode_smoke():
    # (a) 2 default-config Base replays: full checks + bitwise vs EXP-001
    replay_preds = {}
    for ds, h in _SMK_REPLAY:
        tag = "replay:%s/H%d" % (ds, h)
        ok = check_run_group("mae_base", ds, h)
        d = X.run_dir(ds, h, "mae_base")
        rdir = os.path.join(d, "results", "_")
        if not ok or not os.path.isdir(rdir):
            continue
        bd = X.base_dir(ds, h)
        pr = np.load(os.path.join(rdir, "pred.npy"))
        replay_preds[(ds, h)] = pr
        chk(tag + ":pred_bitwise_vs_exp001",
            np.array_equal(pr, np.load(os.path.join(bd, "results", "_",
                                                    "pred.npy"))),
            "same code+data+GPU+arch+shape -> bitwise expected (t41/t43/t44 "
            "precedent)")
        chk(tag + ":true_bitwise_vs_exp001",
            np.array_equal(np.load(os.path.join(rdir, "true.npy")),
                           np.load(os.path.join(bd, "results", "_",
                                                "true.npy"))))
        m_r = np.load(os.path.join(rdir, "metrics.npy"))
        m_b = np.load(os.path.join(bd, "results", "_", "metrics.npy"))
        chk(tag + ":metrics_vs_exp001",
            abs(float(m_r[1]) - float(m_b[1])) <= 1e-9
            and abs(float(m_r[0]) - float(m_b[0])) <= 1e-9)
    # (b) the 4 smoke new-arch configs are queue groups: full checks
    for arch, ds, h in SMOKE_NEW:
        check_run_group(arch, ds, h)
    # (c) arch effectivity: different weights -> different predictions
    #     (bitwise inequality), holding everything else frozen
    for arch, ds, h in SMOKE_NEW:
        pr = np.load(os.path.join(X.run_dir(ds, h, arch), "results", "_",
                                  "pred.npy"))
        short = X.ARCH_SHORT[arch]
        if (ds, h) in replay_preds:
            chk("smoke:effect:%s_vs_base_%s" % (short, ds),
                not np.array_equal(pr, replay_preds[(ds, h)]),
                "pred(%s) != pred(base) bitwise -> arch effective" % short)
        other = "mae_huge" if arch == "mae_large" else "mae_large"
        op = os.path.join(X.run_dir(ds, h, other), "results", "_", "pred.npy")
        if os.path.isfile(op):
            chk("smoke:effect:%s_vs_%s_%s" % (short,
                                              X.ARCH_SHORT[other], ds),
                not np.array_equal(pr, np.load(op)),
                "pred(%s) != pred(%s) bitwise" % (short,
                                                  X.ARCH_SHORT[other]))


# ---- MG3/4/5/6/8: all 32 new groups -------------------------------------------
def mode_new():
    for arch in ("mae_large", "mae_huge"):
        for ds in X.ETT4:
            for h in X.HORIZONS:
                check_run_group(arch, ds, h)
    # geometry identity where the formula predicts it: same (ds,H), same
    # patch family (base vs large share num_patch=14/patch16; huge differs
    # by construction) -- base/large geometry records must be identical
    for ds in X.ETT4:
        for h in X.HORIZONS:
            subs = {}
            for arch in X.ARCHS:
                srp = os.path.join(X.run_dir(ds, h, arch),
                                   "shape_record.jsonl")
                if not os.path.isfile(srp):
                    subs = None
                    break
                recs = [json.loads(ln) for ln in open(srp)
                        if '"update_config"' in ln]
                subs[arch] = X.geometry_subset(recs[-1]) if recs else None
            if subs is None or not chk(
                    "new:geom_records:%s/H%d" % (ds, h),
                    all(v is not None for v in subs.values())):
                continue
            chk("new:geometry_base_eq_large:%s/H%d" % (ds, h),
                subs["mae_base"] == subs["mae_large"],
                "num_patch=14/patch16 shared -> identical geometry records")
            chk("new:geometry_huge_differs:%s/H%d" % (ds, h),
                subs["mae_huge"] != subs["mae_base"],
                "num_patch=16/patch14 -> npi/mask/resize differ per formula")


# ---- MG10: reentry negative ---------------------------------------------------
def mode_reentry():
    sandbox_rel = "save_t45_base_replay/_reentry_neg/ETTh1_96"
    sandbox = os.path.join(X.ROOT, sandbox_rel)
    os.makedirs(sandbox, exist_ok=True)
    ref = X.run_dir("ETTh1", 96, "mae_base")
    refsp = os.path.join(ref, "SUCCESS.txt")
    if not chk("reentry:smoke_success_exists", os.path.isfile(refsp), refsp):
        return
    poisoned = X.read_fingerprint(refsp)
    poisoned[0] = poisoned[0].replace("pred_len=96", "pred_len=999")
    X.write_text(os.path.join(sandbox, "SUCCESS.txt"), poisoned)
    before = X.read_fingerprint(os.path.join(sandbox, "SUCCESS.txt"))
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "run_one_t45.sh")
    r = subprocess.run(["bash", script, "ETTh1", "96", "base", sandbox_rel],
                       capture_output=True, text=True)
    chk("reentry:abort_exit4", r.returncode == 4,
        "exit=%d stdout=%s" % (r.returncode, r.stdout.strip()[:200]))
    chk("reentry:file_untouched",
        X.read_fingerprint(os.path.join(sandbox, "SUCCESS.txt")) == before)


# ---- MG11a: per-group acceptance coverage -------------------------------------
GROUP_ACCEPT_JSONL = os.path.join(X.LOG_DIR, "group_accept.jsonl")


def mode_group_accept():
    short, ds, h = A.group
    arch = X.ARCH_SHORT_INV[short]
    h = int(h)
    ok = check_run_group(arch, ds, h)
    rec = {"arch": arch, "ds": ds, "h": int(h), "ok": bool(ok),
           "n_checks_run": len(CHECKS),
           "n_fail": sum(1 for c in CHECKS if not c["ok"]),
           "t": time.time()}
    with open(GROUP_ACCEPT_JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if not ok:
        raise SystemExit("[verify --group] FAIL %s %s H%s" % (short, ds, h))


def queue_entries():
    """All 34 runs: 32 new-arch groups + 2 base replays."""
    out = [("new", arch, ds, h) for arch in ("mae_large", "mae_huge")
           for ds in X.ETT4 for h in X.HORIZONS]
    out += [("replay", "mae_base", ds, h) for ds, h in _SMK_REPLAY]
    return out


def mode_mg11a():
    entries = []
    if os.path.isfile(GROUP_ACCEPT_JSONL):
        for ln in open(GROUP_ACCEPT_JSONL):
            entries.append(json.loads(ln))
    want = queue_entries()
    chk("mg11a:queue_size", len(want) == 34, str(len(want)))
    seen = {}
    for e in entries:
        seen[(e["arch"], e["ds"], e["h"])] = e
    missing = [w for w in want if (w[1], w[2], w[3]) not in seen]
    chk("mg11a:group_accept_coverage", not missing,
        "%d/%d accepted, missing=%s" % (len(want) - len(missing),
                                        len(want), str(missing[:5])))
    bad = [k for k, e in seen.items() if not e["ok"]]
    chk("mg11a:group_accept_all_ok", not bad, str(bad[:5]))


def mode_mg11():
    """FINAL gate (correction #1): runs AFTER timing + collect.  Consumes
    logs/means_t45.json and logs/timing_t45.json."""
    mode_mg11a()
    # (a) means json vs metrics.npy <=1e-12 for all 48 rows
    mp = os.path.join(X.LOG_DIR, "means_t45.json")
    if not os.path.isfile(mp):
        chk("mg11:means_json", False, mp + " missing (run collect first)")
        return
    means = X.load_json(mp)
    nbad = []
    for arch, table in means.items():
        for ds, h, mse_m, mae_m in table:
            src = X.base_dir(ds, h) if arch == "mae_base" \
                else X.run_dir(ds, h, arch)
            mets = np.load(os.path.join(src, "results", "_", "metrics.npy"))
            if abs(mse_m - float(mets[1])) > 1e-12 \
                    or abs(mae_m - float(mets[0])) > 1e-12:
                nbad.append("%s %s H%d" % (arch, ds, h))
    n_rows = sum(len(v) for v in means.values())
    chk("mg11:means_vs_npy_1e-12", not nbad and n_rows == 48,
        "%d rows checked (want 48), bad=%s" % (n_rows, str(nbad[:5])))
    # (b) Base overall anchor recheck from the means json
    by_ds = {}
    for ds, h, mse_m, mae_m in means.get("mae_base", []):
        by_ds.setdefault(ds, []).append((mse_m, mae_m))
    if len(by_ds) == 4 and all(len(v) == 4 for v in by_ds.values()):
        ds_avg = [(np.mean([m for m, _ in v]), np.mean([a for _, a in v]))
                  for v in by_ds.values()]
        ov = (float(np.mean([v[0] for v in ds_avg])),
              float(np.mean([v[1] for v in ds_avg])))
        chk("mg11:base_overall_anchor",
            abs(ov[0] - X.BASE_OVERALL["overall_mse"]) <= X.SELF_TOL
            and abs(ov[1] - X.BASE_OVERALL["overall_mae"]) <= X.SELF_TOL,
            "overall %.17g/%.17g vs pinned (tol %.0e)"
            % (ov[0], ov[1], X.SELF_TOL))
    else:
        chk("mg11:base_overall_anchor", False,
            "means json mae_base incomplete: %s" % str({k: len(v)
                                                        for k, v in
                                                        by_ds.items()}))
    # (c) timing json completeness + protocol fields
    tp = os.path.join(X.LOG_DIR, "timing_t45.json")
    if not os.path.isfile(tp):
        chk("mg11:timing_json", False, tp + " missing (run timing first)")
        return
    tj = X.load_json(tp)
    rows = tj.get("rows", [])
    meta = tj.get("meta", {})
    bad_rows = []
    for r in rows:
        for k in ("ds", "h", "arch", "mean_s", "min_s", "peak_gb",
                  "load_s"):
            if k not in r or r[k] is None:
                bad_rows.append("%s %s missing %s" % (r.get("ds"),
                                                      r.get("arch"), k))
                break
        if isinstance(r.get("mean_s"), float) and not np.isfinite(r["mean_s"]):
            bad_rows.append("%s nonfinite mean_s" % (r.get("ds"),))
    chk("mg11:timing_rows_48", len(rows) == 48 and not bad_rows,
        "%d rows (want 48), bad=%s" % (len(rows), str(bad_rows[:5])))
    chk("mg11:timing_protocol",
        meta.get("warmup") == 3 and meta.get("repeats") == 5
        and meta.get("cuda_sync") is True
        and meta.get("inference_mode") is True
        and meta.get("eval") is True
        and meta.get("released_between_archs") is True
        and meta.get("batch_size") == 32,
        "eval()+inference_mode(), warmup/repeats/sync/sequential-release "
        "recorded (correction #6): %s" % json.dumps(meta)[:200])


# ---- MG1/MG2 ------------------------------------------------------------------
def mode_mg1_mg2():
    cur = X.framework_md5_dict()
    ref = os.path.join(X.p1_dir("ETTh1", 96), "SUCCESS.txt")
    recorded = {}
    for ln in X.read_fingerprint(ref):
        m = re.match(r"md5:(.+)=([0-9a-f]{32})$", ln)
        if m and m.group(1) in cur:
            recorded[m.group(1)] = m.group(2)
    bad = [f for f in X.FP_FILES if recorded.get(f) != cur[f]]
    chk("mg1:framework_md5_lineage", not bad, str(bad))
    for arch in X.ARCHS:
        p = X.CKPT_FILE[arch]
        okc = os.path.isfile(p) \
            and X.sha256_of(p) == X.CKPTS[arch]["sha256"] \
            and os.path.getsize(p) == X.CKPTS[arch]["bytes"]
        chk("mg1:ckpt_live_sha_%s" % X.ARCH_SHORT[arch], okc, p)
    pf = os.path.join(X.LOG_DIR, "preflight_t45.json")
    okp = os.path.isfile(pf) and X.load_json(pf).get("all_preflight_pass")
    chk("mg2:preflight", bool(okp), pf)
    chk("mg2:geometry_table_csv", os.path.isfile(X.GEOMETRY_TABLE_CSV),
        X.GEOMETRY_TABLE_CSV)


# ---- main -----------------------------------------------------------------------
def main():
    t0 = time.time()
    modes = []
    summary = None
    if A.mg11:
        mode_mg11()
        modes.append("mg11")
        summary = {"modes": modes, "n_checks": len(CHECKS),
                   "n_fail": sum(1 for c in CHECKS if not c["ok"]),
                   "all_pass": all(c["ok"] for c in CHECKS),
                   "wall_s": round(time.time() - t0, 1), "checks": CHECKS}
        X.dump_json(os.path.join(X.LOG_DIR, "verify_t45_mg11.json"), summary)
    elif A.group:
        mode_group_accept()
        modes.append("group")
    else:
        if A.reuse:
            mode_reuse()
            modes.append("reuse")
        if A.smoke:
            mode_smoke()
            modes.append("smoke")
        if A.new:
            mode_new()
            modes.append("new")
        if A.reentry:
            mode_reentry()
            modes.append("reentry")
        if A.all:
            mode_mg1_mg2()
            modes.append("mg1_mg2")
            mode_mg11a()
            modes.append("mg11a")
        n_fail = sum(1 for c in CHECKS if not c["ok"])
        summary = {"modes": modes, "n_checks": len(CHECKS), "n_fail": n_fail,
                   "all_pass": n_fail == 0,
                   "wall_s": round(time.time() - t0, 1), "checks": CHECKS}
        if A.all:
            X.dump_json(os.path.join(X.LOG_DIR, "verify_t45_all.json"),
                        summary)
    if summary is not None:
        X.dump_json(os.path.join(X.LOG_DIR, "verify_t45.json"), summary)
        print("[verify] modes=%s checks=%d fail=%d all_pass=%s wall=%.1fs"
              % (modes, summary["n_checks"], summary["n_fail"],
                 summary["all_pass"], summary["wall_s"]))
        for c in CHECKS:
            if not c["ok"]:
                print("  FAIL %s %s" % (c["gate"], c["detail"][:300]))
        if summary["n_fail"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
