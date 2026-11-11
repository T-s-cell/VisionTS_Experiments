"""EXP-043 gates.  Modes (combinable):
  --valfull      48 validation configs: window count, truth bitwise vs
                 truth_from_std (independent recompute), finiteness,
                 metric recompute consistency, geometry vs formula,
                 fingerprint freshness, explicit split fields  [VG3-VG6, VG10]
  --smoke        the 6 smoke val units incl. --crosspath bitwise evidence +
                 test-path smoke run vs EXP-001 archived group (bitwise)
                 [VG13]
  --ledger       ledger internal consistency (sha256 chain, mean recompute,
                 argmin+tie rule) and ORDERING: acceptance json and LEDGER_OK
                 mtimes precede every new-test SUCCESS  [VG9, VG11]
  --reuse_exp001 16 archived default-P groups: config fingerprint, ckpt sha,
                 framework md5 == current, truth bitwise, shapes, finite,
                 f64 metric recompute <=1e-6  [VG7]
  --reuse_exp030 16 archived P=1 groups: same + metrics json cross-check
                 [VG8]
  --newtest      selected-arm new runs: fingerprint == live --fingerprint
                 (binds ledger sha), shape_record geometry, truth bitwise,
                 f64 recompute  [VG6 subset]
  --reentry      negative skip-if-done test: mutated SUCCESS.txt -> ABORT
                 exit 4, file untouched  [VG12]
  --all          everything above -> logs/verify_t43.json

NO paper-value comparisons are gates (user correction #4): deviations from
paper numbers are collected by collect_t43.py as report-only observations.
Gates verify frozen archives and self-consistency only.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t43_common as X

AP = argparse.ArgumentParser()
AP.add_argument("--valfull", action="store_true")
AP.add_argument("--smoke", action="store_true")
AP.add_argument("--ledger", action="store_true")
AP.add_argument("--reuse_exp001", action="store_true")
AP.add_argument("--reuse_exp030", action="store_true")
AP.add_argument("--newtest", action="store_true")
AP.add_argument("--reentry", action="store_true")
AP.add_argument("--all", action="store_true")
A = AP.parse_args()
if A.all:
    A.valfull = A.smoke = A.ledger = A.reuse_exp001 = \
        A.reuse_exp030 = A.newtest = A.reentry = True

SMOKE_UNITS = [("ETTh1", 96, 24), ("ETTh1", 96, 168), ("ETTh1", 96, 1),
               ("ETTm1", 96, 96), ("ETTm1", 96, 672), ("ETTm1", 96, 1)]

_TRUTH_CACHE = {}


def truth(ds, target, h):
    key = (ds, target, h)
    if key not in _TRUTH_CACHE:
        _TRUTH_CACHE[key] = X.truth_from_std(ds, target, h)
    return _TRUTH_CACHE[key]


CHECKS = []


def chk(gate, ok, detail=""):
    CHECKS.append({"gate": gate, "ok": bool(ok), "detail": str(detail)})
    return ok


def fail_hard(gate, detail):
    chk(gate, False, detail)
    raise SystemExit("[verify] HARD FAIL %s: %s" % (gate, detail))


# ---- valfull ----------------------------------------------------------------
def val_one(ds, P, h, need_crosspath=False):
    tag = "val:%s/P%d/H%d" % (ds, P, h)
    jp = X.vjson_path(ds, P, h)
    if not chk(tag + ":json", os.path.isfile(jp), jp):
        return
    j = X.load_json(jp)
    exp_n = X.expected_n_windows(ds, h)
    chk(tag + ":n_windows", j["n_windows"] == exp_n,
        "%d vs %d" % (j["n_windows"], exp_n))
    sp = X.split_rows(ds)
    chk(tag + ":split_fields",
        j["split"]["K"] == X.month_rows(ds)
        and j["split"]["val_start"] == sp["val_start"]
        and j["split"]["val_end"] == sp["val_end"]
        and j["split"]["last_target_row_end"] == sp["val_end"],
        "explicit [12K,16K) target region")
    arrp = os.path.join(X.val_dir(ds, h, P), "arr_val.npz")
    if not chk(tag + ":arr", os.path.isfile(arrp), arrp):
        return
    z = np.load(arrp)
    pred, tr = z["pred"], z["true"]
    chk(tag + ":shape", pred.shape == (exp_n, h, 7)
        and tr.shape == (exp_n, h, 7), str(pred.shape))
    tref = truth(ds, "val", h)
    chk(tag + ":truth_bitwise", np.array_equal(tr, tref),
        "independent recompute vs arr_val true")
    chk(tag + ":finite", bool(np.isfinite(pred).all() and np.isfinite(tr).all()))
    p64, t64 = pred.astype(np.float64), tr.astype(np.float64)
    sse = float(np.sum((p64 - t64) ** 2))
    sae = float(np.sum(np.abs(p64 - t64)))
    chk(tag + ":sse_recompute",
        abs(sse - j["sse_f64"]) <= 1e-6 * max(1.0, abs(j["sse_f64"]))
        and abs(sae - j["sae_f64"]) <= 1e-6 * max(1.0, abs(j["sae_f64"])),
        "f64 recompute vs recorded")
    chk(tag + ":metric_consistency",
        abs(j["mse_metric"] - j["mse_from_sse"])
        <= 1e-6 * max(1.0, abs(j["mse_from_sse"]))
        and abs(j["mae_metric"] - j["mae_from_sae"])
        <= 1e-6 * max(1.0, abs(j["mae_from_sae"])),
        "metric() vs SSE/SAE f64 estimators")
    geo_ok, geo_notes = X.geometry_matches(j["geometry_observed"],
                                           X.expected_geometry(ds, h, P))
    chk(tag + ":geometry", geo_ok and j["geometry_ok"], geo_notes)
    want = X.fingerprint_lines(ds, h, P,
                               extra_files=["repro_t43_period_val/val_pass.py"])
    chk(tag + ":fingerprint_fresh", j["fingerprint"] == want,
        "code unchanged since val run")
    if need_crosspath:
        cp = j.get("crosspath")
        chk(tag + ":crosspath", bool(cp and cp.get("pred_bitwise")
                                     and cp.get("true_bitwise")),
            "same val batches via verbatim exp.test loop body")


def mode_valfull():
    for ds in X.ETT4:
        for P in X.CANDIDATE_P[ds]:
            for h in X.HORIZONS:
                val_one(ds, P, h)


# ---- smoke ------------------------------------------------------------------
def mode_smoke():
    for ds, h, P in SMOKE_UNITS:
        val_one(ds, P, h, need_crosspath=True)
    # test-path smoke run (P=24 == paper P, ETTh1/H96) vs EXP-001 archived
    sd = os.path.join(X.SMOKE_BASE, "ETTh1_96", "results", "_")
    bd = X.base_dir("ETTh1", 96)
    ok_files = all(os.path.isfile(os.path.join(sd, f))
                   for f in ("metrics.npy", "pred.npy", "true.npy"))
    if not chk("smoke:testrun:outputs", ok_files, sd):
        return
    m_s = np.load(os.path.join(sd, "metrics.npy"))
    m_b = np.load(os.path.join(bd, "results", "_", "metrics.npy"))
    chk("smoke:testrun:metrics_vs_exp001",
        abs(float(m_s[1]) - float(m_b[1])) <= 1e-9
        and abs(float(m_s[0]) - float(m_b[0])) <= 1e-9,
        "mse/mae bitwise-level vs archived EXP-001")
    chk("smoke:testrun:pred_bitwise",
        np.array_equal(np.load(os.path.join(sd, "pred.npy")),
                       np.load(os.path.join(bd, "results", "_", "pred.npy"))),
        "same code+data+GPU+shape -> bitwise expected (t41 same-shape rerun)")
    chk("smoke:testrun:true_bitwise",
        np.array_equal(np.load(os.path.join(sd, "true.npy")),
                       truth("ETTh1", "test", 96)),
        "and == archived true (recompute pipeline validated)")


# ---- ledger -----------------------------------------------------------------
def mode_ledger():
    if not chk("ledger:LEDGER_OK", os.path.isfile(X.LEDGER_OK), X.LEDGER_OK):
        return
    ok_lines = X.read_fingerprint(X.LEDGER_OK)
    sha_rec = {}
    for ln in ok_lines:
        m = re.match(r"sha256:(.+)=([0-9a-f]{64})$", ln)
        if m:
            sha_rec[m.group(1)] = m.group(2)
    for name, path in (("val_table.csv", X.LEDGER_CSV),
                       ("selection.json", X.LEDGER_JSON)):
        chk("ledger:sha:" + name,
            name in sha_rec and sha_rec[name] == X.sha256_of(path), path)
    for ds in X.ETT4:
        for P in X.CANDIDATE_P[ds]:
            for h in X.HORIZONS:
                n = "val_%s_P%d_H%d.json" % (ds, P, h)
                chk("ledger:sha:" + n,
                    n in sha_rec
                    and sha_rec[n] == X.sha256_of(X.vjson_path(ds, P, h)))
    sel = X.load_json(X.LEDGER_JSON)
    for ds in X.ETT4:
        for P in X.CANDIDATE_P[ds]:
            ms = [X.load_json(X.vjson_path(ds, P, h))["mse_from_sse"]
                  for h in X.HORIZONS]
            rec = sel["per_ds"][ds]["candidates"][str(P)]["mean_mse"]
            chk("ledger:mean_recompute:%s/P%d" % (ds, P),
                abs(rec - sum(ms) / len(ms)) <= 1e-12)
        chosen = sel["per_ds"][ds]["chosen_P"]
        means = {P: sel["per_ds"][ds]["candidates"][str(P)]["mean_mse"]
                 for P in X.CANDIDATE_P[ds]}
        best = min(means.values())
        tied = [p for p, v in means.items() if v == best]
        want = X.PAPER_P[ds] if X.PAPER_P[ds] in tied else min(tied)
        chk("ledger:argmin+tie:%s" % ds, chosen == want,
            "chosen=%s tied=%s" % (chosen, tied))
    # ordering: acceptance -> ledger -> every new-test SUCCESS
    t_ledger = os.path.getmtime(X.LEDGER_OK)
    t_vf = os.path.getmtime(X.VALFULL_JSON)
    chk("ledger:order:valfull_before_ledger", t_vf <= t_ledger,
        "accept-before-select")
    later = []
    for f in glob.glob(os.path.join(X.SAVE_BASE, "*", "SUCCESS.txt")):
        if os.path.getmtime(f) < t_ledger:
            later.append(f)
    chk("ledger:order:ledger_before_all_tests", not later, str(later))


# ---- reuse arms ---------------------------------------------------------------
def parse_fp(path):
    fp = {}
    for ln in X.read_fingerprint(path):
        if "=" in ln and not ln.startswith("md5:"):
            k, v = ln.split("=", 1)
            fp[k] = v
        elif ln.startswith("md5:"):
            k, v = ln[4:].rsplit("=", 1)
            fp["md5:" + k] = v
    return fp


def check_archived_group(arm, ds, h, P):
    tag = "%s:%s/H%d" % (arm, ds, h)
    d = X.base_dir(ds, h) if arm == "reuse_exp001" else X.p1_dir(ds, h)
    rdir = os.path.join(d, "results", "_")
    if not chk(tag + ":dir", os.path.isdir(rdir), rdir):
        return
    fp = parse_fp(os.path.join(d, "SUCCESS.txt"))
    chk(tag + ":cfg",
        fp.get("dataset") == ds and fp.get("pred_len") == str(h)
        and fp.get("seq_len") == str(X.DATASETS[ds]["ctx"])
        and fp.get("periodicity") == str(P)
        and fp.get("norm_const") == "0.4" and fp.get("align_const") == "0.4"
        and fp.get("batch_size") == "32" and fp.get("train_epochs") == "0"
        and fp.get("seed") == "2021"
        and fp.get("root_path") == X.DATASETS[ds]["root_path"]
        and fp.get("data_path") == X.DATASETS[ds]["data_path"]
        and fp.get("data") == X.DATASETS[ds]["data"]
        and fp.get("features") == "M",
        "config identity vs frozen protocol")
    chk(tag + ":ckpt", fp.get("ckpt_sha256") == X.CKPT_SHA256)
    cur = X.framework_md5_dict()
    bad = [f for f in X.FP_FILES
           if fp.get("md5:" + f) != cur[f]]
    chk(tag + ":framework_md5", not bad, str(bad))
    mets = np.load(os.path.join(rdir, "metrics.npy"))
    chk(tag + ":metrics_layout", mets.shape == (7,) and mets[5] == -1
        and mets[6] == -1 and np.isfinite(mets[:5]).all(),
        "[mae,mse,rmse,mape,mspe,-1,-1]")
    pred = np.load(os.path.join(rdir, "pred.npy"))
    tr = np.load(os.path.join(rdir, "true.npy"))
    n = X.expected_n_windows(ds, h)
    chk(tag + ":shape", pred.shape == (n, h, 7) and tr.shape == (n, h, 7),
        "%s vs (%d,%d,7)" % (pred.shape, n, h))
    chk(tag + ":truth_bitwise", np.array_equal(tr, truth(ds, "test", h)),
        "archived test targets vs truth_from_std [16K,20K)")
    chk(tag + ":finite", bool(np.isfinite(pred).all()))
    metric = X.load_metric()
    mae_r, mse_r = metric(pred, tr)[0], metric(pred, tr)[1]
    chk(tag + ":recompute",
        abs(float(mse_r) - float(mets[1])) <= 1e-6
        and abs(float(mae_r) - float(mets[0])) <= 1e-6,
        "f64 vs archived %.10f/%.10f" % (mse_r, mae_r))
    if arm == "reuse_exp030":
        jj = X.load_json(os.path.join(d, "metrics_%s_%d.json" % (ds, h)))
        chk(tag + ":json_cross",
            abs(jj["p1"]["mse"] - float(mets[1])) <= 1e-9
            and abs(jj["p1"]["mae"] - float(mets[0])) <= 1e-9,
            "EXP-030 acceptance json vs metrics.npy")


def mode_reuse_exp001():
    for ds, h in X.group_ids():
        check_archived_group("reuse_exp001", ds, h, X.PAPER_P[ds])


def mode_reuse_exp030():
    for ds, h in X.group_ids():
        check_archived_group("reuse_exp030", ds, h, 1)


# ---- newtest ------------------------------------------------------------------
def selected_arm_groups():
    sel = X.load_json(X.LEDGER_JSON)
    out = []
    for ds in X.ETT4:
        P = sel["per_ds"][ds]["chosen_P"]
        if P in (X.PAPER_P[ds], 1):
            continue
        for h in X.HORIZONS:
            out.append((ds, h, P))
    return out


def mode_newtest():
    groups = selected_arm_groups()
    chk("newtest:count", len(groups) <= 16, str(len(groups)))
    if not groups:
        chk("newtest:none_needed", True,
            "selected P coincides with default or 1 for every ds")
        return
    fp_cmd = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "run_one_t43.sh")
    for ds, h, P in groups:
        tag = "newtest:%s/H%d/P%d" % (ds, h, P)
        d = X.t43_run_dir(ds, h)
        sp = os.path.join(d, "SUCCESS.txt")
        if not chk(tag + ":success", os.path.isfile(sp), sp):
            continue
        live = subprocess.run(["bash", fp_cmd, "--fingerprint", ds, str(h),
                               str(P)], capture_output=True, text=True).stdout
        want = [ln for ln in live.splitlines() if ln.strip()]
        got = [ln for ln in X.read_fingerprint(sp)
               if not re.match(r"^(wall_time_s=|Peak GPU|nvidia_smi_"
                               r"device_peak_mib|log_metrics=|shape_record="
                               r"|finished=)", ln)]
        chk(tag + ":fingerprint", got == want,
            "incl. ledger_sha256 binding")
        recs = [json.loads(ln) for ln in
                open(os.path.join(d, "shape_record.jsonl"))
                if '"update_config"' in ln]
        chk(tag + ":geometry",
            len(recs) >= 1
            and X.geometry_matches(recs[-1], X.expected_geometry(ds, h, P))[0],
            "observed update_config vs formula")
        rdir = os.path.join(d, "results", "_")
        pred = np.load(os.path.join(rdir, "pred.npy"))
        tr = np.load(os.path.join(rdir, "true.npy"))
        n = X.expected_n_windows(ds, h)
        chk(tag + ":shape", pred.shape == (n, h, 7) and tr.shape == (n, h, 7))
        chk(tag + ":truth_bitwise", np.array_equal(tr, truth(ds, "test", h)))
        chk(tag + ":finite", bool(np.isfinite(pred).all()))
        mets = np.load(os.path.join(rdir, "metrics.npy"))
        metric = X.load_metric()
        mse_r = metric(pred, tr)[1]
        mae_r = metric(pred, tr)[0]
        chk(tag + ":recompute",
            abs(float(mse_r) - float(mets[1])) <= 1e-6
            and abs(float(mae_r) - float(mets[0])) <= 1e-6)
        lp = os.path.join(X.LOG_DIR, "%s_%d_P%d.log" % (ds, h, P))
        lm = X.parse_log_metrics(open(lp).read()) \
            if os.path.isfile(lp) else None
        if lm:
            chk(tag + ":log_metrics",
                abs(lm["mse"] - float(mets[1])) <= 1e-12
                and abs(lm["mae"] - float(mets[0])) <= 1e-12)


# ---- reentry (negative) -------------------------------------------------------
def mode_reentry():
    sandbox = os.path.join(X.SMOKE_BASE, "_reentry_neg", "ETTh1_96")
    os.makedirs(sandbox, exist_ok=True)
    ref = os.path.join(X.SMOKE_BASE, "ETTh1_96", "SUCCESS.txt")
    if not chk("reentry:smoke_success_exists", os.path.isfile(ref), ref):
        return
    poisoned = X.read_fingerprint(ref)
    poisoned[0] = poisoned[0].replace("pred_len=96", "pred_len=999")
    X.write_text(os.path.join(sandbox, "SUCCESS.txt"), poisoned)
    before = X.read_fingerprint(os.path.join(sandbox, "SUCCESS.txt"))
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "run_one_t43.sh")
    env = dict(os.environ, LEDGER_REQUIRED="0")
    r = subprocess.run(["bash", script, "ETTh1", "96",
                        "save_t43_period_val_smoke/_reentry_neg", "24"],
                       capture_output=True, text=True, env=env)
    chk("reentry:abort_exit4", r.returncode == 4,
        "exit=%d stdout=%s" % (r.returncode, r.stdout.strip()[:200]))
    chk("reentry:file_untouched",
        X.read_fingerprint(os.path.join(sandbox, "SUCCESS.txt")) == before)


# ---- main -----------------------------------------------------------------------
def main():
    t0 = time.time()
    modes = []
    if A.valfull:
        mode_valfull()
        modes.append("valfull")
    if A.smoke:
        mode_smoke()
        modes.append("smoke")
    if A.ledger:
        mode_ledger()
        modes.append("ledger")
    if A.reuse_exp001:
        mode_reuse_exp001()
        modes.append("reuse_exp001")
    if A.reuse_exp030:
        mode_reuse_exp030()
        modes.append("reuse_exp030")
    if A.newtest:
        mode_newtest()
        modes.append("newtest")
    if A.reentry:
        mode_reentry()
        modes.append("reentry")
    n_fail = sum(1 for c in CHECKS if not c["ok"])
    summary = {"modes": modes, "n_checks": len(CHECKS), "n_fail": n_fail,
               "all_pass": n_fail == 0, "wall_s": round(time.time() - t0, 1),
               "checks": CHECKS}
    if modes == ["valfull"]:
        X.dump_json(X.VALFULL_JSON, summary)
    X.dump_json(X.VERIFY_JSON, summary)
    print("[verify] modes=%s checks=%d fail=%d all_pass=%s"
          % (modes, len(CHECKS), n_fail, n_fail == 0))
    for c in CHECKS:
        if not c["ok"]:
            print("  FAIL %s %s" % (c["gate"], c["detail"][:300]))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
