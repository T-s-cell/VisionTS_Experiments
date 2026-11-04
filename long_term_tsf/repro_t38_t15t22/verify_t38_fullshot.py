"""EXP-038 task B gates: 16 configs (4 ETT datasets x H{96,192,336,720})
x 2 arms (All / Rand-VM, seed 2021), plus the frozen EXP-037 reuse pair
(ETTh1_96). Appends a 'fullshot' section to logs/verify_t38.json.
Exit 0 only if every hard gate passes.

Gates per config (hard):
  BG1 provenance    fingerprint code_md5/data_sha256 == the config's own
                    EXP-003 fingerprint; ckpt sha == frozen pin; args
                    deltas only {model_id, save_dir, ft_type,
                    vm_pretrained, gpu}
  BG2 truth         results/_/true.npy shape == formula and bitwise ==
                    EXP-003 seed2021 truth (both arms)
  BG3 metrics       float64 recompute of MSE/MAE from pred/true (channel
                    view restored) vs SUCCESS.txt <= 1e-6 (both arms)
  BG4 init          All init == ckpt per-key (0 differ); Rand differs;
                    init canonical shas differ; trainable name lists equal
  BG5 RNG           torch/np/python state hashes identical across arms
                    (after VisionTS.__init__ and at train entry)
  BG6 sample order  exactly 1 RandomSampler epoch per arm; epoch index
                    count == formula 7*(12*month_rows - L - h + 1);
                    index sequence sha identical across arms
  BG7 first batch   first train-batch tensor hashes identical across arms
  BG8 final ckpt    sha recorded, differs from frozen ckpt and across arms
  BG9 LN anchor     the config's EXP-003 SUCCESS.txt == LN16 pin
                    (seed2021; EXP-033 LN column is a 3-seed mean and is
                    NOT used — user correction 2026-11-04)
  BG10 reuse pair   ETTh1_96 read from the frozen EXP-037 archive
                    (read-only): BG4-BG7-style equality evidence re-read,
                    SUCCESS values == REUSE pins

Aggregation (report, not gate): per dataset the 4-horizon equal-weight
mean of MSE/MAE for All/Rand/LN/Freeze -> 4-dataset equal-weight mean,
compared at the 4H-mean layer against paper Table 22 (single seed,
Rand-VM full-train as this round's setting).
"""
import os
import sys

import numpy as np

import t38_common as C
import t22_common as T

# 30 NEW trainings (15 configs x 2 arms); ETTh1_96 comes from the frozen
# EXP-037 reuse pair and is verified separately (BG10), never re-trained.
CONFIGS = [(ds, h) for ds in C.ETT4 for h in C.HORIZONS
           if not (ds == "ETTh1" and h == 96)]


def gate(rows, key, ok, detail):
    rows.append({"gate": key, "verdict": "PASS" if ok else "FAIL",
                 "detail": detail})
    return ok


def check_config(ds, h, rows, summary):
    tag = "%s_%d" % (ds, h)
    recs = {}
    for arm in C.B_ARMS:
        p = C.train_json(ds, h, arm)
        if not os.path.isfile(p):
            gate(rows, "BG0_presence[%s_%s]" % (tag, arm), False,
                 "train json missing: %s" % p)
            return
        recs[arm] = C.load_json(p)
    gate(rows, "BG0_presence[%s]" % tag, True, "both arm train jsons present")

    fp003_path = os.path.join(C.exp003_job(ds, h), "fingerprint.json")
    fp003 = C.load_json(fp003_path)
    for arm in C.B_ARMS:
        fp = recs[arm]["fingerprint"]
        gate(rows, "BG1_code_md5[%s_%s]" % (tag, arm),
             fp["code_md5"] == fp003["code_md5"],
             "12 code files md5 %s EXP-003(%s)" %
             ("==" if fp["code_md5"] == fp003["code_md5"] else "!=",
              os.path.dirname(fp003_path)))
        gate(rows, "BG1_data_sha[%s_%s]" % (tag, arm),
             fp["data_sha256"] == fp003["data_sha256"],
             "%s sha %s" % (ds, list(fp["data_sha256"].values())[0][:16]))
        got_ckpt = list(fp["ckpt_sha256"].values())[0]
        gate(rows, "BG1_ckpt_sha[%s_%s]" % (tag, arm),
             got_ckpt == C.CKPT_SHA256,
             "fingerprint ckpt %s... pinned %s..." %
             (got_ckpt[:16], C.CKPT_SHA256[:16]))
        adv = recs[arm]["args_delta_check"]
        gate(rows, "BG1_args_delta[%s_%s]" % (tag, arm),
             adv["verdict"] == "ONLY_ALLOWED_DELTAS" and not adv["unexpected"],
             "deltas %s" % sorted(adv["deltas"]))

    n = C.expected_n_windows(ds, h)
    true003 = np.load(os.path.join(C.exp003_job(ds, h), "results", "_",
                                   "true.npy"))
    gate(rows, "BG2_exp003_truth_shape[%s]" % tag,
         true003.shape == (n * 7, h, 1), "EXP-003 true %s (formula n=%d)" %
         (true003.shape, n))
    metric = T.load_metric()
    for arm in C.B_ARMS:
        res_dir = os.path.join(C.b_job_dir(ds, h, arm), "results", "_")
        pred = np.load(os.path.join(res_dir, "pred.npy"))
        true = np.load(os.path.join(res_dir, "true.npy"))
        gate(rows, "BG2_truth_bitwise[%s_%s]" % (tag, arm),
             true.shape == true003.shape and np.array_equal(true, true003),
             "true %s bitwise==EXP-003: %s" %
             (true.shape, bool(np.array_equal(true, true003))))
        pr = pred[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
        tr = true[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
        m = T.metrics_from_arrays(metric, pr, tr)
        succ = recs[arm]["success"]
        dmse = abs(float(m[1]) - succ["mse"])
        dmae = abs(float(m[0]) - succ["mae"])
        gate(rows, "BG3_metrics[%s_%s]" % (tag, arm),
             dmse <= C.METRIC_TOL and dmae <= C.METRIC_TOL,
             "recomp %.10f/%.10f vs SUCCESS %.10f/%.10f (d %.2e/%.2e)" %
             (m[1], m[0], succ["mse"], succ["mae"], dmse, dmae))
        summary.setdefault(ds, {}).setdefault(h, {})[arm] = {
            "mse": succ["mse"], "mae": succ["mae"],
            "mse_recomp": float(m[1]), "mae_recomp": float(m[0])}

    ia, ir = recs["all"]["init"], recs["rand"]["init"]
    gate(rows, "BG4_all_eq_ckpt[%s]" % tag,
         ia["vs_ckpt"]["verdict"] == "ALL_KEYS_EQUAL_TO_CKPT",
         "%d/%d keys == ckpt" % (ia["vs_ckpt"]["n_equal"],
                                 ia["vs_ckpt"]["n_keys"]))
    gate(rows, "BG4_rand_differs[%s]" % tag,
         ir["vs_ckpt"]["verdict"] == "RANDOM_INIT_RECORDED",
         "%d/%d keys differ from ckpt" % (ir["vs_ckpt"]["n_differ"],
                                          ir["vs_ckpt"]["n_keys"]))
    gate(rows, "BG4_init_shas_differ[%s]" % tag,
         ia["canonical_sha256"] != ir["canonical_sha256"],
         "all %s... rand %s..." % (ia["canonical_sha256"][:16],
                                   ir["canonical_sha256"][:16]))
    gate(rows, "BG4_trainable_lists_equal[%s]" % tag,
         ia["trainable_names"] == ir["trainable_names"]
         and ia["trainable_count"] == ir["trainable_count"],
         "%d trainable params, identical name lists" % ia["trainable_count"])

    for key in ("rng_after_init", "rng_at_train_entry"):
        a = recs["all"]["init"][key] if key == "rng_after_init" \
            else recs["all"][key]
        r = recs["rand"]["init"][key] if key == "rng_after_init" \
            else recs["rand"][key]
        gate(rows, "BG5_%s[%s]" % (key, tag), a == r,
             "identical across arms")

    sa, sr = recs["all"]["sampler"], recs["rand"]["sampler"]
    exp_epoch = C.expected_epoch_len(ds, h)
    ok = sa["n_randomsampler_iters"] == 1 and \
        sr["n_randomsampler_iters"] == 1
    gate(rows, "BG6_single_epoch_iter[%s]" % tag, ok,
         "iters all=%d rand=%d" % (sa["n_randomsampler_iters"],
                                   sr["n_randomsampler_iters"]))
    ea, er = sa["iters"][0], sr["iters"][0]
    gate(rows, "BG6_epoch_len[%s]" % tag,
         ea["n"] == er["n"] == exp_epoch,
         "n_idx %d/%d expect %d (7 x train windows)" % (ea["n"], er["n"],
                                                        exp_epoch))
    gate(rows, "BG6_order_identical[%s]" % tag,
         ea["sha256"] == er["sha256"],
         "epoch index sha %s..." % ea["sha256"][:16])

    fa, fr = recs["all"]["first_batch"], recs["rand"]["first_batch"]
    ok = fa.get("done") and fr.get("done") and \
        [t["sha256"] for t in fa["tensors"]] == \
        [t["sha256"] for t in fr["tensors"]]
    gate(rows, "BG7_first_batch_identical[%s]" % tag, ok,
         "tensors %s" % [t["shape"] for t in fa.get("tensors", [])])

    sha_a = recs["all"]["final_ckpt_sha256"]
    sha_r = recs["rand"]["final_ckpt_sha256"]
    gate(rows, "BG8_final_ckpts[%s]" % tag,
         sha_a != sha_r and sha_a != C.CKPT_SHA256 and
         sha_r != C.CKPT_SHA256,
         "all %s... rand %s..." % (sha_a[:16], sha_r[:16]))

    succ_path = os.path.join(C.exp003_job(ds, h), "SUCCESS.txt")
    mse, mae = C.parse_success_mse_mae(succ_path)
    pin_mse, pin_mae = C.LN16[(ds, h)]
    gate(rows, "BG9_ln_anchor[%s]" % tag,
         mse == pin_mse and mae == pin_mae,
         "EXP-003 SUCCESS %.16f/%.16f == pin %s" %
         (mse, mae, str((pin_mse, pin_mae))[:60]))

    for arm in C.B_ARMS:
        s = recs[arm]["sampler_len_check"]
        gate(rows, "BG6_formula_check[%s_%s]" % (tag, arm), s["ok"],
             "sampler %d vs formula %d" % (s["got"],
                                           s["expected_epoch_len"]))


def check_reuse_pair(rows, summary):
    """ETTh1_96: frozen EXP-037 archive (read-only) + REUSE pins."""
    tag = "ETTh1_96_reuse"
    okall = True
    vals = {}
    for arm in C.B_ARMS:
        p = C.t37_train_json(arm)
        if not os.path.isfile(p):
            okall = gate(rows, "BG10_presence[%s_%s]" % (tag, arm), False,
                         "t37 train json missing: %s" % p)
            continue
        rec = C.load_json(p)
        succ = rec["success"]
        pin = C.REUSE[("ETTh1", 96, arm)]
        ok = succ["mse"] == pin["mse"] and succ["mae"] == pin["mae"]
        okall &= gate(rows, "BG10_pin[%s_%s]" % (tag, arm), ok,
                      "t37 SUCCESS %.16f/%.16f vs REUSE pin" %
                      (succ["mse"], succ["mae"]))
        vals[arm] = {"mse": succ["mse"], "mae": succ["mae"],
                     "train_json": p}
    ia = None
    for arm in C.B_ARMS:
        p = C.t37_train_json(arm)
        if not os.path.isfile(p):
            continue
        rec = C.load_json(p)
        if rec["init"]["vs_ckpt"]["verdict"] == "ALL_KEYS_EQUAL_TO_CKPT":
            ia = rec["init"]
        okall &= gate(rows, "BG4_evidence[%s_%s]" % (tag, arm),
                      rec["init"]["called"] == 1,
                      "t37 archived init evidence called=%d" %
                      rec["init"]["called"])
    irp = None
    for arm in C.B_ARMS:
        p = C.t37_train_json(arm)
        if os.path.isfile(p) and \
                C.load_json(p)["init"]["vs_ckpt"]["verdict"] == \
                "RANDOM_INIT_RECORDED":
            irp = C.load_json(p)["init"]
    if ia and irp:
        okall &= gate(rows, "BG4_init_shas_differ[%s]" % tag,
                      ia["canonical_sha256"] != irp["canonical_sha256"],
                      "t37 archived init shas differ")
        okall &= gate(rows, "BG4_trainable_lists_equal[%s]" % tag,
                      ia["trainable_names"] == irp["trainable_names"],
                      "t37 archived trainable lists equal")
    sa = C.load_json(C.t37_train_json("all"))["sampler"] \
        if os.path.isfile(C.t37_train_json("all")) else None
    sr = C.load_json(C.t37_train_json("rand"))["sampler"] \
        if os.path.isfile(C.t37_train_json("rand")) else None
    if sa and sr:
        okall &= gate(rows, "BG6_order_identical[%s]" % tag,
                      sa["iters"][0]["sha256"] == sr["iters"][0]["sha256"]
                      and sa["iters"][0]["n"] == sr["iters"][0]["n"] ==
                      C.expected_epoch_len("ETTh1", 96),
                      "t37 archived epoch index sha equal, n=%d" %
                      sa["iters"][0]["n"])
    exp003_true = os.path.join(C.exp003_job("ETTh1", 96), "results", "_",
                               "true.npy")
    t003 = np.load(exp003_true)
    for arm in C.B_ARMS:
        jp = C.t37_job_dir(arm)
        tp = os.path.join(jp, "results", "_", "true.npy")
        if not os.path.isfile(tp):
            okall &= gate(rows, "BG2_truth_bitwise[%s_%s]" % (tag, arm),
                          False, "t37 true.npy missing: %s" % tp)
            continue
        t = np.load(tp)
        okall &= gate(rows, "BG2_truth_bitwise[%s_%s]" % (tag, arm),
                      t.shape == t003.shape and np.array_equal(t, t003),
                      "t37 true %s bitwise==EXP-003" % (t.shape,))
    if vals:
        summary.setdefault("ETTh1", {}).setdefault(96, {}).update(vals)
    return okall


def aggregate(summary, rows):
    """Per-dataset 4H equal means -> overall; freeze column from
    summary_t22.csv; LN from EXP-003 pins; compare vs paper Table 22."""
    freeze16 = C.load_freeze16()
    per_ds = {}
    for ds in C.ETT4:
        ms = {k: {"mse": 0.0, "mae": 0.0} for k in
              ("all", "rand", "ln", "freeze")}
        cnt = 0
        for h in C.HORIZONS:
            cell = summary.get(ds, {}).get(h, {})
            if "all" not in cell or "rand" not in cell:
                continue
            cnt += 1
            for arm in ("all", "rand"):
                ms[arm]["mse"] += cell[arm]["mse"]
                ms[arm]["mae"] += cell[arm]["mae"]
            ln_mse, ln_mae = C.LN16[(ds, h)]
            ms["ln"]["mse"] += ln_mse
            ms["ln"]["mae"] += ln_mae
            fz_mse, fz_mae, _src = freeze16[(ds, h)]
            ms["freeze"]["mse"] += fz_mse
            ms["freeze"]["mae"] += fz_mae
        if cnt != 4:
            gate(rows, "AGG_coverage[%s]" % ds, False,
                 "only %d/4 horizons present" % cnt)
            continue
        per_ds[ds] = {k: {"mse": v["mse"] / 4.0, "mae": v["mae"] / 4.0}
                      for k, v in ms.items()}
        per_ds[ds]["n_horizons"] = 4
    overall = {k: {"mse": sum(per_ds[ds][k]["mse"] for ds in per_ds) /
                   len(per_ds),
                   "mae": sum(per_ds[ds][k]["mae"] for ds in per_ds) /
                   len(per_ds)}
               for k in ("all", "rand", "ln", "freeze")} \
        if len(per_ds) == 4 else {}
    cmp_rows = []
    if per_ds:
        for ds in per_ds:
            for k in ("all", "rand", "ln", "freeze"):
                pmse, pmae = C.PAPER_T22[ds][k]
                cmp_rows.append({"ds": ds, "col": k,
                                 "mse_ours": per_ds[ds][k]["mse"],
                                 "mae_ours": per_ds[ds][k]["mae"],
                                 "mse_paper": pmse, "mae_paper": pmae})
        for k in ("all", "rand", "ln", "freeze"):
            pmse, pmae = C.PAPER_T22_OVERALL[k]
            cmp_rows.append({"ds": "Overall", "col": k,
                             "mse_ours": overall[k]["mse"],
                             "mae_ours": overall[k]["mae"],
                             "mse_paper": pmse, "mae_paper": pmae})
    return per_ds, overall, cmp_rows


def main():
    rows, summary = [], {}
    for ds, h in CONFIGS:
        check_config(ds, h, rows, summary)
    check_reuse_pair(rows, summary)
    freeze16 = C.load_freeze16()
    gate(rows, "BG9_freeze16_coverage",
         len(freeze16) == 16 and all(("ETTm1", h) in freeze16
                                     for h in C.HORIZONS),
         "16 freeze rows from %s; %s" % (C.t22_summary_csv(),
                                         C.FREEZE16_SOURCE_NOTE))
    per_ds, overall, cmp_rows = aggregate(summary, rows)

    n_fail = sum(1 for g in rows if g["verdict"] == "FAIL")
    out = {
        "all_pass": n_fail == 0,
        "n_gates": len(rows), "n_pass": sum(1 for g in rows
                                            if g["verdict"] == "PASS"),
        "gates": rows,
        "summary": {ds: {str(h): v for h, v in hs.items()}
                    for ds, hs in summary.items()},
        "per_dataset_4h_mean": per_ds,
        "overall_4d_mean": overall,
        "table22_comparison": cmp_rows,
        "timing": {},
        "gpu_peak": {},
    }
    for ds, h in [(ds, h) for ds in C.ETT4 for h in C.HORIZONS]:
        for arm in C.B_ARMS:
            p = C.train_json(ds, h, arm)
            if os.path.isfile(p):
                rec = C.load_json(p)
                out["timing"]["%s_%d_%s" % (ds, h, arm)] = rec["timing"]
                out["gpu_peak"]["%s_%d_%s" % (ds, h, arm)] = rec.get(
                    "gpu_peak")
    C.merge_json(C.VERIFY_JSON, "fullshot", out)
    for g in rows:
        if g["verdict"] == "FAIL":
            print("[t38][B][FAIL] %s %s" % (g["gate"], g["detail"]))
    print("[t38][B] %d/%d gates pass (%d FAIL) -> %s" %
          (out["n_pass"], out["n_gates"], n_fail, C.VERIFY_JSON))
    if per_ds:
        for ds in C.ETT4:
            if ds not in per_ds:
                continue
            print("[t38][B] %s 4H mean: all %.4f/%.4f rand %.4f/%.4f "
                  "ln %.4f/%.4f freeze %.4f/%.4f (paper all %.3f/%.3f "
                  "rand %.3f/%.3f)" %
                  (ds, per_ds[ds]["all"]["mse"], per_ds[ds]["all"]["mae"],
                   per_ds[ds]["rand"]["mse"], per_ds[ds]["rand"]["mae"],
                   per_ds[ds]["ln"]["mse"], per_ds[ds]["ln"]["mae"],
                   per_ds[ds]["freeze"]["mse"], per_ds[ds]["freeze"]["mae"],
                   C.PAPER_T22[ds]["all"][0], C.PAPER_T22[ds]["all"][1],
                   C.PAPER_T22[ds]["rand"][0], C.PAPER_T22[ds]["rand"][1]))
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
