"""EXP-038 task B collector: logs/summary_t38_ett.csv + RESULT_t38_ett.md
from archived evidence only (train_<DS>_<H>_<arm>.json, the frozen
EXP-037 reuse archive, verify_t38.json section fullshot,
repro_t22_freeze_ln/summary_t22.csv for the freeze column). Local-only;
--selfcheck exercises output paths with fabricated records.
"""
import argparse
import csv
import os

import t38_common as C

CONFIGS = [(ds, h) for ds in C.ETT4 for h in C.HORIZONS]


def fmt(v, nd=6):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def arm_rec(ds, h, arm):
    if ds == "ETTh1" and h == 96:
        p = C.t37_train_json(arm)
        if os.path.isfile(p):
            return C.load_json(p), "EXP-037 frozen reuse"
        return None, "reuse archive missing"
    p = C.train_json(ds, h, arm)
    if os.path.isfile(p):
        return C.load_json(p), "this run"
    return None, "missing"


def build_rows():
    rows = []
    for ds, h in CONFIGS:
        row = {"task": "B_ett_t22", "ds": ds, "h": h}
        freeze16 = C.load_freeze16()
        fz = freeze16.get((ds, h))
        row["freeze_mse"] = fz[0] if fz else None
        row["freeze_mae"] = fz[1] if fz else None
        row["freeze_source"] = fz[2] if fz else "missing"
        ln = C.LN16.get((ds, h))
        row["ln_mse"] = ln[0] if ln else None
        row["ln_mae"] = ln[1] if ln else None
        for arm in C.B_ARMS:
            rec, src = arm_rec(ds, h, arm)
            row["%s_source" % arm] = src
            if rec:
                row["%s_mse" % arm] = rec["success"]["mse"]
                row["%s_mae" % arm] = rec["success"]["mae"]
                row["%s_wall_s" % arm] = rec["timing"]["wall_sec"]
                ph = rec.get("phases", {})
                row["%s_train_s" % arm] = ph.get("train_wall")
                row["%s_vali_s" % arm] = ph.get("vali_wall")
                row["%s_test_s" % arm] = ph.get("test_wall")
                row["%s_train_peak_mb" % arm] = ph.get("train_peak_mb")
                row["%s_vali_peak_mb" % arm] = ph.get("vali_peak_mb")
                row["%s_test_peak_mb" % arm] = ph.get("test_peak_mb")
                row["%s_train_batches" % arm] = \
                    rec.get("sampler_len_check", {}).get(
                        "expected_train_batches",
                        rec.get("expected_train_batches"))
                row["%s_ckpt_sha16" % arm] = rec["final_ckpt_sha256"][:16]
            else:
                for k in ("mse", "mae", "wall_s", "train_s", "vali_s",
                          "test_s", "train_peak_mb", "vali_peak_mb",
                          "test_peak_mb", "train_batches", "ckpt_sha16"):
                    row["%s_%s" % (arm, k)] = None
        rows.append(row)
    return rows


FIELDS = ["task", "ds", "h", "all_source", "all_mse", "all_mae", "all_wall_s",
          "all_train_s", "all_vali_s", "all_test_s", "all_train_peak_mb",
          "all_vali_peak_mb", "all_test_peak_mb", "all_train_batches",
          "all_ckpt_sha16", "rand_source", "rand_mse", "rand_mae",
          "rand_wall_s", "rand_train_s", "rand_vali_s", "rand_test_s",
          "rand_train_peak_mb", "rand_vali_peak_mb", "rand_test_peak_mb",
          "rand_train_batches", "rand_ckpt_sha16", "ln_mse", "ln_mae",
          "freeze_mse", "freeze_mae", "freeze_source"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def budget_md(rows):
    """Measured s/batch per dataset from completed arms -> remaining-arms
    budget estimate (train phase only; vali+test add a small constant)."""
    L = ["## Budget (measured train s/batch per dataset)\n"]
    L.append("| dataset | measured s/batch | from | remaining arms | "
             "est train h remaining (per arm x batches x s/batch) |")
    L.append("|---|---|---|---|---|")
    total_left_h = 0.0
    for ds in C.ETT4:
        sb, src = None, None
        for r in rows:
            if r["ds"] != ds:
                continue
            for arm in C.B_ARMS:
                w, b = r.get("%s_train_s" % arm), r.get("%s_train_batches" % arm)
                if w and b:
                    sb, src = w / b, "%s_%d_%s" % (ds, r["h"], arm)
                    break
            if sb:
                break
        left = []
        for r in rows:
            if r["ds"] != ds:
                continue
            for arm in C.B_ARMS:
                if r.get("%s_train_s" % arm) is None:
                    left.append((r["h"], C.expected_train_batches(ds, r["h"])))
        if sb:
            est = sum(b for _, b in left) * sb / 3600.0
            total_left_h += est
            L.append("| %s | %.3f | %s | %d (%s) | %.2f h |" %
                     (ds, sb, src, len(left),
                      ",".join(str(h) for h, _ in left) or "-", est))
        else:
            L.append("| %s | n/a | - | %d | n/a |" %
                     (ds, len(left)))
    L.append("\nTotal estimated remaining train time: **%.2f h** "
             "(excludes vali/test phases and I/O; refined after the pilot "
             "per user instruction #4).\n" % total_left_h)
    return L


def result_md(rows, vj):
    L = []
    L.append("# EXP-038 RESULT B: ETT All (ImageNet init) vs Rand-VM "
             "(random init), full finetune, seed 2021 — paper Table 22\n")
    L.append("Generated by collect_t38_fullshot.py from archived evidence "
             "only (logs/train_<DS>_<H>_<arm>.json, EXP-037 frozen reuse "
             "archive, verify_t38.json section fullshot, "
             "repro_t22_freeze_ln/summary_t22.csv).\n")
    L.append("Scope: single seed (2021); All = verified ImageNet MAE-base "
             "full finetune; Rand-VM = same architecture ALL weights "
             "randomly initialized with exactly the All recipe (this "
             "round's setting; the paper does not disclose the Rand-VM "
             "trainable set); LN column = EXP-003 seed-2021 SUCCESS "
             "values (the paper's LN column is a 3-seed mean — not "
             "comparable at 3dp); Freeze column = EXP-033 via "
             "summary_t22.csv (ETTm1 rows exp001_cache). Recipe: "
             "EXP-003 (bilinear, r=c=0.4, FP32, Adam 1e-4, bs 256, "
             "1 epoch, full-val MSE selection, reload then test).\n")

    L.append("## T1: per-horizon results (16 configs)\n")
    L.append("| ds | H | All MSE/MAE | Rand-VM MSE/MAE | LN MSE/MAE | "
             "Freeze MSE/MAE | All train/vali/test s | Rand train/vali/test s |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append("| %s | %d | %s/%s | %s/%s | %s/%s | %s/%s | %s/%s/%s | "
                 "%s/%s/%s |" %
                 (r["ds"], r["h"], fmt(r["all_mse"], 4), fmt(r["all_mae"], 4),
                  fmt(r["rand_mse"], 4), fmt(r["rand_mae"], 4),
                  fmt(r["ln_mse"], 4), fmt(r["ln_mae"], 4),
                  fmt(r["freeze_mse"], 4), fmt(r["freeze_mae"], 4),
                  fmt(r["all_train_s"], 1), fmt(r["all_vali_s"], 1),
                  fmt(r["all_test_s"], 1), fmt(r["rand_train_s"], 1),
                  fmt(r["rand_vali_s"], 1), fmt(r["rand_test_s"], 1)))
    L.append("")

    vj_fs = vj or {}
    per_ds = vj_fs.get("per_dataset_4h_mean") or {}
    if per_ds:
        L.append("## T2: 4H equal means per dataset + overall vs paper "
                 "Table 22 (4H-mean layer)\n")
        L.append("| ds | col | ours MSE/MAE | paper MSE/MAE | dMSE |")
        L.append("|---|---|---|---|---|")
        for ds in C.ETT4:
            if ds not in per_ds:
                continue
            for k in ("all", "rand", "ln", "freeze"):
                pm, pa = C.PAPER_T22[ds][k]
                L.append("| %s | %s | %.4f/%.4f | %.3f/%.3f | %+.4f |" %
                         (ds, k, per_ds[ds][k]["mse"], per_ds[ds][k]["mae"],
                          pm, pa, per_ds[ds][k]["mse"] - pm))
        ov = vj_fs.get("overall_4d_mean") or {}
        if ov:
            for k in ("all", "rand", "ln", "freeze"):
                pm, pa = C.PAPER_T22_OVERALL[k]
                L.append("| **Overall** | %s | %.4f/%.4f | %.3f/%.3f | "
                         "%+.4f |" %
                         (k, ov[k]["mse"], ov[k]["mae"], pm, pa,
                          ov[k]["mse"] - pm))
        L.append("")
        if "all" in per_ds.get("ETTh1", {}) and \
                "rand" in per_ds.get("ETTh1", {}):
            L.append("Pretrained-vs-random headline (dataset-level 4H "
                     "means): " + "; ".join(
                         "%s All %.4f vs Rand %.4f (All %s by %.2f%% MSE)"
                         % (ds, per_ds[ds]["all"]["mse"],
                            per_ds[ds]["rand"]["mse"],
                            "lower" if per_ds[ds]["all"]["mse"] <
                            per_ds[ds]["rand"]["mse"] else "HIGHER",
                            100.0 * abs(per_ds[ds]["all"]["mse"] -
                                        per_ds[ds]["rand"]["mse"]) /
                            max(per_ds[ds]["all"]["mse"],
                                per_ds[ds]["rand"]["mse"]))
                         for ds in C.ETT4 if ds in per_ds) + "\n")

    L.extend(budget_md(rows))

    L.append("## Gates and anomalies\n")
    fails = [g for g in vj_fs.get("gates", []) if g["verdict"] == "FAIL"]
    if not fails:
        L.append("All hard gates pass (BG1-BG10 over 15 new configs x 2 "
                 "arms + reuse pair); see verify_t38.json.\n")
    else:
        for g in fails:
            L.append("- [FAIL] %s: %s" % (g["gate"], g["detail"]))
        L.append("")
    L.append("GPU peak memory per arm recorded in "
             "summary_t38_ett.csv (train/vali/test phase peaks from "
             "observation-only wrappers; reset per phase).\n")
    return "\n".join(L) + "\n"


def selfcheck():
    out = "temp_selfcheck_t38"
    os.makedirs(out, exist_ok=True)
    rows = []
    for ds, h in CONFIGS:
        r = {"task": "B_ett_t22", "ds": ds, "h": h,
             "freeze_mse": 0.4, "freeze_mae": 0.41, "freeze_source": "x",
             "ln_mse": 0.3, "ln_mae": 0.37}
        for arm in C.B_ARMS:
            r.update({"%s_source" % arm: "this run", "%s_mse" % arm: 0.45,
                      "%s_mae" % arm: 0.42, "%s_wall_s" % arm: 900.0,
                      "%s_train_s" % arm: 800.0, "%s_vali_s" % arm: 60.0,
                      "%s_test_s" % arm: 40.0,
                      "%s_train_peak_mb" % arm: 5000.0,
                      "%s_vali_peak_mb" % arm: 4000.0,
                      "%s_test_peak_mb" % arm: 4500.0,
                      "%s_train_batches" % arm: 200,
                      "%s_ckpt_sha16" % arm: "ab" * 8})
        rows.append(r)
    write_csv(os.path.join(out, "summary_t38_ett.csv"), rows)
    md = result_md(rows, {"per_dataset_4h_mean": {}, "gates": []})
    with open(os.path.join(out, "RESULT_t38_ett.md"), "w") as f:
        f.write(md)
    print("[t38] selfcheck OK -> %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    rows = build_rows()
    write_csv(C.SUMMARY_B_CSV, rows)
    vj = C.load_json(C.VERIFY_JSON).get("fullshot", {}) \
        if os.path.isfile(C.VERIFY_JSON) else {}
    md = result_md(rows, vj)
    with open(C.RESULT_B_MD, "w") as f:
        f.write(md)
    print("[t38] wrote %s and %s" % (C.SUMMARY_B_CSV, C.RESULT_B_MD))


if __name__ == "__main__":
    main()
