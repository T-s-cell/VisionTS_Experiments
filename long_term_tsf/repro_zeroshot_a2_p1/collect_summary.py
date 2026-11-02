"""EXP-030 summary: paper A.2 Table 7 vs measured (P=1 and original-P).
Reads the 16 metrics_<DS>_<H>.json produced by post_eval.py, averages the
four unrounded horizon values per dataset, then the four datasets, and
writes summary_a2_p1.csv + RESULT_a2_p1.md (paper / measured / delta and
the exception list of cells with |delta| > 0.001).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2p1_common as C


def mean4(vals):
    return sum(vals) / 4.0


def main():
    rows = []
    per_ds = {}
    for ds in C.ETT4:
        hs = []
        for h in C.HORIZONS:
            j = C.load_json(os.path.join(C.p1_dir(ds, h),
                                         "metrics_%s_%d.json" % (ds, h)))
            hs.append(j)
            rows.append({
                "dataset": ds, "pred_len": h,
                "P_orig": C.DATASETS[ds]["period"],
                "p1_mse": j["p1"]["mse"], "p1_mae": j["p1"]["mae"],
                "orig_mse": j["baseline_p_orig"]["mse"],
                "orig_mae": j["baseline_p_orig"]["mae"],
                "d_mse": j["delta_p1_minus_orig"]["mse"],
                "d_mae": j["delta_p1_minus_orig"]["mae"],
                "rel_d_mse_pct": 100 * j["rel_p1_minus_orig"]["mse"],
                "rel_d_mae_pct": 100 * j["rel_p1_minus_orig"]["mae"],
                "n_windows": j["n_windows"]})
        per_ds[ds] = {
            "p1_mse": mean4([x["p1"]["mse"] for x in hs]),
            "p1_mae": mean4([x["p1"]["mae"] for x in hs]),
            "orig_mse": mean4([x["baseline_p_orig"]["mse"] for x in hs]),
            "orig_mae": mean4([x["baseline_p_orig"]["mae"] for x in hs]),
        }

    avg = {k: mean4([per_ds[ds][k] for ds in C.ETT4])
           for k in ("p1_mse", "p1_mae", "orig_mse", "orig_mae")}

    # ---- exception cells vs paper (|delta| > 0.001) ----
    exceptions = []
    for ds in C.ETT4:
        for col, mk, pk in (("orig", "orig_mse", "orig"), ("orig", "orig_mae", "orig"),
                            ("p1", "p1_mse", "p1"), ("p1", "p1_mae", "p1")):
            metric = "MSE" if mk.endswith("mse") else "MAE"
            measured = per_ds[ds][mk]
            paper = C.PAPER_T7[ds][pk][0 if metric == "MSE" else 1]
            d = measured - paper
            if abs(d) > 0.001:
                exceptions.append((ds, "P=1" if col == "p1" else "P_orig",
                                   metric, paper, measured, d))
    for mk, pk in (("p1_mse", "p1"), ("p1_mae", "p1")):
        metric = "MSE" if mk.endswith("mse") else "MAE"
        measured = avg[mk]
        paper = C.PAPER_T7_AVG[pk][0 if metric == "MSE" else 1]
        d = measured - paper
        if abs(d) > 0.001:
            exceptions.append(("Average", "P=1", metric, paper, measured, d))
    for mk in ("orig_mse", "orig_mae"):
        metric = "MSE" if mk.endswith("mse") else "MAE"
        measured = avg[mk]
        paper = C.PAPER_T7_AVG["orig"][0 if metric == "MSE" else 1]
        d = measured - paper
        if abs(d) > 0.001:
            exceptions.append(("Average", "P_orig", metric, paper, measured, d))

    # ---- csv ----
    csv_path = os.path.join(C.PKG, "summary_a2_p1.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
        avg_row = {"dataset": "Average", "pred_len": "", "P_orig": "",
                   "p1_mse": avg["p1_mse"], "p1_mae": avg["p1_mae"],
                   "orig_mse": avg["orig_mse"], "orig_mae": avg["orig_mae"],
                   "d_mse": avg["p1_mse"] - avg["orig_mse"],
                   "d_mae": avg["p1_mae"] - avg["orig_mae"],
                   "rel_d_mse_pct": 100 * (avg["p1_mse"] / avg["orig_mse"] - 1),
                   "rel_d_mae_pct": 100 * (avg["p1_mae"] / avg["orig_mae"] - 1),
                   "n_windows": sum(r["n_windows"] for r in rows)}
        w.writerow(avg_row)

    # ---- markdown ----
    L = ["# EXP-030｜VisionTS A.2 表 7 复现：P=1 vs 原周期 P（零样本，ETT 4 数据集 × 4H）", ""]
    L += ["## 1. 逐数据集（4H 未舍入等权平均）vs 论文表 7", "",
          "| 数据集 | 原P | 论文 原 P MSE/MAE | 实测 原 P MSE/MAE | Δ | 论文 P=1 MSE/MAE | 实测 P=1 MSE/MAE | Δ |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in C.ETT4:
        d = per_ds[ds]
        po, pp = C.PAPER_T7[ds]["orig"], C.PAPER_T7[ds]["p1"]
        L.append("| %s | %d | %.3f/%.3f | %.4f/%.4f | %+.4f/%+.4f | %.3f/%.3f | %.4f/%.4f | %+.4f/%+.4f |"
                 % (ds, C.DATASETS[ds]["period"], po[0], po[1],
                    d["orig_mse"], d["orig_mae"],
                    d["orig_mse"] - po[0], d["orig_mae"] - po[1],
                    pp[0], pp[1], d["p1_mse"], d["p1_mae"],
                    d["p1_mse"] - pp[0], d["p1_mae"] - pp[1]))
    L.append("| **平均** | — | %.3f/%.3f | %.4f/%.4f | %+.4f/%+.4f | %.3f/%.3f | %.4f/%.4f | %+.4f/%+.4f |"
             % (C.PAPER_T7_AVG["orig"][0], C.PAPER_T7_AVG["orig"][1],
                avg["orig_mse"], avg["orig_mae"],
                avg["orig_mse"] - C.PAPER_T7_AVG["orig"][0],
                avg["orig_mae"] - C.PAPER_T7_AVG["orig"][1],
                C.PAPER_T7_AVG["p1"][0], C.PAPER_T7_AVG["p1"][1],
                avg["p1_mse"], avg["p1_mae"],
                avg["p1_mse"] - C.PAPER_T7_AVG["p1"][0],
                avg["p1_mae"] - C.PAPER_T7_AVG["p1"][1]))
    L.append("")

    L += ["## 2. P=1 相对原周期 P 的实测收益（负值=P=1 更差）", "",
          "| 数据集 | ΔMSE (P=1 − 原P) | ΔMAE | 相对 ΔMSE | 相对 ΔMAE |",
          "|---|---:|---:|---:|---:|"]
    for ds in C.ETT4:
        d = per_ds[ds]
        L.append("| %s | %+.4f | %+.4f | %+.2f%% | %+.2f%% |"
                 % (ds, d["p1_mse"] - d["orig_mse"], d["p1_mae"] - d["orig_mae"],
                    100 * (d["p1_mse"] / d["orig_mse"] - 1),
                    100 * (d["p1_mae"] / d["orig_mae"] - 1)))
    L.append("| **平均** | %+.4f | %+.4f | %+.2f%% | %+.2f%% |"
             % (avg["p1_mse"] - avg["orig_mse"], avg["p1_mae"] - avg["orig_mae"],
                100 * (avg["p1_mse"] / avg["orig_mse"] - 1),
                100 * (avg["p1_mae"] / avg["orig_mae"] - 1)))
    L.append("")

    L += ["## 3. 异常表：单列 |Δ| > 0.001（vs 论文表 7）", ""]
    if exceptions:
        L += ["| 数据集 | 列 | 指标 | 论文值 | 实测值 | Δ |", "|---|---|---|---:|---:|---:|"]
        for ds, col, met, pv, mv, d in exceptions:
            L.append("| %s | %s | %s | %.3f | %.4f | %+.4f |" % (ds, col, met, pv, mv, d))
    else:
        L.append("（无：全部表 7 单列偏差均 ≤ 0.001）")
    L.append("")

    L += ["## 4. 逐组明细（标准化空间，16 组）", "",
          "| 组 | 原P | 窗数 | P=1 MSE | P=1 MAE | 原P MSE | 原P MAE | ΔMSE | ΔMAE |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append("| %s_%d | %d | %d | %.6f | %.6f | %.6f | %.6f | %+.6f | %+.6f |"
                 % (r["dataset"], r["pred_len"], r["P_orig"], r["n_windows"],
                    r["p1_mse"], r["p1_mae"], r["orig_mse"], r["orig_mae"],
                    r["d_mse"], r["d_mae"]))
    L.append("")
    L += ["口径：与 EXP-001 完全一致（同划分、同 scaler、同测试窗、标准化空间、原 metric()）；"
          "原 P 列 = EXP-001 实测（preflight F3 已证 metrics.npy 可复算）；"
          "P=1 列 = 本轮推理；P=1 相对原 P 的唯一差异为成像周期。",
          "测试窗总数：%d" % sum(r["n_windows"] for r in rows), ""]

    md_path = os.path.join(C.PKG, "RESULT_a2_p1.md")
    with open(md_path, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print("\n[collect_summary] wrote %s and %s" % (csv_path, md_path))


if __name__ == "__main__":
    main()
