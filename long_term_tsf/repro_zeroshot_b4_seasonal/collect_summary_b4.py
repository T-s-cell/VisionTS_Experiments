"""EXP-031 summary: paper B.4 Table 12 vs measured, at every level the
paper prints (per-H cells, per-dataset 4H-average rows, overall average).
Measured means use unrounded values (4H equal-weight mean, then 4-dataset
mean); paper means are taken from the table as printed. Writes
summary_b4_seasonal.csv + RESULT_b4_seasonal.md with:
  table 1  per-dataset 4H averages (3 methods) vs paper avg rows
  table 2  per-H comparison (3 methods x 4 datasets x 4 horizons)
  table 3  exception list: every cell with |delta| > 0.001, any level
  table 4  measured VisionTS (orig P) advantage over the seasonal baselines
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b4_common as C


def mean4(vals):
    return sum(vals) / 4.0


def load_all():
    """meas[ds][h][method] = (mse, mae) unrounded; paper from PAPER_T12."""
    meas = {}
    for ds in C.ETT4:
        meas[ds] = {}
        for h in C.HORIZONS:
            md = C.method_dir(ds, h, "naive")
            jn = C.load_json(os.path.join(md, "metrics_%s_%d.json" % (ds, h)))
            ja = C.load_json(os.path.join(C.method_dir(ds, h, "avg"),
                                          "metrics_%s_%d.json" % (ds, h)))
            bm = np.load(os.path.join(C.base_dir(ds, h), "results", "_",
                                      "metrics.npy"))
            meas[ds][h] = {
                "naive": (jn["mse"], jn["mae"]),
                "avg": (ja["mse"], ja["mae"]),
                "visionts": (float(bm[1]), float(bm[0])),
                "n_windows": jn["n_windows"]}
    return meas


def main():
    meas = load_all()
    methods = ["visionts", "naive", "avg"]
    mlabel = {"visionts": "VisionTS", "naive": "S-Naive", "avg": "S-Avg"}

    # per-dataset 4H unrounded means + overall
    per_ds = {ds: {m: (mean4([meas[ds][h][m][0] for h in C.HORIZONS]),
                   mean4([meas[ds][h][m][1] for h in C.HORIZONS]))
               for m in methods} for ds in C.ETT4}
    overall = {m: (mean4([per_ds[ds][m][0] for ds in C.ETT4]),
               mean4([per_ds[ds][m][1] for ds in C.ETT4])) for m in methods}

    # ---- exceptions at all levels ----
    exceptions = []  # (scope_ds, scope_h, method, metric, paper, measured, d)

    def add_exc(ds, h, m, mi, paper, measured):
        d = measured - paper
        if abs(d) > 0.001:
            exceptions.append((ds, h, m, mi, paper, measured, d))

    for ds in C.ETT4:
        for h in C.HORIZONS:
            for m in methods:
                for mi, name in ((0, "MSE"), (1, "MAE")):
                    add_exc(ds, str(h), m, name,
                            C.PAPER_T12[ds][h][m][mi], meas[ds][h][m][mi])
        for m in methods:
            for mi, name in ((0, "MSE"), (1, "MAE")):
                add_exc(ds, "avg", m, name,
                        C.PAPER_T12[ds]["avg"][m][mi], per_ds[ds][m][mi])
    for m in methods:
        for mi, name in ((0, "MSE"), (1, "MAE")):
            add_exc("Average", "avg", m, name,
                    C.PAPER_T12_OVERALL[m][mi], overall[m][mi])

    # ---- csv ----
    csv_path = os.path.join(C.PKG, "summary_b4_seasonal.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "pred_len", "method", "period", "lookback",
                    "paper_mse", "paper_mae", "meas_mse", "meas_mae",
                    "d_mse", "d_mae", "n_windows"])
        for ds in C.ETT4:
            for h in C.HORIZONS:
                for m in methods:
                    pk, pm = C.PAPER_T12[ds][h][m]
                    mk, mm = meas[ds][h][m]
                    w.writerow([ds, h, mlabel[m], C.DATASETS[ds]["period"],
                                C.DATASETS[ds]["ctx"], pk, pm, mk, mm,
                                mk - pk, mm - pm, meas[ds][h]["n_windows"]])
        for ds in C.ETT4:
            for m in methods:
                pk, pm = C.PAPER_T12[ds]["avg"][m]
                mk, mm = per_ds[ds][m]
                w.writerow([ds, "avg", mlabel[m], C.DATASETS[ds]["period"],
                            C.DATASETS[ds]["ctx"], pk, pm, mk, mm,
                            mk - pk, mm - pm, ""])
        for m in methods:
            pk, pm = C.PAPER_T12_OVERALL[m]
            w.writerow(["Average", "avg", mlabel[m], "", "", pk, pm,
                        overall[m][0], overall[m][1],
                        overall[m][0] - pk, overall[m][1] - pm, ""])

    # ---- markdown ----
    L = ["# EXP-031｜VisionTS B.4 表 12 复现：Seasonal Naïve / Seasonal Avg "
         "周期基线（零样本，ETT 4 数据集 × 4H）", ""]
    L += ["## 1. 逐数据集 4H 平均 vs 论文表 12（论文均值取原表 avg 行；实测为"
          "未舍入 4H 等权平均）", "",
          "| 数据集 | 方法 | 论文 MSE/MAE | 实测 MSE/MAE | Δ（实测−论文） |",
          "|---|---|---:|---:|---:|"]
    for ds in C.ETT4:
        for m in methods:
            pk, pm = C.PAPER_T12[ds]["avg"][m]
            mk, mm = per_ds[ds][m]
            L.append("| %s | %s | %.3f/%.3f | %.4f/%.4f | %+.4f/%+.4f |"
                     % (ds, mlabel[m], pk, pm, mk, mm, mk - pk, mm - pm))
    L.append("| **平均** | %s | %.3f/%.3f | %.4f/%.4f | %+.4f/%+.4f |"
             % (mlabel["naive"], C.PAPER_T12_OVERALL["naive"][0],
                C.PAPER_T12_OVERALL["naive"][1], overall["naive"][0],
                overall["naive"][1],
                overall["naive"][0] - C.PAPER_T12_OVERALL["naive"][0],
                overall["naive"][1] - C.PAPER_T12_OVERALL["naive"][1]))
    for m in ("avg", "visionts"):
        L.append("| | %s | %.3f/%.3f | %.4f/%.4f | %+.4f/%+.4f |"
                 % (mlabel[m], C.PAPER_T12_OVERALL[m][0],
                    C.PAPER_T12_OVERALL[m][1], overall[m][0], overall[m][1],
                    overall[m][0] - C.PAPER_T12_OVERALL[m][0],
                    overall[m][1] - C.PAPER_T12_OVERALL[m][1]))
    L.append("")

    L += ["## 2. 逐 H 对照（论文表 12 逐 H 值 vs 实测）", ""]
    for ds in C.ETT4:
        L += ["**%s**（L=%d，P=%d）" % (ds, C.DATASETS[ds]["ctx"],
                                        C.DATASETS[ds]["period"]), "",
              "| H | 方法 | 论文 MSE | 实测 MSE | ΔMSE | 论文 MAE | 实测 MAE | ΔMAE |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for h in C.HORIZONS:
            for m in methods:
                pk, pm_ = C.PAPER_T12[ds][h][m]
                mk, mm = meas[ds][h][m]
                L.append("| %d | %s | %.3f | %.4f | %+.4f | %.3f | %.4f | %+.4f |"
                         % (h, mlabel[m], pk, mk, mk - pk, pm_, mm, mm - pm_))
        L.append("")

    L += ["## 3. 异常表：所有层级单列 |Δ| > 0.001（vs 论文表 12）", ""]
    if exceptions:
        L += ["| 数据集 | 层级 | 方法 | 指标 | 论文值 | 实测值 | Δ |",
              "|---|---|---|---|---:|---:|---:|"]
        for ds, hs, m, mi, pv, mv, d in exceptions:
            L.append("| %s | %s | %s | %s | %.3f | %.4f | %+.4f |"
                     % (ds, hs, mlabel[m], mi, pv, mv, d))
    else:
        L.append("（无：全部对照单元偏差均 ≤ 0.001）")
    L.append("")

    L += ["## 4. VisionTS（原周期 P）相对周期基线的实测优势", "",
          "Δ = VisionTS − 基线（负值 = VisionTS 误差更低，即更优）。", "",
          "**MSE**", "",
          "| 数据集 | VisionTS | S-Naive | S-Avg | Δ vs Naive | 相对 | Δ vs Avg | 相对 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in C.ETT4 + ["**平均**"]:
        v, n, a = (per_ds[ds]["visionts"][0] if ds != "**平均**"
                   else overall["visionts"][0],
                   per_ds[ds]["naive"][0] if ds != "**平均**"
                   else overall["naive"][0],
                   per_ds[ds]["avg"][0] if ds != "**平均**"
                   else overall["avg"][0])
        L.append("| %s | %.4f | %.4f | %.4f | %+.4f | %+.2f%% | %+.4f | %+.2f%% |"
                 % (ds, v, n, a, v - n, 100 * (v / n - 1),
                    v - a, 100 * (v / a - 1)))
    L += ["", "**MAE**", "",
          "| 数据集 | VisionTS | S-Naive | S-Avg | Δ vs Naive | 相对 | Δ vs Avg | 相对 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in C.ETT4 + ["**平均**"]:
        v, n, a = (per_ds[ds]["visionts"][1] if ds != "**平均**"
                   else overall["visionts"][1],
                   per_ds[ds]["naive"][1] if ds != "**平均**"
                   else overall["naive"][1],
                   per_ds[ds]["avg"][1] if ds != "**平均**"
                   else overall["avg"][1])
        L.append("| %s | %.4f | %.4f | %.4f | %+.4f | %+.2f%% | %+.4f | %+.2f%% |"
                 % (ds, v, n, a, v - n, 100 * (v / n - 1),
                    v - a, 100 * (v / a - 1)))
    L.append("")

    L += ["口径：与 EXP-001 完全一致（同划分、同 sklearn StandardScaler 训练段拟合、"
          "同 7 变量列序、同测试窗 N=4k−H+1、标准化空间、原 metric()）；"
          "历史 = 预测起点前 L 步 f32 标准化行（与模型所见一致）；"
          "两基线真值与 EXP-001 true.npy 逐元素 bitwise 相等（16/16 已核对）；"
          "Seasonal Naïve 手算核对逐位一致，Seasonal Avg 以独立 float64 累加为参考"
          "（rtol=1e-5, atol=1e-6）核对通过；"
          "VisionTS 原P 列 = EXP-001 实测（metrics.npy 复算精确相等）。", ""]

    md_path = os.path.join(C.PKG, "RESULT_b4_seasonal.md")
    with open(md_path, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print("\n[collect_summary] wrote %s and %s" % (csv_path, md_path))


if __name__ == "__main__":
    main()
