#!/usr/bin/env python3
"""汇总 zero-shot 复现结果并与论文 Table 9 对照。
用法: python collect_results.py --save_base save_repro_zeroshot [--out_dir repro_zeroshot]
metrics.npy 布局 (exp_long_term_forecasting.py): [mae, mse, rmse, mape, mspe, best_valid_loss, best_valid_epoch]
"""
import argparse
import csv
import math
import os

import numpy as np

PAPER_TABLE9 = {  # (mse, mae) from VisionTS paper Table 9, VISIONTS zero-shot column
    "ETTh1": {96: (0.353, 0.383), 192: (0.392, 0.410), 336: (0.407, 0.423), 720: (0.406, 0.441)},
    "ETTh2": {96: (0.271, 0.328), 192: (0.328, 0.367), 336: (0.345, 0.381), 720: (0.388, 0.422)},
    "ETTm1": {96: (0.341, 0.347), 192: (0.360, 0.360), 336: (0.377, 0.374), 720: (0.416, 0.405)},
    "ETTm2": {96: (0.228, 0.282), 192: (0.262, 0.305), 336: (0.293, 0.328), 720: (0.343, 0.370)},
    "Weather": {96: (0.220, 0.257), 192: (0.244, 0.275), 336: (0.280, 0.299), 720: (0.330, 0.337)},
    "Electricity": {96: (0.177, 0.266), 192: (0.188, 0.277), 336: (0.207, 0.296), 720: (0.256, 0.337)},
}
PAPER_TABLE1_AVG = {  # (mse, mae) averaged over 4 pred lengths
    "ETTh1": (0.390, 0.414), "ETTh2": (0.333, 0.375), "ETTm1": (0.374, 0.372),
    "ETTm2": (0.282, 0.321), "Weather": (0.269, 0.292), "Electricity": (0.207, 0.294),
}
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"]
PRED_LENS = [96, 192, 336, 720]
TOLERANCE = 0.01  # 自定验收容差 (绝对偏差), 非误差理论界


def read_run(save_base, ds, pred):
    run_dir = os.path.join(save_base, f"{ds}_{pred}")
    ok_path = os.path.join(run_dir, "SUCCESS.txt")
    metrics_path = os.path.join(run_dir, "results", "_", "metrics.npy")
    if not os.path.isfile(metrics_path):
        return None
    m = np.load(metrics_path)
    mae, mse = float(m[0]), float(m[1])
    if not (math.isfinite(mae) and math.isfinite(mse)):
        return None
    fp_ok = os.path.isfile(ok_path)
    return {"mae": mae, "mse": mse, "fingerprint_ok": fp_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_base", default="save_repro_zeroshot")
    ap.add_argument("--out_dir", default=None, help="默认与 save_base 同级的 repro_zeroshot 目录")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(args.save_base)) if os.path.isdir(args.save_base) else "."
    out_dir = args.out_dir or os.path.join(root, "repro_zeroshot")
    os.makedirs(out_dir, exist_ok=True)

    rows = []  # (ds, pred, mse, mae, paper_mse, paper_mae)
    missing = []
    for ds in DATASETS:
        for pred in PRED_LENS:
            r = read_run(args.save_base, ds, pred)
            pmse, pmae = PAPER_TABLE9[ds][pred]
            if r is None:
                missing.append(f"{ds}_{pred}")
                continue
            rows.append((ds, pred, r["mse"], r["mae"], pmse, pmae, r["fingerprint_ok"]))

    # ---- summary.csv ----
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "pred_len", "mse_repro", "mse_paper", "mse_abs_diff", "mse_rel_diff_pct",
                    "mae_repro", "mae_paper", "mae_abs_diff", "mae_rel_diff_pct", "fingerprint_ok"])
        for ds, pred, mse, mae, pmse, pmae, fp in rows:
            w.writerow([ds, pred, f"{mse:.4f}", pmse, f"{mse - pmse:+.4f}", f"{(mse - pmse) / pmse * 100:+.2f}",
                        f"{mae:.4f}", pmae, f"{mae - pmae:+.4f}", f"{(mae - pmae) / pmae * 100:+.2f}", fp])
        for ds in DATASETS:
            drows = [r for r in rows if r[0] == ds]
            if len(drows) == 4:
                mse = sum(r[2] for r in drows) / 4
                mae = sum(r[3] for r in drows) / 4
                pmse, pmae = PAPER_TABLE1_AVG[ds]
                w.writerow([ds, "avg", f"{mse:.4f}", pmse, f"{mse - pmse:+.4f}", f"{(mse - pmse) / pmse * 100:+.2f}",
                            f"{mae:.4f}", pmae, f"{mae - pmae:+.4f}", f"{(mae - pmae) / pmae * 100:+.2f}", ""])

    # ---- results_comparison.md ----
    lines = []
    lines.append("# VisionTS Zero-Shot 复现 vs 论文 Table 9 对照")
    lines.append("")
    lines.append(f"- 数据: `{args.save_base}/`; 汇总时间: 见文件时间戳")
    lines.append("- 口径: MSE/MAE 在 StandardScaler 归一化空间; ETT 划分 12/4/4 个月(共 20 个月), Weather/Electricity 划分 0.7/0.1/0.2; test drop_last=False。")
    lines.append(f"- 验收容差 {TOLERANCE} 为自定绝对偏差阈值, 非误差理论界; 平均值不掩盖单点异常, 全部 24 项逐格列出。")
    lines.append("")
    lines.append("| Dataset | Pred | MSE (复现/论文/Δ/Δ%) | MAE (复现/论文/Δ/Δ%) |")
    lines.append("|---|---|---|---|")
    over = []
    for ds, pred, mse, mae, pmse, pmae, fp in rows:
        dm, da = mse - pmse, mae - pmae
        flag = " ⚠" if (abs(dm) > TOLERANCE or abs(da) > TOLERANCE) else ""
        if flag:
            over.append(f"{ds}/{pred}")
        fpm = "" if fp else " (无指纹标记)"
        lines.append(f"| {ds}{fpm} | {pred} | {mse:.3f} / {pmse:.3f} / {dm:+.3f} / {(dm / pmse * 100):+.1f}% "
                     f"| {mae:.3f} / {pmae:.3f} / {da:+.3f} / {(da / pmae * 100):+.1f}% |{flag}")
    for ds in DATASETS:
        drows = [r for r in rows if r[0] == ds]
        if len(drows) == 4:
            mse = sum(r[2] for r in drows) / 4
            mae = sum(r[3] for r in drows) / 4
            pmse, pmae = PAPER_TABLE1_AVG[ds]
            lines.append(f"| **{ds} avg** | — | **{mse:.3f} / {pmse:.3f} / {mse - pmse:+.3f}** "
                         f"| **{mae:.3f} / {pmae:.3f} / {mae - pmae:+.3f}** |")
    n = len(rows)
    if n:
        avg_mse = sum(r[2] for r in rows) / n
        avg_mae = sum(r[3] for r in rows) / n
        lines.append(f"| **总平均** ({n}/24) | — | **{avg_mse:.4f}** | **{avg_mae:.4f}** |")
    lines.append("")
    lines.append("## 结论")
    if missing:
        lines.append(f"- 缺失运行: {', '.join(missing)} (未完成或失败, 不计入统计)")
    if n == 24:
        n_over = len(over)
        lines.append(f"- 24 项中 {24 - n_over} 项绝对偏差 ≤ {TOLERANCE}, {n_over} 项超出: {', '.join(over) if over else '无'}。")
        lines.append("- 偏差来源未预设归因; 若超出容差, 逐项排查: 数据划分/预处理一致性、双线性插值实现差异、"
                     "权重文件完整性 (SUCCESS.txt 中 ckpt_sha256)、torch 版本数值行为等, 再下结论。")
    else:
        lines.append("- 运行未完成, 结论待全部 24 项就绪后给出。")
    md_path = os.path.join(out_dir, "results_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"written: {csv_path}\nwritten: {md_path}")
    print(f"completed runs: {n}/24" + (f", missing: {missing}" if missing else ""))
    for l in lines[-6:]:
        print(l)


if __name__ == "__main__":
    main()
