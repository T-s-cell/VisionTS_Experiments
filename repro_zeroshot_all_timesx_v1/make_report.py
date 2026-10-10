#!/usr/bin/env python3
"""Render results/REPORT.md from pipeline artifacts (read-only)."""
import json
import os

import common as C
import protocol as P


def load(p):
    path = os.path.join(C.HERE, p)
    return C.load_json(path) if os.path.isfile(path) else None


def main():
    freeze = load("frozen/freeze_report.json")
    reuse = load("reuse_final.json")
    pf = load("preflight_report.json")
    asm = load("results/assemble_report.json")
    ov = load("results/overall.json")
    s1677 = load("results/subset_1677.json")
    s2474 = load("results/subset_2474.json")
    acc = load("results/acceptance.json")
    doms = []
    dom_path = os.path.join(C.RESULTS_DIR, "domain_level.csv")
    if os.path.isfile(dom_path):
        import csv
        with open(dom_path) as f:
            doms = list(csv.DictReader(f))

    L = []
    L.append("# VisionTS × TimesX — 在 TimesX 全部原生窗口上的固定 P=1 zero-shot 补充评估")
    L.append("")
    L.append(f"Protocol: `{P.PROTOCOL_VERSION}` | 数据 zip sha256 `{P.ZIP_SHA256[:12]}…` | "
             f"权重 sha256 `{P.CKPT_SHA256[:12]}…` | r=c=0.4, 224×224, patch16, bilinear, fp32, "
             f"TF32 off, seed 2021, batch 64")
    L.append("")
    if freeze:
        L.append(f"- 冻结: **{freeze['checks']['total']['got']}** 原生窗口 / "
                 f"{freeze['checks']['n_vars']['got']} 变量 / {freeze['n_domains']} 领域"
                 f"（test {P.N_TEST} + val {P.N_VAL} + 跨边界剔除 {P.N_CROSS_REMOVED} + "
                 f"train 区 {P.N_TRAIN_REGION}）；零方差窗 {len(freeze['zero_variance_census'])} 个全部走 calib 回退")
    if pf:
        s = pf["stats"]
        fail_tag = (f"失败变量 {s['n_failed']} 个（其窗口 c1/c2 超出阈值，整变量重推）"
                    if s["n_failed"] else "全部通过双判据")
        L.append(f"- 预检: 复推 {s['n_checked']} 窗（compare={s['n_compare']}），"
                 f"c1_max={s['c1_max']:.2e} (阈值 {P.TOL_PRED_REL:g}), c2_max={s['c2_max']:.2e} "
                 f"(阈值 {P.TOL_PRED_OVER_D:g})，{fail_tag}")
    if reuse:
        t = reuse["totals"]
        L.append(f"- 缓存复用: {t['reuse_vars']} 变量 / {t['reuse_windows']} 窗（S2 绑定校验+逐窗核对，"
                 f"来源文件 sha256 全记录）；补推理 {t['fill_windows']} 窗")
    if asm:
        L.append(f"- 汇总: {asm['totals']['windows']} 窗（cache={asm['totals']['cache']}, "
                 f"new={asm['totals']['new']}），每窗恰一条预测")
    if ov:
        L.append("")
        L.append("## 总体（标准化尺度；19 域等权）")
        L.append("")
        L.append("| MSE | MAE |")
        L.append("|---|---|")
        L.append(f"| {ov['std_mse']:.6f} | {ov['std_mae']:.6f} |")
        L.append("")
        L.append("原始尺度 MSE/MAE 只在变量层报告（variable_level.csv），不跨单位平均。")
    if s1677:
        pc = s1677["pure_cache"]
        L.append("")
        L.append("## 子集对照")
        L.append("")
        L.append(f"- **1,677 旧 test 子集（纯缓存复算）**: MSE {pc['std_mse']:.6f} / "
                 f"MAE {pc['std_mae']:.6f}，vs 参考 {pc['reference']['std_mse']:.6f}/"
                 f"{pc['reference']['std_mae']:.6f}，Δ=({pc['delta_vs_ref']['std_mse']:.1e}, "
                 f"{pc['delta_vs_ref']['std_mae']:.1e})，≤5e-7: **{pc['within_5e-7']}**")
        L.append(f"- 1,677 子集（最终预测口径）: MSE {s1677['final_preds']['std_mse']:.6f} / "
                 f"MAE {s1677['final_preds']['std_mae']:.6f}"
                 + (f"，受重推变量影响: {s1677['reinfer_vars_touching_subset']}"
                    if s1677["reinfer_vars_touching_subset"] else "（无重推变量触及）"))
        if s2474:
            L.append(f"- **2,474 旧 LN test 子集（信息性）**: MSE {s2474['overall']['std_mse']:.6f} / "
                     f"MAE {s2474['overall']['std_mae']:.6f} "
                     f"(cache={s2474['n_cache']}, new={s2474['n_new']})")
    if doms:
        L.append("")
        L.append("## 分领域（域内变量等权）")
        L.append("")
        L.append("| Domain | vars | MSE | MAE |")
        L.append("|---|---|---|---|")
        for r in sorted(doms, key=lambda x: x["domain"]):
            L.append(f"| {r['domain']} | {r['n_vars']} | {float(r['std_mse']):.4f} | "
                     f"{float(r['std_mae']):.4f} |")
    if acc:
        L.append("")
        L.append(f"## 验收: **{'ALL PASS' if acc['pass'] else 'FAILED'}** "
                 f"({sum(acc['checks'].values())}/{len(acc['checks'])} 项)")
        for k, v in acc["checks"].items():
            L.append(f"- {'PASS' if v else 'FAIL'} {k}")
    L.append("")
    L.append("> 本实验为「在 TimesX 全部原生窗口上的固定 P=1 zero-shot 补充评估」，"
             "原 test 集实验（repro_zeroshot_ext）结论保持独立。")
    out = os.path.join(C.RESULTS_DIR, "REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[report] -> {out}")


if __name__ == "__main__":
    main()
