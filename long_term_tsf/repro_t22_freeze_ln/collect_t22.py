"""EXP-033 step 3: assemble summary_t22.csv + RESULT_t22_freeze_ln.md.

Freeze column = recomputed metric(pred, true) from the arrays of each
group (source: EXP-001 cache for verified ETTm1 groups, else this
experiment's save_t22_freeze run). LN column = EXP-003 aggregate.csv
snapshot (3-seed means, seeds 2021/2022/2023), pinned by sha256.
Aggregation: per dataset = equal mean over the 4 horizons; overall =
equal mean over the 4 datasets. delta = LN - Freeze (positive = LN
worse = finetuning hurt). Paper anchors reported alongside; |gap| > 0.001
flagged honestly, not gated (EXP-030 precedent). LN overall self-check
<=1e-6 against the user's recomputation is a hard gate.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t22_common as C

VERIFY_JSON = os.path.join(C.PKG, "logs", "verify_t22.json")
RESULT_MD = os.path.join(C.PKG, "RESULT_t22_freeze_ln.md")
SUMMARY_CSV = os.path.join(C.PKG, "summary_t22.csv")
ETT4 = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


def fail(msg):
    raise SystemExit("[collect_t22] FAIL: %s" % msg)


def freeze_metrics_all(metric):
    """{(ds,h): {"mse","mae","source","n"}} from verify evidence + arrays."""
    vj = C.load_json(VERIFY_JSON)
    out = {}
    for ds in ETT4:
        for h in C.HORIZONS:
            ck = vj.get("cache_%s_%d" % (ds, h))
            gk = vj.get("group_%s_%d" % (ds, h))
            if ds in C.CACHE_DS and ck and ck.get("status") == "cache_ok":
                src_dir, source = C.exp001_dir(ds, h), "exp001_cache"
                g = ck.get("arrays")
                if not g:
                    fail("cache %s_%d ok but array evidence missing" % (ds, h))
            else:
                if not gk:
                    fail("missing verified Freeze group %s_%d "
                         "(run+verify all groups first)" % (ds, h))
                g = gk
                src_dir = C.freeze_dir(ds, h)
                source = ("t22_rerun" if (ck and ck.get("status")
                                          == "rerun_needed") else "t22_new")
            rd = os.path.join(src_dir, "results", "_")
            pred = np.load(os.path.join(rd, "pred.npy"))
            true = np.load(os.path.join(rd, "true.npy"))
            rec = C.metrics_from_arrays(metric, pred, true)
            if not (g["true_bitwise_vs_std_array"]
                    and g["true_bitwise_vs_exp003_restored"]
                    and g["metrics_recompute_equals_stored"]):
                fail("%s_%d verification gates not all green" % (ds, h))
            out[(ds, h)] = {"mse": float(rec[1]), "mae": float(rec[0]),
                            "source": source, "n": g["n"]}
    return out


def fmt(x):
    return "%.6f" % x


def main():
    metric = C.load_metric()
    agg = C.load_aggregate()
    ln_ds, ln_overall = C.ln_aggregates(agg)   # hard self-check gate inside
    fz = freeze_metrics_all(metric)

    fz_ds, rows_g = {}, []
    for ds in ETT4:
        mses = [fz[(ds, h)]["mse"] for h in C.HORIZONS]
        maes = [fz[(ds, h)]["mae"] for h in C.HORIZONS]
        fz_ds[ds] = (float(np.mean(mses)), float(np.mean(maes)))
    fz_overall = (float(np.mean([fz_ds[d][0] for d in ETT4])),
                  float(np.mean([fz_ds[d][1] for d in ETT4])))

    # ---------- summary csv ----------
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "dataset", "horizon", "freeze_mse", "freeze_mae",
                    "ln_mse", "ln_mae", "delta_mse", "delta_mae",
                    "freeze_source"])
        for ds in ETT4:
            for h in C.HORIZONS:
                a = fz[(ds, h)]
                m = agg[(ds, h)]
                w.writerow(["group", ds, h, fmt(a["mse"]), fmt(a["mae"]),
                            fmt(m[0]), fmt(m[1]), fmt(m[0] - a["mse"]),
                            fmt(m[1] - a["mae"]), a["source"]])
        for ds in ETT4:
            a, m = fz_ds[ds], ln_ds[ds]
            w.writerow(["dataset", ds, "", fmt(a[0]), fmt(a[1]), fmt(m[0]),
                        fmt(m[1]), fmt(m[0] - a[0]), fmt(m[1] - a[1]), ""])
        a, m = fz_overall, ln_overall
        w.writerow(["overall", "", "", fmt(a[0]), fmt(a[1]), fmt(m[0]),
                    fmt(m[1]), fmt(m[0] - a[0]), fmt(m[1] - a[1]), ""])

    # ---------- RESULT md ----------
    L = [
        "# EXP-033:VisionTS 表 22 — 完全冻结 vs 仅微调 LayerNorm(ETT 4 数据集)",
        "",
        "配置:Table 19 回看长度 L(ETTh1/ETTh2 1152, ETTm1 2304, ETTm2 1152)"
        "与周期 P(24/24/96/96),r=c=0.4,双线性,原始 MAE-base 权重;标准化空"
        "间 MSE/MAE。**Freeze** = 冻结推理(train_epochs=0,seed 2021,bs 32);"
        "**LN** = 复用 EXP-003(`repro_fullshot/`,commit `b40d663`,`aggregate"
        ".csv` 3-seed 均值,seeds 2021/2022/2023;快照 sha256 `%s`)。"
        % C.AGG_SHA256,
        "",
        "**口径**:逐数据集 = 4 个 H 等权平均;总体 = 4 数据集等权平均;"
        "**Δ = LN − Freeze**(正值 = 微调后更差)。LN 为 3-seed 均值、"
        "Freeze 为单次(seed 2021)推理,逐 H 表只列实测值;表 22 锚点只在数"
        "据集层与总体层给出(3 位小数)。窗口数(用户核对 EXP-031 win_start,"
        "16 组共 109,840 窗):小时 N=2881−H,首预测行 11520;分钟 N=11521−H,"
        "首预测行 46080,与 L 无关。",
        "",
        "## 1. LN 列自检",
        "",
        "由快照 aggregate.csv 复算:总体 LN MSE/MAE = %s/%s,与用户复算 "
        "0.33235446/0.36986023 差 ≤1e-6,通过(硬门禁)。"
        % (fmt(ln_overall[0]), fmt(ln_overall[1])),
        "",
        "## 2. 逐 H 实测(16 组)",
        "",
        "| 数据集 | H | Freeze MSE | Freeze MAE | LN MSE | LN MAE "
        "| ΔMSE | ΔMAE | Freeze 来源 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ds in ETT4:
        for h in C.HORIZONS:
            a, m = fz[(ds, h)], agg[(ds, h)]
            L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |"
                     % (ds, h, fmt(a["mse"]), fmt(a["mae"]), fmt(m[0]),
                        fmt(m[1]), fmt(m[0] - a["mse"]), fmt(m[1] - a["mae"]),
                        {"exp001_cache": "EXP-001 缓存",
                         "t22_new": "本轮新跑",
                         "t22_rerun": "本轮补跑"}[a["source"]]))
    L += [
        "", "## 3. 数据集层(4-H 均值)与论文表 22 对照", "",
        "| 数据集 | Freeze 实测 | 论文 Freeze | |差| | LN 实测 | 论文 LN "
        "| |差| | Δ=LN−Freeze |",
        "|---|---|---|---|---|---|---|---|",
    ]
    flags = []
    for ds in ETT4:
        a, m, pf, pl = fz_ds[ds], ln_ds[ds], C.PAPER_T22[ds]["freeze"], \
            C.PAPER_T22[ds]["ln"]
        df_, dm = m[0] - a[0], m[1] - a[1]
        for mi, metric_name in ((0, "MSE"), (1, "MAE")):
            if abs(a[mi] - pf[mi]) > 0.001:
                flags.append("%s Freeze %s 实测 %s vs 论文 %.3f(差 %.4f)"
                             % (ds, metric_name, fmt(a[mi]), pf[mi],
                                a[mi] - pf[mi]))
            if abs(m[mi] - pl[mi]) > 0.001:
                flags.append("%s LN %s 实测 %s vs 论文 %.3f(差 %.4f)"
                             % (ds, metric_name, fmt(m[mi]), pl[mi],
                                m[mi] - pl[mi]))
        L.append("| %s | %s/%s | %.3f/%.3f | %.3f/%.3f | %s/%s | %.3f/%.3f "
                 "| %.3f/%.3f | %s/%s |"
                 % (ds, fmt(a[0]), fmt(a[1]), pf[0], pf[1],
                    abs(a[0] - pf[0]), abs(a[1] - pf[1]),
                    fmt(m[0]), fmt(m[1]), pl[0], pl[1],
                    abs(m[0] - pl[0]), abs(m[1] - pl[1]),
                    fmt(df_), fmt(dm)))
    a, m, pf, pl = fz_overall, ln_overall, C.PAPER_T22_OVERALL["freeze"], \
        C.PAPER_T22_OVERALL["ln"]
    for mi, metric_name in ((0, "MSE"), (1, "MAE")):
        if abs(a[mi] - pf[mi]) > 0.001:
            flags.append("总体 Freeze %s 实测 %s vs 论文 %.3f(差 %.4f)"
                         % (metric_name, fmt(a[mi]), pf[mi], a[mi] - pf[mi]))
        if abs(m[mi] - pl[mi]) > 0.001:
            flags.append("总体 LN %s 实测 %s vs 论文 %.3f(差 %.4f)"
                         % (metric_name, fmt(m[mi]), pl[mi], m[mi] - pl[mi]))
    L += ["| **总体** | %s/%s | %.3f/%.3f | %.3f/%.3f | %s/%s | %.3f/%.3f "
          "| %.3f/%.3f | **%s/%s** |"
          % (fmt(a[0]), fmt(a[1]), pf[0], pf[1], abs(a[0] - pf[0]),
             abs(a[1] - pf[1]), fmt(m[0]), fmt(m[1]), pl[0], pl[1],
             abs(m[0] - pl[0]), abs(m[1] - pl[1]), fmt(m[0] - a[0]),
             fmt(m[1] - a[1])),
          "",
          "## 4. 退步清单(Δ=LN−Freeze > 0,微调后变差)",
          ""]
    degr = []
    for ds in ETT4:
        for mi, name in ((0, "MSE"), (1, "MAE")):
            d = ln_ds[ds][mi] - fz_ds[ds][mi]
            if d > 0:
                degr.append(("dataset", ds, "", name, fz_ds[ds][mi],
                             ln_ds[ds][mi], d))
    for ds in ETT4:
        for h in C.HORIZONS:
            for mi, name, fkey in ((0, "MSE", "mse"), (1, "MAE", "mae")):
                d = agg[(ds, h)][mi] - fz[(ds, h)][fkey]
                if d > 0:
                    degr.append(("group", ds, h, name, fz[(ds, h)][fkey],
                                 agg[(ds, h)][mi], d))
    if degr:
        L.append("| 层级 | 数据集 | H | 指标 | Freeze | LN | Δ |")
        L.append("|---|---|---|---|---|---|---|")
        for lv, ds, h, name, a_, m_, d in degr:
            L.append("| %s | %s | %s | %s | %s | %s | %s |"
                     % ({"dataset": "数据集", "group": "逐H"}[lv], ds, h,
                        name, fmt(a_), fmt(m_), fmt(d)))
    else:
        L.append("无退步项:LN 在全部数据集层与逐 H 项上均不劣于 Freeze。")
    L += ["", "## 5. 论文对照标注(实测 − 论文,|差|>0.001 逐项列出)", ""]
    if flags:
        for x in flags:
            L.append("- %s" % x)
    else:
        L.append("全部数据集层与总体层 Freeze/LN 实测与论文锚点之差 ≤0.001。")
    vj = C.load_json(VERIFY_JSON)
    cache_lines = []
    for h in C.HORIZONS:
        ck = vj.get("cache_ETTm1_%d" % h)
        if ck:
            cache_lines.append("- ETTm1_%d:%s%s" % (
                h, ck["status"],
                ("(%s)" % ck["reason"]) if ck.get("reason") else ""))
    L += ["", "## 6. 核验与证据", "",
          "- 每组门禁:`logs/verify_t22.json`(SUCCESS 指纹逐文件复算、"
          "ckpt sha256、Table-19 L/P、train_epochs=0、无 checkpoint 产物、"
          "N=预期、true 双 bitwise(vs std_array 与 vs EXP-003 seed2021 还原)"
          "、metric 复算==metrics.npy、首预测行 %s)。"
          % "/".join(str(C.first_pred_abs_row0(d)) for d in ("ETTh1", "ETTm1")),
          "- ETTm1 缓存核验(逐组,缺失/漂移只补跑该组):",
          ] + cache_lines + [
          "- 汇总数值:`summary_t22.csv`(16 组 + 4 数据集 + 总体)。",
          "- Freeze 大数组:`save_t22_freeze/{DS}_{H}/results/_/{pred,true,"
          "metrics}.npy`;EXP-001 缓存:`save_repro_zeroshot/ETTm1_{H}/`;"
          "EXP-003 LN:`save_fullshot/{DS}_{H}_seed2021/`(真值 layout "
          "(N·7,H,1),按 7·窗口+变量展平固定还原)。",
          ""]
    with open(RESULT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("[collect_t22] wrote %s" % RESULT_MD)
    print("[collect_t22] overall Freeze %s/%s  LN %s/%s  dMSE=%s dMAE=%s"
          % (fmt(fz_overall[0]), fmt(fz_overall[1]), fmt(ln_overall[0]),
             fmt(ln_overall[1]), fmt(ln_overall[0] - fz_overall[0]),
             fmt(ln_overall[1] - fz_overall[1])))
    print("[collect_t22] degradation items: %d; paper flags: %d"
          % (len(degr), len(flags)))


if __name__ == "__main__":
    main()
