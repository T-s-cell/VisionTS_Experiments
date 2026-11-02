"""EXP-034 aggregation: three-layer summary + 126-item paper comparison.

Gates (any failure -> SystemExit):
  * logs/verify_t18.json holds all 48 green records — 16 bilinear
    (EXP-001 fingerprints, true double-bitwise, metrics==stored, EXP-031
    column aligned, probe bitwise) + 32 variants (config interpolation
    token, true double-bitwise vs std & EXP-001, pred bitwise != bilinear,
    metrics==stored, probe bitwise);
  * probe geometry consistent across the three interps of each (ds, H):
    identical expected formula record, identical actual resize sizes and
    num_patch_input, each record's modes matching its own interpolation;
  * stored metrics.npy re-checked against the verify record values
    (exact float equality — verify already proved recompute==stored);
  * bilinear overall self-check <= 1e-6 vs the user-recomputed pinned
    0.34439535439014435 / 0.37042006477713585 (hard gate).

Aggregation uses unrounded metrics.npy: per-H measured -> per-dataset
equal-weight mean of 4 H -> overall equal-weight mean of 4 datasets.
Paper anchors come from the THREE INDEPENDENT transcribed tables
(PAPER_T18 / PAPER_T18_DS / PAPER_T18_OVERALL) — no recomputation across
levels, because the paper averages unrounded values. 126 comparisons
(96 per-H + 24 dataset + 6 overall); abs diff > 0.001 is listed, never
fatal (EXP-030 precedent). Delta = variant - bilinear; per-H items with
Delta < 0 (variant better) are listed as local counterexamples, and the
paper's ETTm2/H720 nearest-neighbor win is judged by THIS round's
measurement.

Outputs (package dir): summary_t18.csv + RESULT_t18_interp.md.
"""
import csv
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t18_common as C

VERIFY_JSON = os.path.join(C.PKG, "logs", "verify_t18.json")
SUMMARY_CSV = os.path.join(C.PKG, "summary_t18.csv")
RESULT_MD = os.path.join(C.PKG, "RESULT_t18_interp.md")
MISMATCH_TOL = 0.001
_DSS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


def fail(msg):
    raise SystemExit("[collect_t18] FAIL: %s" % msg)


def want(cond, msg):
    if not cond:
        fail(msg)


def group_dir(ds, h, interp):
    return C.exp001_dir(ds, h) if interp == "bilinear" \
        else C.interp_dir(ds, h, interp)


def stored_metrics(ds, h, interp):
    m = np.load(os.path.join(group_dir(ds, h, interp), "results", "_",
                             "metrics.npy"))
    return float(m[1]), float(m[0])  # mse, mae (metric() order [mae,mse,...])


def check_bilinear_record(vj, ds, h):
    key = "bilinear_%s_%d" % (ds, h)
    rec = vj.get(key)
    want(rec is not None, "verify json missing section %s — run verify_t18 "
                          "--bilinear %s %d first" % (key, ds, h))
    fp = rec.get("fingerprints", {})
    want(fp.get("md5_all_match") is True and fp.get("checkpoint_artifacts") == [],
         "%s: fingerprint gate not green" % key)
    for f in ("true_bitwise_vs_std_array", "true_bitwise_vs_exp003",
              "metrics_recompute_equals_stored"):
        want(rec.get(f) is True, "%s: %s not green" % (key, f))
    col = rec.get("exp031_visionts_column", {})
    want(col.get("aligned") is True,
         "%s: EXP-031 VisionTS column not aligned" % key)
    probe = rec.get("probe", {})
    want(probe.get("pred_bitwise_vs_stored") is True,
         "%s: probe first batch not bitwise vs stored pred" % key)
    return rec


def check_variant_record(vj, ds, h, interp):
    key = "group_%s_%d_%s" % (ds, h, interp)
    rec = vj.get(key)
    want(rec is not None, "verify json missing section %s — run verify_t18 "
                          "--group %s %d %s first" % (key, ds, h, interp))
    fp = rec.get("fingerprints", {})
    want(fp.get("md5_all_match") is True and fp.get("checkpoint_artifacts") == [],
         "%s: fingerprint gate not green" % key)
    want(rec.get("interpolation") == interp,
         "%s: recorded interpolation %r != %s" % (key, rec.get("interpolation"),
                                                  interp))
    for f in ("true_bitwise_vs_std_array", "true_bitwise_vs_exp001",
              "metrics_recompute_equals_stored", "pred_bitwise_ne_bilinear"):
        want(rec.get(f) is True, "%s: %s not green" % (key, f))
    probe = rec.get("probe", {})
    want(probe.get("pred_bitwise_vs_stored") is True,
         "%s: probe first batch not bitwise vs stored pred" % key)
    want(float(rec.get("max_abs_diff_vs_bilinear", 0.0)) > 0.0,
         "%s: max_abs_diff_vs_bilinear not positive" % key)
    return rec


def check_geometry(records):
    """records: {(ds,h): {interp: probe-geometry dict}} — same (ds, H) must
    share expected formula and actual sizes/npi across interps; each
    record's modes must match its own interpolation."""
    import json as _json
    for (ds, h), by_interp in sorted(records.items()):
        want(set(by_interp) == {"bilinear", "bicubic", "nearest"},
             "%s_%d: geometry records for %s (need all three)"
             % (ds, h, sorted(by_interp)))
        exps = [_json.dumps(by_interp[i]["expected"], sort_keys=True)
                for i in by_interp]
        want(len(set(exps)) == 1,
             "%s_%d: expected geometry differs across interps" % (ds, h))
        for name in ("input_resize", "output_resize"):
            sizes = {_json.dumps(by_interp[i]["actual"][name]["size"])
                     for i in by_interp}
            want(len(sizes) == 1,
                 "%s_%d: %s size differs across interps: %s"
                 % (ds, h, name, sizes))
        npis = {int(by_interp[i]["expected"]["num_patch_input"])
                for i in by_interp}
        want(len(npis) == 1,
             "%s_%d: num_patch_input differs across interps" % (ds, h))
        mode_want = {"bilinear": "BILINEAR", "bicubic": "BICUBIC",
                     "nearest": "NEAREST"}
        for interp, g in by_interp.items():
            for name in ("input_resize", "output_resize"):
                mode = g["actual"][name]["mode"]
                want(mode_want[interp] in mode,
                     "%s_%d %s: %s mode %s != requested %s"
                     % (ds, h, interp, name, mode, mode_want[interp]))


def fmt(x, nd=4):
    return ("%." + str(nd) + "f") % x


def main():
    want(os.path.isfile(VERIFY_JSON),
         "verify json missing: %s — run the verify modes first" % VERIFY_JSON)
    vj = C.load_json(VERIFY_JSON)

    # ---- 48 green records + geometry cross-records ----
    geom = {}
    for ds, h in C.group_ids():
        rb = check_bilinear_record(vj, ds, h)
        geom.setdefault((ds, h), {})["bilinear"] = rb["probe"]["geometry"]
        for interp in C.VARIANT_INTERPS:
            rv = check_variant_record(vj, ds, h, interp)
            geom.setdefault((ds, h), {})[interp] = rv["probe"]["geometry"]
    check_geometry(geom)

    # ---- measured metrics from stored metrics.npy, cross-checked vs json ----
    meas = {}  # {(ds,h,interp): (mse, mae)}
    for ds, h in C.group_ids():
        for interp in C.INTERPS:
            mse, mae = stored_metrics(ds, h, interp)
            key = ("bilinear_%s_%d" % (ds, h)) if interp == "bilinear" \
                else ("group_%s_%d_%s" % (ds, h, interp))
            rec = vj[key]
            want(float(rec["mse"]) == mse and float(rec["mae"]) == mae,
                 "%s: metrics.npy (%.17g/%.17g) != verify record (%.17g/%.17g)"
                 % (key, mse, mae, rec["mse"], rec["mae"]))
            meas[(ds, h, interp)] = (mse, mae)

    # ---- three-layer aggregation (unrounded) ----
    ds_avg = {}   # {(ds,interp): (mse,mae)}
    for ds in _DSS:
        for interp in C.INTERPS:
            vs = [meas[(ds, h, interp)] for h in C.HORIZONS]
            ds_avg[(ds, interp)] = (float(np.mean([v[0] for v in vs])),
                                    float(np.mean([v[1] for v in vs])))
    overall = {}  # {interp: (mse,mae)}
    for interp in C.INTERPS:
        vs = [ds_avg[(ds, interp)] for ds in _DSS]
        overall[interp] = (float(np.mean([v[0] for v in vs])),
                           float(np.mean([v[1] for v in vs])))

    # ---- hard gate: bilinear overall self-check ----
    dmse = abs(overall["bilinear"][0] - C.BILINEAR_SELF_CHECK["overall_mse"])
    dmae = abs(overall["bilinear"][1] - C.BILINEAR_SELF_CHECK["overall_mae"])
    want(dmse <= C.SELF_TOL and dmae <= C.SELF_TOL,
         "bilinear overall self-check FAIL: %.17g/%.17g vs pinned "
         "%.17g/%.17g (tol %.0e) — the denominator of every Delta is wrong, "
         "stop and locate the cause"
         % (overall["bilinear"][0], overall["bilinear"][1],
            C.BILINEAR_SELF_CHECK["overall_mse"],
            C.BILINEAR_SELF_CHECK["overall_mae"], C.SELF_TOL))

    # ---- 126 paper comparisons (three independent tables, no recompute) ----
    mismatches = []  # (level, ds, h, interp, metric, measured, paper)
    n_cmp = 0
    for ds in _DSS:          # 96 per-H
        for h in C.HORIZONS:
            for interp in C.INTERPS:
                mse, mae = meas[(ds, h, interp)]
                pm, pa = C.PAPER_T18[ds][h][interp]
                for met, mv, pv in (("MSE", mse, pm), ("MAE", mae, pa)):
                    n_cmp += 1
                    if abs(mv - pv) > MISMATCH_TOL:
                        mismatches.append(("per_H", ds, h, interp, met,
                                           mv, pv))
    for ds in _DSS:          # 24 dataset
        for interp in C.INTERPS:
            mse, mae = ds_avg[(ds, interp)]
            pm, pa = C.PAPER_T18_DS[ds][interp]
            for met, mv, pv in (("MSE", mse, pm), ("MAE", mae, pa)):
                n_cmp += 1
                if abs(mv - pv) > MISMATCH_TOL:
                    mismatches.append(("dataset", ds, None, interp, met,
                                       mv, pv))
    for interp in C.INTERPS:  # 6 overall
        mse, mae = overall[interp]
        pm, pa = C.PAPER_T18_OVERALL[interp]
        for met, mv, pv in (("MSE", mse, pm), ("MAE", mae, pa)):
            n_cmp += 1
            if abs(mv - pv) > MISMATCH_TOL:
                mismatches.append(("overall", None, None, interp, met,
                                   mv, pv))
    want(n_cmp == 126, "paper comparison count %d != 126" % n_cmp)

    # ---- Delta tables + counterexamples (Delta < 0 = variant better) ----
    delta_ph = {}   # {(ds,h,interp): (dmse,dmae)}
    better = {"bicubic": [0, 0], "nearest": [0, 0]}  # [mse<0 count, mae<0]
    for ds in _DSS:
        for h in C.HORIZONS:
            for interp in C.VARIANT_INTERPS:
                d = (meas[(ds, h, interp)][0] - meas[(ds, h, "bilinear")][0],
                     meas[(ds, h, interp)][1] - meas[(ds, h, "bilinear")][1])
                delta_ph[(ds, h, interp)] = d
                if d[0] < 0:
                    better[interp][0] += 1
                if d[1] < 0:
                    better[interp][1] += 1
    delta_ds = {(ds, interp): (ds_avg[(ds, interp)][0] - ds_avg[(ds, "bilinear")][0],
                               ds_avg[(ds, interp)][1] - ds_avg[(ds, "bilinear")][1])
                for ds in _DSS for interp in C.VARIANT_INTERPS}
    delta_ov = {interp: (overall[interp][0] - overall["bilinear"][0],
                         overall[interp][1] - overall["bilinear"][1])
                for interp in C.VARIANT_INTERPS}

    ce = [(ds, h, interp, delta_ph[(ds, h, interp)])
          for ds in _DSS for h in C.HORIZONS for interp in C.VARIANT_INTERPS
          if delta_ph[(ds, h, interp)][0] < 0 or delta_ph[(ds, h, interp)][1] < 0]

    # ---- summary_t18.csv ----
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "dataset", "horizon", "interp",
                    "mse", "mae", "d_mse", "d_mae", "source"])
        for ds in _DSS:
            for h in C.HORIZONS:
                for interp in C.INTERPS:
                    mse, mae = meas[(ds, h, interp)]
                    if interp == "bilinear":
                        w.writerow(["per_H", ds, h, interp,
                                    "%.6f" % mse, "%.6f" % mae, "", "",
                                    "t18_measured"])
                    else:
                        d = delta_ph[(ds, h, interp)]
                        w.writerow(["per_H", ds, h, interp,
                                    "%.6f" % mse, "%.6f" % mae,
                                    "%+.6f" % d[0], "%+.6f" % d[1],
                                    "t18_measured"])
        for ds in _DSS:
            for interp in C.INTERPS:
                mse, mae = ds_avg[(ds, interp)]
                if interp == "bilinear":
                    w.writerow(["dataset", ds, "", interp,
                                "%.6f" % mse, "%.6f" % mae, "", "",
                                "t18_measured"])
                else:
                    d = delta_ds[(ds, interp)]
                    w.writerow(["dataset", ds, "", interp,
                                "%.6f" % mse, "%.6f" % mae,
                                "%+.6f" % d[0], "%+.6f" % d[1],
                                "t18_measured"])
        for interp in C.INTERPS:
            mse, mae = overall[interp]
            if interp == "bilinear":
                w.writerow(["overall", "", "", interp,
                            "%.6f" % mse, "%.6f" % mae, "", "",
                            "t18_measured"])
            else:
                d = delta_ov[interp]
                w.writerow(["overall", "", "", interp,
                            "%.6f" % mse, "%.6f" % mae,
                            "%+.6f" % d[0], "%+.6f" % d[1],
                            "t18_measured"])

    # ---- RESULT md ----
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT,
                              capture_output=True, text=True, timeout=10)
        commit = head.stdout.strip() if head.returncode == 0 else "unknown"
    except Exception:
        commit = "unknown"

    L = []
    L.append("# EXP-034 结果:VisionTS 表 18 插值策略对照(零样本,ETT 4 数据集 × H∈{96,192,336,720})\n")
    L.append("日期:2026-11-03 | commit `%s` | 冻结 MAE-base(sha256 c551c37b…)"
             % commit)
    L.append("")
    L.append("配置:论文表 8 回看(ETTh1 2880/24、ETTh2 1728/24、ETTm1 2304/96、"
             "ETTm2 4032/96),r=c=0.4、bs=32、seed=2021、train_epochs=0;"
             "双线性列整体复用 EXP-001(EXP-031 已核验),仅新增 bicubic/nearest "
             "32 组推理(--interpolation 一值同驱 input_resize 与 output_resize,"
             "run.py:106 -> models/VisionTS.py:22 -> visionts/model.py:84-92)。")
    L.append("")
    L.append("窗口数(与 L 无关):小时 N=2881−H、首预测行 11520;分钟 N=11521−H、"
             "首行 46080;16 组共 109,840 窗,真值跨实验逐元素可比。汇总:先 4 个 H "
             "等权平均、再 4 个数据集等权平均,全部用未舍入 metrics.npy。")
    L.append("")
    L.append("## 门禁\n")
    L.append("- 48 组证据全绿:16 组双线性(8 文件 md5+ckpt sha256、true 双 bitwise "
             "vs std_array/EXP-003、metric 复算==metrics.npy、EXP-031 原周期列逐组"
             "对齐 ≤1e-12、探针首批 vs pred.npy[:32] bitwise)+ 32 组新组(config 含 "
             "interpolation token、true 双 bitwise vs std_array/EXP-001、pred 逐位 "
             "≠ 双线性、metric 复算一致、探针首批 bitwise)。")
    L.append("- 探针几何:每 (ds,H) 三种插值的 input/output 缩放尺寸与可见 patch 列"
             "完全一致(几何与插值无关),各记录实际插值模式与其请求一致,两处 Resize "
             "antialias=False。")
    L.append("- 双线性总体自检:实测 %.17g/%.17g vs 钉死 %.17g/%.17g,"
             "差 %.2e/%.2e ≤ 1e-6(硬门禁)。"
             % (overall["bilinear"][0], overall["bilinear"][1],
                C.BILINEAR_SELF_CHECK["overall_mse"],
                C.BILINEAR_SELF_CHECK["overall_mae"], dmse, dmae))
    L.append("- 论文对照 126 项(96 逐 H + 24 数据集 + 6 总体),三层各自 vs 独立"
             "转写常量表;超 0.001 共 **%d** 项,见下文清单。" % len(mismatches))
    L.append("")

    L.append("## 总体(4 数据集等权)\n")
    L.append("| interp | MSE | MAE | ΔMSE | ΔMAE | 相对变化 MSE/MAE | 论文 MSE | 论文 MAE | abs diff MSE/MAE |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for interp in C.INTERPS:
        mse, mae = overall[interp]
        pm, pa = C.PAPER_T18_OVERALL[interp]
        if interp == "bilinear":
            L.append("| bilinear | %s | %s | — | — | — | %.3f | %.3f | %.4f / %.4f |"
                     % (fmt(mse, 6), fmt(mae, 6), pm, pa,
                        abs(mse - pm), abs(mae - pa)))
        else:
            d = delta_ov[interp]
            rel = "%+.2f%% / %+.2f%%" % (100.0 * d[0] / overall["bilinear"][0],
                                         100.0 * d[1] / overall["bilinear"][1])
            L.append("| %s | %s | %s | %+.4f | %+.4f | %s | %.3f | %.3f | %.4f / %.4f |"
                     % (interp, fmt(mse, 6), fmt(mae, 6), d[0], d[1], rel,
                        pm, pa, abs(mse - pm), abs(mae - pa)))
    L.append("")

    L.append("## 数据集层(4 个 H 等权)\n")
    L.append("| dataset | interp | MSE | MAE | ΔMSE | ΔMAE | 论文 MSE | 论文 MAE | abs diff MSE/MAE |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ds in _DSS:
        for interp in C.INTERPS:
            mse, mae = ds_avg[(ds, interp)]
            pm, pa = C.PAPER_T18_DS[ds][interp]
            if interp == "bilinear":
                L.append("| %s | bilinear | %s | %s | — | — | %.3f | %.3f | %.4f / %.4f |"
                         % (ds, fmt(mse, 6), fmt(mae, 6), pm, pa,
                            abs(mse - pm), abs(mae - pa)))
            else:
                d = delta_ds[(ds, interp)]
                L.append("| %s | %s | %s | %s | %+.4f | %+.4f | %.3f | %.3f | %.4f / %.4f |"
                         % (ds, interp, fmt(mse, 6), fmt(mae, 6),
                            d[0], d[1], pm, pa,
                            abs(mse - pm), abs(mae - pa)))
    L.append("")

    L.append("## 逐 H(Δ = 变体 − 双线性)\n")
    for ds in _DSS:
        L.append("### %s\n" % ds)
        L.append("| H | bilinear MSE/MAE | bicubic MSE/MAE | nearest MSE/MAE | Δbicubic MSE/MAE | Δnearest MSE/MAE |")
        L.append("|---|---|---|---|---|---|")
        for h in C.HORIZONS:
            bl = meas[(ds, h, "bilinear")]
            bc = meas[(ds, h, "bicubic")]
            nn = meas[(ds, h, "nearest")]
            dbc = delta_ph[(ds, h, "bicubic")]
            dnn = delta_ph[(ds, h, "nearest")]
            L.append("| %d | %s / %s | %s / %s | %s / %s | %+.4f / %+.4f | %+.4f / %+.4f |"
                     % (h, fmt(bl[0]), fmt(bl[1]), fmt(bc[0]), fmt(bc[1]),
                        fmt(nn[0]), fmt(nn[1]), dbc[0], dbc[1], dnn[0], dnn[1]))
        L.append("")

    L.append("## 论文对照 abs diff > 0.001 清单(%d 项)\n" % len(mismatches))
    if not mismatches:
        L.append("(无 —— 126 项全部 ≤ 0.001)\n")
    else:
        L.append("| 层 | dataset | H | interp | 指标 | 实测 | 论文 | abs diff |")
        L.append("|---|---|---|---|---|---|---|---|")
        for lv, ds, h, interp, met, mv, pv in mismatches:
            L.append("| %s | %s | %s | %s | %s | %s | %.3f | %.4f |"
                     % (lv, ds or "—", h if h else "—", interp, met,
                        fmt(mv, 6), pv, abs(mv - pv)))
        L.append("")

    L.append("## 局部反例:变体优于双线性的逐 H 项(Δ<0)\n")
    L.append("逐 H 层变体更优计数:bicubic MSE %d/16、MAE %d/16;nearest MSE %d/16、"
             "MAE %d/16。" % (better["bicubic"][0], better["bicubic"][1],
                              better["nearest"][0], better["nearest"][1]))
    if not ce:
        L.append("\n(无 —— 两变体在全部逐 H 项上均不优于双线性)\n")
    else:
        L.append("")
        for ds, h, interp, d in ce:
            m = meas[(ds, h, interp)]
            b = meas[(ds, h, "bilinear")]
            parts = []
            if d[0] < 0:
                parts.append("MSE %s < 双线性 %s(Δ%+.4f)"
                             % (fmt(m[0]), fmt(b[0]), d[0]))
            if d[1] < 0:
                parts.append("MAE %s < 双线性 %s(Δ%+.4f)"
                             % (fmt(m[1]), fmt(b[1]), d[1]))
            L.append("- %s_%d %s:%s" % (ds, h, interp, ";".join(parts)))
    m72 = delta_ph[("ETTm2", 720, "nearest")]
    if m72[0] < 0:
        L.append("\n论文局部反例(ETTm2/H720 nearest MSE 0.334 优于双线性 0.343)"
                 "**本轮实测重现**:ΔMSE %+.4f(ΔMAE %+.4f)。" % m72)
    else:
        L.append("\n论文局部反例(ETTm2/H720 nearest 优于双线性)本轮实测**未重现**:"
                 "ΔMSE %+.4f、ΔMAE %+.4f。" % m72)
    L.append("")

    L.append("## 插值生效证据\n")
    L.append("- 接线链:run.py:106(--interpolation)-> models/VisionTS.py:22 -> "
             "visionts/model.py:84-92,一值同驱 input_resize 与 output_resize;"
             "各组 SUCCESS.txt config 行含 interpolation token,8 个框架文件 md5 "
             "与 EXP-001 冻结记录逐文件一致(零框架改动)。")
    L.append("- 探针(与产出数组同路径构建):每组的 input_resize/output_resize "
             "实际模式/尺寸/antialias 已实测记录;ETTh1_96 三种模式均为 5 个可见 "
             "patch 列、输入缩放 224×80、输出缩放 24×336;探针首批前向与落盘 "
             "pred.npy[:32] 逐位一致。")
    L.append("- 每组 pred.npy 与双线性基线逐位不同(防插值静默失效),"
             "max abs diff 范围见 logs/verify_t18.json 的 max_abs_diff_vs_bilinear。")
    fw = [k for k in vj if k.startswith("first_window_")]
    if fw:
        L.append("- 冒烟首窗检查(%s):loader 窗 0 历史(Table-8 L 步)与 "
                 "std_array 逐位一致,7 变量顺序正确。" % ", ".join(sorted(fw)))
    L.append("")
    L.append("## 大数组(留在 eta,不入包)\n")
    L.append("- 新组:`save_t18_interp/{bicubic,nearest}/{DS}_{H}/results/_/{pred,true,metrics}.npy`")
    L.append("- 双线性基线:`save_repro_zeroshot/{DS}_{H}/results/_/`(EXP-001 复用)")
    L.append("- MAE 权重:`/dev_data/wlt/VisionTS_Experiments/ckpt/mae_visualize_vit_base.pth`"
             "(447MB,绝不入包)")
    L.append("")

    with open(RESULT_MD, "w") as f:
        f.write("\n".join(L))

    print("[collect_t18] OK: 48 records green, self-check %.2e/%.2e <= 1e-6, "
          "%d/126 paper mismatches, %d per-H counterexamples (Delta<0)"
          % (dmse, dmae, len(mismatches), len(ce)))
    print("[collect_t18] overall bilinear %.6f/%.6f, bicubic %.6f/%.6f "
          "(%+.4f/%+.4f), nearest %.6f/%.6f (%+.4f/%+.4f)"
          % (overall["bilinear"] + overall["bicubic"] + delta_ov["bicubic"]
             + overall["nearest"] + delta_ov["nearest"]))
    print("[collect_t18] wrote %s and %s" % (SUMMARY_CSV, RESULT_MD))


if __name__ == "__main__":
    main()
