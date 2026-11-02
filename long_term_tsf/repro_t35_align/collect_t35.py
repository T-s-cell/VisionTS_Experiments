"""EXP-035 aggregation: c-sensitivity tables + trend reading + compare fig.

Gates (any failure -> SystemExit):
  * logs/verify_t35.json holds all 36 green records — 4 baseline
    (EXP-001 fingerprints, true double-bitwise vs std & EXP-003 restore,
    metrics==stored, EXP-031 column aligned <=1e-12, probe bitwise at
    c=0.4) + 32 new groups (config align_const token, true double-bitwise
    vs std & EXP-001, finite values, pred bitwise != c=0.4,
    metrics==stored, probe bitwise at the requested c);
  * probe geometry per (H, c) matches the exact update_config formula
    recomputed here (npi, input/output sizes, bilinear modes);
  * mandatory smoke evidence present and green (hard gate):
    first_window_ETTm1_96_0.6 (history [43776,46080) + future truth
    [46080,46096) bitwise, variable order) and smoke_hash_ETTm1_96_0.6
    (inputs bitwise identical across the two protocols, both replays
    bitwise vs stored, pinned geometry pairs 5->8 / 224x80->224x128 /
    96x67->96x42, input + two prediction SHA-256 recorded, prediction
    hashes differ);
  * stored metrics.npy re-checked against verify records (exact float
    equality);
  * baseline four-H equal-weight self-check <= 1e-6 vs the user-recomputed
    pinned 0.373517170548439 / 0.37162137776613235 (hard gate — the
    denominator of every Delta).

Aggregation uses unrounded metrics.npy. Two tables: 36 per-config rows
(c, H, mse, mae, Delta vs c=0.4 same H, measured npi / sizes) + 9 rows of
four-H equal-weight means (no window-count weighting). Trend reading:
lowest c, rising/falling intervals from consecutive deltas — qualitative
only; the paper Figure 9(c) panel has no printed numbers, so the
comparison is trend-level (argmin position and non-monotonicity), with
all observed discrepancies reported as-is. fig9c_compare.png: measured
curve next to the paper panel image (matplotlib, --fig regenerates).

Outputs (package dir): summary_t35.csv + RESULT_t35_align.md.
"""
import csv
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t35_common as C

VERIFY_JSON = os.path.join(C.PKG, "logs", "verify_t35.json")
SUMMARY_CSV = os.path.join(C.PKG, "summary_t35.csv")
RESULT_MD = os.path.join(C.PKG, "RESULT_t35_align.md")
FIG_PNG = os.path.join(C.PKG, "fig9c_compare.png")


def fail(msg):
    raise SystemExit("[collect_t35] FAIL: %s" % msg)


def want(cond, msg):
    if not cond:
        fail(msg)


def stored_metrics(h, c):
    d = C.exp001_dir(h) if c == C.BASELINE_C else C.group_dir(h, c)
    rd = os.path.join(d, "results", "_")
    m = np.load(os.path.join(rd, "metrics.npy"))
    return float(m[1]), float(m[0])  # mse, mae (metric() order [mae,mse,...])


def check_baseline_record(vj, h):
    key = "baseline_%d" % h
    rec = vj.get(key)
    want(rec is not None, "verify json missing section %s — run verify_t35 "
                          "--baseline %d first" % (key, h))
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


def check_group_record(vj, h, c):
    cs = C.c_str(c)
    key = "group_%d_%s" % (h, cs)
    rec = vj.get(key)
    want(rec is not None, "verify json missing section %s — run verify_t35 "
                          "--group %d %s first" % (key, h, cs))
    fp = rec.get("fingerprints", {})
    want(fp.get("md5_all_match") is True and fp.get("checkpoint_artifacts") == [],
         "%s: fingerprint gate not green" % key)
    want(str(rec.get("align_const")) == cs,
         "%s: recorded align_const %r != %s" % (key, rec.get("align_const"), cs))
    for f in ("true_bitwise_vs_std_array", "true_bitwise_vs_exp001",
              "metrics_recompute_equals_stored", "pred_bitwise_ne_baseline"):
        want(rec.get(f) is True, "%s: %s not green" % (key, f))
    probe = rec.get("probe", {})
    want(probe.get("pred_bitwise_vs_stored") is True,
         "%s: probe first batch not bitwise vs stored pred" % key)
    return rec


def check_geometry(vj, h, c, rec_key):
    """Probe geometry must equal the formula recomputed HERE (independent
    of verify-time computation)."""
    import json as _json
    probe = vj[rec_key]["probe"]
    got = probe["geometry"]["expected"]
    want_exp = C.expected_geometry(h, c)
    want(_json.dumps(got, sort_keys=True) == _json.dumps(want_exp, sort_keys=True),
         "%s: expected geometry %s != formula recomputed %s"
         % (rec_key, got, want_exp))
    actual = probe["geometry"]["actual"]
    want(actual["input_resize"]["size"] == want_exp["input_size"]
         and actual["output_resize"]["size"] == want_exp["output_size"],
         "%s: actual resize sizes %s/%s != formula %s/%s"
         % (rec_key, actual["input_resize"]["size"],
            actual["output_resize"]["size"], want_exp["input_size"],
            want_exp["output_size"]))
    for name in ("input_resize", "output_resize"):
        mode = actual[name]["mode"]
        want("BILINEAR" in mode, "%s: %s mode %s != bilinear" % (rec_key, name, mode))
        aa = actual[name]["antialias"]
        want(aa is False or aa is None,
             "%s: %s antialias=%r != False" % (rec_key, name, aa))
    return want_exp


def check_smoke_evidence(vj):
    """Mandatory (user gate): window-0 record + three SHA-256 pair record,
    both green, for ETTm1_96 c=0.4->0.6."""
    fw_key = "first_window_96_0.6"
    fw = vj.get(fw_key)
    want(fw is not None, "verify json missing %s — run verify_t35 "
                         "--first_window 96 0.6 (mandatory smoke evidence)"
                         % fw_key)
    t0 = C.first_pred_abs_row0()
    want(fw.get("history_bitwise_vs_std_array") is True
         and fw.get("future_truth_bitwise_vs_std_array") is True,
         "%s: bitwise gates not green" % fw_key)
    want(fw.get("history_rows") == [t0 - C.ETTM1["ctx"], t0]
         and fw.get("future_truth_rows") == [t0, t0 + 96],
         "%s: rows %s/%s != [%d,%d)/[%d,%d)"
         % (fw_key, fw.get("history_rows"), fw.get("future_truth_rows"),
            t0 - C.ETTM1["ctx"], t0, t0, t0 + 96))

    sh_key = "smoke_hash_96_0.6"
    sh = vj.get(sh_key)
    want(sh is not None, "verify json missing %s — run verify_t35 "
                         "--smoke_hash 96 0.6 (mandatory smoke evidence)"
                         % sh_key)
    want(sh.get("inputs_bitwise_identical") is True,
         "%s: cross-protocol inputs not bitwise identical" % sh_key)
    want(sh.get("pred_hashes_differ") is True,
         "%s: prediction hashes not proven different" % sh_key)
    for cc in ("0.4", "0.6"):
        want(sh.get("replay_bitwise_vs_stored", {}).get(cc) is True,
             "%s: replay not bitwise vs stored for c=%s" % (sh_key, cc))
        want(sh.get("pred_sha256", {}).get(cc), "%s: pred sha256 missing "
             "for c=%s" % (sh_key, cc))
    want(sh.get("input_sha256"), "%s: input sha256 missing" % sh_key)
    want(sh.get("pinned_geometry_pairs_ok") is True,
         "%s: pinned geometry pairs not green" % sh_key)
    return fw, sh


def trend_paragraph(mean_rows):
    """Lowest c + consecutive rising/falling intervals (measured only)."""
    cs = [r["c"] for r in mean_rows]
    ys = [r["mse"] for r in mean_rows]
    lo = int(np.argmin(ys))
    steps = []
    for i in range(len(cs) - 1):
        d = ys[i + 1] - ys[i]
        arrow = "升" if d > 0 else "降"
        steps.append("%s→%s %s(%+.4f)" % (C.c_str(cs[i]), C.c_str(cs[i + 1]),
                                          arrow, d))
    return cs[lo], ys[lo], steps


def make_figure(mean_rows, out_png):
    """Measured curve next to the paper Figure 9(c) panel jpg."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot([r["c"] for r in mean_rows], [r["mse"] for r in mean_rows],
            "o-", color="#1f77b4")
    ax.axvline(C.BASELINE_C, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("align constant c")
    ax.set_ylabel("equal-weight MSE over H in {96,192,336,720}")
    ax.set_title("VisionTS ETTm1 zero-shot (EXP-035 measured)")
    ax.set_xticks([round(c, 1) for c in C.ALIGN_C])
    ax.grid(alpha=0.3)
    ax2 = axes[1]
    img = mpimg.imread(C.FIG9C_IMAGE)
    ax2.imshow(img)
    ax2.set_title("paper Figure 9(c) ETTm1 (trend only)")
    ax2.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    want(os.path.isfile(VERIFY_JSON),
         "verify json missing: %s — run the verify modes first" % VERIFY_JSON)
    vj = C.load_json(VERIFY_JSON)

    # ---- 36 green records + geometry vs formula ----
    geom = {}  # {(h, c): expected geometry}
    for h in C.HORIZONS:
        rb = check_baseline_record(vj, h)
        geom[(h, C.BASELINE_C)] = check_geometry(vj, h, C.BASELINE_C,
                                                 "baseline_%d" % h)
        for c in C.VARIANT_C:
            rv = check_group_record(vj, h, c)
            geom[(h, c)] = check_geometry(vj, h, c, "group_%d_%s" % (h, C.c_str(c)))

    # ---- mandatory smoke evidence ----
    fw, sh = check_smoke_evidence(vj)

    # ---- measured metrics from stored metrics.npy, cross-checked vs json ----
    meas = {}  # {(h, c): (mse, mae)}
    for h in C.HORIZONS:
        mse, mae = stored_metrics(h, C.BASELINE_C)
        rec = vj["baseline_%d" % h]
        want(float(rec["mse"]) == mse and float(rec["mae"]) == mae,
             "baseline_%d: metrics.npy (%.17g/%.17g) != verify record "
             "(%.17g/%.17g)" % (h, mse, mae, rec["mse"], rec["mae"]))
        meas[(h, C.BASELINE_C)] = (mse, mae)
        for c in C.VARIANT_C:
            mse, mae = stored_metrics(h, c)
            rec = vj["group_%d_%s" % (h, C.c_str(c))]
            want(float(rec["mse"]) == mse and float(rec["mae"]) == mae,
                 "group_%d_%s: metrics.npy (%.17g/%.17g) != verify record "
                 "(%.17g/%.17g)" % (h, C.c_str(c), mse, mae, rec["mse"], rec["mae"]))
            meas[(h, c)] = (mse, mae)

    # ---- four-H equal-weight means + hard self-check vs pinned ----
    mean_rows = []
    for c in C.ALIGN_C:
        vs = [meas[(h, c)] for h in C.HORIZONS]
        mean_rows.append({"c": c,
                          "mse": float(np.mean([v[0] for v in vs])),
                          "mae": float(np.mean([v[1] for v in vs]))})
    base_mean = next(r for r in mean_rows if r["c"] == C.BASELINE_C)
    dmse = abs(base_mean["mse"] - C.BASELINE_SELF_CHECK["avg_mse"])
    dmae = abs(base_mean["mae"] - C.BASELINE_SELF_CHECK["avg_mae"])
    want(dmse <= C.SELF_TOL and dmae <= C.SELF_TOL,
         "baseline four-H self-check FAIL: %.17g/%.17g vs pinned "
         "%.17g/%.17g (tol %.0e) — the denominator of every Delta is wrong, "
         "stop and locate the cause"
         % (base_mean["mse"], base_mean["mae"],
            C.BASELINE_SELF_CHECK["avg_mse"], C.BASELINE_SELF_CHECK["avg_mae"],
            C.SELF_TOL))

    # ---- deltas vs c=0.4 same H ----
    delta = {}  # {(h, c): (dmse, dmae)}
    for h in C.HORIZONS:
        for c in C.VARIANT_C:
            delta[(h, c)] = (meas[(h, c)][0] - meas[(h, C.BASELINE_C)][0],
                             meas[(h, c)][1] - meas[(h, C.BASELINE_C)][1])
    for r in mean_rows:
        r["d_mse"] = r["mse"] - base_mean["mse"]
        r["d_mae"] = r["mae"] - base_mean["mae"]

    # ---- summary_t35.csv: 36 per-config rows + 9 mean rows ----
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "c", "horizon", "mse", "mae", "d_mse", "d_mae",
                    "num_patch_input", "input_size", "output_size"])
        for h in C.HORIZONS:
            for c in C.ALIGN_C:
                mse, mae = meas[(h, c)]
                g = geom[(h, c)]
                if c == C.BASELINE_C:
                    w.writerow(["per_config", C.c_str(c), h, "%.6f" % mse,
                                "%.6f" % mae, "", "",
                                g["num_patch_input"],
                                "%dx%d" % tuple(g["input_size"]),
                                "%dx%d" % tuple(g["output_size"])])
                else:
                    d = delta[(h, c)]
                    w.writerow(["per_config", C.c_str(c), h, "%.6f" % mse,
                                "%.6f" % mae, "%+.6f" % d[0], "%+.6f" % d[1],
                                g["num_patch_input"],
                                "%dx%d" % tuple(g["input_size"]),
                                "%dx%d" % tuple(g["output_size"])])
        for r in mean_rows:
            npis = [geom[(h, r["c"])]["num_patch_input"] for h in C.HORIZONS]
            if r["c"] == C.BASELINE_C:
                w.writerow(["h_mean", C.c_str(r["c"]), "", "%.6f" % r["mse"],
                            "%.6f" % r["mae"], "", "",
                            "/".join(str(x) for x in npis), "", ""])
            else:
                w.writerow(["h_mean", C.c_str(r["c"]), "", "%.6f" % r["mse"],
                            "%.6f" % r["mae"], "%+.6f" % r["d_mse"],
                            "%+.6f" % r["d_mae"],
                            "/".join(str(x) for x in npis), "", ""])

    # ---- trend reading (measured only; no eyeballed paper numbers) ----
    lo_c, lo_y, steps = trend_paragraph(mean_rows)

    # ---- RESULT md ----
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT,
                              capture_output=True, text=True, timeout=10)
        commit = head.stdout.strip() if head.returncode == 0 else "unknown"
    except Exception:
        commit = "unknown"

    n_groups = 4 + 4 * len(C.VARIANT_C)
    L = []
    L.append("# EXP-035 结果:VisionTS 附录 B.9 图 9(c) ETTm1 对齐系数 c 敏感性"
             "(零样本,H∈{96,192,336,720} × c∈{0.2..1.0})\n")
    L.append("日期:2026-11-03 | commit `%s` | 冻结 MAE-base(sha256 c551c37b…)"
             % commit)
    L.append("")
    L.append("配置:EXP-001 零样本协议 ETTm1 L=2304/P=96、r=0.4、双线性、原始图像"
             "方向、features=M、bs=32、seed=2021、train_epochs=0;唯一变量 "
             "--align_const(run.py:108 -> models/VisionTS.py:22 -> "
             "visionts/model.py npi=int(input_ratio×14×c))。c=0.4 四组整体复用 "
             "EXP-001(EXP-031 已核验),仅新增 %d 组推理,输出 "
             "save_t35_align/c{C}/ETTm1_{H}/。" % (n_groups - 4))
    L.append("")
    L.append("窗口数 N=11521−H=11425/11329/11185/10801,首预测行 46080,窗 0 历史 "
             "[43776,46080);汇总先 4 个 H 等权平均(不按窗口数加权),全部用未舍入 "
             "metrics.npy;Δ = 该 c − c=0.4 同 H。")
    L.append("")
    L.append("## 门禁\n")
    L.append("- %d 组证据全绿:4 组基线(8 文件 md5+ckpt sha256、true 双 bitwise "
             "vs std_array/EXP-003 还原、metric 复算==metrics.npy、EXP-031 列逐组"
             "对齐 ≤1e-12、探针首批 vs pred.npy[:32] bitwise)+ 32 组新组(config 含 "
             "align_const token、true 双 bitwise vs std_array/EXP-001 同 H、数值"
             "全有限、pred 逐位 ≠ c=0.4、metric 复算一致、探针首批 bitwise)。"
             % n_groups)
    L.append("- 探针几何:每 (H,c) 的 npi/输入输出缩放尺寸 == update_config 精确"
             "公式(collect 内独立复算),两处 Resize 均 bilinear、antialias=False;"
             "H=336/720 有 48 步右填充(pad_right=48)。")
    L.append("- **冒烟证据(必查项,硬门禁)**:first_window_ETTm1_96_0.6 — 窗 0 "
             "历史 [43776,46080) 与未来真值 [46080,46096) 对 std_array 逐位一致,"
             "7 变量顺序正确;smoke_hash_ETTm1_96_0.6 — 两协议首批输入逐位相同,"
             "两路回放各自与落盘 pred.npy[:32] 逐位一致,钉死几何对 npi 5→8、输入 "
             "224×80→224×128、输出 96×67→96×42 实测吻合,输入/两路预测三份 "
             "SHA-256 在录(%s…/%s…/%s…),两路预测哈希不同。"
             % (sh["input_sha256"][:12], sh["pred_sha256"]["0.4"][:12],
                sh["pred_sha256"]["0.6"][:12]))
    L.append("- c=0.4 基线四 H 等权自检:实测 %.17g/%.17g vs 钉死 %.17g/%.17g,"
             "差 %.2e/%.2e ≤ 1e-6(硬门禁)。"
             % (base_mean["mse"], base_mean["mae"],
                C.BASELINE_SELF_CHECK["avg_mse"], C.BASELINE_SELF_CHECK["avg_mae"],
                dmse, dmae))
    L.append("")

    L.append("## 四 H 等权平均(主表,9 行)\n")
    L.append("| c | MSE | MAE | ΔMSE | ΔMAE | 可见列数 H96/H192/H336/H720 |")
    L.append("|---|---|---|---|---|---|")
    for r in mean_rows:
        npis = "/".join(str(geom[(h, r["c"])]["num_patch_input"])
                        for h in C.HORIZONS)
        if r["c"] == C.BASELINE_C:
            L.append("| %s | %s | %s | — | — | %s |"
                     % (C.c_str(r["c"]), "%.6f" % r["mse"], "%.6f" % r["mae"],
                        npis))
        else:
            L.append("| %s | %s | %s | %+.4f | %+.4f | %s |"
                     % (C.c_str(r["c"]), "%.6f" % r["mse"], "%.6f" % r["mae"],
                        r["d_mse"], r["d_mae"], npis))
    L.append("")

    L.append("## 逐配置(36 行,Δ = 该 c − c=0.4 同 H)\n")
    for h in C.HORIZONS:
        L.append("### ETTm1_%d(N=%d 窗)\n" % (h, C.expected_n_windows(h)))
        L.append("| c | MSE | MAE | ΔMSE | ΔMAE | npi | 输入缩放 | 输出缩放 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in C.ALIGN_C:
            mse, mae = meas[(h, c)]
            g = geom[(h, c)]
            if c == C.BASELINE_C:
                L.append("| %s | %s | %s | — | — | %d | %d×%d | %d×%d |"
                         % (C.c_str(c), "%.6f" % mse, "%.6f" % mae,
                            g["num_patch_input"], g["input_size"][0],
                            g["input_size"][1], g["output_size"][0],
                            g["output_size"][1]))
            else:
                d = delta[(h, c)]
                L.append("| %s | %s | %s | %+.4f | %+.4f | %d | %d×%d | %d×%d |"
                         % (C.c_str(c), "%.6f" % mse, "%.6f" % mae,
                            d[0], d[1], g["num_patch_input"],
                            g["input_size"][0], g["input_size"][1],
                            g["output_size"][0], g["output_size"][1]))
        L.append("")

    L.append("## 趋势判读(实测;论文图 9(c) 原图无数值,只对趋势)\n")
    lo_row = next(r for r in mean_rows if r["c"] == lo_c)
    L.append("- 实测最低点:c=%s(四 H 等权 MSE %.6f,MAE %.6f);论文正文称默认 "
             "c=0.4 附近表现稳健(B.9/正文 \"around 0.4\"),实测最低点%s。"
             % (C.c_str(lo_c), lo_y, lo_row["mae"],
                "与论文一致" if lo_c == C.BASELINE_C
                else "**偏离 0.4,见下方差异说明**"))
    L.append("- 相邻 c 区间走向(未舍入差值):%s。" % "、".join(steps))
    L.append("- 形态:npi=int(·) 的整数量化使曲线呈锯齿(相邻 c 可见列数可能跳变"
             "或持平);论文原图同样非单调,本判读只按最低点与升降区间对比,不拟合"
             "平滑。")
    L.append("- 与论文图 9(c) 原图的差异如实保留:若实测最优 c 偏离 0.4 或形态"
             "完全不符,见上表数值本身;原图仅作趋势对照(fig9c_compare.png 并排)。")
    L.append("")

    L.append("## c 生效证据链\n")
    L.append("- 接线:run.py:108(--align_const,float 默认 0.4)-> "
             "models/VisionTS.py:22 update_config -> visionts/model.py "
             "npi=int(input_ratio×num_patch×c),一值驱动全部几何;各组 "
             "SUCCESS.txt config 行含 align_const token,8 个框架文件 md5 与 "
             "EXP-001 冻结记录逐文件一致(零框架改动)。")
    L.append("- 每组探针实测 npi 与两处缩放尺寸 == 公式(collect 独立复算);"
             "每组 pred.npy 与 c=0.4 基线逐位不同(防 align_const 静默失效),"
             "max abs diff 见 logs/verify_t35.json 的 max_abs_diff_vs_baseline。")
    L.append("- 冒烟对照(c=0.4→0.6 同一批输入):npi 5→8、输入 224×80→224×128、"
             "输出 96×67→96×42;三份 SHA-256 与首窗记录为 collect 必查项(见门禁)。")
    L.append("")

    L.append("## 大数组(留在 eta,不入包)\n")
    L.append("- 新组:`save_t35_align/c{C}/ETTm1_{H}/results/_/{pred,true,metrics}.npy`")
    L.append("- c=0.4 基线:`save_repro_zeroshot/ETTm1_{H}/results/_/`(EXP-001 复用)")
    L.append("- MAE 权重:`/dev_data/wlt/VisionTS_Experiments/ckpt/mae_visualize_vit_base.pth`"
             "(447MB,绝不入包)")
    L.append("")

    with open(RESULT_MD, "w") as f:
        f.write("\n".join(L))

    # ---- figure (numbers above already written; fig failure not fatal for
    # the CSV/RESULT, but S6 completion requires the png) ----
    try:
        make_figure(mean_rows, FIG_PNG)
        fig_msg = "fig9c_compare.png written"
    except (ImportError, OSError) as e:
        # numbers are already written above; S6 completion still needs the png
        fig_msg = ("FIGURE SKIPPED (%s: %s) — provide the paper panel jpg / "
                   "matplotlib, then rerun `python3 collect_t35.py --fig`"
                   % (type(e).__name__, e))
    print("[collect_t35] OK: %d records green, smoke evidence green, "
          "self-check %.2e/%.2e <= 1e-6" % (n_groups, dmse, dmae))
    print("[collect_t35] c-sweep 4H-mean MSE: %s"
          % " ".join("%s=%.6f" % (C.c_str(r["c"]), r["mse"]) for r in mean_rows))
    print("[collect_t35] lowest c=%s (%.6f); %s" % (C.c_str(lo_c), lo_y, fig_msg))
    print("[collect_t35] wrote %s and %s" % (SUMMARY_CSV, RESULT_MD))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fig":
        want(os.path.isfile(VERIFY_JSON),
             "verify json missing: %s — run the verify modes first" % VERIFY_JSON)
        vj = C.load_json(VERIFY_JSON)
        rows = []
        for c in C.ALIGN_C:
            vs = [stored_metrics(h, c) for h in C.HORIZONS]
            rows.append({"c": c, "mse": float(np.mean([v[0] for v in vs])),
                         "mae": float(np.mean([v[1] for v in vs]))})
        make_figure(rows, FIG_PNG)
        print("[collect_t35] regenerated %s" % FIG_PNG)
    elif len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        # min-stage hard gate: pinned four-H equal-weight average from the
        # EXP-001 c=0.4 arrays only (no verify json needed)
        vs = [stored_metrics(h, C.BASELINE_C) for h in C.HORIZONS]
        mse = float(np.mean([v[0] for v in vs]))
        mae = float(np.mean([v[1] for v in vs]))
        d0 = abs(mse - C.BASELINE_SELF_CHECK["avg_mse"])
        d1 = abs(mae - C.BASELINE_SELF_CHECK["avg_mae"])
        want(d0 <= C.SELF_TOL and d1 <= C.SELF_TOL,
             "baseline four-H self-check FAIL: %.17g/%.17g vs pinned "
             "%.17g/%.17g (tol %.0e) — the denominator of every Delta is "
             "wrong, stop and locate the cause"
             % (mse, mae, C.BASELINE_SELF_CHECK["avg_mse"],
                C.BASELINE_SELF_CHECK["avg_mae"], C.SELF_TOL))
        print("[collect_t35] selfcheck OK: 4H-mean %.17g/%.17g vs pinned, "
              "diff %.2e/%.2e <= %.0e" % (mse, mae, d0, d1, C.SELF_TOL))
    else:
        main()
