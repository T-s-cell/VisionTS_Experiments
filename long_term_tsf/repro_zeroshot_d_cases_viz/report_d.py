"""EXP-032 step 3: assemble RESULT_d_cases.md from the selection and replay
evidence jsons. Observation text per case lives in the OBS dict below and is
filled after viewing the PNGs (rerun this script only).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d_common as D

OBS = {
    "success": "[待查图后填写: VisionTS 延续了哪种历史图案; 两条周期基线为何差]",
    "failure": "[待查图后填写: VisionTS 从哪里开始偏离真值; 基线为何更接近]",
}


def fmt_mae(d):
    return "%.6f" % d if isinstance(d, float) else str(d)


def main():
    selj = D.load_json(os.path.join(D.PKG, "logs",
                                    "selection_verification.json"))
    repj = D.load_json(os.path.join(D.PKG, "logs", "replay_evidence.json"))

    L = [
        "# EXP-032:VisionTS 附录 D 成功与失败案例可视化(定性复现)",
        "",
        "案例定义: ETTh1 / OT / L=2880 / H=96 / P=24。复用 EXP-001 原周期"
        "预测与冻结权重、EXP-031 两条周期基线数组与逐窗 MAE npz。",
        "",
        "**口径**:本报告为**新选窗口的定性复现**(窗口由固定规则在本轮重新"
        "选取,非论文原图窗口)。本轮胜负判据 = VisionTS 与两条周期基线 "
        "(Seasonal Naïve / Seasonal Avg) 的 OT MAE 比较;**论文图 14 的失败"
        "判据则是 VisionTS MAE 高于 MOIRAI-Large**,两者不同。",
        "",
        "## 1. 选窗复核",
        "",
        "规则:成功组 = VisionTS 同时低于两条基线;失败组 = 同时高于两条"
        "基线;组内按 |VisionTS − 较接近基线| 升序,并列按 window_index 升序,"
        "取下中位 (n−1)//2,无回退分位数。与用户冻结表任一不符即停。",
        "",
        "| 案例 | 候选 n | 选中 win | 起始行 | 起始时间 | 末行 | 末时间 | "
        "vts MAE | naive | avg | 冻结表 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in ("success", "failure"):
        r = selj["results"][c]
        t = r["timestamps"]
        fz = D.FROZEN[c]
        L.append(
            "| %s | %d | %d | %d | %s | %d | %s | %.6f | %.6f | %.6f | "
            "n=%d win=%d |"
            % (c, r["n_cand"], r["window_index"], t["start_row"],
               t["start_time"], t["end_row"], t["end_time"],
               r["mae_npz"]["visionts"], r["mae_npz"]["naive"],
               r["mae_npz"]["avg"], fz["n_cand"], fz["win"]))

    L += ["", "## 2. 回放与还原核验", "",
          "按用户修正 ②,预测从**原始解码图**(MAE decoder tokens 的 "
          "unpatchify)以与 `visionts/model.py` L135-149 完全一致的算子序列"
          "还原(grey 均值 → output_resize → rearrange → 切 [2880:2976] → "
          "×stdev + means),**不使用**展示版补全图(可见区换回输入;首列双线"
          "性插值会跨过可见区边界 x=80 取到输入像素)。同一次前向中捕获:真实"
          "MAE 输入(掩区=0)、固定 patch mask、原始解码图。",
          "",
          "验收(c):还原 96 步 vs 同次 forward 输出 max|d| ≤ 1e-6;"
          "(d)回放输出 vs EXP-001 冻结预测 max|d| ≤ 1e-4。",
          "",
          "| 案例 | win | batch/位 | ①还原vs同次输出 max|d| | ②回放vs冻结 "
          "max|d| (7变量) | ②OT | 回放 OT MAE | 冻结 vts MAE | 门禁 |",
          "|---|---|---|---|---|---|---|---|---|",
          ]
    for c in ("success", "failure"):
        w = repj["windows"][c]
        L.append("| %s | %d | %d/%d | %.3e | %.3e | %.3e | %.6f | %.4f | %s |"
                 % (c, w["window_index"], w["batch"], w["pos_in_batch"],
                    w["ver_a_recon_vs_same_forward_maxdiff"],
                    w["ver_b_replay_vs_frozen_maxdiff_7var"],
                    w["ver_b_replay_vs_frozen_maxdiff_ot"],
                    w["mae_ot_from_replay_pred"],
                    D.FROZEN[c]["mae"]["visionts"],
                    "PASS" if w["gates"]["pass"] else "FAIL"))

    L += ["", "### 图像几何", "",
          "- 可见区 = patch 0-4 → 224px 图 x∈[0,80)(红虚线);掩区 patch "
          "5-13(126/196)在真实 MAE 输入中填 0,导出图中的 −2 仅为显示标记。",
          "- 未来 96 步 = 24×336 缩放图的列 [120,124),对应 224px 图 "
          "x∈[80, 82.67)(红带)。",
          "", "## 3. 每例观察", ""]
    for c in ("success", "failure"):
        w = repj["windows"][c]
        L.append("### %s(win=%d,%s → %s)" % (c, w["window_index"],
                                             w["timestamps"]["start_time"],
                                             w["timestamps"]["end_time"]))
        L.append("")
        L.append("图:`%s`;证据:`%s`、`%s`"
                 % (w["figure_png"], os.path.relpath(
                     os.path.join(D.ROOT, w["evidence_npz"]), D.PKG),
                    "logs/replay_evidence.json"))
        L.append("")
        L.append(OBS[c])
        L.append("")

    env = repj.get("env", {})
    L += ["## 4. 环境", "",
          "- device: %s(%s), torch %s, seed %d"
          % (env.get("device"), env.get("cuda_device"), env.get("torch"),
             env.get("seed")),
          "- ckpt sha256: `%s`" % env.get("ckpt_sha256"),
          "- 运行时指纹文件 md5 见 `logs/replay_evidence.json` 的 code_md5;",
          "- 每窗证据 NPZ(图像张量/反归一化参数/OT 真值/三方法预测)与 "
          "PNG 位于 `save_d_cases/ETTh1_96/`。",
          ""]
    out = os.path.join(D.PKG, "RESULT_d_cases.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("[report_d] wrote %s" % out)


if __name__ == "__main__":
    main()
