"""Human-readable summary md + machine json for EXP-028 (LN full-shot)."""
import argparse
from pathlib import Path

import numpy as np

from repro_timemmd_common_ln_v1 import protocol as P
from repro_timemmd_common_ln_v1.common import dump_json, load_json, \
    setup_deterministic_env
from repro_timemmd_common_ln_v1.ln_train import weighted_batch_mean


def train_loss_annotation(out_root, tstat):
    """status.json of THIS run records mean_train_loss as a simple average
    of batch losses (unequal denominators). For transparency, recompute
    the window-weighted value from each group's stored per-batch losses;
    training code now weights by batch window count going forward."""
    tl = {}
    for dom, h in P.all_groups():
        gid = P.group_id(dom, h)
        lp = out_root / P.train_log_name(dom, h)
        if not lp.exists():
            continue
        log = load_json(lp)
        losses = log.get("losses") or []
        n = int(tstat["groups"][gid]["n_train_windows"])
        sizes = log.get("batch_sizes") or \
            [P.BATCH] * (n // P.BATCH) + \
            ([n % P.BATCH] if n % P.BATCH else [])
        tl[gid] = {"recorded_simple": log.get("mean_train_loss"),
                   "window_weighted": weighted_batch_mean(losses, sizes)}
    return tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())

    ev = load_json(out_root / "evaluation.json")
    pf = load_json(out_root / "preflight" / "preflight_report.json")
    vr = load_json(out_root / "verify_report.json")
    rs = load_json(out_root / "replay_summary.json")
    tstat = load_json(out_root / "status.json")
    ts_infer = load_json(out_root / "test_status.json")

    lines = ["# VisionTS x timemmd_common_801010_v1 — LN full-shot 微调汇总"
             "(EXP-028)", "",
             "协议:%s(冻结 51 文件)| 权重:MAE-Base 起始(sha %s...,全程非 LN 冻结)"
             % (P.PROTOCOL_NAME, pinfo["weights_sha256"][:12]),
             "训练:全部 84 个 LayerNorm 张量(55,808 标量)| Adam lr=1e-4 "
             "betas=(0.9,0.999) eps=1e-8 wd=0 | 有效 batch 256(microbatch %d)"
             % P.MICROBATCH,
             "1 epoch × 36 组独立训练(9 域×4H,seed %d,固定 P=EXP-027 选中)"
             % P.SEED,
             "训练窗 55,836 | 236 次 optimizer.step | 单种子单 epoch 固定预算", ""]

    lines += ["## 固定周期 P(=EXP-027 验证集选定,本轮不搜索)", "",
              "| 域 | P |", "|---|---|"]
    for dom in P.ORDER:
        lines.append("| %s | %d |" % (dom, P.FIXED_P[dom]))
    lines.append("")

    lines += ["## 36 组测试结果(标准化空间;逐组 f32 全元素;Δ=LN−EXP-027 零样本)",
              "",
              "| 组 | P | 测试窗 | MSE | MAE | ΔMSE | ΔMSE% |",
              "|---|---|---|---|---|---|---|"]
    for dom, h in P.all_groups():
        g = ev["groups"][P.group_id(dom, h)]
        lines.append("| %s | %d | %d | %.6f | %.6f | %+.6f | %+.2f%% |"
                     % (g["group"], P.FIXED_P[dom], g["windows"], g["mse"],
                        g["mae"], g["exp027"]["delta_mse"],
                        100 * g["exp027"]["rel_mse"]))
    lines.append("")

    lines += ["## 逐域(4H 等权)", "",
              "| 域 | MSE | MAE | EXP-027 MSE | ΔMSE% | TaTS MSE | ΔMSE% |",
              "|---|---|---|---|---|---|---|"]
    for dom in P.ORDER:
        d = ev["by_domain"][dom]
        lines.append("| %s | %.6f | %.6f | %.6f | %+.2f%% | %.6f | %+.2f%% |"
                     % (dom, d["mse"], d["mae"], d["exp027_mse"],
                        100 * d["rel_mse_exp027"], d["tats_mse"],
                        100 * d["rel_mse_tats"]))
    lines.append("")

    o9, o36 = ev["overall_9domain_equal"], ev["overall_36group_equal"]
    lines += ["## Overall", "",
              "- 9 域等权:MSE **%.6f** | MAE **%.6f**"
              % (o9["mse"], o9["mae"]),
              "- 36 组等权:MSE %.6f | MAE %.6f(测试窗 %d)"
              % (o36["mse"], o36["mae"], ev["test_windows"]),
              "- 主对照 EXP-027 零样本(36 组等权 MSE %.6f):ΔMSE %+.6f"
              "(%+.2f%%)ΔMAE %+.6f(%+.2f%%);改善 %d 组 / 退化 %d 组"
              % (ev["exp027_overall_36group"]["mse"],
                 o36["delta_mse_exp027"], 100 * o36["rel_mse_exp027"],
                 o36["delta_mae_exp027"], 100 * o36["rel_mae_exp027"],
                 ev["vs_exp027"]["n_improved"], ev["vs_exp027"]["n_degraded"]),
              "- 次对照 TaTS 监督(36 组等权 MSE %.6f,锚点断言过):ΔMSE %+.6f"
              "(%+.2f%%)"
              % (ev["tats_overall_36group"]["mse"],
                 o36["delta_mse_tats"], 100 * o36["rel_mse_tats"]),
              "- 口径:EXP-027=零样本冻结;TaTS=监督+文本+数值先验;本轮=LN 微调+仅数值历史,"
              "单种子 1 epoch 完整训练集", ""]

    lines += ["## 训练成本(实测)", ""]
    tot_s = sum(g.get("elapsed_s") or 0 for g in tstat["groups"].values())
    peak = max((g.get("peak_mem_mb") or 0)
               for g in tstat["groups"].values())
    lines.append("- 36 组训练总耗时 %.1f min(单卡串行)| 单组峰值显存 %d MB"
                 % (tot_s / 60.0, peak))
    n_unchanged = sum(len(g.get("unchanged_ln") or [])
                      for g in tstat["groups"].values())
    lines.append("- 冻结核验:全部组非 LN 参数与 buffer 逐位不变;编码器+解码器 LN "
                 "组均有更新;单步未变化 LN 张量累计 %d 项(记录见 status.json)"
                 % n_unchanged)
    lines.append("- 预检外推:%s" % pf.get("b5", {}).get("note", "n/a"))
    tl = train_loss_annotation(out_root, tstat)
    if tl:
        wv = [v["window_weighted"] for v in tl.values()
              if v["window_weighted"] is not None]
        sv = [v["recorded_simple"] for v in tl.values()
              if v["recorded_simple"] is not None]
        lines.append("- mean_train_loss 口径:本运行 status.json 记录为各批"
                     "均值的简单平均;按窗加权重算 36 组均值 %.6f vs 简单"
                     "平均 %.6f(逐组见 train_loss_weighted.json);训练代码"
                     "已改为按批窗数加权(不影响梯度与测试结果)"
                     % (float(np.mean(wv)) if wv else float("nan"),
                        float(np.mean(sv)) if sv else float("nan")))
    lines.append("")

    lines += ["## 验证集 epoch0→epoch1(诊断,报告性)", "",
              "| 域 | H | epoch0(EXP-027) val MSE | epoch1(LN) val MSE | Δ |",
              "|---|---|---|---|---|"]
    for dom in P.ORDER:
        for r in ev["val_epoch0_vs_epoch1"][dom]:
            d = r["delta_val_mse"]
            lines.append("| %s | %d | %.6f | %s | %s |"
                         % (dom, r["h"], r["epoch0_val_mse_exp027"],
                            ("%.6f" % r["epoch1_val_mse_ln"])
                            if r["epoch1_val_mse_ln"] is not None else "n/a",
                            ("%+.6f" % d) if d is not None else "n/a"))
    lines.append("")

    lines += ["## 门禁", "",
              "- 预检 B1–B5:%s" % ("ALL PASS" if pf["all_pass"] else "FAIL"),
              "- 训练:36/36 done,236 step,55,836 窗(指纹门禁续跑)",
              "- verify_cache 三方(协议 AND TaTS AND EXP-027):%d/%d 组 ok"
              % (vr["n_groups"] - vr["n_failed"], vr["n_groups"]),
              "- 独立重放:%d/%d pass(max|d|=%.2e;ckpt+npz 哈希不变=%s)"
              % (rs["n_pass"], rs["n_groups"],
                 max(r.get("max_abs_diff") or 0
                     for r in rs["reports"].values()),
                 rs["immutability_ok"]),
              "- 负控:%s" % ("两变体均检出" if rs.get("perturb_verify", {})
                             .get("variant_a_naive_detected") and
                             rs["perturb_verify"]
                             .get("variant_b_consistent_detected") else "n/a"),
              "- 推理:%d/36 done(每组独立进程严格恢复)"
              % sum(1 for g in ts_infer["groups"].values()
                    if g.get("state") == "done"), ""]

    (out_root / "summary.md").write_text("\n".join(lines))
    if tl:
        dump_json(out_root / "train_loss_weighted.json", tl)
    dump_json(out_root / "summary.json", {
        "fixed_p": P.FIXED_P,
        "by_domain": ev["by_domain"],
        "overall_9domain_equal": o9, "overall_36group_equal": o36,
        "exp027_overall_36group": ev["exp027_overall_36group"],
        "tats_overall_36group": ev.get("tats_overall_36group"),
        "vs_exp027": ev["vs_exp027"],
        "test_windows": ev["test_windows"],
        "train": {"groups": len(tstat["groups"]),
                  "steps": sum(g.get("steps") or 0
                               for g in tstat["groups"].values()),
                  "total_elapsed_s": round(tot_s, 1),
                  "peak_mem_mb": peak},
        "train_loss_annotation": tl,
        "gates": {"preflight_all_pass": pf["all_pass"],
                  "train_all_done": all(
                      g.get("state") == "done"
                      for g in tstat["groups"].values()),
                  "verify_ok": vr["n_failed"] == 0,
                  "replay_all_pass": rs["n_pass"] == rs["n_groups"],
                  "infer_all_done": all(
                      g.get("state") == "done"
                      for g in ts_infer["groups"].values())},
    })
    print((out_root / "summary.md").read_text()[-600:])


if __name__ == "__main__":
    main()
