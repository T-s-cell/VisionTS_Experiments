"""Human-readable summary md + machine json from evaluation.json,
selected_p.json, preflight/replay/verify reports, and cost accounting."""
import argparse
from pathlib import Path

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import dump_json, load_json, \
    setup_deterministic_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())

    ev = load_json(out_root / "evaluation.json")
    sel = load_json(out_root / "selected_p.json")
    pf = load_json(out_root / "preflight_report.json")
    rp = load_json(out_root / "replay_report.json")
    vr = load_json(out_root / "verify_report.json")

    lines = ["# VisionTS x timemmd_common_801010_v1 — zero-shot 汇总", "",
             "协议:%s(冻结 51 文件,EXP-026)| 权重:MAE-Base 冻结(sha %s...)"
             % (P.PROTOCOL_NAME, pinfo["weights_sha256"][:12]),
             "选 P:仅由各域验证集确定(argmin 4H 等权 val MSE,并列取小 P)| seed %d | batch %d"
             % (P.SEED, P.BATCH), ""]

    lines += ["## 选中周期 P", "", "| 域 | P | 各 P 的 4H 等权 val MSE |",
              "|---|---|---|"]
    for dom in P.ORDER:
        s = sel["selection"][dom]
        tab = ", ".join("P%s=%.4f" % (k, v)
                        for k, v in sorted(s["mean_val_mse_by_P"].items(),
                                           key=lambda kv: int(kv[0])))
        lines.append("| %s | **%d** | %s |" % (dom, s["selected"], tab))
    lines.append("")

    lines += ["## 36 组测试结果(标准化空间;逐组 f32 全元素)", "",
              "| 组 | P | 测试窗 | MSE | MAE | MSE(f64) |", "|---|---|---|---|---|---|"]
    for dom, h in P.all_groups():
        g = ev["groups"][P.group_id(dom, h)]
        lines.append("| %s | %d | %d | %.6f | %.6f | %.6f |"
                     % (g["group"], sel["selection"][dom]["selected"],
                        g["windows"], g["mse"], g["mae"], g["mse_f64"]))
    lines.append("")

    lines += ["## 逐域(4H 等权)", "", "| 域 | MSE | MAE |" +
              (" | TaTS MSE | ΔMSE | ΔMSE% |" if ev["tats_comparison_used"]
               else ""), "|---|---|---|" +
              ("---|---|---|" if ev["tats_comparison_used"] else "")]
    for dom in P.ORDER:
        d = ev["by_domain"][dom]
        row = "| %s | %.6f | %.6f " % (dom, d["mse"], d["mae"])
        if ev["tats_comparison_used"]:
            row += "| %.6f | %+.6f | %+.2f%% " % (
                d["tats_mse"], d["delta_mse"], 100 * d["rel_mse"])
        lines.append(row + "|")
    lines.append("")

    o9, o36 = ev["overall_9domain_equal"], ev["overall_36group_equal"]
    lines += ["## Overall", "",
              "- 9 域等权:MSE **%.6f** | MAE **%.6f**"
              % (o9["mse"], o9["mae"]),
              "- 36 组等权:MSE %.6f | MAE %.6f(测试窗 %d)"
              % (o36["mse"], o36["mae"], ev["test_windows"])]
    if ev["tats_comparison_used"]:
        t = ev["tats_overall_36group"]
        lines.append("- 对照 TaTS(36 组等权,同协议同空间):TaTS MSE %.6f / "
                     "MAE %.6f → VisionTS ΔMSE %+.6f(%+.2f%%)ΔMAE %+.6f"
                     "(%+.2f%%)"
                     % (t["mse"], t["mae"], o36["delta_mse"],
                        100 * o36["rel_mse"], o36["delta_mae"],
                        100 * o36["rel_mae"]))
        lines.append("- 口径:TaTS=监督训练+文本+数值先验;VisionTS=零样本冻结权重,仅数值历史")
    lines.append("")

    a5 = pf["gates"]["A5"]
    a2 = pf["gates"]["A2"]
    lines += ["## 成本(实测)", "",
              "- 预检 A2(4 组合):%.1fs;单窗 %.4f s" % (a2["seconds"],
                                                        a5["sec_per_window_measured"]),
              "- P 搜索(136 组合,28,868 预测):见 psearch/status.json 合计",
              "- 36 组测试外推 %.0f s;重放外推 %.0f s(实测见 replay_report.json)"
              % (a5["test_seconds_extrapolated"],
                 a5["replay_seconds_extrapolated"]),
              "- 峰显存 %.3f GiB | batch 探针 max|d|=%.2e(atol 1e-6/rtol 1e-5)"
              % (a5.get("peak_mem_gib", 0), a5["batch_probe_max_abs_diff"]),
              ""]
    lines += ["## 门禁", "",
              "- 预检 A1–A5:%s" % ("ALL PASS" if pf["all_pass"] else "FAIL"),
              "- verify_cache:%d/%d 组 ok" % (vr["n_groups"] - vr["n_failed"],
                                              vr["n_groups"]),
              "- 独立重放:%s(max|d|=%.2e;缓存 npz 哈希不变=%s;负控=%s)"
              % ("PASS" if rp["all_pass"] else "FAIL",
                 max(v["max_abs_diff"] for v in rp["groups"].values()),
                 rp["cached_npz_immutable"],
                 "检测成功" if rp.get("negative_control", {})
                 .get("max_abs_diff") else "n/a"), ""]

    (out_root / "summary.md").write_text("\n".join(lines))
    dump_json(out_root / "summary.json", {
        "selected_p": {d: sel["selection"][d]["selected"]
                       for d in P.ORDER},
        "by_domain": ev["by_domain"],
        "overall_9domain_equal": o9, "overall_36group_equal": o36,
        "tats_overall_36group": ev.get("tats_overall_36group"),
        "test_windows": ev["test_windows"],
        "gates": {"preflight_all_pass": pf["all_pass"],
                  "verify_ok": vr["n_failed"] == 0,
                  "replay_all_pass": rp["all_pass"]},
    })
    print((out_root / "summary.md").read_text()[-600:])


if __name__ == "__main__":
    main()
