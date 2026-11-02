"""EXP-032 step 1: selection verification for Appendix D case windows.
Runs the fixed selection rule on the three EXP-031 win npz (ETTh1/OT), checks
every item against the user-frozen table (counts / indices / MAEs), and writes
logs/selection_verification.json + save_d_cases/ETTh1_96/selection_list.md.
Any mismatch -> SystemExit (no fallback quantile; investigate indexing/sort).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d_common as D


def main():
    sel = D.select_windows()          # SystemExit on any frozen mismatch
    for case in ("success", "failure"):
        rec = sel[case]
        win = rec["window_index"]
        f64 = D.mae_f64_from_arrays(win)
        ts = D.timestamps(win)
        rec["mae_f64_from_arrays"] = {m: f64[m] for m in
                                      ("visionts", "naive", "avg")}
        rec["timestamps"] = ts
        rec["batch"] = D.batch_of(win)
        rec["code_md5"] = D.code_md5()

    os.makedirs(os.path.join(D.PKG, "logs"), exist_ok=True)
    out = {
        "case_definition": "ETTh1 / OT / L=2880 / H=96 / P=24",
        "selection_rule": ("success: vts<naive AND vts<avg; failure: "
                           "vts>naive AND vts>avg (unrounded OT MAE); "
                           "within group sort by |vts - nearest baseline| "
                           "then window_index; take lower median (n-1)//2; "
                           "no fallback"),
        "criterion_note": ("win/loss judged against the two seasonal "
                           "baselines (EXP-031); paper Fig.14 judged "
                           "against MOIRAI-Large instead"),
        "results": {c: sel[c] for c in ("success", "failure")},
    }
    p_json = os.path.join(D.PKG, "logs", "selection_verification.json")
    D.dump_json(p_json, out)

    lines = [
        "# EXP-032 选窗清单(定性复现,新选窗口)", "",
        "案例定义: ETTh1 / OT / L=2880 / H=96 / P=24", "",
        "判据说明: 本轮胜负判据 = VisionTS 与两条周期基线 "
        "(Seasonal Naïve / Seasonal Avg, EXP-031) 的 OT MAE 比较;"
        "论文图 14 的失败判据则是 VisionTS MAE 高于 MOIRAI-Large,"
        "两者不同。", "",
        "选窗规则: 成功组 = VisionTS 同时低于两条基线;失败组 = 同时高于"
        "两条基线;组内按 |VisionTS − 较接近基线| 升序,并列按 window_index "
        "升序,取下中位 (n−1)//2。无任何回退分位数。", "",
    ]
    for case in ("success", "failure"):
        rec = sel[case]
        f64, ts, b = (rec["mae_f64_from_arrays"], rec["timestamps"],
                      rec["batch"])
        lines += [
            "## %s" % case, "",
            "| 项目 | 值 |", "|---|---|",
            "| window_index | %d |" % rec["window_index"],
            "| 候选窗数 n | %d |" % rec["n_cand"],
            "| 组内名次 rank_pos | %d ((n−1)//2) |" % rec["rank_pos"],
            "| |vts − 较近基线| | %.6f |" % rec["diff_to_nearest"],
            "| 起始行 (csv) | %d |" % ts["start_row"],
            "| 起始时间 | %s |" % ts["start_time"],
            "| 结束行 | %d |" % ts["end_row"],
            "| 结束时间 | %s |" % ts["end_time"],
            "| 历史展示起点行 | %d (%s) |" % (ts["hist_start_row"],
                                             ts["hist_start_time"]),
            "| batch / 批内位 | %d / %d (行 %d-%d) |"
            % (b["batch"], b["pos_in_batch"], b["batch_rows"][0],
               b["batch_rows"][1]), "",
            "| 方法 | npz f32 MAE | 冻结数组 f64 重算 | 用户冻结表 |",
            "|---|---|---|---|",
        ]
        exp = D.FROZEN[case]["mae"]
        for m in ("visionts", "naive", "avg"):
            lines.append("| %s | %.6f | %.6f | %.4f |"
                         % (m, rec["mae_npz"][m], f64[m], exp[m]))
        lines.append("")
    p_md = os.path.join(D.SAVE_D, "ETTh1_96", "selection_list.md")
    os.makedirs(os.path.dirname(p_md), exist_ok=True)
    with open(p_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[select_windows_d] OK -> %s" % p_json)
    print("[select_windows_d] OK -> %s" % p_md)
    for case in ("success", "failure"):
        r = sel[case]
        print("  %s: n=%d win=%d rank=%d vts=%.6f naive=%.6f avg=%.6f"
              % (case, r["n_cand"], r["window_index"], r["rank_pos"],
                 r["mae_npz"]["visionts"], r["mae_npz"]["naive"],
                 r["mae_npz"]["avg"]))


if __name__ == "__main__":
    main()
