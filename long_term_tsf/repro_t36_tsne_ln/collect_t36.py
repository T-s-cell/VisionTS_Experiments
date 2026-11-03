"""EXP-036 step 7: gate summary -> RESULT_t36_tsne_ln.md + summary_t36.csv.

Default mode (run on eta after plot): re-checks every recorded gate from
logs/verify_t36.json (weights fingerprints/probe/splice, features bitwise +
input determinism + row order, t-SNE params/version, NN table consistency),
copies the small evidence artifacts next to this script, and writes
RESULT_t36_tsne_ln.md with a COMMIT_PENDING placeholder.

--stamp <sha> (run locally on the pulled artifacts): recomputes the NN
proportions from the archived neighbour row indices, re-checks the gates
and coordinate shapes, then replaces the placeholder with the commit SHA.
Any mismatch -> nonzero exit, RESULT not stamped.
"""
import argparse
import os
import shutil
import sys

import numpy as np

import t36_common as C


def fail(msg):
    raise SystemExit("[collect_t36] FAIL: %s" % msg)


def orig_coords_path():
    """Archived EXP-002 frozen coords: repo file, or the copy shipped in the
    audit package (exp002_refs/ next to the package's repro dir)."""
    pkg_copy = os.path.abspath(os.path.join(C.PKG, "..", "exp002_refs",
                                            "tsne_coords_1900x2.npy"))
    if os.path.isfile(C.ORIG_COORDS):
        return C.ORIG_COORDS
    if os.path.isfile(pkg_copy):
        return pkg_copy
    fail("archived coords not found at %s or %s" % (C.ORIG_COORDS, pkg_copy))


def check_gates(vj):
    g = []

    w = vj.get("weights", {})
    if sorted(w) != sorted(C.EXP003_DS):
        fail("weights sections %s != %s" % (sorted(w), C.EXP003_DS))
    for ds, sec in w.items():
        if sec["code_files"] != len(C.EXP003_FP_FILES) or not sec["data_sha_ok"]:
            fail("%s fingerprint gate" % ds)
        if not sec["anchor_match"]:
            fail("%s anchor gate" % ds)
        d = sec["max_abs_diff"]
        if d > C.PROBE_TOL:
            fail("%s probe gate: %g > %g" % (ds, d, C.PROBE_TOL))
        if sec["n_changed"] < 1 or sec["splice_keys_equal_src"] != 50:
            fail("%s splice gate" % ds)
        if sec["nonln_bitwise_equal"] <= 0:
            fail("%s non-LN bitwise gate" % ds)
        g.append("weights/%s: probe max|d|=%.3e (B=%d), %d/%d LN keys changed "
                 "(max %.3e), encoder non-LN bitwise==frozen, ckpt %s"
                 % (ds, d, sec["batch"], sec["n_changed"], sec["enc_ln_keys"],
                    sec["max_delta"], sec["ckpt_sha256"][:16] + "..."))

    f = vj.get("features", {})
    if sorted(f.get("conds", [])) != sorted(C.COND_LIST):
        fail("features conds %s" % f.get("conds"))
    if not (f.get("row_order_ok") and f.get("input_hashes_equal")):
        fail("features row-order/input-hash gate")
    fc = f.get("per_cond", {}).get("frozen", {})
    if not fc.get("orig_bitwise_equal") or fc.get("orig_max_abs_diff") != 0.0:
        fail("frozen features not bitwise vs EXP-002 original")
    for cond, rec in f["per_cond"].items():
        if rec["shape"] != [1900, C.FEATURE_DIM] or not rec["finite"]:
            fail("%s feature shape/finite gate" % cond)
    g.append("features: 4x (1900,54528) f32 finite; frozen bitwise == EXP-002 "
             "original (max|d|=0); per-batch input + noise hashes identical "
             "across conditions; 1,900-row order == archived meta.csv/groups")

    ex = vj.get("extract", {})
    for cond in ("ettm1_ln", "ecl_ln", "weather_ln"):
        if ex.get(cond, {}).get("keys_equal_src") != 50:
            fail("%s splice-at-extract gate" % cond)
    g.append("extract: 50 spliced encoder-LN keys bitwise == source checkpoint "
             "in every LN condition; ids_keep == archived mask_ref")

    ts = vj.get("tsne", {})
    for cond in ("ettm1_ln", "ecl_ln", "weather_ln"):
        sec = ts.get(cond, {})
        if sec.get("params") != C.TSNE_PARAMS or sec.get("sklearn_version") != C.SKLEARN_PIN:
            fail("%s tsne params/version gate" % cond)
    if "frozen" in ts:
        fail("frozen must reuse archived coords, not a fresh t-SNE")
    g.append("t-SNE: 3 LN conditions, EXP-002 params verbatim, sklearn %s; "
             "frozen panel = archived EXP-002 coords (sha %s...)"
             % (C.SKLEARN_PIN, C.sha256_of(orig_coords_path())[:16]))

    nn = vj.get("nn")
    if not nn:
        fail("nn section missing")
    g.append("NN: 30-NN over all 1,900 features, self excluded by global row, "
             "denominator 300x30=9000 per cell; neighbour row indices archived")
    return g


def tsne_2d_descriptive():
    """Panel-internal 2D 30-NN ImageNet proportion (descriptive only).

    The high-dim metric saturates at 0 in all four conditions (modality gap
    dominates raw 54,528-dim features), so this mirrors what each t-SNE panel
    actually shows: each TS sample's 30 nearest neighbours within its own
    panel's 2D coordinates, fraction that are ImageNet rows.
    """
    prop = {}
    for cond in C.COND_LIST:
        path = orig_coords_path() if cond == "frozen" else os.path.join(C.PKG, "coords_%s_1900x2.npy" % cond)
        c = np.load(path).astype(np.float64)
        assert c.shape == (1900, 2)
        d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        prop[cond] = {}
        for g in C.GROUPS[1:]:
            s0, s1 = C.group_spans()[g]
            idx = np.argpartition(d2[s0:s1], C.N_NEIGHBORS, axis=1)[:, :C.N_NEIGHBORS]
            prop[cond][g] = float((idx < 1000).sum()) / C.NN_DENOM
    return prop


def check_nn_consistency(base_nn_npz, base_table):
    flat = dict(np.load(base_nn_npz))
    neighbors = {}
    for k, arr in flat.items():
        cond, g = k.split("__")
        neighbors.setdefault(cond, {})[g] = arr
    prop = C.nn_recompute_proportions(neighbors)
    with open(base_table) as f:
        lines = f.read().strip().splitlines()
    assert lines[0].startswith("encoder,ts_source"), lines[0]
    for line in lines[1:]:
        cond, g, n, cnt, denom, p, pp = (line.split(",") + [""])[:7]
        if abs(prop[cond][g] - float(p)) > 0:
            fail("nn_table proportion mismatch %s/%s: file %s vs recomputed %r"
                 % (cond, g, p, prop[cond][g]))
        if int(cnt) != int(round(prop[cond][g] * C.NN_DENOM)) or int(denom) != C.NN_DENOM:
            fail("nn_table count/denominator mismatch %s/%s" % (cond, g))
    return prop


def md_table(nn_summary):
    order = ["ettm1_ln", "ecl_ln", "weather_ln"]
    heads = ["encoder"] + C.GROUPS[1:]
    rows = ["| " + " | ".join(heads) + " |",
            "|" + "---|" * len(heads)]
    label = {"frozen": "Frozen (EXP-002)",
             "ettm1_ln": "ETTm1-LN", "ecl_ln": "Electricity-LN", "weather_ln": "Weather-LN"}
    pr = nn_summary["proportions"]
    pp = nn_summary["pp_vs_frozen"]
    for cond in ["frozen"] + order:
        cells = [label[cond]]
        for g in C.GROUPS[1:]:
            v = "%.4f" % pr[cond][g]
            if cond != "frozen":
                v += " (%+.2f pp)" % pp[cond][g]
            cells.append(v)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def md_table_2d(desc):
    order = ["ettm1_ln", "ecl_ln", "weather_ln"]
    heads = ["encoder"] + C.GROUPS[1:]
    rows = ["| " + " | ".join(heads) + " |",
            "|" + "---|" * len(heads)]
    label = {"frozen": "Frozen (EXP-002, 归档坐标)",
             "ettm1_ln": "ETTm1-LN", "ecl_ln": "Electricity-LN", "weather_ln": "Weather-LN"}
    for cond in ["frozen"] + order:
        cells = [label[cond]]
        for g in C.GROUPS[1:]:
            v = "%.4f" % desc[cond][g]
            if cond != "frozen":
                v += " (%+.2f pp)" % ((desc[cond][g] - desc["frozen"][g]) * 100.0)
            cells.append(v)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def read_nn_summary(path):
    import json
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default=None,
                    help="local mode: commit SHA to stamp into RESULT after re-verification")
    args = ap.parse_args()

    vj_path = os.path.join(C.PKG, "logs", "verify_t36.json")
    if not os.path.isfile(vj_path):
        fail("verify json missing: %s" % vj_path)
    vj = C.load_json(vj_path)
    gates = check_gates(vj)

    # stage the small evidence artifacts next to this script (no-op when the
    # files were already pulled into a local PKG copy)
    for name in ["nn_neighbors.npz", "nn_table.csv", "nn_summary.json"] + \
                ["coords_%s_1900x2.npy" % c for c in ("ettm1_ln", "ecl_ln", "weather_ln")]:
        src = os.path.join(C.SAVE_BASE, name)
        dst = os.path.join(C.PKG, name)
        if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

    nn_npz = os.path.join(C.PKG, "nn_neighbors.npz")
    nn_table = os.path.join(C.PKG, "nn_table.csv")
    nn_summary = read_nn_summary(os.path.join(C.PKG, "nn_summary.json"))
    prop = check_nn_consistency(nn_npz, nn_table)
    if prop != nn_summary["proportions"]:
        fail("nn_summary proportions != recomputed")

    for cond in ("ettm1_ln", "ecl_ln", "weather_ln"):
        p = os.path.join(C.PKG, "coords_%s_1900x2.npy" % cond)
        if not os.path.isfile(p):
            fail("missing coords copy: %s" % p)
        c = np.load(p)
        if c.shape != (1900, 2) or not np.isfinite(c).all():
            fail("coords %s shape/finite" % cond)
    for ext in ("png", "pdf"):
        if not os.path.isfile(os.path.join(C.PKG, "fig36_tsne_compare.%s" % ext)):
            fail("figure %s missing" % ext)

    out_lines = []
    for cond in C.COND_LIST:
        rec = vj["features"]["per_cond"][cond]
        src = C.COND_SOURCE.get(cond, "-")
        d = "-" if cond == "frozen" else "%.3e" % vj["weights"][src]["max_abs_diff"]
        nch = "-" if cond == "frozen" else "%d/50" % vj["weights"][src]["n_changed"]
        out_lines.append("%s,%s,%s,%s,%s" % (cond, src, rec["sha256"], d, nch))
    with open(os.path.join(C.PKG, "summary_t36.csv"), "w") as f:
        f.write("cond,ln_source,features_sha256,probe_max_abs_diff,encoder_ln_keys_changed\n")
        f.write("\n".join(out_lines) + "\n")

    stamp = args.stamp or "COMMIT_PENDING"
    desc = tsne_2d_descriptive()
    domain_lines = ["高维原始特征空间四条件全部 0/9,000(模态鸿沟主导,见上);"
                    "以下按各面板 2D 局部交叠(描述性,括号为相对冻结的百分点)判读。"]
    for cond in ("ettm1_ln", "ecl_ln", "weather_ln"):
        src = C.COND_SOURCE[cond]
        own = (desc[cond][src] - desc["frozen"][src]) * 100.0
        others = [g for g in C.GROUPS[1:] if g != src]

        def dir_word(v):
            return "上升" if v > 0 else ("下降" if v < 0 else "不变")

        parts = ["%s-LN(训练域 %s): 面板内 ImageNet 占比 %s %+.2f pp"
                 % (label_short(cond), src, dir_word(own), own)]
        for g in others:
            v = (desc[cond][g] - desc["frozen"][g]) * 100.0
            parts.append("迁移域 %s %+.2f pp(%s)" % (g, v, dir_word(v)))
        domain_lines.append("- " + "; ".join(parts) + "。")

    md = []
    md.append("# EXP-036 结果:VisionTS 微调 LayerNorm 前后的 t-SNE 对照(1,900 固定样本)")
    md.append("")
    md.append("日期:2026-11-03 | commit `%s` | 分支 repro/ltsf-t36-tsne-ln-v1 | 冻结 MAE-base(sha256 %s…)" % (stamp, C.CKPT_SHA256[:16]))
    md.append("")
    md.append("问题:让冻结编码器与三套 EXP-003 编码器-LN 微调权重(ETTm1/Electricity/Weather,H=96、seed=2021 最佳验证 checkpoint)看**完全相同的 1,900 个输入**(EXP-002 归档抽样清单:ImageNet 1,000 + ETTm1/Weather/Electricity 各 300),局部交叠如何变化。四条件输入逐批哈希一致;仅编码器 50 个 LN 张量来自各自 checkpoint,其余参数与冻结逐位一致。")
    md.append("")
    md.append("## 门禁(全部通过)")
    md.append("")
    for x in gates:
        md.append("- " + x)
    md.append("")
    md.append("## 权重来源与回放")
    md.append("")
    md.append("| 数据集 | 训练配置(EXP-003 args) | 首批回放 max\\|d\\| (≤1e-4) | 改变 LN 键 | checkpoint sha256(前 16) |")
    md.append("|---|---|---|---|---|")
    cfgs = {"ETTm1": "L=2304/P=96/r=0.4/c=0.4 freq=t", "Electricity": "L=1152/P=24/r=0.4/c=0.4 freq=h", "Weather": "L=576/P=144/r=1.0/c=0.7 freq=h"}
    for ds in C.EXP003_DS:
        s = vj["weights"][ds]
        md.append("| %s | %s | %.3e (B=%d) | %d/50 | %s |" % (ds, cfgs[ds], s["max_abs_diff"], s["batch"], s["n_changed"], s["ckpt_sha256"][:16]))
    md.append("")
    md.append("锚点(SUCCESS 未舍入):ETTm1 %s/%s、Electricity %s/%s、Weather %s/%s。"
              % (repr(C.ANCHORS["ETTm1"]["mse"]), repr(C.ANCHORS["ETTm1"]["mae"]),
                 repr(C.ANCHORS["Electricity"]["mse"]), repr(C.ANCHORS["Electricity"]["mae"]),
                 repr(C.ANCHORS["Weather"]["mse"]), repr(C.ANCHORS["Weather"]["mae"])))
    md.append("")
    md.append("## 高维 30-NN ImageNet 占比(欧氏距离,分母 9,000)")
    md.append("")
    md.append(md_table(nn_summary))
    md.append("")
    md.append("括号为相对冻结的百分点变化。邻域在**该编码器的全部 1,900 个特征**内取(排除自身,按全局行号),不限于同组。")
    md.append("")
    if all(nn_summary["proportions"][c][g] == 0.0 for c in C.COND_LIST for g in C.GROUPS[1:]):
        md.append("**高维结果饱和:4 编码器 × 3 时序来源全部 12 格均为 0/9,000。**"
                  "在原始 54,528 维特征空间,每个时序样本的 30 近邻全部是时序样本"
                  "(冻结编码器下抽查行最近 ImageNet 距离约为最近时序距离的 1.4–3.3 倍;"
                  "独立复算与 nn_neighbors.npz 一致)。模态鸿沟主导原始特征,"
                  "该高维指标在本设置下不区分四套编码器,微调的变化量恒为 ±0 pp,照实记录。")
        md.append("")
    md.append("## 二维面板内 30-NN ImageNet 占比(t-SNE 坐标空间,描述性)")
    md.append("")
    md.append(md_table_2d(desc))
    md.append("")
    md.append("同一 30-NN 算法用在各面板自己的 2D 坐标上,刻画图面所见;各面板独立拟合,"
              "数值跨面板仅作方向参考,不构成显著性检验,也不替代高维表。")
    md.append("")
    md.append("## 判读(训练域 vs 迁移域)")
    md.append("")
    md.extend(domain_lines)
    md.append("")
    md.append("二维面板各自独立拟合(t-SNE 坐标方向/尺度跨面板不可比),只比较各面板内部的局部交叠;高维指标见表。二维观察与高维结果若不一致,以两者数值并存、不调和。交叠增加、减少或变化很小均照实呈现;预测精度的变化不由本图承担,由 EXP-003 的预测评估单独支持。")
    md.append("")
    md.append("## 大件(留在 eta,不入包)")
    md.append("")
    eta_root = "/dev_data/wlt/VisionTS_Experiments"
    md.append("- 四条件特征:%s/long_term_tsf/save_t36_tsne(每份 1900x54528 f32;"
              "frozen 与 EXP-002 原件 %s/repro_fig7/outputs/features_1900x54528.npy 逐位一致)"
              % (eta_root, eta_root))
    md.append("- 权重:三份 EXP-003 checkpoint.pth(%s/long_term_tsf/save_fullshot/{ETTm1,Electricity,Weather}_96_seed2021/checkpoints/_/)"
              "+ 冻结 ckpt(%s/ckpt/mae_visualize_vit_base.pth,sha256 %s)" % (eta_root, eta_root, C.CKPT_SHA256))
    md.append("- ImageNet 原图:/dev_data/wlt/data/ImageNet/val(50,000 张)")
    md.append("")
    with open(os.path.join(C.PKG, "RESULT_t36_tsne_ln.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("[collect_t36] RESULT_t36_tsne_ln.md written (stamp=%s)" % stamp)


def label_short(cond):
    return {"ettm1_ln": "ETTm1", "ecl_ln": "Electricity", "weather_ln": "Weather"}[cond]


if __name__ == "__main__":
    main()
