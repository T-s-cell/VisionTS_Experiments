#!/usr/bin/env python3
"""Stage 04: high-dim check (the primary quantitative evidence).

For every one of the 1677 test windows: mean cosine distance to the 10 nearest
of the 2000 masked-ImageNet reference features, computed BEFORE any
dimensionality reduction on L2-normalized 768-d features. Aggregate per
variable (mean) -> 190 points; Spearman (primary) and Pearson (supplementary)
against std_mse_P1 / std_mae_P1. Frequency split is supplementary only.
Caveats recorded: overlapping test windows are not independent; correlation
p-values ignore within-domain dependence (exploratory only).
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

from common import DIRS, ensure_dirs, l2n

ZERO = os.path.dirname(HERE)


def main():
    ensure_dirs()
    d = np.load(os.path.join(DIRS["features"], "timesx_all.npz"), allow_pickle=False)
    r = np.load(os.path.join(DIRS["features"], "imagenet_masked.npz"), allow_pickle=False)
    X = l2n(d["feat"])
    R = l2n(r["feat"])
    sims = X @ R.T                      # cosine similarity on unit spheres
    dist = 1.0 - sims
    assert dist.min() >= -1e-9 and dist.max() <= 2 + 1e-9
    k = 10
    nn = np.sort(dist, axis=1)[:, :k]
    mean_d = nn.mean(1)
    assert mean_d.min() >= 0 and mean_d.max() <= 2

    vks = [str(v) for v in d["var_key"]]
    sids = [str(s) for s in d["sample_id"]]

    metrics = {}
    for row in csv.DictReader(open(os.path.join(ZERO, "results", "test_variable_level.csv"))):
        if row["P"] == "1":
            metrics[row["var_key"]] = (float(row["std_mse"]), float(row["std_mae"]),
                                       row["domain"], row["group"])

    per_var = {}
    with open(os.path.join(DIRS["highdim"], "window_distances.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["var_key", "sample_id", "mean_cos_d_10nn", "std_mse_P1", "std_mae_P1"])
        for i, (vk, sid) in enumerate(zip(vks, sids)):
            mse, mae, _, _ = metrics[vk]
            w.writerow([vk, sid, f"{mean_d[i]:.8f}", mse, mae])
            per_var.setdefault(vk, []).append(mean_d[i])
    assert len(per_var) == 190

    from scipy.stats import pearsonr, spearmanr

    def pack(r):
        return {"stat": float(r[0]), "p": float(r[1])}

    xs = sorted(per_var)
    agg = np.array([np.mean(per_var[v]) for v in xs])
    mse = np.array([metrics[v][0] for v in xs])
    mae = np.array([metrics[v][1] for v in xs])
    res = {"n_vars": len(xs), "k": k, "space": "L2-normalized 768-d (pre-DR)",
           "spearman_primary": {
               "dist_vs_mse": pack(spearmanr(agg, mse)),
               "dist_vs_mae": pack(spearmanr(agg, mae))},
           "pearson_supplementary": {
               "dist_vs_mse": pack(pearsonr(agg, mse)),
               "dist_vs_mae": pack(pearsonr(agg, mae))},
           "caveats": [
               "overlapping test windows are not independent samples",
               "p-values ignore within-domain variable dependence; exploratory only, "
               "not evidence that feature proximity causes forecast error"]}

    with open(os.path.join(DIRS["highdim"], "var_correlations.json"), "w") as f:
        json.dump(res, f, indent=1)

    s = res["spearman_primary"]
    p = res["pearson_supplementary"]
    with open(os.path.join(DIRS["highdim"], "var_correlations.md"), "w") as f:
        f.write("# 高维特征检验(主定量证据)\n\n"
                "- 口径:每个测试窗口在 **L2 归一化 768 维**(降维前)特征空间中,到 2000 张"
                "相同遮挡(P=1, 56 可见 patch)ImageNet 参考的最近 10 邻平均余弦距离;"
                "按变量取均值 → 190 点。\n"
                f"- **Spearman(主)**: dist~MSE ρ={s['dist_vs_mse']['stat']:.4f} "
                f"(p={s['dist_vs_mse']['p']:.3e});dist~MAE ρ={s['dist_vs_mae']['stat']:.4f} "
                f"(p={s['dist_vs_mae']['p']:.3e})\n"
                f"- Pearson(补充): dist~MSE r={p['dist_vs_mse']['stat']:.4f} "
                f"(p={p['dist_vs_mse']['p']:.3e});dist~MAE r={p['dist_vs_mae']['stat']:.4f} "
                f"(p={p['dist_vs_mae']['p']:.3e})\n"
                "- 频率分层仅为补充分析;不替代全 190 变量主结果。\n"
                "- 注意:重叠测试窗口非独立样本;p 值未处理领域内变量依赖,仅作探索性参考;"
                "观察到关联 ≠ 证明性能差异的原因。\n")

    freq = {}
    for v in xs:
        freq.setdefault(metrics[v][3], []).append(v)
    with open(os.path.join(DIRS["highdim"], "frequency_split.md"), "w") as f:
        f.write("# 频率分层(补充,非主结果)\n\n| 分组 | 变量数 | Spearman dist~MSE ρ | Spearman dist~MAE ρ |\n|---|---|---|---|\n")
        for g, vs in sorted(freq.items()):
            a = np.array([np.mean(per_var[v]) for v in vs])
            f.write(f"| {g} | {len(vs)} | {spearmanr(a, [metrics[v][0] for v in vs]).statistic:.4f} "
                    f"| {spearmanr(a, [metrics[v][1] for v in vs]).statistic:.4f} |\n")

    print(json.dumps({"spearman_dist_mse": s["dist_vs_mse"]["stat"],
                      "spearman_dist_mae": s["dist_vs_mae"]["stat"]}))
    print("highdim outputs ->", DIRS["highdim"])


if __name__ == "__main__":
    main()
