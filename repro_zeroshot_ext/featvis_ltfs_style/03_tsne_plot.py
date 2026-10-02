#!/usr/bin/env python3
"""Stage 03: joint dimensionality reduction + figures + report (CPU).

- main fit (LTFS style): the 1117x43776 raw flatten features go through ONE
  t-SNE directly — no mean pooling, no L2, no standardization, no pre-PCA
  (init='pca' only seeds the optimization and is NOT a PCA reduction).
- ctrl fit (pooled): the SAME rows' pooled features (CLS excluded, 56-patch
  mean) -> row L2 -> PCA(50, random_state=2021) -> ONE t-SNE, identical
  hyper-parameters. The two arms differ as a WHOLE pipeline (flatten+CLS kept
  vs pooling+CLS dropped+L2+PCA50); any difference between them is attributed
  to the pipeline as a whole, never to mean pooling alone.
- The two fits are separate: absolute coordinates are NOT comparable across
  them. Within each fit the three panels filter rows of the ONE coordinate
  set, so ImageNet background points/range/ratio are identical.
- t-SNE params (both fits): n_components=2, perplexity=30, learning_rate=200,
  init='pca', early_exaggeration=12, max_iter=1000, random_state=2021,
  metric='euclidean'.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager

from common import (DIRS, FEATVIS_DIR, FLAT_DIM, PRED_RTOL, SEED, ensure_dirs,
                    l2n, load_fig7_imagenet_json, timesx_windows)


def _cjk_font():
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK SC", "Droid Sans Fallback",
                 "WenQuanYi Zen Hei", "AR PL UMing CN"):
        if cand in avail:
            return cand
    return None


_CJK = _cjk_font()
if _CJK:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [_CJK, "DejaVu Sans"]
else:
    print("[03] WARNING: no CJK font found; Chinese annotations will not render")

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

GROUPS = ["best", "middle", "worst"]
GROUP_CN = {"best": "最好 (rank 1-6)", "middle": "中间 (rank 93-98)",
            "worst": "最差 (rank 185-190)"}
BG = dict(color="0.78", s=10, alpha=0.45, linewidths=0)
TSNE_KW = dict(n_components=2, perplexity=30, learning_rate=200, init="pca",
               early_exaggeration=12, max_iter=1000, random_state=SEED,
               metric="euclidean")
FITS = [("main", "主实验:LTFS 式 flatten 43776 维(含 CLS,无池化/无 L2/无 PCA)"),
        ("ctrl", "对照:排除 CLS 均值池化 768 维 → L2 → PCA50")]


def tsne_params_dict():
    import torch
    _, sha_list, _ = load_fig7_imagenet_json()
    return {"fits": {name: {"ref": "fig7_imagenet_1000_masked_P1like",
                            "tsne": TSNE_KW,
                            "pca": None if name == "main"
                            else {"n_components": 50, "random_state": SEED},
                            "l2_normalize": name == "ctrl",
                            "rows": 1117, "imagenet": 1000, "timesx": 117}
                    for name, _ in FITS},
            "weights_sha256_prefix": json.load(
                open(os.path.join(DIRS["logs"], "extract_env.json")))
                ["ckpt_sha256_prefix"],
            "imagenet_list_sha256_prefix": sha_list[:16],
            "pred_rtol_declared": PRED_RTOL,
            "software": {"python": sys.version.split()[0],
                         "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "torch": torch.__version__},
            "notes": ["main and ctrl fits are separate; absolute coordinates "
                      "not comparable across the two arms",
                      "init='pca' only initializes the optimization; no PCA "
                      "reduction is applied to the flatten features"]}


def run_fits():
    fl = np.load(os.path.join(DIRS["features"], "flatten_1117.npz"),
                 allow_pickle=False)
    po = np.load(os.path.join(DIRS["features"], "pooled_1117.npz"),
                 allow_pickle=False)
    for key, shape in (("feat", (1117, FLAT_DIM)),):
        assert fl[key].shape == shape, fl[key].shape
    assert po["feat"].shape == (1117, 768), po["feat"].shape
    assert [str(s) for s in fl["id"]] == [str(s) for s in po["id"]]
    assert [str(s) for s in fl["source"]] == [str(s) for s in po["source"]]
    ids = [str(s) for s in fl["id"]]
    src = [str(s) for s in fl["source"]]
    assert src == ["imagenet"] * 1000 + ["timesx"] * 117
    assert len(set(ids)) == 1117

    win = {w["sample_id"]: w for w in timesx_windows()}
    cmap = {r["var_key"]: int(r["color_idx"]) for r in csv.DictReader(
        open(os.path.join(FEATVIS_DIR, "tsne", "color_map.csv")))}

    X = {"main": fl["feat"],
         "ctrl": PCA(n_components=50, random_state=SEED).fit_transform(
             l2n(po["feat"]))}
    params = tsne_params_dict()
    for name, _ in FITS:
        Z = TSNE(**TSNE_KW).fit_transform(X[name])
        assert Z.shape == (1117, 2) and np.isfinite(Z).all()
        out = os.path.join(DIRS["tsne"], f"coords_{name}.csv")
        with open(out, "w") as f:
            f.write("row_id,source,masked,id,var_key,sel_group,color_idx,x,y\n")
            for j in range(1117):
                if src[j] == "imagenet":
                    f.write(f"{j},imagenet,1,{ids[j]},,,,")
                else:
                    w = win[ids[j]]
                    f.write(f"{j},timesx,1,{ids[j]},{w['var_key']},"
                            f"{w['sel_group']},{cmap[w['var_key']]},")
                f.write(f"{'%.8f' % Z[j, 0]},{'%.8f' % Z[j, 1]}\n")
        print(f"{name}: tsne done rows=1117 -> {out}")

    with open(os.path.join(DIRS["tsne"], "params.json"), "w") as f:
        json.dump(params, f, indent=1)


def read_coords(name):
    rows = list(csv.DictReader(open(os.path.join(DIRS["tsne"],
                                                 f"coords_{name}.csv"))))
    ref = [r for r in rows if r["source"] == "imagenet"]
    tsx = [r for r in rows if r["source"] == "timesx"]
    assert len(ref) == 1000 and len(tsx) == 117
    xy_ref = np.array([[float(r["x"]), float(r["y"])] for r in ref])
    return rows, tsx, xy_ref


def draw_fit(name, desc, out_prefix):
    rows, tsx, xy_ref = read_coords(name)
    win = {w["sample_id"]: w for w in timesx_windows()}
    freq_cn = lambda vk: "周频" if win[vk]["frequency"] == "weekly" else "日频"
    short = lambda vk: vk.split("__", 1)[1].rsplit("_96_12", 1)[0]
    colors = matplotlib.colormaps["tab20"]
    allxy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    xlim = (allxy[:, 0].min(), allxy[:, 0].max())
    ylim = (allxy[:, 1].min(), allxy[:, 1].max())

    def draw_panel(ax, group):
        ax.scatter(xy_ref[:, 0], xy_ref[:, 1], **BG, zorder=1)
        for r in tsx:
            if win[r["id"]]["sel_group"] != group:
                continue
            ax.scatter([float(r["x"])], [float(r["y"])], s=30, zorder=2,
                       linewidths=0.3, color=colors(int(r["color_idx"])))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        n = sum(1 for r in tsx if win[r["id"]]["sel_group"] == group)
        ax.set_title(f"{GROUP_CN[group]} · n={n} 窗口", fontsize=10)
        ax.tick_params(labelsize=7)

    def legend_items(groups_):
        items = [Line2D([0], [0], marker="o", linestyle="", color=BG["color"],
                        alpha=0.8, markersize=5,
                        label="ImageNet (LTFS 同批 1000 张,56 patch 遮挡)")]
        seen = set()
        for r in tsx:
            vk = r["var_key"]
            if vk in seen or win[r["id"]]["sel_group"] not in groups_:
                continue
            seen.add(vk)
            items.append(Line2D([0], [0], marker="o", linestyle="",
                                color=colors(int(r["color_idx"])), markersize=6,
                                label=f"[{win[r['id']]['sel_group']}] "
                                      f"{short(vk)} ({freq_cn(r['id'])})"))
        return items

    for g in GROUPS:
        fig, ax = plt.subplots(figsize=(8.2, 5.8))
        draw_panel(ax, g)
        ax.legend(handles=legend_items([g]), fontsize=6.5,
                  bbox_to_anchor=(1.03, 1.0), loc="upper left",
                  framealpha=0.85)
        fig.suptitle(f"VisionTS TimesX zero-shot · t-SNE({name};{desc})",
                     fontsize=10)
        fig.text(0.01, 0.01, "ImageNet: 复用 LTFS 实验 1000 张清单,未按本轮重抽;"
                             "主实验与对照各自独立拟合,绝对坐标不可跨套比较",
                 fontsize=6.5, color="0.35")
        fig.savefig(os.path.join(DIRS["figures"], f"{out_prefix}_{g}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.6))
    for ax, g in zip(axes, GROUPS):
        draw_panel(ax, g)
    axes[2].legend(handles=legend_items(GROUPS), fontsize=6.5,
                   bbox_to_anchor=(1.02, 1.0), loc="upper left",
                   framealpha=0.85)
    fig.suptitle(f"VisionTS TimesX zero-shot · t-SNE({name},三组共用坐标;"
                 f"{desc})", fontsize=11)
    fig.text(0.01, 0.01, "ImageNet: 复用 LTFS 实验 1000 张清单,未按本轮重抽;"
                         "主实验与对照各自独立拟合,绝对坐标不可跨套比较",
             fontsize=7, color="0.35")
    fig.savefig(os.path.join(DIRS["figures"], f"{out_prefix}_triple.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def separation_stats(name):
    """Descriptive only: t-SNE 2-D distances are not a formal metric."""
    import math
    import random
    _, tsx, xy_ref = read_coords(name)
    pts = [tuple(q) for q in xy_ref]

    def nn_d(p, pts_):
        return min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in pts_)

    d_tx = sorted(nn_d((float(r["x"]), float(r["y"])), pts) for r in tsx)
    random.seed(0)
    idx = random.sample(range(len(pts)), 300)
    d_im = sorted(nn_d(pts[i], pts[:i] + pts[i + 1:]) for i in idx)
    return {"timesx_nn_median": d_tx[58], "timesx_nn_min": d_tx[0],
            "timesx_nn_max": d_tx[-1],
            "imagenet_nn_median_sampled": d_im[150],
            "imagenet_nn_p95_sampled": d_im[285]}


def write_report(stats):
    env = json.load(open(os.path.join(DIRS["logs"], "extract_env.json")))
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write("# LTFS 式 flatten 对照实验报告(featvis_ltfs_style)\n\n"
                "## 目的\n\n检验 featvis 主实验观察到的 TimesX/ImageNet 分离是否"
                "依赖可视化处理方式:改用旧 Long-term TSF 实验(repro_fig7)的特征"
                "口径(含 CLS 全 token flatten,无池化/无 L2/无预 PCA,直接 "
                "t-SNE)重画同一批 117 窗口,并用同样本的池化流程作对照。\n\n"
                "## 口径\n\n"
                "- 同一批编码器输出两种特征:main=flatten 43776(含 CLS);"
                "ctrl=排除 CLS → 56 patch 均值 → L2 → PCA50。两组 t-SNE 超参完全"
                "相同(perplexity=30, lr=200, init='pca', early_exaggeration=12, "
                "max_iter=1000, seed=2021, metric='euclidean'),各自独立联合拟合,"
                "**绝对坐标不可跨套比较**。\n"
                "- 预测配置与 featvis 冻结口径一致(固定 P=1,96→12,r=c=0.4,"
                "mae_base,TF32 关闭);117 窗口名单与 featvis 完全一致。\n"
                "- ImageNet 参考复用旧 LTFS 实验的 1000 张清单"
                f"(sha256 前缀 {env['imagenet_list_sha256_prefix']}),顺序原样,"
                "mask 换成本轮 56 可见 patch;每个 batch(含尾部不足 64 的 "
                "batch)均断言可见 token **顺序**逐元素一致。\n"
                "- 预测核验:全部被前向行与冻结 cache 比较,最大相对误差 "
                f"{env['worst_cache_pred_rel']:.2e}(声明 rtol≤{PRED_RTOL})。\n\n"
                "## 结果\n\n"
                + "".join(f"- **{name}**: TimesX→最近 ImageNet 二维距离 "
                          f"median={s['timesx_nn_median']:.2f}, "
                          f"min={s['timesx_nn_min']:.2f};ImageNet 内部最近邻"
                          f"(300 抽样)median={s['imagenet_nn_median_sampled']:.2f}。"
                          f"(见图 figures/{name}_triple.png)\n"
                          for (name, _), s in zip(FITS, stats)) +
                "- 分离/混合是否随流程改变:对比两套图内 TimesX 簇与 ImageNet 云"
                "的相对关系;t-SNE 二维距离仅作描述,不是正式距离指标。\n\n"
                "## 结论限定(必读)\n\n"
                "- 两组**同时**改变了:展平 vs 池化、CLS 保留与否、L2 归一化、"
                "预先 PCA。两套结果的任何差异只能归因于**整套处理流程**,不能仅凭"
                "这两组把变化单独归因于平均池化。\n"
                "- 本轮已对齐可视化流程,但与旧 Long-term 实验仍存在原生差异:"
                "**数据集(TimesX vs LTFS)、原生窗口、P 配置、可见 patch 数量"
                "(56 vs 70)、样本比例(旧联合拟合 = 1000 张 ImageNet + 900 个"
                "时序点,每数据集 300;本轮 = 1000 张 ImageNet + 117 个时序点)**"
                "。\n"
                "- 本轮为事后诊断性探索:观察到的是分布关系,不得解释为采样频率"
                "或预测性能差异的因果证据。\n")


def main():
    ensure_dirs()
    run_fits()
    stats = [separation_stats(name) for name, _ in FITS]
    for (name, _), s in zip(FITS, stats):
        print(f"separation[{name}]: {s}")
    draw_fit("main", FITS[0][1], "main")
    draw_fit("ctrl", FITS[1][1], "ctrl")
    write_report(stats)
    print("figures ->", DIRS["figures"], "; report -> REPORT.md")


if __name__ == "__main__":
    main()
