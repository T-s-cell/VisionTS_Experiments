#!/usr/bin/env python3
"""Stage 06: figures.

- For each fit (main, supp_nomask, 4 stability): three panels (best/middle/
  worst) filtering rows from the ONE joint coordinate set + one triple
  composite. ImageNet background points are identical across panels of a fit
  (same array by construction; row counts asserted). Axes limits/style equal.
- Annotations: ImageNet sampling label; mask condition per fit; "main and
  supplementary fits are separate -- absolute coords not comparable across the
  two figure sets".
- Representative windows: 2 per group: history curve, the EXACT encoder input
  image (independent rebuild verified against the hook in stage 02),
  pred-vs-true overlay from the frozen cache.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager


def _cjk_font():
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK SC", "Droid Sans Fallback", "WenQuanYi Zen Hei",
                 "AR PL UMing CN"):
        if cand in avail:
            return cand
    return None


_CJK = _cjk_font()
if _CJK:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [_CJK, "DejaVu Sans"]
else:
    print("[06] WARNING: no CJK font found; Chinese annotations will not render")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from common import (DIRS, build_ts_image, ensure_dirs, load_model)

ZERO = os.path.dirname(HERE)
GROUPS = ["best", "middle", "worst"]
GROUP_CN = {"best": "最好 (rank 1-6)", "middle": "中间 (rank 93-98)",
            "worst": "最差 (rank 185-190)"}
BG = dict(color="0.78", s=10, alpha=0.45, linewidths=0)


def read_coords(name):
    rows = list(csv.DictReader(open(os.path.join(DIRS["tsne"], f"coords_{name}.csv"))))
    ref = [r for r in rows if r["source"] == "imagenet"]
    tsx = [r for r in rows if r["source"] == "timesx"]
    assert len(ref) == 2000 and len(tsx) == 117
    xy_ref = np.array([[float(r["x"]), float(r["y"])] for r in ref])
    return rows, tsx, xy_ref


def draw_panel(ax, tsx, xy_ref, group, meta, colors, xlim, ylim):
    ax.scatter(xy_ref[:, 0], xy_ref[:, 1], **BG, zorder=1)
    for vk, m in meta.items():
        if m["sel_group"] != group:
            continue
        pts = np.array([[float(r["x"]), float(r["y"])] for r in tsx if r["var_key"] == vk])
        ax.scatter(pts[:, 0], pts[:, 1], s=30, zorder=2, linewidths=0.3,
                   color=colors(m["color_idx"]))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    n = sum(1 for r in tsx if meta[r["var_key"]]["sel_group"] == group)
    ax.set_title(f"{GROUP_CN[group]} · n={n} 窗口", fontsize=10)
    ax.tick_params(labelsize=7)


def short(vk):
    return vk.split("__", 1)[1].rsplit("_96_12", 1)[0]


def draw_fit(name, ref_label, cross_note, out_prefix):
    rows, tsx, xy_ref = read_coords(name)
    entries = list(csv.DictReader(open(os.path.join(DIRS["tsne"], "color_map.csv"))))
    colors = matplotlib.colormaps["tab20"]
    meta = {e["var_key"]: {"sel_group": e["sel_group"], "frequency": e["frequency"],
                           "color_idx": int(e["color_idx"])} for e in entries}
    allxy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    xlim, ylim = (allxy[:, 0].min(), allxy[:, 0].max()), (allxy[:, 1].min(), allxy[:, 1].max())

    def legend_items(groups):
        items = [Line2D([0], [0], marker="o", linestyle="", color=BG["color"],
                        alpha=0.8, markersize=5, label=f"ImageNet ({ref_label})")]
        for vk, m in meta.items():
            if m["sel_group"] in groups:
                freq_cn = "周频" if m["frequency"] == "weekly" else "日频"
                pre = f"[{m['sel_group']}] " if len(groups) > 1 else ""
                items.append(Line2D([0], [0], marker="o", linestyle="",
                                    color=colors(m["color_idx"]), markersize=6,
                                    label=f"{pre}{short(vk)} ({freq_cn})"))
        return items

    for g in GROUPS:
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        draw_panel(ax, tsx, xy_ref, g, meta, colors, xlim, ylim)
        ax.legend(handles=legend_items([g]), fontsize=6.2, loc="best", framealpha=0.85)
        fig.suptitle(f"VisionTS TimesX zero-shot · MAE 编码器特征 t-SNE（{name}）", fontsize=11)
        fig.text(0.01, 0.01, "ImageNet: ILSVRC2012 val 随机子集 2000 张，未按类别分层  |  " + cross_note,
                 fontsize=6.5, color="0.35")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(DIRS["figures"], f"{out_prefix}_{g}.png"), dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.8))
    for ax, g in zip(axes, GROUPS):
        draw_panel(ax, tsx, xy_ref, g, meta, colors, xlim, ylim)
    axes[2].legend(handles=legend_items(GROUPS), fontsize=6.4, loc="best", framealpha=0.85)
    fig.suptitle(f"VisionTS TimesX zero-shot · MAE 编码器特征 t-SNE（{name}，三组共用坐标）", fontsize=12)
    fig.text(0.01, 0.01, "ImageNet: ILSVRC2012 val 随机子集 2000 张，未按类别分层  |  " + cross_note,
             fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(DIRS["figures"], f"{out_prefix}_triple.png"), dpi=200)
    plt.close(fig)


def representative_windows(device):
    picks = {"best": ["traffic__tour_de_france_96_12_4_10events",
                      "shopping__air_conditioner_96_12_4_10events"],
             "middle": ["climate__endangered_species_96_12_4_10events",
                        "CropsAndStaples__rice_usd_cwt_96_12_12_10events"],
             "worst": ["economy__cost_of_living_96_12_4_10events",
                       "science__nobel_prize_96_12_4_10events"]}
    rows = [(g, vk) for g in GROUPS for vk in picks[g]]
    d = np.load(os.path.join(DIRS["features"], "timesx_sel117.npz"), allow_pickle=False)
    first_sid = {}
    for vk, sid in zip([str(v) for v in d["var_key"]], [str(s) for s in d["sample_id"]]):
        first_sid.setdefault(vk, sid)

    model, _ = load_model(device)  # image rebuild only; verified == hook in 02
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 2.7 * len(rows)))
    from timesx_data import find_zip, iter_variables
    need = {vk: first_sid[vk] for _, vk in rows}
    pasts = {}
    for v in iter_variables(find_zip(None)):
        if v.var_key in need:
            by = {s.sample_id: s for s in v.samples}
            pasts[v.var_key] = np.asarray(by[need[v.var_key]].past_val, dtype=float)
    for r, (g, vk) in enumerate(rows):
        arr = np.load(os.path.join(ZERO, "cache", f"{vk}__P1.npz"), allow_pickle=False)
        ids = [str(s) for s in arr["sample_ids"]]
        i = ids.index(first_sid[vk])
        past, pred, true = pasts[vk], arr["pred"][i], arr["true"][i]

        ax = axes[r, 0]
        ax.plot(range(96), past, lw=1)
        ax.set_ylabel(f"[{g}] {short(vk)}", fontsize=7)

        ax = axes[r, 1]
        x = torch.tensor(past[None, :, None], dtype=torch.float32, device=device)
        img = build_ts_image(model, x)[0].mean(0).cpu().numpy()  # grey [224,224]
        ax.imshow(img, cmap="grey", vmin=-8, vmax=8)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[r, 2]
        ax.plot(range(12), true, lw=1.4, label="true")
        ax.plot(range(12), pred, lw=1.4, ls="--", label="pred")
        ax.legend(fontsize=6)
    for ax in axes.flat:
        ax.tick_params(labelsize=6)
    fig.suptitle("代表窗口：历史 96 步 · 编码器实际输入图（P=1，右 10 列为零 mask）· 预测 vs 真值", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(DIRS["figures"], "representative_windows.png"), dpi=200)
    plt.close(fig)


def main():
    ensure_dirs()
    device = "cpu"  # display rebuilds only; GPU-exactness was asserted in stage 02
    cross = "主图与无-mask 补充图分别拟合，两套坐标不可直接比较"
    draw_fit("main", "P=1 相同遮挡参考：56 可见 patch", cross, "main")
    draw_fit("supp_nomask", "无 mask 参考：196 patch 全可见",
             "ImageNet 无 mask；TimesX 保留实际预测 mask  |  " + cross, "supp")
    for s in ("sens_seed2022", "sens_seed2023", "sens_perp20", "sens_perp50"):
        draw_fit(s, "P=1 相同遮挡参考：56 可见 patch", cross, s)
    representative_windows(device)
    print("figures ->", DIRS["figures"])


if __name__ == "__main__":
    main()
