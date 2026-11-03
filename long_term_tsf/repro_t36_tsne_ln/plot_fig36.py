"""EXP-036 step 6: 2x2 candidate figure.

Top-left = the archived EXP-002 frozen coordinates (PPT page-5 right figure,
repro_fig7/outputs/tsne_coords_1900x2.npy) reused verbatim; the other three
panels are the new per-condition t-SNE runs. Source colors, marker size,
alpha and legend are identical across panels (plot_fig7.py styling); each
panel is an independent joint t-SNE, so axes are NOT comparable across
panels — only within-panel local overlap is read, plus the high-dim NN
table. Writes fig36_tsne_compare.png / .pdf next to this script.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import t36_common as C

COLORS = {
    "ImageNet": "#4C72B0",
    "ETTm1": "#DD8452",
    "Weather": "#55A868",
    "Electricity": "#C44E52",
}

PANELS = [("frozen", "Frozen MAE encoder\n(EXP-002, PPT p.5, archived coords)", C.ORIG_COORDS),
          ("ettm1_ln", "ETTm1-LN encoder\n(encoder-LN finetuned on ETTm1, H=96 seed=2021)", None),
          ("ecl_ln", "Electricity-LN encoder\n(encoder-LN finetuned on Electricity, H=96 seed=2021)", None),
          ("weather_ln", "Weather-LN encoder\n(encoder-LN finetuned on Weather, H=96 seed=2021)", None)]


def main():
    groups = np.load(C.ORIG_GROUPS, allow_pickle=True)
    assert len(groups) == 1900

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13.5))
    for ax, (cond, title, archived) in zip(axes.ravel(), PANELS):
        path = archived or C.cond_coords(cond)
        coords = np.load(path)
        assert coords.shape == (1900, 2), (cond, coords.shape)
        for g in C.GROUPS:
            m = groups == g
            ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.55,
                       c=COLORS[g], label="%s (n=%d)" % (g, int(m.sum())), linewidths=0)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="best", framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Same 1,900 inputs, four encoders: frozen vs LayerNorm-finetuned "
                 "(EXP-036)\nper-panel joint t-SNE on raw 54,528-dim features — "
                 "axes are per-panel independent; compare overlap within each panel",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        p = os.path.join(C.PKG, "fig36_tsne_compare.%s" % ext)
        fig.savefig(p, dpi=220)
        print("saved", p, flush=True)


if __name__ == "__main__":
    main()
