"""EXP-041 step 5: figures — our joint 2,200-row t-SNE (5 sources), the
paper's original Fig.7 side-by-side (sha-pinned asset), and the
rs2022/2023 stability view.  Qualitative only: 2-D overlap makes no
prediction or causal claim.

Outputs (logs/): fig41_tsne_monash.png
"""
import os

import numpy as np

import t41_common as X


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    main_p = os.path.join(X.SAVE_BASE, "tsne_coords_2200x2.npy")
    assert os.path.isfile(main_p), "missing %s (run run_tsne_joint first)" % main_p
    emb = np.load(main_p)
    assert emb.shape == (2200, 2)
    groups_col = np.load(X.GROUPS_MONASH, allow_pickle=True)
    assert len(list(groups_col)) == 300

    paper_img_path = os.path.join(X.REPO_ROOT, X.PAPER_FIG7_IMG_REL)
    paper_ok = os.path.isfile(paper_img_path)
    if paper_ok:
        sha = X.sha256_of(paper_img_path)
        assert sha == X.PAPER_FIG7_IMG_SHA256, "paper Fig.7 asset sha drift"
    else:
        X.log("paper Fig.7 asset not found at %s — panel skipped" % paper_img_path)

    n_panel = 3 if paper_ok else 2
    fig, axes = plt.subplots(1, n_panel, figsize=(5.0 * n_panel, 4.6))

    ax = axes[0]
    for g in X.GROUPS:
        s0, s1 = X.SPANS[g]
        sel = np.arange(s0, s1)
        ax.scatter(emb[sel, 0], emb[sel, 1], s=6, alpha=0.55,
                   c=X.PLOT_COLORS[g], label=g,
                   marker="o" if g != "ImageNet" else "o")
    ax.legend(fontsize=8, loc="best")
    ax.set_title("EXP-041 joint t-SNE (2,200; Monash added)")
    ax.set_xticks([]), ax.set_yticks([])

    if paper_ok:
        ax = axes[1]
        ax.imshow(mpimg.imread(paper_img_path))
        ax.set_title("paper Fig.7 (original, 1,900)")
        ax.axis("off")

    ax = axes[n_panel - 1]
    for rs in X.STABILITY_SEEDS:
        p = os.path.join(X.SAVE_BASE, "tsne_coords_2200x2_rs%d.npy" % rs)
        assert os.path.isfile(p), "missing stability coords %s" % p
        e = np.load(p)
        sel = np.arange(*X.SPANS["Monash"])
        ax.scatter(e[sel, 0], e[sel, 1], s=6, alpha=0.55,
                   c=X.PLOT_COLORS["Monash"],
                   label="Monash rs%d" % rs)
    sel = np.arange(0, 1900)
    ax.scatter(emb[sel, 0], emb[sel, 1], s=4, alpha=0.15, c="#bbbbbb",
               label="others rs2021")
    ax.legend(fontsize=8)
    ax.set_title("stability (Monash rs2022/2023)")
    ax.set_xticks([]), ax.set_yticks([])

    fig.tight_layout()
    out = os.path.join(X.LOG_DIR, "fig41_tsne_monash.png")
    fig.savefig(out, dpi=150)
    X.log("wrote %s" % out)


if __name__ == "__main__":
    main()
