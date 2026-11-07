"""EXP-041 step 5: figures — our joint 2,200-row t-SNE (5 sources), the
paper's original Fig.7 side-by-side (sha-pinned asset), and the stability
view as THREE full panels (rs2021/2022/2023), each showing ALL five
sources.  Different fits are never overlaid: t-SNE embeddings from
separate runs are not aligned, so cross-seed overlay would be meaningless
— stability is judged qualitatively per-panel (cluster structure).
Qualitative only: 2-D overlap makes no prediction or causal claim.

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

    fig = plt.figure(figsize=(15.0, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    def draw_all(ax, e, title):
        for g in X.GROUPS:
            s0, s1 = X.SPANS[g]
            sel = np.arange(s0, s1)
            ax.scatter(e[sel, 0], e[sel, 1], s=6, alpha=0.55,
                       c=X.PLOT_COLORS[g], label=g)
        ax.legend(fontsize=7, loc="best", markerscale=1.8)
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 0])
    draw_all(ax, emb, "EXP-041 joint t-SNE (rs%d, 2,200; Monash added)"
             % X.SEED)

    if paper_ok:
        ax = fig.add_subplot(gs[0, 1:])
        ax.imshow(mpimg.imread(paper_img_path))
        ax.set_title("paper Fig.7 (original, 2,200)")
        ax.axis("off")

    for j, rs in enumerate((X.SEED,) + tuple(X.STABILITY_SEEDS)):
        if rs == X.SEED:
            e = emb
        else:
            p = os.path.join(X.SAVE_BASE, "tsne_coords_2200x2_rs%d.npy" % rs)
            assert os.path.isfile(p), "missing stability coords %s" % p
            e = np.load(p)
        ax = fig.add_subplot(gs[1, j])
        draw_all(ax, e, "rs%d (all 2,200 rows)" % rs)

    fig.tight_layout()
    out = os.path.join(X.LOG_DIR, "fig41_tsne_monash.png")
    fig.savefig(out, dpi=150)
    X.log("wrote %s" % out)


if __name__ == "__main__":
    main()
