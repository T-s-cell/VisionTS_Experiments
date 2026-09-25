"""Step 5: plot the joint t-SNE figure (PNG + PDF)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {
    'ImageNet': '#4C72B0',
    'ETTm1': '#DD8452',
    'Weather': '#55A868',
    'Electricity': '#C44E52',
}


def main():
    coords = np.load(os.path.join(common.OUT_DIR, 'tsne_coords_1900x2.npy'))
    groups = np.load(os.path.join(common.OUT_DIR, 'groups.npy'), allow_pickle=True)
    assert len(coords) == len(groups) == 1900

    fig, ax = plt.subplots(figsize=(8.5, 7))
    for g in common.GROUPS:
        m = groups == g
        ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.55,
                   c=COLORS[g], label=f'{g} (n={int(m.sum())})', linewidths=0)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.set_title('VisionTS Fig. 7 partial reproduction\n'
                 'MAE encoder features, ImageNet + ETTm1 + Weather + Electricity (joint t-SNE)')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(common.OUT_DIR, f'fig7_tsne.{ext}')
        fig.savefig(p, dpi=220)
        print('saved', p, flush=True)


if __name__ == '__main__':
    main()
