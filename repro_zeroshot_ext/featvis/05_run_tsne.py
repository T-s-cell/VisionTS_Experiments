#!/usr/bin/env python3
"""Stage 05: joint dimensionality reduction (one PCA + one t-SNE per fit).

Main fit: [2000 masked-ImageNet + 117 TimesX windows] -> row L2-normalize ->
PCA(50) fit on ALL rows -> single joint t-SNE. The three panels later filter
rows from this ONE coordinate set, so ImageNet background points are identical
across panels.

Supplementary fit: the SAME features with the ImageNet block swapped to the
no-mask features; ONE separate joint fit (same hyper-params). The main and
supplementary fits are fitted separately: their absolute coordinates are NOT
comparable across the two sets (only distribution-level consistency may be
observed; never interpret position changes across sets as feature movement).

Stability fits: main pairing re-run with seeds {2022, 2023} and perplexities
{20, 50}; each internally joint; archived, never cherry-picked.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common import DIRS, SEED, ensure_dirs, l2n

FITS = [
    ("main", "masked", 2021, 30),
    ("supp_nomask", "nomask", 2021, 30),
    ("sens_seed2022", "masked", 2022, 30),
    ("sens_seed2023", "masked", 2023, 30),
    ("sens_perp20", "masked", 2021, 20),
    ("sens_perp50", "masked", 2021, 50),
]


def main():
    ensure_dirs()
    import csv
    import sklearn
    sel = np.load(os.path.join(DIRS["features"], "timesx_sel117.npz"), allow_pickle=False)
    im = {k: np.load(os.path.join(DIRS["features"], f"imagenet_{k}.npz"), allow_pickle=False)
          for k in ("masked", "nomask")}

    sel_ids = [str(s) for s in sel["sample_id"]]
    assert len(set(sel_ids)) == 117
    im_files = {k: [str(f) for f in im[k]["fname"]] for k in im}
    for k in im:
        assert len(set(im_files[k])) == 2000
        assert im_files[k] == im_files["masked"]  # same batch of images for both controls

    # frozen per-variable color map (selection order: best 6, middle 6, worst 6)
    seen = {}
    for vk, g in zip([str(v) for v in sel["var_key"]], [str(v) for v in sel["group"]]):
        seen.setdefault(vk, g)
    assert len(seen) == 18
    var_color = {vk: i for i, vk in enumerate(seen)}
    meta = {r["var_key"]: r for r in
            csv.DictReader(open(os.path.join(DIRS["selection"], "selected_vars.csv")))}
    with open(os.path.join(DIRS["tsne"], "color_map.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["var_key", "sel_group", "frequency", "color_idx"])
        for vk, g in seen.items():
            w.writerow([vk, g, meta[vk]["frequency"], var_color[vk]])

    params = {}
    for name, ref, seed, perp in FITS:
        F_ref = l2n(im[ref]["feat"])
        F_tsx = l2n(sel["feat"])
        X = np.concatenate([F_ref, F_tsx], 0)
        X50 = PCA(n_components=50, random_state=seed).fit_transform(X)
        Z = TSNE(n_components=2, perplexity=perp, learning_rate=200, init="pca",
                 max_iter=1000, random_state=seed).fit_transform(X50)
        n_ref = F_ref.shape[0]
        assert Z.shape == (2117, 2)

        out = os.path.join(DIRS["tsne"], f"coords_{name}.csv")
        with open(out, "w") as f:
            f.write("row_id,source,masked,id,var_key,sel_group,color_idx,x,y\n")
            for i in range(n_ref):
                f.write(f"{i},imagenet,{1 if ref == 'masked' else 0},"
                        f"{im_files[ref][i]},,,,{'%.8f' % Z[i, 0]},{'%.8f' % Z[i, 1]}\n")
            for j in range(117):
                i = n_ref + j
                vk = str(sel["var_key"][j])
                f.write(f"{4000 + int(sel['row_id'][j])},timesx,1,{sel_ids[j]},"
                        f"{vk},{sel['group'][j]},{var_color[vk]},"
                        f"{'%.8f' % Z[i, 0]},{'%.8f' % Z[i, 1]}\n")
        params[name] = {"ref": ref, "seed": seed, "perplexity": perp,
                        "learning_rate": 200, "init": "pca", "max_iter": 1000,
                        "rows": int(Z.shape[0]), "pca_dim": 50,
                        "sklearn": sklearn.__version__,
                        "note": "main and supplementary fits are separate; "
                                "absolute coords not comparable across fits"}
        print(f"{name}: rows=2117 ref={ref} seed={seed} perp={perp} -> {out}")

    with open(os.path.join(DIRS["tsne"], "params.json"), "w") as f:
        json.dump(params, f, indent=1)
    print("t-SNE done ->", DIRS["tsne"])


if __name__ == "__main__":
    main()
