"""EXP-041 step 4: high-dim 30-NN ImageNet-proportion metric on the joint
2,200-row frozen-encoder features (nn_metric_t36.py mechanics, +Monash).

For each of the 4 time-series sources (ETTm1 / Weather / Electricity from
the frozen 1,900 rows; Monash from the new 300), every row's 30 nearest
neighbours are taken over ALL 2,200 rows (raw Euclidean, float64
computation, self excluded by global row index).  Numerator = # ImageNet
neighbours summed over the source's rows; denominator = n_source x 30
(= 9,000 per cell).  Neighbour row indices are archived for independent
recomputation; the 2-D overlap is reported qualitatively ONLY.

Outputs (save_t41_monash/): nn_neighbors.npz, nn_table.csv, nn_summary.json
"""
import json
import os

import numpy as np

import t41_common as X


def main():
    fp = X.frozen_features_path()
    frozen = np.load(fp)
    assert frozen.shape == (1900, X.FEATURE_DIM)
    mon = np.load(X.FEATURES_MONASH)
    n_mon = mon.shape[0]
    assert mon.shape[1] == X.FEATURE_DIM
    merged = np.vstack([frozen, mon]).astype(np.float64)
    assert merged.shape == (1900 + n_mon, X.FEATURE_DIM)

    sq = (merged ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (merged @ merged.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, np.inf)     # self excluded by global row index
    dist = np.sqrt(d2)

    s_im = X.SPANS["ImageNet"]
    neighbors, proportions = {}, {}
    for g in X.TS_SOURCES:
        s0, s1 = X.SPANS[g]
        r1 = min(s1, 1900 + n_mon)   # smoke runs have fewer Monash rows
        idx = np.argpartition(dist[s0:r1], X.N_NEIGHBORS,
                              axis=1)[:, :X.N_NEIGHBORS].astype(np.int32)
        assert (idx != np.arange(s0, r1)[:, None]).all(), \
            "self in neighbours for %s" % g
        neighbors[g] = idx
        num = int(((idx >= s_im[0]) & (idx < s_im[1])).sum())
        proportions[g] = {
            "numerator_imagenet_neighbors": num,
            "denominator": len(idx) * X.N_NEIGHBORS,
            "proportion": num / float(len(idx) * X.N_NEIGHBORS)}

    flat = {"Monash__%s" % g: neighbors[g] for g in X.TS_SOURCES}
    np.savez_compressed(os.path.join(X.SAVE_BASE, "nn_neighbors.npz"), **flat)

    lines = ["ts_source,n_rows,imagenet_neighbors,denominator,proportion"]
    for g in X.TS_SOURCES:
        p = proportions[g]
        lines.append("%s,%d,%d,%d,%.6f" % (
            g, min(X.SPANS[g][1], 1900 + n_mon) - X.SPANS[g][0],
            p["numerator_imagenet_neighbors"],
            p["denominator"], p["proportion"]))
    with open(os.path.join(X.SAVE_BASE, "nn_table.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    summary = {
        "n_neighbors": X.N_NEIGHBORS,
        "distance": "euclidean on raw 54528-dim features, float64 computation",
        "self_exclusion": "fill_diagonal(inf) + per-source assert",
        "rows_total": 1900 + n_mon, "imagenet_rows": [0, 1000],
        "proportions": proportions,
        "neighbors_npz": os.path.join(X.SAVE_BASE, "nn_neighbors.npz"),
        "row_spans": {g: list(v) for g, v in X.SPANS.items()},
        "note": "frozen-encoder units only; the 2-D t-SNE overlap is a "
                "qualitative statement and is never used for prediction or "
                "causal claims.",
    }
    with open(os.path.join(X.SAVE_BASE, "nn_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    for line in lines:
        X.log("  " + line)
    X.log("DONE")


if __name__ == "__main__":
    main()
