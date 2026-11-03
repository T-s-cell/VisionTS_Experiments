"""EXP-036 step 5: high-dim 30-NN ImageNet-proportion metric.

For each encoder condition and each time-series source (300 rows in the
fixed order), each sample's 30 nearest neighbours are taken over ALL 1,900
features with raw Euclidean distance (computed in float64 for a stable
distance; t-SNE itself ran on the float32 features), excluding self by
global row index. Proportion = (# ImageNet neighbours) / (300*30 = 9000).
Neighbour row indices are archived for independent recomputation.
Writes:
  save_t36_tsne/nn_neighbors.npz   keys '{cond}__{group}' (300x30 int32)
  save_t36_tsne/nn_table.csv
  save_t36_tsne/nn_summary.json
"""
import json
import os
import time

import numpy as np

import t36_common as C


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    spans = C.group_spans()
    feats = {}
    for cond in C.COND_LIST:
        p = C.cond_features(cond)
        x = np.load(p).astype(np.float64)
        assert x.shape == (1900, C.FEATURE_DIM)
        feats[cond] = x
    log("loaded 4 x (1900, 54528); computing float64 euclidean 30-NN")

    neighbors = {}
    rows = {}
    for cond, x in feats.items():
        sq = (x ** 2).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
        np.maximum(d2, 0.0, out=d2)
        np.fill_diagonal(d2, np.inf)  # exclude self by global row index
        dist = np.sqrt(d2)
        per = {}
        for g in C.GROUPS[1:]:
            s0, s1 = spans[g]
            idx = np.argpartition(dist[s0:s1], C.N_NEIGHBORS, axis=1)[:, :C.N_NEIGHBORS]
            idx = idx.astype(np.int32)
            assert (idx != np.arange(s0, s1)[:, None]).all(), "self in neighbours"
            per[g] = idx
        neighbors[cond] = per
        log("%s: neighbours done" % cond)
        del dist, d2

    proportions = C.nn_recompute_proportions(neighbors)

    npz_path = os.path.join(C.SAVE_BASE, "nn_neighbors.npz")
    flat = {}
    for cond, per in neighbors.items():
        for g, arr in per.items():
            flat["%s__%s" % (cond, g)] = arr
    np.savez_compressed(npz_path, **flat)

    csv_path = os.path.join(C.SAVE_BASE, "nn_table.csv")
    lines = ["encoder,ts_source,n_samples,imagenet_neighbors,denominator,proportion,pp_vs_frozen"]
    for cond in C.COND_LIST:
        for g in C.GROUPS[1:]:
            cnt = int(round(proportions[cond][g] * C.NN_DENOM))
            prop = proportions[cond][g]
            pp = "" if cond == "frozen" else "%.4f" % ((prop - proportions["frozen"][g]) * 100.0)
            lines.append("%s,%s,%d,%d,%d,%.6f,%s"
                         % (cond, g, C.N_TS, cnt, C.NN_DENOM, prop, pp))
    with open(csv_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    summary = {
        "n_neighbors": C.N_NEIGHBORS,
        "denominator_per_cell": C.NN_DENOM,
        "distance": "euclidean on raw 54528-dim features, float64 computation",
        "proportions": {c: {g: proportions[c][g] for g in C.GROUPS[1:]} for c in C.COND_LIST},
        "pp_vs_frozen": {c: {g: (proportions[c][g] - proportions["frozen"][g]) * 100.0
                             for g in C.GROUPS[1:]} for c in C.COND_LIST if c != "frozen"},
        "neighbors_npz": npz_path,
        "row_spans": spans,
    }
    with open(os.path.join(C.SAVE_BASE, "nn_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")

    data = C.load_json(C.VERIFY_JSON) if os.path.isfile(C.VERIFY_JSON) else {}
    data["nn"] = {"table": csv_path, "neighbors": npz_path,
                  "proportions": summary["proportions"],
                  "pp_vs_frozen": summary["pp_vs_frozen"]}
    C.dump_json(C.VERIFY_JSON, data)

    log("nn_table.csv:")
    for line in lines:
        log("  " + line)
    log("DONE")


if __name__ == "__main__":
    main()
