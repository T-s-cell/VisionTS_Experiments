"""EXP-036 step 4: t-SNE for one condition on its raw 1900x54528 features.

EXP-002 parameters verbatim (sklearn TSNE, perplexity=30, learning_rate=200,
init='pca', early_exaggeration=12, max_iter=1000, random_state=2021,
euclidean; no pre-PCA, no per-group standardization) and the same sklearn
version (pinned 1.9.1, checked in the driver preflight). The frozen panel
reuses the archived tsne_coords_1900x2.npy — only the three LN conditions
run here. CPU-bound; intended under nohup. Writes:
  save_t36_tsne/coords_{cond}_1900x2.npy  (float64)
"""
import argparse
import os
import sys
import time

import numpy as np

import t36_common as C


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=C.COND_LIST)
    ap.add_argument("--force", action="store_true",
                    help="allow the frozen condition (verification only; the "
                         "figure uses the archived EXP-002 coords)")
    args = ap.parse_args()
    cond = args.cond
    if cond == "frozen" and not args.force:
        raise SystemExit("frozen panel uses the archived EXP-002 coords; "
                         "pass --force to run a verification t-SNE")

    import sklearn
    from sklearn.manifold import TSNE
    if sklearn.__version__ != C.SKLEARN_PIN:
        raise SystemExit("sklearn %s != pinned %s (EXP-002 env); t-SNE replication "
                         "requires the same version" % (sklearn.__version__, C.SKLEARN_PIN))

    feats = np.load(C.cond_features(cond))
    assert feats.shape == (1900, C.FEATURE_DIM), feats.shape
    assert np.isfinite(feats).all()
    log("%s: features %s dtype=%s; TSNE %s" % (cond, feats.shape, feats.dtype, C.TSNE_PARAMS))
    t0 = time.time()
    emb = TSNE(**C.TSNE_PARAMS).fit_transform(feats)
    el = time.time() - t0
    log("TSNE done in %.1f min" % (el / 60.0))

    np.save(C.cond_coords(cond), emb.astype(np.float64))
    data = C.load_json(C.VERIFY_JSON) if os.path.isfile(C.VERIFY_JSON) else {}
    data.setdefault("tsne", {})[cond] = {
        "params": C.TSNE_PARAMS,
        "sklearn_version": sklearn.__version__,
        "input_shape": list(feats.shape),
        "input_dtype": str(feats.dtype),
        "coords": C.cond_coords(cond),
        "elapsed_s": round(el, 1),
    }
    C.dump_json(C.VERIFY_JSON, data)
    log("saved %s" % C.cond_coords(cond))


if __name__ == "__main__":
    main()
