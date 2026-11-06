"""EXP-041 step 3: joint t-SNE over the FULL 2,200 x 54528 matrix
(frozen 1,900 + new 300 Monash) — never append coords to the old run.

Checks before fitting:
  merged[:1900] == frozen EXP-002 features  element-for-element
  (np.array_equal; the frozen FILE sha was separately pinned at
  preflight/extract — array bytes vs file bytes are NOT conflated).
Parameters are EXP-002's run_tsne.py TSNE_PARAMS verbatim with
random_state=2021 (main).  The main fit is run TWICE and the two coord
matrices must be bitwise identical (determinism gate FG5); rs=2022/2023
runs provide the stability view.  CPU-bound, nohup-friendly.

Outputs (save_t41_monash/):
  tsne_coords_2200x2_rs<rs>.npy   float64 coords (rs=2021 saved twice as
                                  tsne_coords_2200x2.npy + _recheck)
  tsne_params.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_PKG = os.path.dirname(os.path.abspath(__file__))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import t41_common as X          # noqa: E402


def fit(feats, rs):
    import sklearn
    from sklearn.manifold import TSNE
    params = dict(X.TSNE_PARAMS)
    params["random_state"] = rs
    t0 = time.time()
    X.log("TSNE fit rs=%d params=%s" % (rs, params))
    emb = TSNE(**params).fit_transform(feats)
    X.log("TSNE rs=%d done in %.1f min" % (rs, (time.time() - t0) / 60.0))
    return emb.astype(np.float64), {"sklearn": sklearn.__version__,
                                    "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=int, nargs="*",
                    default=[X.SEED] + list(X.STABILITY_SEEDS))
    ap.add_argument("--no-recheck", action="store_true",
                    help="skip the rs2021 double-fit determinism check")
    a = ap.parse_args()

    fp = X.frozen_features_path()
    frozen = np.load(fp)
    assert frozen.shape == (1900, X.FEATURE_DIM), frozen.shape
    assert frozen.dtype == np.float32
    manifest = X.load_json(X.MANIFEST_JSON) \
        if os.path.isfile(X.MANIFEST_JSON) else {}
    smoke = str(manifest.get("mode", "")).startswith("SMOKE")
    mon = np.load(X.FEATURES_MONASH)
    assert mon.shape == ((X.N_MONASH if not smoke else len(
        manifest.get("samples", []))), X.FEATURE_DIM), mon.shape
    assert mon.dtype == np.float32
    assert np.isfinite(frozen).all() and np.isfinite(mon).all()
    merged = np.vstack([frozen, mon]).astype(np.float32)
    assert merged.shape == (1900 + mon.shape[0], X.FEATURE_DIM)
    assert np.array_equal(merged[:1900], frozen), \
        "merged[:1900] != frozen features (row drift)"
    X.log("merged %s (mode=%s); frozen rows verified element-identical"
          % (merged.shape, manifest.get("mode")))

    runs = {}
    for rs in a.rs:
        emb, info = fit(merged, rs)
        if rs == X.SEED:
            np.save(os.path.join(X.SAVE_BASE, "tsne_coords_2200x2.npy"), emb)
            if not a.no_recheck:
                emb2, info2 = fit(merged, rs)
                assert np.array_equal(emb, emb2), \
                    "FG5 FAIL: same-rs refit coords differ"
                np.save(os.path.join(X.SAVE_BASE,
                                     "tsne_coords_2200x2_recheck.npy"), emb2)
                info["recheck"] = info2
                info["recheck_bitwise_equal"] = True
        else:
            np.save(os.path.join(X.SAVE_BASE,
                                 "tsne_coords_2200x2_rs%d.npy" % rs), emb)
        runs["rs%d" % rs] = info

    params = {
        "params": dict(X.TSNE_PARAMS, random_state=X.SEED),
        "stability_random_states": list(X.STABILITY_SEEDS),
        "input_shape": list(merged.shape), "input_dtype": str(merged.dtype),
        "frozen_rows": [0, 1900], "monash_rows": [1900, 2200],
        "frozen_features_sha256": X.sha256_of(fp),
        "group_order": X.GROUPS,
        "group_spans": {g: list(v) for g, v in X.SPANS.items()},
        "note": "one joint refit over all 2,200 samples (the frozen 1,900 "
                "rows enter the fit; their EXP-002 coords are NOT reused); "
                "no pre-PCA, no per-group standardization; parameters are "
                "the EXP-002 supplementary settings.",
        "runs": runs, "finished": X.time_str(),
    }
    with open(os.path.join(X.SAVE_BASE, "tsne_params.json"), "w") as f:
        json.dump(params, f, indent=1)
        f.write("\n")
    X.log("DONE: coords for rs %s" % (a.rs,))


if __name__ == "__main__":
    main()
