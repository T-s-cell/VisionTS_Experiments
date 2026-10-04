#!/usr/bin/env python3
"""Stage 02 — CPU t-SNE (EXP-009): 38 independent joint fits.

Each fit: X = concat([imagenet(1000), timesx(n_d)]) -> ONE TSNE fit with the
frozen directive parameters; before (Z0) / after (F0_s2021) fit independently,
no coordinate locking. No preprocessing (no pooling/L2/PCA).

Thread limits are set at the very top (before numpy/sklearn) so every worker
stays single-threaded; parallelism (--jobs) is chosen from the small-trial
measured per-process RSS/CPU, never assumed.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "exp009_common", os.path.join(HERE, "common.py"))
E9 = importlib.util.module_from_spec(_spec)
sys.modules["exp009_common"] = E9
_spec.loader.exec_module(E9)

LOG_PATH = os.path.join(E9.LOGS, "tsne.jsonl")
TSNE_PARAMS = {k: E9.CFG["tsne"][k] for k in (
    "n_components", "perplexity", "learning_rate", "init",
    "early_exaggeration", "max_iter", "random_state", "metric",
    "method", "angle")}


def log(row):
    E9.append_jsonl(LOG_PATH, {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **row})
    print(f"[tsne] {json.dumps(row, ensure_ascii=False)}", flush=True)


def feats_sha(name):
    p = os.path.join(E9.FEATURES, name)
    assert os.path.isfile(p), f"missing feature artifact: {name}"
    return E9.sha256_file(p)


def artifact_tag(state, domain):
    """Feature files carry the model tag: Z0 or F0__<domain>__s2021."""
    return state if state == E9.STATE_BEFORE else \
        f"F0__{domain}__s{E9.F0_SEED}"


def fit_key(domain, state):
    """Resume key: frozen params + exact input artifact hashes."""
    tag = artifact_tag(state, domain)
    comps = {"domain": domain, "state": state, "tsne": TSNE_PARAMS,
             "flat_dim": E9.FLAT_DIM,
             "imagenet_sha": feats_sha(f"{tag}__imagenet.npz"),
             "timesx_sha": feats_sha(f"{tag}__timesx.npz")}
    blob = json.dumps(comps, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest(), comps


def load_xy(state, domain):
    tag = artifact_tag(state, domain)
    img = np.load(os.path.join(E9.FEATURES, f"{tag}__imagenet.npz"),
                  allow_pickle=False)
    tsx = np.load(os.path.join(E9.FEATURES, f"{tag}__timesx.npz"),
                  allow_pickle=False)
    X_img = img["feats"]
    fnames = [str(s) for s in img["filenames"]]
    dom = np.array([str(s) for s in tsx["domains"]])
    mask = dom == domain
    X_tsx = tsx["feats"][mask]
    sids = [str(s) for s in tsx["sample_ids"][mask]]
    vks = [str(s) for s in tsx["var_keys"][mask]]
    assert X_img.shape == (E9.IMAGENET_N, E9.FLAT_DIM)
    assert X_tsx.shape[0] == len(sids) > 0
    return X_img, fnames, X_tsx, sids, vks


def run_fit(task):
    domain, state = task
    out_csv = os.path.join(E9.COORDS, f"{domain}__{state}.csv")
    out_par = os.path.join(E9.COORDS, f"{domain}__{state}.params.json")
    key, comps = fit_key(domain, state)
    if (os.path.isfile(out_csv) and os.path.isfile(out_par)
            and json.load(open(out_par)).get("fit_key") == key):
        return {"domain": domain, "state": state, "event": "skip"}

    import sklearn
    from sklearn.manifold import TSNE

    t0 = time.time()
    X_img, fnames, X_tsx, sids, vks = load_xy(state, domain)
    X = np.concatenate([X_img, X_tsx], axis=0)
    assert np.isfinite(X).all(), f"{domain}/{state}: non-finite t-SNE input"
    ts = TSNE(verbose=1, **TSNE_PARAMS)
    emb = ts.fit_transform(X)
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    rows = []
    for i, fn in enumerate(fnames):
        rows.append([fn, "imagenet", domain, "", state,
                     repr(float(emb[i, 0])), repr(float(emb[i, 1]))])
    for j in range(len(sids)):
        i = E9.IMAGENET_N + j
        rows.append([sids[j], "timesx", domain, vks[j], state,
                     repr(float(emb[i, 0])), repr(float(emb[i, 1]))])
    assert len(rows) == X.shape[0]
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "source", "domain", "var_key",
                    "model_state", "x", "y"])
        w.writerows(rows)
    os.replace(tmp, out_csv)
    E9.atomic_write_json(
        {"domain": domain, "state": state, "tsne": TSNE_PARAMS,
         "kl_divergence": float(ts.kl_divergence_),
         "n_iter": int(ts.n_iter_), "n_imagenet": len(fnames),
         "n_timesx": len(sids), "sklearn_version": sklearn.__version__,
         "threads": "1 per process", "fit_key": key,
         "inputs": {k: v for k, v in comps.items()
                    if k.endswith("_sha")},
         "wall_secs": round(time.time() - t0, 1),
         "peak_rss_mib": round(rss_mib, 1)},
        out_par)
    return {"domain": domain, "state": state, "event": "fit",
            "n": int(X.shape[0]), "kl": float(ts.kl_divergence_),
            "n_iter": int(ts.n_iter_), "secs": round(time.time() - t0, 1),
            "peak_rss_mib": round(rss_mib, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel fits; choose from measured RSS/CPU of the "
                         "small trial (each worker is single-threaded)")
    ap.add_argument("--domains", default=None, help="comma list (default all)")
    ap.add_argument("--states", default=f"{E9.STATE_BEFORE},{E9.STATE_AFTER}")
    args = ap.parse_args()

    domains = args.domains.split(",") if args.domains else E9.DOMAIN_NAMES
    assert set(domains) <= set(E9.DOMAIN_NAMES)
    states = args.states.split(",")
    tasks = [(d, s) for d in domains for s in states]
    log({"event": "plan", "fits": len(tasks), "jobs": args.jobs,
         "note": "wall times are estimates; parallelism set from small trial"})

    done = 0
    if args.jobs <= 1:
        for t in tasks:
            r = run_fit(t)
            log(r)
            done += 1
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_fit, t): t for t in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                log(r)
                done += 1
    log({"event": "tsne_complete", "fits": done})


if __name__ == "__main__":
    main()
