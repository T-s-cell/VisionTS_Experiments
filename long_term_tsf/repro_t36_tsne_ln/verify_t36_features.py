"""EXP-036 step 3: verify extracted features.

  1. frozen condition features are BITWISE equal to the EXP-002 archived
     original (repro_fig7/outputs/features_1900x54528.npy on this tree);
  2. every condition: shape (1900, 54528) float32, all finite, sha256 in
     verify json matches the file on disk;
  3. input determinism: per-group per-batch input tensor hashes and the
     mask-noise hash are identical across all conditions;
  4. archived row order re-check (rebuilt meta rows == meta.csv, groups order).

--conds accepts a subset (min smoke checks frozen,ettm1_ln; main checks all).
Merges into logs/verify_t36.json section 'features'.
"""
import argparse
import os
import sys

import numpy as np

import t36_common as C


def fail(msg):
    raise SystemExit("[verify_t36_features] FAIL: %s" % msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default=",".join(C.COND_LIST))
    args = ap.parse_args()
    conds = args.conds.split(",")

    C.check_meta_rows()
    C.check_groups_order()

    sec = {"conds": conds, "row_order_ok": True, "per_cond": {}, "input_hashes_equal": None}

    for cond in conds:
        p = C.cond_features(cond)
        if not os.path.isfile(p):
            fail("missing features file: %s" % p)
        sha = C.sha256_of(p)
        feats = np.load(p)
        if feats.shape != (1900, C.FEATURE_DIM) or feats.dtype != np.float32:
            fail("%s shape/dtype %s %s != (1900, 54528) float32" % (cond, feats.shape, feats.dtype))
        if not np.isfinite(feats).all():
            fail("%s non-finite features" % cond)
        rec = {"sha256": sha, "shape": list(feats.shape), "finite": True}
        vjson = C.load_json(C.VERIFY_JSON) if os.path.isfile(C.VERIFY_JSON) else {}
        rec_sha = vjson.get("extract", {}).get(cond, {}).get("features_sha256")
        if rec_sha is not None and rec_sha != sha:
            fail("%s features sha drift vs verify json: %s != %s" % (cond, sha, rec_sha))
        if cond == "frozen":
            if not os.path.isfile(C.ORIG_FEATURES):
                fail("EXP-002 original features missing: %s" % C.ORIG_FEATURES)
            orig = np.load(C.ORIG_FEATURES)
            if orig.shape != feats.shape:
                fail("original features shape %s != %s" % (orig.shape, feats.shape))
            bitwise = bool(np.array_equal(orig, feats))
            md = float(np.abs(orig.astype(np.float64) - feats.astype(np.float64)).max())
            rec["orig_bitwise_equal"] = bitwise
            rec["orig_max_abs_diff"] = md
            rec["orig_sha256"] = C.sha256_of(C.ORIG_FEATURES)
            if not bitwise:
                fail("frozen features NOT bitwise vs EXP-002 original (max|d|=%.3e) "
                     "-> stop, keep the site" % md)
        sec["per_cond"][cond] = rec
        print("[verify_t36_features] %s OK (sha256 %s)" % (cond, sha))

    hashes = {}
    for cond in conds:
        p = C.cond_inputs_json(cond)
        if not os.path.isfile(p):
            fail("missing input hashes: %s" % p)
        hashes[cond] = C.load_json(p)
    ref = hashes[conds[0]]
    for cond in conds[1:]:
        if hashes[cond] != ref:
            for g in C.GROUPS:
                a = ref["groups"][g]["batch_sha256"]
                b = hashes[cond]["groups"][g]["batch_sha256"]
                if a != b:
                    fail("input hash mismatch %s vs %s group %s (first differing batch %d)"
                         % (conds[0], cond, g, next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)))
            fail("input hash mismatch %s vs %s (noise or structure)" % (conds[0], cond))
    sec["input_hashes_equal"] = True
    sec["noise_sha256"] = ref["noise_sha256"]

    data = C.load_json(C.VERIFY_JSON) if os.path.isfile(C.VERIFY_JSON) else {}
    data["features"] = sec
    C.dump_json(C.VERIFY_JSON, data)
    print("[verify_t36_features] ALL PASS for %s (inputs identical across conditions)"
          % conds)


if __name__ == "__main__":
    main()
