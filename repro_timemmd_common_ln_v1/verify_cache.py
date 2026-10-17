"""Cached-array verification, three-way (runs after run_infer).

Per group: npz keys complete; window_index == 0..N-1 exactly once;
N == splits.json; identity (s/orig_row/label_dates) bitwise == protocol
recompute AND == EXP-027 runs npz; truth bitwise == protocol recompute
AND == TaTS reference npz AND == EXP-027 npz; pred finite; metrics json
present with matching npz sha. Exit 1 on any failure.
"""
import argparse
from pathlib import Path

import numpy as np

from repro_timemmd_common_ln_v1 import protocol as P
from repro_timemmd_common_ln_v1.common import bitwise_equal, dump_json, \
    load_json, setup_deterministic_env, sha256_of
from repro_timemmd_common_ln_v1.data_timemmd import DomainData

NPZ_KEYS = ["pred", "truth", "s", "orig_row", "label_dates", "window_index"]


def tats_path(gid):
    ref = P.tats_ref_dir()
    for cand in (ref / gid / "test_arrays.npz",
                 ref / ("%s_seed2021" % gid) / "test_arrays.npz"):
        if cand.exists():
            return cand
    return None


def verify_group(gdir, dom, h, dd, tats_npz, e27_npz):
    errs = []
    npz = gdir / P.test_npz_name(dom, h)
    if not npz.exists():
        return ["missing %s" % npz.name]
    arr = dict(np.load(str(npz)))
    for k in NPZ_KEYS:
        if k not in arr:
            errs.append("missing key %s" % k)
    if errs:
        return errs
    n = arr["pred"].shape[0]
    if not np.array_equal(np.sort(arr["window_index"]), np.arange(n)):
        errs.append("window_index not a 0..n-1 permutation")
    sp = dd.load_split(h, "test")
    if n != len(sp["X"]):
        errs.append("count %d != protocol %d" % (n, len(sp["X"])))
        return errs
    for k in ("s", "orig_row", "label_dates", "window_index"):
        if not bitwise_equal(arr[k], sp[k]):
            errs.append("%s mismatch vs protocol" % k)
    if not bitwise_equal(arr["truth"], sp["Y"]):
        errs.append("truth mismatch vs protocol recompute")
    if not np.isfinite(arr["pred"].astype(np.float64)).all():
        errs.append("non-finite pred")
    if e27_npz is not None and e27_npz.exists():
        a27 = dict(np.load(str(e27_npz)))
        for k in ("s", "orig_row", "label_dates", "window_index"):
            if k in a27 and not bitwise_equal(arr[k], a27[k]):
                errs.append("%s mismatch vs EXP-027 npz" % k)
        if "truth" in a27 and not bitwise_equal(arr["truth"],
                                                a27["truth"]):
            errs.append("truth mismatch vs EXP-027 npz")
    else:
        errs.append("EXP-027 npz missing")
    if tats_npz is not None:
        t = dict(np.load(str(tats_npz)))
        for k in ("truth", "s", "orig_row", "label_dates"):
            if k not in t:
                errs.append("tats npz missing %s" % k)
                continue
            a, b = arr[k], t[k]
            if a.ndim == 3 and b.ndim == 2:
                a = a[:, :, 0]
            if b.ndim == 3 and a.ndim == 2:
                b = b[:, :, 0]
            if not bitwise_equal(np.asarray(a), np.asarray(b)):
                errs.append("%s mismatch vs TaTS reference" % k)
    else:
        errs.append("TaTS reference npz missing")
    mj = gdir / ("metrics__%s.json" % P.group_id(dom, h))
    if not mj.exists():
        errs.append("missing metrics json")
    else:
        m = load_json(mj)
        if m.get("npz_sha256") != sha256_of(npz):
            errs.append("npz sha != metrics json")
        if m.get("evaluated_windows") != n:
            errs.append("metrics window count mismatch")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--groups", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())

    groups = P.all_groups()
    if args.groups:
        want = set(args.groups.split(","))
        groups = [g for g in groups if P.group_id(*g) in want]

    report = {"groups": {}}
    n_err = 0
    domcache = {}
    for dom, h in groups:
        gid = P.group_id(dom, h)
        if dom not in domcache:
            domcache[dom] = DomainData(pinfo["protocol_dir"],
                                       pinfo["splits"], dom)
        e27 = P.exp027_out() / "runs" / gid / P.test_npz_name(dom, h)
        errs = verify_group(out_root / "runs" / gid, dom, h,
                            domcache[dom], tats_path(gid), e27)
        report["groups"][gid] = {"errors": errs,
                                 "tats_npz": str(tats_path(gid)),
                                 "exp027_npz": str(e27)}
        if errs:
            n_err += 1
            print("FAIL %s: %s" % (gid, "; ".join(errs)), flush=True)
        else:
            print("OK %s (3-way truth+identity)" % gid, flush=True)

    report["n_groups"] = len(groups)
    report["n_failed"] = n_err
    dump_json(out_root / "verify_report.json", report)
    print("VERIFY_CACHE: %d/%d groups ok" % (len(groups) - n_err,
                                             len(groups)))
    if n_err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
