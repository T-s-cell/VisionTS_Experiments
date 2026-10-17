"""Cached-array verification (runs after run_infer, before replay).

Per group:
  - npz keys complete; window_index == 0..N-1 exactly once; N == splits.json
  - identity (s / orig_row / label_dates) bitwise == protocol recompute in
    manifest order
  - truth bitwise == protocol recompute AND bitwise == TaTS reference npz
  - pred finite; metrics json present with matching npz sha
Exit 1 on any failure; writes verify_report.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import bitwise_equal, dump_json, \
    load_json, setup_deterministic_env, sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData

NPZ_KEYS = ["pred", "truth", "s", "orig_row", "label_dates",
            "window_index"]


def verify_group(gdir, dom, h, dd, tats_npz=None):
    errs = []
    npz = gdir / P.test_npz_name(dom, h)
    if not npz.exists():
        return ["missing %s" % npz.name]
    arr = dict(np.load(npz))
    for k in NPZ_KEYS:
        if k not in arr:
            errs.append("missing key %s" % k)
    if errs:
        return errs
    n = arr["pred"].shape[0]
    idx = arr["window_index"]
    if not np.array_equal(np.sort(idx), np.arange(n)):
        errs.append("window_index not a 0..n-1 permutation")
    sp = dd.load_split(h, "test")
    if n != len(sp["X"]):
        errs.append("count %d != protocol %d" % (n, len(sp["X"])))
        return errs
    if not bitwise_equal(arr["s"], sp["s"]):
        errs.append("s mismatch vs protocol")
    if not bitwise_equal(arr["orig_row"], sp["orig_row"]):
        errs.append("orig_row mismatch vs protocol")
    if not bitwise_equal(arr["label_dates"], sp["label_dates"]):
        errs.append("label_dates mismatch vs protocol")
    if not bitwise_equal(arr["truth"], sp["Y"]):
        errs.append("truth mismatch vs protocol recompute")
    if not np.isfinite(arr["pred"].astype(np.float64)).all():
        errs.append("non-finite pred")
    if tats_npz is not None:
        t = dict(np.load(tats_npz))
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
    ap.add_argument("--tats_ref", default=None,
                    help="dir with TaTS test_arrays.npz files; '' disables")
    ap.add_argument("--groups", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())

    tats_ref = args.tats_ref
    if tats_ref is None:
        tats_ref = str(P.tats_ref_dir())
    use_tats = bool(tats_ref) and Path(tats_ref).exists()

    groups = P.all_groups()
    if args.groups:
        want = set(args.groups.split(","))
        groups = [g for g in groups if P.group_id(*g) in want]

    report = {"use_tats_ref": use_tats, "groups": {}}
    n_err = 0
    domcache = {}
    for dom, h in groups:
        gid = P.group_id(dom, h)
        if dom not in domcache:
            domcache[dom] = DomainData(pinfo["protocol_dir"],
                                       pinfo["splits"], dom)
        tats_npz = None
        if use_tats:
            for cand in (Path(tats_ref) / gid / "test_arrays.npz",
                         Path(tats_ref) / ("%s_seed2021" % gid) /
                         "test_arrays.npz"):
                if cand.exists():
                    tats_npz = cand
                    break
        errs = verify_group(out_root / "runs" / gid, dom, h,
                            domcache[dom], tats_npz)
        report["groups"][gid] = {"errors": errs,
                                 "tats_npz": str(tats_npz) if tats_npz
                                 else None}
        if errs:
            n_err += 1
            print("FAIL %s: %s" % (gid, "; ".join(errs)), flush=True)
        else:
            print("OK %s%s" % (gid, " (incl TaTS truth)" if tats_npz
                               else ""), flush=True)

    report["n_groups"] = len(groups)
    report["n_failed"] = n_err
    dump_json(out_root / "verify_report.json", report)
    print("VERIFY_CACHE: %d/%d groups ok" % (len(groups) - n_err,
                                             len(groups)))
    if n_err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
