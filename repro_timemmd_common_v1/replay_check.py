"""Independent-process replay of all 36 test groups + comparator negative
control. This module IS the independent verifier: run it as a separate
process after run_infer (same weights, same config, fresh process).

  - reruns every group -> replay__{Domain}__H{h}.npz (distinct names)
  - compares pred vs cached test npz with atol=1e-6, rtol=1e-5
    (user-corrected policy: tolerance, not bitwise; out-of-tolerance is a
    hard failure to investigate, never auto-masked)
  - identity arrays must match exactly
  - immutability: sha256 of every cached test npz unchanged across replay
  - --perturb-verify: perturbed copy of replay preds MUST fail the
    comparator (negative control that the comparator can detect faults)
Exit 0 all pass; exit 2 mismatch; exit 3 negative control not detected.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_visionts, configure, \
    predict_split
from repro_timemmd_common_v1.common import bitwise_equal, dump_json, \
    identity_equal, load_json, pred_close, seed_everything, \
    setup_deterministic_env, sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData


def compare_group(replay_npz, cached_npz, atol, rtol):
    """Returns dict verdict. Cached npz is the reference; never written."""
    r = dict(np.load(replay_npz))
    c = dict(np.load(cached_npz))
    close, max_abs, max_viol = pred_close(r["pred"], c["pred"], atol, rtol)
    ident = all(identity_equal(r[k], c[k])
                for k in ("s", "orig_row", "label_dates", "window_index"))
    truth_ok = bitwise_equal(r["truth"], c["truth"])
    return {"max_abs_diff": max_abs, "max_tol_violation": max_viol,
            "pred_close": close, "identity_equal": ident,
            "truth_bitwise_equal": truth_ok,
            "pass": bool(close and ident and truth_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--perturb_verify", action="store_true",
                    help="after replay, verify the comparator FAILS on a "
                         "perturbed copy (negative control)")
    ap.add_argument("--perturb", type=float, default=0.01)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())
    runs = out_root / "runs"

    sel_path = out_root / "selected_p.json"
    selected = {d: v["selected"] for d, v in
                load_json(sel_path)["selection"].items()}

    groups = P.all_groups()
    if args.groups:
        want = set(args.groups.split(","))
        groups = [g for g in groups if P.group_id(*g) in want]

    hashes_before = {P.group_id(d, h):
                     sha256_of(runs / P.group_id(d, h) /
                               P.test_npz_name(d, h))
                     for d, h in groups}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_visionts()
    model.to(dev)
    domcache = {}
    report = {"atol": P.ATOL, "rtol": P.RTOL, "groups": {}}
    n_bad = 0
    t0 = time.time()
    for dom, h in groups:
        gid = P.group_id(dom, h)
        if dom not in domcache:
            domcache[dom] = DomainData(pinfo["protocol_dir"],
                                       pinfo["splits"], dom)
        dd = domcache[dom]
        configure(model, dd.L, h, selected[dom])
        sp = dd.load_split(h, "test")
        seed_everything(P.SEED)
        pred = predict_split(model, sp["X"], dev, batch=P.BATCH)
        pred = pred[:, :, None] if pred.ndim == 2 else pred
        rnpz = runs / gid / P.replay_npz_name(dom, h)
        np.savez_compressed(rnpz, pred=pred, truth=sp["Y"], s=sp["s"],
                            orig_row=sp["orig_row"],
                            label_dates=sp["label_dates"],
                            window_index=sp["window_index"])
        v = compare_group(rnpz, runs / gid / P.test_npz_name(dom, h),
                          P.ATOL, P.RTOL)
        report["groups"][gid] = v
        if not v["pass"]:
            n_bad += 1
            print("FAIL %s: %s" % (gid, v), flush=True)
        else:
            print("REPLAY OK %s max|d|=%.2e" % (gid, v["max_abs_diff"]),
                  flush=True)

    hashes_after = {P.group_id(d, h):
                    sha256_of(runs / P.group_id(d, h) /
                              P.test_npz_name(d, h))
                    for d, h in groups}
    report["cached_npz_immutable"] = hashes_before == hashes_after
    report["replay_seconds"] = round(time.time() - t0, 1)
    report["n_failed"] = n_bad
    report["all_pass"] = bool(n_bad == 0 and report["cached_npz_immutable"])
    dump_json(out_root / "replay_report.json", report)

    if not report["all_pass"]:
        print("REPLAY: FAIL (n_bad=%d immutable=%s)"
              % (n_bad, report["cached_npz_immutable"]))
        raise SystemExit(2)

    if args.perturb_verify:
        gid0 = P.group_id(*groups[0])
        dom0, h0 = groups[0]
        r = dict(np.load(runs / gid0 / P.replay_npz_name(dom0, h0)))
        r["pred"] = r["pred"] + args.perturb
        pnpz = runs / gid0 / ("replay_perturb__%s__H%d.npz" % (dom0, h0))
        np.savez_compressed(pnpz, **r)
        v = compare_group(pnpz, runs / gid0 / P.test_npz_name(dom0, h0),
                          P.ATOL, P.RTOL)
        report["negative_control"] = v
        dump_json(out_root / "replay_report.json", report)
        if v["pass"]:
            print("[COMPARATOR BROKEN] perturbation not detected")
            raise SystemExit(3)
        print("NEGATIVE CONTROL OK (perturbation %.3g detected, max|d|="
              "%.2e)" % (args.perturb, v["max_abs_diff"]))

    print("REPLAY ALL PASS: %d groups, %.1fs, cached npz immutable"
          % (len(groups), report["replay_seconds"]))


if __name__ == "__main__":
    main()
