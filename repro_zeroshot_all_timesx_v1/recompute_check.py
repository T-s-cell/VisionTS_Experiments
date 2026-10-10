#!/usr/bin/env python3
"""Independent recomputation of the three-level aggregation.

Deliberately shares NO code with the pipeline: imports neither common/protocol/
evaluate/metrics_util/aggregate nor the old experiment modules. Reads only the
frozen artifacts, rewrites the formulas from scratch, and hard-checks:
  - per-window identity/order/coverage vs the frozen inventory
  - per-window target EXACT equality with the frozen target array
  - per-window d vs recomputed std(frozen past, ddof=0) incl. fallback rule
  - variable metrics recomputed from pred/target
  - domain / overall aggregated FROM THE RECOMPUTED variable metrics
    (equal weight, None-safe) vs the published tables
Every deviation is a FAILURE: the result JSON records pass=false and the
process exits nonzero. Tolerances: 1e-9 (variable), 1e-12 (domain/overall/d).
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-8
TOL_D = 1e-12
TOL_VAR = 1e-9
TOL_AGG = 1e-12
TOTAL_EXPECT = 8106
DOMS = {
    "CropsAndStaples", "EnergyAndFuels", "LivestockAndFoodProducts",
    "RawMaterialsAndConstruction", "SpecialtyAndAdvancedMaterials",
    "StrategicAndHighValueMaterials", "Currency",
    "arts", "climate", "economy", "electronic_technology", "finance", "pets",
    "public_health", "public_policy", "science", "shopping", "society", "traffic",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=os.path.join(HERE, "frozen", "inventory.json"))
    ap.add_argument("--arrays", default=os.path.join(HERE, "frozen", "frozen_arrays.npz"))
    ap.add_argument("--vars", default=os.path.join(HERE, "frozen", "vars.json"))
    ap.add_argument("--preds", default=os.path.join(HERE, "results", "preds.jsonl.gz"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "recompute_check.json"))
    args = ap.parse_args()

    t0 = time.time()
    fails = []
    inv = json.load(open(args.inventory))
    windows = inv["windows"]
    var_calib = {v["var_key"]: v["calib_std"] for v in json.load(open(args.vars))}
    arr = np.load(args.arrays)
    past, target = arr["past"], arr["target"]

    def fail(msg):
        fails.append(msg)
        if len(fails) <= 20:
            print(f"[recompute] FAIL: {msg}", file=sys.stderr)

    # ---- structural: flat_index identity, order, coverage
    flat_ok = bool((arr["flat_index"] == np.arange(len(windows))).all())
    if not flat_ok:
        fail("frozen flat_index array is not 0..T-1 in order")
    if past.shape[0] != TOTAL_EXPECT or target.shape[0] != TOTAL_EXPECT:
        fail(f"frozen arrays have {past.shape[0]} rows != {TOTAL_EXPECT}")

    win_d_max = 0.0
    var_acc = {}
    seen = set()
    n_rows = 0
    for line_no, line in enumerate(gzip.open(args.preds, "rt")):
        r = json.loads(line)
        n_rows += 1
        fi = r["flat_index"]
        if not (0 <= fi < len(windows)):
            fail(f"row {line_no}: flat_index {fi} out of range")
            continue
        w = windows[fi]
        if (r["sample_id"] != w["sample_id"] or r["var_key"] != w["var_key"]
                or r["domain"] != w["domain"]):
            fail(f"row {line_no}: identity mismatch vs inventory "
                 f"({r['var_key']}/{r['sample_id']}/{r['domain']} vs "
                 f"{w['var_key']}/{w['sample_id']}/{w['domain']})")
        if fi in seen:
            fail(f"row {line_no}: duplicate flat_index {fi}")
        seen.add(fi)
        pastv = past[fi]
        d_new = float(pastv.std(ddof=0))
        if d_new < EPS:
            d_new = float(var_calib[r["var_key"]])
            src = "calib_std"
        else:
            src = "window"
        if src != r["denom_source"]:
            fail(f"row {line_no} ({r['sample_id']}): denom_source "
                 f"{r['denom_source']} != recomputed {src}")
        dev = abs(d_new - r["d"]) / max(1e-12, abs(d_new))
        win_d_max = max(win_d_max, dev)
        if dev > TOL_D:
            fail(f"row {line_no} ({r['sample_id']}): d deviation {dev:.3e} > {TOL_D:g} "
                 f"(recomputed {d_new!r} vs stored {r['d']!r})")
        # target must EXACTLY equal the frozen array
        tv = np.asarray(r["target"], dtype=np.float64)
        if tv.shape != target[fi].shape or not np.array_equal(tv, target[fi]):
            fail(f"row {line_no} ({r['sample_id']}): target != frozen target array")
        if len(r["pred"]) != 12 or not np.isfinite(np.asarray(r["pred"], dtype=np.float64)).all():
            fail(f"row {line_no} ({r['sample_id']}): pred invalid")
        diffs = [p - t for p, t in zip(r["pred"], r["target"])]
        raw_mse = sum(x * x for x in diffs) / len(diffs)
        raw_mae = sum(abs(x) for x in diffs) / len(diffs)
        rec = var_acc.setdefault(r["var_key"], {"domain": r["domain"], "n": 0, "sm": 0.0,
                                                "sa": 0.0, "rm": 0.0, "ra": 0.0, "undef": 0})
        rec["n"] += 1
        if d_new >= EPS:
            rec["sm"] += raw_mse / (d_new * d_new)
            rec["sa"] += raw_mae / d_new
        else:
            rec["undef"] += 1
        rec["rm"] += raw_mse
        rec["ra"] += raw_mae

    if n_rows != TOTAL_EXPECT:
        fail(f"preds rows {n_rows} != {TOTAL_EXPECT}")
    if len(seen) != TOTAL_EXPECT:
        missing = len(windows) - len(seen)
        fail(f"{missing} inventory windows missing from preds")
    if set(var_acc) != set(var_calib):
        fail(f"variable set mismatch: only-in-preds "
             f"{sorted(set(var_acc) - set(var_calib))[:5]} only-in-vars "
             f"{sorted(set(var_calib) - set(var_acc))[:5]}")

    # ---- variable level: own recompute vs published
    var_rows = []
    for vk, rec in var_acc.items():
        n = rec["n"]
        var_rows.append({"var_key": vk, "domain": rec["domain"], "n": n,
                         "n_std_undefined": rec["undef"],
                         "raw_mse": rec["rm"] / n, "raw_mae": rec["ra"] / n,
                         "std_mse": rec["sm"] / n if rec["undef"] == 0 else None,
                         "std_mae": rec["sa"] / n if rec["undef"] == 0 else None})

    published = {r["var_key"]: r for r in csv.DictReader(
        open(os.path.join(HERE, "results", "variable_level.csv")))}
    var_dev = 0.0
    for rec in var_rows:
        vk = rec["var_key"]
        pub = published.get(vk)
        if pub is None:
            fail(f"variable {vk} missing from published table")
            continue
        for f, tol in (("std_mse", TOL_VAR), ("std_mae", TOL_VAR),
                       ("raw_mse", TOL_VAR), ("raw_mae", TOL_VAR)):
            a = rec[f]
            b = float(pub[f]) if pub[f] not in ("", None) else None
            if (a is None) != (b is None):
                fail(f"{vk}.{f}: recomputed {a!r} vs published {b!r}")
                continue
            if a is None:
                continue
            dev = abs(a - b) / max(1e-12, abs(b))
            var_dev = max(var_dev, dev)
            if dev > tol:
                fail(f"{vk}.{f}: recomputed {a!r} vs published {b!r} dev {dev:.2e} > {tol:g}")
    if set(published) - set(var_acc):
        fail(f"published table has {sorted(set(published) - set(var_acc))[:5]} extra variables")

    # ---- domain / overall aggregated FROM THE RECOMPUTED variable metrics
    def aggregate_none_safe(rows):
        by_dom = defaultdict(list)
        for r in rows:
            by_dom[r["domain"]].append(r)
        if set(by_dom) != DOMS:
            fail(f"domain coverage mismatch: {set(by_dom) ^ DOMS}")
        dm, da = [], []
        for d in sorted(by_dom):
            vs = by_dom[d]
            ms = [r["std_mse"] for r in vs if r["std_mse"] is not None]
            as_ = [r["std_mae"] for r in vs if r["std_mae"] is not None]
            if ms:
                dm.append(sum(ms) / len(ms))
            if as_:
                da.append(sum(as_) / len(as_))
        return (sum(dm) / len(dm) if dm else None, sum(da) / len(da) if da else None,
                {d: (sum([r["std_mse"] for r in by_dom[d] if r["std_mse"] is not None]) /
                     max(1, len([r for r in by_dom[d] if r["std_mse"] is not None])),
                     sum([r["std_mae"] for r in by_dom[d] if r["std_mae"] is not None]) /
                     max(1, len([r for r in by_dom[d] if r["std_mae"] is not None])))
                 for d in by_dom})

    own_overall_m, own_overall_a, own_dom = aggregate_none_safe(var_rows)
    pub_domains = {r["domain"]: r for r in
                   csv.DictReader(open(os.path.join(HERE, "results", "domain_level.csv")))}
    agg_dev = 0.0
    for d, (m, a) in sorted(own_dom.items()):
        pd_ = pub_domains.get(d)
        if pd_ is None:
            fail(f"domain {d} missing from published domain table")
            continue
        for mine, theirs, f in ((m, float(pd_["std_mse"]) if pd_["std_mse"] else None, "std_mse"),
                                (a, float(pd_["std_mae"]) if pd_["std_mae"] else None, "std_mae")):
            if (mine is None) != (theirs is None):
                fail(f"domain {d}.{f}: recomputed {mine!r} vs published {theirs!r}")
                continue
            if mine is None:
                continue
            dev = abs(mine - theirs) / max(1e-12, abs(theirs))
            agg_dev = max(agg_dev, dev)
            if dev > TOL_AGG:
                fail(f"domain {d}.{f}: recomputed {mine!r} vs published {theirs!r} "
                     f"dev {dev:.2e} > {TOL_AGG:g}")
    if len(pub_domains) != len(DOMS):
        fail(f"published domain table has {len(pub_domains)} rows != {len(DOMS)}")

    ov = json.load(open(os.path.join(HERE, "results", "overall.json")))
    for mine, theirs, f in ((own_overall_m, ov.get("std_mse"), "std_mse"),
                            (own_overall_a, ov.get("std_mae"), "std_mae")):
        if (mine is None) != (theirs is None):
            fail(f"overall.{f}: recomputed {mine!r} vs published {theirs!r}")
            continue
        if mine is None:
            continue
        dev = abs(mine - theirs) / max(1e-12, abs(theirs))
        agg_dev = max(agg_dev, dev)
        if dev > TOL_AGG:
            fail(f"overall.{f}: recomputed {mine!r} vs published {theirs!r} "
                 f"dev {dev:.2e} > {TOL_AGG:g}")

    result = {
        "stage": "independent_recompute",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {os.path.basename(p): sha256(p) for p in
                   (args.inventory, args.arrays, args.vars, args.preds)},
        "flat_index_order_ok": flat_ok,
        "max_rel_dev": {"per_window_d": win_d_max, "variable_level": var_dev,
                        "domain_overall": agg_dev},
        "tolerances": {"per_window_d": TOL_D, "variable_level": TOL_VAR,
                       "domain_overall": TOL_AGG},
        "n_vars": len(var_rows),
        "n_rows": n_rows,
        "n_failures": len(fails),
        "failures": fails[:100],
        "pass": not fails,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    if fails:
        print(f"[recompute] FAILED with {len(fails)} deviations "
              f"(d_dev={win_d_max:.2e} var_dev={var_dev:.2e} agg_dev={agg_dev:.2e}) "
              f"-> {args.out}")
        sys.exit(1)
    print(f"[recompute] PASS: d_dev={win_d_max:.2e} var_dev={var_dev:.2e} "
          f"agg_dev={agg_dev:.2e} ({result['elapsed_s']}s)")
    os.makedirs(HERE + "/.state", exist_ok=True)

    def md5_local(p):
        h = hashlib.md5()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    # input guards: MUST stay identical to common.stage_input_guards("recompute")
    # (this script deliberately imports no pipeline code)
    input_guards = {}
    for rel in ("frozen/inventory.json", "frozen/frozen_arrays.npz", "frozen/vars.json",
                "results/preds.jsonl.gz", "results/variable_level.csv",
                "results/domain_level.csv", "results/overall.json"):
        input_guards[rel] = sha256(os.path.join(HERE, rel))
    input_guards["code_md5:recompute_check.py"] = md5_local(os.path.abspath(__file__))

    with open(os.path.join(HERE, ".state", "recompute.done.tmp"), "w") as f:
        json.dump({"status": "done",
                   "sha": {"results/recompute_check.json": sha256(args.out),
                           **{os.path.relpath(p, HERE): sha256(p) for p in
                              (args.inventory, args.arrays, args.vars, args.preds)}},
                   "input_guards": input_guards}, f, indent=1)
    os.replace(os.path.join(HERE, ".state", "recompute.done.tmp"),
               os.path.join(HERE, ".state", "recompute.done"))


if __name__ == "__main__":
    main()
