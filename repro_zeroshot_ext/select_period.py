#!/usr/bin/env python3
"""Stage: select period P per variable using ONLY validation rows.

Writes P_selection.json containing per-variable choices plus a fingerprint
block binding it to splits.json, the data zip, the code, the checkpoint and
the frozen run config (r=c=0.4 explicit). Test metrics are NOT computed here.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

from metrics_util import sample_metrics
from record_fingerprint import binding_diffs, make_fingerprint, verify_fingerprint
from timesx_data import find_zip


def load_npz(cache_dir, key):
    d = np.load(os.path.join(cache_dir, f"{key}.npz"), allow_pickle=False)
    return {k: d[k] for k in d.files}


def split_metrics(npz, rows, calib_std, split_name):
    """Aggregate sample_metrics over rows of one split; returns per-sample list + summary."""
    ids = [str(s) for s in npz["sample_ids"]]
    lab = [str(s) for s in npz["splits"]]
    pos = {}
    for i, (sid, sp) in enumerate(zip(ids, lab)):
        if sp == split_name:
            pos[sid] = i
    pred, true, win_std = npz["pred"], npz["true"], npz["win_std"]
    per_sample, agg = [], {"n": 0, "std_mse": [], "std_mae": [], "raw_mse": [],
                           "raw_mae": [], "n_std_undefined": 0, "denom_sources": {}}
    for r in rows:
        i = pos[r["sample_id"]]
        m = sample_metrics(pred[i].tolist(), true[i].tolist(), float(win_std[i]), calib_std)
        m["sample_id"] = r["sample_id"]
        per_sample.append(m)
        agg["n"] += 1
        if m["std_defined"]:
            agg["std_mse"].append(m["std_mse"])
            agg["std_mae"].append(m["std_mae"])
        else:
            agg["n_std_undefined"] += 1
        agg["denom_sources"][m["denom_source"]] = agg["denom_sources"].get(m["denom_source"], 0) + 1
        agg["raw_mse"].append(m["raw_mse"])
        agg["raw_mae"].append(m["raw_mae"])
    summary = {"n": agg["n"],
               "std_mse": (sum(agg["std_mse"]) / len(agg["std_mse"])) if agg["std_mse"] else None,
               "std_mae": (sum(agg["std_mae"]) / len(agg["std_mae"])) if agg["std_mae"] else None,
               "raw_mse": sum(agg["raw_mse"]) / len(agg["raw_mse"]),
               "raw_mae": sum(agg["raw_mae"]) / len(agg["raw_mae"]),
               "n_std_undefined": agg["n_std_undefined"],
               "denom_sources": agg["denom_sources"]}
    return per_sample, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--splits", default=os.path.join(HERE, "splits.json"))
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--out", default=os.path.join(HERE, "P_selection.json"))
    ap.add_argument("--force", action="store_true",
                    help="regenerate P_selection.json even when its recorded bindings changed")
    args = ap.parse_args()

    zip_path = find_zip(args.zip)
    splits = json.load(open(args.splits))

    # --- cache -> selection linkage: manifest must bind to current inputs/code ---
    manifest_path = os.path.join(args.cache_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"cache manifest not found: {manifest_path} — run infer.py first")
    manifest = json.load(open(manifest_path))
    verify_fingerprint(manifest.get("fingerprint", {}), zip_path, args.splits,
                       args.ckpt_dir, what="cache/manifest.json")
    missing = [f"{vk}__P{P}" for vk, e in splits["variables"].items()
               for P in ([1, 5] if e["group"] in ("CommodityPrice", "Currency") else [1, 52])
               if f"{vk}__P{P}" not in manifest.get("entries", {})]
    if missing:
        raise SystemExit(f"cache missing {len(missing)} (var,P) entries, e.g. {missing[:3]}; "
                         f"run infer.py first")
    print(f"[select] cache manifest verified: {len(manifest['entries'])} entries, "
          f"forward bindings match")

    # --- overwrite protection for the frozen selection record ---
    if os.path.isfile(args.out) and not args.force:
        old_fp = json.load(open(args.out)).get("fingerprint", {})
        diffs, _ = binding_diffs(old_fp, zip_path, args.splits, args.ckpt_dir,
                                 require_selection=True)
        legacy = diffs == ["code_md5_selection:missing(legacy P_selection — rerun select_period.py)"]
        if diffs and not legacy:
            raise SystemExit(f"{args.out} exists but its bindings changed: {diffs}; "
                             f"pass --force to regenerate deliberately")
        if legacy:
            print("[select] legacy P_selection.json (pre code_md5_selection) — regenerating")

    selections, tie_count, fallback_count = {}, 0, 0
    for var_key, entry in splits["variables"].items():
        cands = [1, 5] if entry["group"] in ("CommodityPrice", "Currency") else [1, 52]
        per_cand, per_sample_val = {}, {}
        for P in cands:
            npz = load_npz(args.cache_dir, f"{var_key}__P{P}")
            rows, summary = split_metrics(npz, entry["val"], entry["calib_std"], "val")
            per_cand[P] = summary
            per_sample_val[P] = rows
        if entry["fallback_p1"]:
            selected, reason = 1, "fallback_val_below_8"
            fallback_count += 1
        else:
            selected, reason = 1, "P1_lower_val_mse"
            for P in cands[1:]:
                if per_cand[P]["std_mse"] is None:
                    selected, reason = 1, "candidate_std_undefined"
                elif per_cand[P]["std_mse"] < per_cand[1]["std_mse"]:
                    selected, reason = P, "val_lower_std_mse"
                elif per_cand[P]["std_mse"] == per_cand[1]["std_mse"]:
                    selected, reason = 1, "tie_kept_P1"
                    tie_count += 1
        selections[var_key] = {
            "domain": entry["domain"], "group": entry["group"],
            "candidates": cands, "val_metrics": {str(P): per_cand[P] for P in cands},
            "selected_P": selected, "reason": reason, "fallback_p1": entry["fallback_p1"],
        }

    doc = {
        "protocol": splits["protocol"],
        "selection_rule": ("argmin val standardized MSE over candidates; tie -> P=1; "
                           "fallback to P=1 when val < 8 (flagged)"),
        "stats": {"n_vars": len(selections),
                  "by_selected_P": _count_by(selections, "selected_P"),
                  "by_reason": _count_by(selections, "reason"),
                  "fallback_vars": fallback_count, "exact_ties": tie_count},
        "fingerprint": make_fingerprint(zip_path, args.splits, args.ckpt_dir),
        "variables": selections,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(json.dumps(doc["stats"], indent=2))
    print("P_selection ->", args.out)


def _count_by(selections, field):
    out = {}
    for s in selections.values():
        k = s[field]
        out[str(k)] = out.get(str(k), 0) + 1
    return out


if __name__ == "__main__":
    main()
