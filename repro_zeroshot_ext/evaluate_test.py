#!/usr/bin/env python3
"""Stage: test metrics, gated by P_selection.json AND its fingerprint linkage.

Verifies that P_selection.json was built from the exact same splits.json,
data zip, code, checkpoint and run config as the current state — a mere
existence check is not enough. Computes test metrics ONLY for the two
protocol configs: fixed P=1 and the per-variable selected P.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from record_fingerprint import RUN_CONFIG, verify_fingerprint
from select_period import load_npz, split_metrics
from timesx_data import find_zip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--splits", default=os.path.join(HERE, "splits.json"))
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--selection", default=os.path.join(HERE, "P_selection.json"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    if not os.path.isfile(args.selection):
        raise SystemExit(f"P_selection.json not found at {args.selection} — run select_period.py first")

    zip_path = find_zip(args.zip)

    # cache -> test linkage: the cache manifest must bind to current inputs/code
    manifest_path = os.path.join(args.cache_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"cache manifest not found: {manifest_path} — run infer.py first")
    manifest = json.load(open(manifest_path))
    verify_fingerprint(manifest.get("fingerprint", {}), zip_path, args.splits,
                       args.ckpt_dir, what="cache/manifest.json")

    # selection -> test linkage: same bindings PLUS selection/scoring code hashes
    sel = json.load(open(args.selection))
    verify_fingerprint(sel["fingerprint"], zip_path, args.splits, args.ckpt_dir,
                       what="P_selection.json linkage", require_selection=True)
    assert sel["fingerprint"]["run_config"] == dict(RUN_CONFIG), "run config drift"
    print("[test] linkage verified: cache manifest + P_selection (incl. "
          "selection/scoring code hashes) match current state")

    splits = json.load(open(args.splits))
    os.makedirs(args.out_dir, exist_ok=True)

    var_rows, sample_lines = [], []
    for var_key, entry in splits["variables"].items():
        s_entry = sel["variables"][var_key]
        selected = int(s_entry["selected_P"])
        per_config = {}
        for P in sorted({1, selected}):
            npz = load_npz(args.cache_dir, f"{var_key}__P{P}")
            rows, summary = split_metrics(npz, entry["test"], entry["calib_std"], "test")
            per_config[P] = {"rows": rows, "summary": summary}
            for m in rows:
                sample_lines.append(json.dumps({
                    "sample_id": m["sample_id"], "var_key": var_key,
                    "domain": entry["domain"], "P": P,
                    "raw_mse": m["raw_mse"], "raw_mae": m["raw_mae"],
                    "std_mse": m["std_mse"], "std_mae": m["std_mae"],
                    "denom_source": m["denom_source"], "std_defined": m["std_defined"]}))
        # both configs must score the identical test sample set
        ids1 = {m["sample_id"] for m in per_config[1]["rows"]}
        for P, cfg in per_config.items():
            assert {m["sample_id"] for m in cfg["rows"]} == ids1, \
                f"{var_key}: test sample set differs between P=1 and P={P}"
        assert len(ids1) == entry["n_test"], f"{var_key}: test row count mismatch"

        for P, cfg in per_config.items():
            s = cfg["summary"]
            var_rows.append({
                "domain": entry["domain"], "group": entry["group"], "var_key": var_key,
                "P": P, "is_selected": int(P == selected), "n_test": s["n"],
                "std_mse": s["std_mse"], "std_mae": s["std_mae"],
                "raw_mse": s["raw_mse"], "raw_mae": s["raw_mae"],
                "n_std_undefined": s["n_std_undefined"],
                "fallback_p1": int(entry["fallback_p1"]),
            })

    with open(os.path.join(args.out_dir, "test_variable_level.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(var_rows[0].keys()))
        w.writeheader()
        w.writerows(var_rows)
    with open(os.path.join(args.out_dir, "test_sample_level.jsonl"), "w") as f:
        f.write("\n".join(sample_lines) + "\n")

    n_pairs = len(var_rows)
    print(f"test metrics for {n_pairs} (var,config) pairs -> {args.out_dir}")
    print("variable-level CSV and sample-level JSONL written")


if __name__ == "__main__":
    main()
