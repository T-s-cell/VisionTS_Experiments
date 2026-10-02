#!/usr/bin/env python3
"""Acceptance checks over frozen splits + produced results (stdlib only).

  A. splits.json internal consistency: sample_id uniqueness, val/test window
     disjointness (real dates), frozen totals.
  B. sample-level predictions: every test sample exactly one prediction per
     applicable config; no duplicates; no unknown ids.
  C. variable-level summaries: exactly one row per variable per reporting
     config; n_test sums to the frozen total.
  D. zero-variance handling census (denominator sources).
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


splits = json.load(open(os.path.join(HERE, "splits.json")))
variables = splits["variables"]

# A. splits internal
ids_all = Counter()
for vk, e in variables.items():
    seen = Counter()
    for row in e["val"] + e["test"]:
        seen[row["sample_id"]] += 1
        ids_all[row["sample_id"]] += 1
    for row in e["calib_remaining_ids"] + e["cross_boundary_removed_ids"]:
        seen[row] += 1
    if dup := [k for k, c in seen.items() if c > 1]:
        fails.append(f"dup ids in {vk}: {dup}")
test_starts = {vk: min(r["future_start"] for r in e["test"]) for vk, e in variables.items()}
overlap = [vk for vk, e in variables.items()
           if any(r["future_end"] >= test_starts[vk] for r in e["val"])]
check("A1 val/test windows disjoint (real dates)", not overlap, f"violations={overlap[:5]}")
check("A2 frozen totals", sum(e["n_val"] for e in variables.values()) == 4736
      and sum(e["n_test"] for e in variables.values()) == 1677
      and sum(e["n_cross_boundary_removed"] for e in variables.values()) == 242
      and sum(int(e["fallback_p1"]) for e in variables.values()) == 0)
check("A3 sample_ids globally unique", len(ids_all) == sum(len(e["val"]) + len(e["test"]) for e in variables.values()),
      f"unique={len(ids_all)}")

# B. sample-level predictions
test_ids = {vk: {r["sample_id"] for r in e["test"]} for vk, e in variables.items()}
sel = json.load(open(os.path.join(HERE, "P_selection.json")))
per_cfg = defaultdict(lambda: defaultdict(list))
unknown = []
for line in open(os.path.join(HERE, "results", "test_sample_level.jsonl")):
    r = json.loads(line)
    key = (r["var_key"], r["P"])
    per_cfg[key][r["sample_id"]].append(r)
    if r["sample_id"] not in test_ids[r["var_key"]]:
        unknown.append(r)
bad_counts, bad_dups = [], []
for (vk, P), ids in per_cfg.items():
    want = test_ids[vk]
    if set(ids) != want:
        bad_counts.append((vk, P, len(ids), len(want)))
    if any(len(v) > 1 for v in ids.values()):
        bad_dups.append((vk, P))
check("B1 each test sample exactly one prediction per applicable config",
      not bad_counts, f"mismatches={bad_counts[:5]}")
check("B2 no duplicate (var,P,sample_id) rows", not bad_dups)
check("B3 all prediction ids belong to frozen test set", not unknown)
expect_cfgs = sum(2 if int(s["selected_P"]) != 1 else 1 for s in sel["variables"].values())
check("B4 (var,config) pair count", len(per_cfg) == expect_cfgs, f"{len(per_cfg)} vs {expect_cfgs}")

# C. variable level
with open(os.path.join(HERE, "results", "test_variable_level.csv")) as f:
    rows = list(csv.DictReader(f))
p1 = [r for r in rows if r["P"] == "1"]
selr = [r for r in rows if r["is_selected"] == "1"]
check("C1 P=1 summary: exactly one row per variable", len(p1) == 190 == len({r['var_key'] for r in p1}))
check("C2 selected summary: exactly one row per variable", len(selr) == 190 == len({r['var_key'] for r in selr}))
check("C3 n_test sums == 1677 (both configs)",
      sum(int(r["n_test"]) for r in p1) == 1677 and sum(int(r["n_test"]) for r in selr) == 1677)
check("C4 identical test samples across P=1 and selected (n_test equal per var)",
      all(r1["n_test"] == rs["n_test"] for r1, rs in zip(
          sorted(p1, key=lambda r: r["var_key"]), sorted(selr, key=lambda r: r["var_key"]))))

# D. zero-variance handling
src = Counter(json.loads(l)["denom_source"] for l in
              open(os.path.join(HERE, "results", "test_sample_level.jsonl")))
undef = sum(1 for l in open(os.path.join(HERE, "results", "test_sample_level.jsonl"))
            if not json.loads(l)["std_defined"])
print(f"[INFO] denominator sources: {dict(src)}; std_undefined rows: {undef}")

print("\n" + ("ALL ACCEPTANCE CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
