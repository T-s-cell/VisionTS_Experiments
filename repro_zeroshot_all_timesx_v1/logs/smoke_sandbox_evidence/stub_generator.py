#!/usr/bin/env python3
"""Sandbox-only: fabricate an all-reuse reuse_final.json and zero-valued
cache_all/ for the fill set, so the S5 chain (assemble/evaluate/recompute/
acceptance) can be plumbing-tested locally without a GPU. Real runs never use
this. Predictions are zeros -> metrics are meaningless; the 1,677 pure-cache
reproduction check is REAL (uses the genuine old cache).
"""
import hashlib
import json
import os
import sys

import numpy as np

SMOKE = "/tmp/vz_smoke"
OLD_CACHE = "/home/wlt/MMTS/VisionTS_Experiments/repro_zeroshot_ext/cache"


def load(p):
    return json.load(open(p))


def dump(o, p):
    with open(p, "w") as f:
        json.dump(o, f, indent=1)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


os.chdir(SMOKE)
inv = load("frozen/inventory.json")
plan = load("reuse_plan.json")
reuse_vars = sorted(plan["variables"])
assert len(reuse_vars) == 190

covered_sids = set()
n_reuse = 0
for vk in reuse_vars:
    n_reuse += plan["variables"][vk]["n"]
    with np.load(os.path.join(OLD_CACHE, plan["variables"][vk]["file"]), allow_pickle=False) as z:
        covered_sids.update(str(s) for s in z["sample_ids"])
assert n_reuse == 6413

final = {
    "stage": "S3_preflight_final",
    "created": "smoke-stub",
    "variables": {vk: {"verdict": "reuse", "reason": "smoke stub"} for vk in reuse_vars},
    "totals": {"reuse_vars": 190, "reinfer_vars": 0,
               "reuse_windows": n_reuse, "fill_windows": 8106 - n_reuse},
}
dump(final, "reuse_final.json")

arrays = np.load("frozen/frozen_arrays.npz")
past_all, target_all = arrays["past"], arrays["target"]
flat_of = {w["sample_id"]: w["flat_index"] for w in inv["windows"]}
var_meta = {v["var_key"]: v for v in load("frozen/vars.json")}

by_var = {}
for w in inv["windows"]:
    if w["sample_id"] not in covered_sids:
        by_var.setdefault(w["var_key"], []).append(w)
n_fill = sum(len(v) for v in by_var.values())
print(f"stub fill windows: {n_fill}")

os.makedirs("cache_all", exist_ok=True)
entries = {}
geom = {"pad_left": 0, "pad_right": 0, "num_patch_input": 4, "num_patch": 14,
        "mask_ratio": 0.714285671710968}
for vk, rows in sorted(by_var.items()):
    rows = sorted(rows, key=lambda w: w["flat_index"])
    sids = [w["sample_id"] for w in rows]
    past_rows = np.stack([past_all[flat_of[s]] for s in sids])
    trues = np.stack([target_all[flat_of[s]] for s in sids])
    preds = np.zeros((len(sids), 12), dtype=np.float64)
    f = os.path.join("cache_all", f"{vk}__P1.npz")
    np.savez_compressed(f, sample_ids=np.array(sids), pred=preds, true=trues,
                        mu=past_rows.mean(axis=1), win_std=past_rows.std(axis=1, ddof=0),
                        calib_std=np.full(len(sids), float(var_meta[vk]["calib_std"])),
                        geom=np.array(json.dumps(geom)))
    entries[f"{vk}__P1"] = {"n": len(sids), "geom": geom,
                            "file": os.path.basename(f), "sha256": sha(f)}
dump({"fingerprint": {"stub": True}, "entries": entries}, "cache_all/manifest.json")
print("stub cache_all ready:", len(entries), "vars")
