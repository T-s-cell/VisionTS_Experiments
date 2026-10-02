#!/usr/bin/env python3
"""Stage 01: freeze sampling lists.

- TimesX: test-window sample_ids for the 18 selected vars (plotting) and for
  all 190 vars (high-dim check), aligned with cache/{var}__P1.npz.
- ImageNet: filenames sorted, then numpy default_rng(2021) choice of 2000
  without replacement. Honest label: random subset of ILSVRC2012 val, NOT
  class-stratified (no ground-truth labels available on either machine).
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DIRS, IMAGENET_N, IMAGENET_VAL_DIR, SEED, ensure_dirs

ZERO = os.path.dirname(HERE)


def main():
    ensure_dirs()
    sel_vars = {r["var_key"]: r for r in
                csv.DictReader(open(os.path.join(DIRS["selection"], "selected_vars.csv")))}

    def windows(var_key):
        d = __import__("numpy").load(os.path.join(ZERO, "cache", f"{var_key}__P1.npz"),
                                     allow_pickle=False)
        ids = [str(s) for s in d["sample_ids"]]
        sp = [str(s) for s in d["splits"]]
        return [(i, s) for i, s in zip(ids, sp) if s == "test"]

    n_sel = 0
    with open(os.path.join(DIRS["lists"], "timesx_windows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["var_key", "sel_group", "frequency", "sample_id"])
        for vk, r in sel_vars.items():
            rows = windows(vk)
            assert len(rows) == int(r["n_test"]), f"{vk}: {len(rows)} != {r['n_test']}"
            for sid, _ in rows:
                w.writerow([vk, r["sel_group"], r["frequency"], sid])
            n_sel += len(rows)
    assert n_sel == 117, n_sel

    n_all = 0
    # all 190 vars, deterministic order
    splits = json.load(open(os.path.join(ZERO, "splits.json")))
    with open(os.path.join(DIRS["lists"], "all_windows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["var_key", "sample_id"])
        for vk in sorted(splits["variables"]):
            for sid, _ in windows(vk):
                w.writerow([vk, sid])
                n_all += 1
    assert n_all == 1677, n_all

    files = sorted(f for f in os.listdir(IMAGENET_VAL_DIR) if f.endswith(".JPEG"))
    assert len(files) == 50000, f"expected 50000 val images, got {len(files)}"
    import numpy as np
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(files), size=IMAGENET_N, replace=False)
    chosen = [files[i] for i in sorted(pick)]
    assert len(set(chosen)) == IMAGENET_N
    man = {
        "seed": SEED,
        "rule": ("sorted filenames of ILSVRC2012 val (flat dir), "
                 "numpy.random.default_rng(2021).choice(50000, 2000, replace=False)"),
        "n_total": len(files),
        "n_selected": IMAGENET_N,
        "stratified": False,
        "note": "ImageNet validation random subset, NOT class-stratified "
                "(no ground-truth labels available; devkit not used per decision)",
        "val_dir": IMAGENET_VAL_DIR,
        "files": chosen,
    }
    with open(os.path.join(DIRS["lists"], "imagenet_manifest.json"), "w") as f:
        json.dump(man, f, indent=1)

    print(json.dumps({"selected_windows": n_sel, "all_windows": n_all,
                      "imagenet": IMAGENET_N}))


if __name__ == "__main__":
    main()
