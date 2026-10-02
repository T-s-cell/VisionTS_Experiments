#!/usr/bin/env python3
"""Stage 00: freeze the sampling lists.

- TimesX: the SAME 117 test windows as featvis (no re-ranking, no re-picking);
  asserted set-identical to featvis/lists/timesx_windows.csv and the 18
  variables/groups identical to featvis/selection/selected_vars.csv.
- ImageNet: the LTFS experiment's frozen 1000 files IN THEIR ORIGINAL ORDER,
  read from repro_fig7/outputs/sampling_lists/ImageNet.json (read-only);
  every file must exist under IMAGENET_VAL_DIR.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (DIRS, FEATVIS_DIR, IMAGENET_VAL_DIR, ensure_dirs,
                    load_fig7_imagenet_json)


def main():
    ensure_dirs()

    fv_sel = list(csv.DictReader(
        open(os.path.join(FEATVIS_DIR, "selection", "selected_vars.csv"))))
    assert len(fv_sel) == 18
    groups = {r["var_key"]: r["sel_group"] for r in fv_sel}
    freqs = {r["var_key"]: r["frequency"] for r in fv_sel}
    assert set(groups.values()) == {"best", "middle", "worst"}
    for g in ("best", "middle", "worst"):
        assert sum(1 for v in groups.values() if v == g) == 6

    fv_win = list(csv.DictReader(
        open(os.path.join(FEATVIS_DIR, "lists", "timesx_windows.csv"))))
    assert len(fv_win) == 117
    rows = [{"var_key": r["var_key"], "sel_group": r["sel_group"],
             "frequency": r["frequency"], "sample_id": r["sample_id"]}
            for r in fv_win]
    assert all(r["sel_group"] == groups[r["var_key"]]
               and r["frequency"] == freqs[r["var_key"]] for r in rows)
    assert len({r["sample_id"] for r in rows}) == 117

    out = os.path.join(DIRS["lists"], "timesx_windows.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["var_key", "sel_group", "frequency",
                                          "sample_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"timesx_windows: 117 rows (== featvis) -> {out}")

    files, md5, rule = load_fig7_imagenet_json()
    missing = [f_ for f_ in files
               if not os.path.isfile(os.path.join(IMAGENET_VAL_DIR, f_))]
    assert not missing, f"{len(missing)} fig7 ImageNet files missing on this host"
    man = {"source": "repro_fig7/outputs/sampling_lists/ImageNet.json",
           "source_sha256": md5, "index_rule": rule, "n": len(files),
           "val_dir": IMAGENET_VAL_DIR, "filenames": files}
    out = os.path.join(DIRS["lists"], "imagenet_list.json")
    with open(out, "w") as f:
        json.dump(man, f, indent=1)
    print(f"imagenet_list: {len(files)} files in LTFS original order "
          f"(sha256 {md5[:16]}) -> {out}")


if __name__ == "__main__":
    main()
