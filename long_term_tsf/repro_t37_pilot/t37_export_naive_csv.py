"""EXP-037 pack fix (user verdict 2026-11-04): the old naive CSVs shared one
filename per subset, so nn5's N1 row was overwritten by N2. Export N1 and N2
into distinct files from the retained logs/naive_<ds>.json (official column
set, identical to the VisionTS CSVs), then remove the ambiguous
naive_monash_<ds>.csv. CPU-only, idempotent; safe to re-run.
Run: python t37_export_naive_csv.py  (any env with numpy/json)
"""
import csv
import os

import t37_common as C

COLUMNS = ["dataset", "MSE[mean]", "MAE[0.5]", "sMAPE[0.5]",
           "MASE[0.5]", "ND[0.5]", "NRMSE[mean]"]


def main():
    for ds in C.SUBSETS:
        rec = C.load_json(C.naive_json(ds))
        rows = {}
        for tag in ("n1", "n2"):
            path = os.path.join(C.SAVE_A_BASE, "naive_%s_monash_%s.csv" % (tag, ds))
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                w.writeheader()
                w.writerow({"dataset": ds,
                            **{k: repr(float(v)) for k, v in rec[tag]["metrics"].items()}})
            rows[tag] = path
        old = C.naive_csv(ds)
        if os.path.isfile(old):
            os.remove(old)
        print("[t37] exported %s / %s (removed ambiguous %s)" %
              (rows["n1"], rows["n2"], os.path.basename(old)))


if __name__ == "__main__":
    main()
