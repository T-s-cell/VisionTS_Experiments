#!/usr/bin/env python3
"""Independent metric recomputation (standalone pass) of THIS run's
predictions (VTS-NEW).

Deliberately does NOT import or reuse evaluate.py's ladder code: reads the
shard npz files directly (pred + saved target), recomputes the ladder with
the SAME documented fixed order (shard asc, CSV row order within a shard;
sequences ascending Idx), and compares against evaluate.py's stored results
at the DOUBLE tolerance rtol=atol=1e-12: an entry passes iff
    |got - ref| <= atol + rtol * |ref|.
(identical reduction order should give bitwise equality; the tolerance only
absorbs library-level differences). Reports the maximum absolute/relative
difference. Writes results/recompute_check.json."""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (HERE,):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

import protocol as P

RTOL = 1e-12
ATOL = 1e-12


def recompute_subset(ds):
    seq_m = {}
    order = []
    for shard in range(P.n_shards(ds)):
        npz_path = P.shard_paths(ds, shard)[0]
        with np.load(npz_path) as z:
            idx = z["idx"]
            pred = z["pred"].astype(np.float64)
            tgt = z["target"].astype(np.float64)
        for i in range(len(idx)):
            ix = int(idx[i])
            d = pred[i] - tgt[i]
            item = (float(np.mean(d ** 2)), float(np.mean(np.abs(d))))
            if ix not in seq_m:
                seq_m[ix] = []
                order.append(ix)
            seq_m[ix].append(item)
    seqs = []
    for ix in sorted(seq_m):
        a = np.array(seq_m[ix], dtype=np.float64)
        seqs.append((ix, a[:, 0].mean(), a[:, 1].mean()))
    mse = float(np.mean([s[1] for s in seqs]))
    mae = float(np.mean([s[2] for s in seqs]))
    return mse, mae, seqs


def close(got, ref):
    got, ref = float(got), float(ref)
    ad = abs(got - ref)
    rd = ad / abs(ref) if ref != 0 else ad
    return (ad <= ATOL + RTOL * abs(ref)), ad, rd


def main():
    rep = {"rtol": RTOL, "atol": ATOL, "subsets": {}}
    all_ok, max_ad = True, 0.0
    for ds in P.ORDER:
        mse, mae, seqs = recompute_subset(ds)
        with open(os.path.join(P.RESULTS_DIR, f"{ds}_metrics.json")) as f:
            ref = json.load(f)["vts_new"]
        with open(os.path.join(P.RESULTS_DIR,
                               f"sequence_level_{ds}.csv")) as f:
            r = list(csv.reader(f))[1:]
        ref_seq = {int(row[0]): (float(row[2]), float(row[3])) for row in r}
        seq_bad, seq_max_ad = 0, 0.0
        for ix, m, a in seqs:
            for got, expect in ((m, ref_seq[ix][0]), (a, ref_seq[ix][1])):
                ok, ad, _ = close(got, expect)
                seq_max_ad = max(seq_max_ad, ad)
                if not ok:
                    seq_bad += 1
        m_ok, m_ad, m_rd = close(mse, ref["mse_seq_equal_weight"])
        a_ok, a_ad, a_rd = close(mae, ref["mae_seq_equal_weight"])
        ok = m_ok and a_ok and seq_bad == 0 and \
            len(seqs) == ref["n_sequences"]
        all_ok &= ok
        max_ad = max(max_ad, m_ad, a_ad, seq_max_ad)
        rep["subsets"][ds] = {
            "mse": {"got": mse, "ref": ref["mse_seq_equal_weight"],
                    "abs_diff": m_ad, "rel_diff": m_rd, "ok": bool(m_ok)},
            "mae": {"got": mae, "ref": ref["mae_seq_equal_weight"],
                    "abs_diff": a_ad, "rel_diff": a_rd, "ok": bool(a_ok)},
            "sequence_entries_mismatched": seq_bad,
            "max_seq_abs_diff": seq_max_ad,
            "n_sequences": len(seqs),
            "ok": bool(ok),
        }
        print(f"[recompute] {ds}: dMSE_abs={m_ad:.3e} dMAE_abs={a_ad:.3e} "
              f"seq_mismatch={seq_bad} -> {'OK' if ok else 'FAIL'}")
    rep["max_abs_diff_overall"] = max_ad
    rep["ok"] = bool(all_ok)
    C_json = os.path.join(P.RESULTS_DIR, "recompute_check.json")
    with open(C_json + ".tmp", "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(C_json + ".tmp", C_json)
    print(f"[recompute] max abs diff overall = {max_ad:.3e} "
          f"(double tolerance rtol=atol={ATOL:.0e}) -> "
          f"{'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
