"""EXP-037 G4 follow-up (user verdict 2026-11-04, task 1): decompose the
FRED-MD MAE difference between the current env and the author-pinned env
into per-series contributions. CPU-only; consumes the two npz files written
by t37_diag_fred_author_env.py.
Writes logs/fred_series_compare.json.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t37_common as C


def main():
    cur = np.load(os.path.join(C.LOG_DIR, "fred_series_current.npz"))
    aut = np.load(os.path.join(C.LOG_DIR, "fred_series_author.npz"))
    truth_eq = bool(np.array_equal(cur["truth"], aut["truth"]))
    assert truth_eq, "truth arrays differ across envs — data path NOT identical, abort"
    preds_eq = bool(np.array_equal(cur["preds"], aut["preds"]))

    per_cur = cur["abs_err"].mean(axis=1)   # (107,)
    per_aut = aut["abs_err"].mean(axis=1)
    diff = per_aut - per_cur                # >0: author env WORSE on this series
    mae_cur = float(per_cur.mean())
    mae_aut = float(per_aut.mean())
    order = np.argsort(-np.abs(diff))
    top = [{"series_idx": int(i),
            "mae_current": float(per_cur[i]), "mae_author": float(per_aut[i]),
            "diff": float(diff[i]),
            "share_of_total_diff_pct": float(100.0 * diff[i] / (mae_aut - mae_cur))}
           for i in order[:15]]
    decomp = {"mae_current": mae_cur, "mae_author": mae_aut,
              "total_diff": mae_aut - mae_cur,
              "mean_per_series_diff": float(diff.mean()),
              "decomposition_check_abs": abs((mae_aut - mae_cur) - float(diff.mean())),
              "n_series_author_better": int((diff < 0).sum()),
              "n_series_author_worse": int((diff > 0).sum()),
              "n_series_bitwise_equal": int((diff == 0).sum())}
    out = {
        "truth_bitwise_equal": truth_eq,
        "preds_bitwise_equal": preds_eq,
        "anchor": C.AUTHOR_ANCHOR["fred_md"]["mae"],
        "rel_diff_current_vs_anchor": abs(mae_cur - C.AUTHOR_ANCHOR["fred_md"]["mae"]) / C.AUTHOR_ANCHOR["fred_md"]["mae"],
        "rel_diff_author_vs_anchor": abs(mae_aut - C.AUTHOR_ANCHOR["fred_md"]["mae"]) / C.AUTHOR_ANCHOR["fred_md"]["mae"],
        "decomposition": decomp,
        "top15_by_abs_diff": top,
        "per_series_mae_current": per_cur.tolist(),
        "per_series_mae_author": per_aut.tolist(),
    }
    C.dump_json(os.path.join(C.LOG_DIR, "fred_series_compare.json"), out)
    print("[t37][fred-compare] truth_bitwise=%s preds_bitwise=%s" % (truth_eq, preds_eq))
    print("[t37][fred-compare] mae current %.6f author %.6f (diff %+.4f); anchor %.6f"
          % (mae_cur, mae_aut, mae_aut - mae_cur, C.AUTHOR_ANCHOR["fred_md"]["mae"]))
    print("[t37][fred-compare] author better on %d series, worse on %d, bitwise-equal %d"
          % (decomp["n_series_author_better"], decomp["n_series_author_worse"],
             decomp["n_series_bitwise_equal"]))
    for t in top[:10]:
        print("[t37][fred-compare]   series %3d: cur %10.2f -> aut %10.2f (diff %+.2f, %.1f%% of total)"
              % (t["series_idx"], t["mae_current"], t["mae_author"], t["diff"],
                 t["share_of_total_diff_pct"]))


if __name__ == "__main__":
    main()
