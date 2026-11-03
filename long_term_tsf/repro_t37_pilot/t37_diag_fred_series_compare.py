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
    diff = per_aut - per_cur                # >0: author env HIGHER MAE (worse)
    n = len(diff)
    mae_cur = float(per_cur.mean())
    mae_aut = float(per_aut.mean())
    total_diff = mae_aut - mae_cur
    # a series shifts the DATASET MAE by diff[i]/n (mean over series);
    # its share of the net diff compares that shift against total_diff
    contrib = diff / n
    order = np.argsort(-np.abs(diff))
    top = [{"series_idx": int(i),
            "mae_current": float(per_cur[i]), "mae_author": float(per_aut[i]),
            "diff": float(diff[i]),
            "contribution_to_total_diff": float(contrib[i]),
            "share_of_net_diff_pct": float(100.0 * contrib[i] / total_diff) if total_diff else 0.0}
           for i in order[:15]]
    decomp = {"mae_current": mae_cur, "mae_author": mae_aut,
              "total_diff": total_diff,
              "mean_per_series_diff": float(diff.mean()),
              "decomposition_check_abs": abs(total_diff - float(diff.mean())),
              "n_series_author_better": int((diff < 0).sum()),
              "n_series_author_worse": int((diff > 0).sum()),
              "n_series_author_mae_lower": int((diff < 0).sum()),
              "n_series_author_mae_higher": int((diff > 0).sum()),
              "n_series_bitwise_equal": int((diff == 0).sum()),
              "pos_contrib_sum": float(np.maximum(contrib, 0).sum()),
              "neg_contrib_sum": float(np.minimum(contrib, 0).sum())}
    t0 = top[0]
    decomp["top1_series_idx"] = t0["series_idx"]
    decomp["top1_contribution"] = t0["contribution_to_total_diff"]
    decomp["top1_share_of_net_diff_pct"] = t0["share_of_net_diff_pct"]
    decomp["others_net_offset"] = total_diff - t0["contribution_to_total_diff"]
    out = {
        "truth_bitwise_equal": truth_eq,
        "preds_bitwise_equal": preds_eq,
        "anchor": C.AUTHOR_ANCHOR["fred_md"]["mae"],
        "rel_diff_current_vs_anchor": abs(mae_cur - C.AUTHOR_ANCHOR["fred_md"]["mae"]) / C.AUTHOR_ANCHOR["fred_md"]["mae"],
        "rel_diff_author_vs_anchor": abs(mae_aut - C.AUTHOR_ANCHOR["fred_md"]["mae"]) / C.AUTHOR_ANCHOR["fred_md"]["mae"],
        "attribution_note": "the WHOLE software env was swapped (python/torch/torchvision/"
                            "timm/numpy/pandas/gluonts/einops); individual deps are not isolated",
        "decomposition": decomp,
        "top15_by_abs_diff": top,
        "per_series_mae_current": per_cur.tolist(),
        "per_series_mae_author": per_aut.tolist(),
    }
    C.dump_json(os.path.join(C.LOG_DIR, "fred_series_compare.json"), out)
    print("[t37][fred-compare] truth_bitwise=%s preds_bitwise=%s" % (truth_eq, preds_eq))
    print("[t37][fred-compare] mae current %.6f author %.6f (diff %+.4f); anchor %.6f"
          % (mae_cur, mae_aut, total_diff, C.AUTHOR_ANCHOR["fred_md"]["mae"]))
    print("[t37][fred-compare] author env MAE LOWER on %d series, HIGHER on %d, bitwise-equal %d"
          % (decomp["n_series_author_mae_lower"], decomp["n_series_author_mae_higher"],
             decomp["n_series_bitwise_equal"]))
    for t in top[:10]:
        print("[t37][fred-compare]   series %3d: cur %10.2f -> aut %10.2f (diff %+.2f, contrib %+.2f, %.2f%% of net)"
              % (t["series_idx"], t["mae_current"], t["mae_author"], t["diff"],
                 t["contribution_to_total_diff"], t["share_of_net_diff_pct"]))


if __name__ == "__main__":
    main()
