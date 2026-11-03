"""EXP-037 task A gates. Reads logs/monash_<ds>.json (+ naive_<ds>.json)
and appends a 'monash' section to logs/verify_t37.json. Exit 0 only if
every present subset passes all gates; a missing/crashed subset is a FAIL.

Gates per subset:
  G1 pred_len runtime check OK and == raw-benchmark horizon (12 / 56)
  G2 raw .tsf meta really carried forecast_horizon == that value (i.e. the
     prediction length did NOT come from the frequency fallback)
  G3 autotune accepted P == paper Table 6 periodicity (1 / 7)
  G4 VisionTS MAE vs author precomputed anchor, rel diff <= 1e-3
     (MSE recorded alongside; MAE is the gate)
  G5 data-version evidence present (raw download sha256 + cache diff)
  G6 naive N1 MAE vs paper Table 15 Naive, rel diff <= 0.05
     (window-alignment check; N2 recorded alongside)
  G7 ratios VisionTS/N1 and VisionTS/N2 present and finite
"""
import os
import sys

import numpy as np

import t37_common as C


def check(ds, gates):
    rec_p = C.monash_json(ds)
    if not os.path.isfile(rec_p):
        gates.append((ds, "G0", "FAIL", "monash json missing (runner crashed?)"))
        return None
    rec = C.load_json(rec_p)
    expect = C.SUBSETS[ds]

    got_pred = rec.get("split", {}).get("meta_prediction_length")
    ok = rec.get("pred_len_check") == "OK" and got_pred == expect["pred_len"]
    gates.append((ds, "G1_pred_len", "PASS" if ok else "FAIL",
                  "got %s want %d" % (got_pred, expect["pred_len"])))

    gen = rec.get("generation", {})
    ok = gen.get("has_forecast_horizon") is True and \
        gen.get("forecast_horizon") == expect["pred_len"]
    gates.append((ds, "G2_raw_horizon", "PASS" if ok else "FAIL",
                  "has_fh=%s fh=%s freq=%s recipe=%s%s" %
                  (gen.get("has_forecast_horizon"),
                   gen.get("forecast_horizon"),
                   gen.get("raw_frequency"),
                   gen.get("recipe_orig_module", "?"),
                   " (eval-local gen!)" if gen.get("recipe_orig_is_eval_local_gen")
                   else "")))

    acc = rec.get("autotune", {}).get("accepted_p")
    ok = acc == expect["p"]
    gates.append((ds, "G3_autotune_P", "PASS" if ok else "FAIL",
                  "accepted %s want %d (events %s)" %
                  (acc, expect["p"],
                   [(e["p"], round(e["valid_mae"], 6), e["accept"])
                    for e in rec.get("autotune", {}).get("events", [])])))

    mae = rec.get("result_csv", {}).get("metrics", {}).get("MAE[0.5]")
    mse = rec.get("result_csv", {}).get("metrics", {}).get("MSE[mean]")
    if mae is None:
        gates.append((ds, "G4_vision_anchor", "FAIL", "no MAE in result csv"))
    else:
        anchor = C.AUTHOR_ANCHOR[ds]["mae"]
        rd = C.rel_diff(mae, anchor)
        ok = rd <= C.ANCHOR_TOL
        gates.append((ds, "G4_vision_anchor", "PASS" if ok else "FAIL",
                      "MAE %.6f vs anchor %.6f rel %.3e (tol %.0e); MSE %.6e vs %.6e"
                      % (mae, anchor, rd, C.ANCHOR_TOL,
                         mse if mse else float("nan"), C.AUTHOR_ANCHOR[ds]["mse"])))

    dv = rec.get("data_version", {})
    ok = bool(gen.get("download_sha256")) and gen.get("downloaded_bytes", 0) > 0
    gates.append((ds, "G5_data_version", "PASS" if ok else "FAIL",
                  "raw %s B sha %s... | cache files new/changed %d" %
                  (gen.get("downloaded_bytes"), str(gen.get("download_sha256"))[:16],
                   dv.get("n_files_new_or_changed", -1))))

    nrec_p = C.naive_json(ds)
    if not os.path.isfile(nrec_p):
        gates.append((ds, "G6_naive_align", "FAIL", "naive json missing"))
        return rec
    nrec = C.load_json(nrec_p)
    n1_mae = nrec.get("n1", {}).get("metrics", {}).get("MAE[0.5]")
    align = nrec.get("n1_vs_paper_naive", {})
    ok = align.get("ok") is True and n1_mae is not None
    gates.append((ds, "G6_naive_align", "PASS" if ok else "FAIL",
                  "N1 %.6f vs paper Naive %.2f rel %.3e (tol %.2f); N2 %.6f" %
                  (n1_mae if n1_mae else float("nan"),
                   align.get("paper_naive_mae", float("nan")),
                   align.get("rel_diff", float("nan")), C.N1_ALIGN_TOL,
                   nrec.get("n2", {}).get("metrics", {}).get("MAE[0.5]",
                                                             float("nan")))))

    ratios = nrec.get("ratios", {})
    ok = np.isfinite(ratios.get("visionts_over_n1", np.nan)) and \
        np.isfinite(ratios.get("visionts_over_n2", np.nan))
    gates.append((ds, "G7_ratios", "PASS" if ok else "FAIL",
                  "V/N1 %.4f V/N2 %.4f" % (ratios.get("visionts_over_n1", np.nan),
                                           ratios.get("visionts_over_n2", np.nan))))
    return rec


def main():
    gates = []
    recs = {}
    for ds in C.SUBSETS:
        recs[ds] = check(ds, gates)

    all_pass = all(g[2] == "PASS" for g in gates)
    summary = {
        "all_pass": all_pass,
        "n_gates": len(gates),
        "n_pass": sum(1 for g in gates if g[2] == "PASS"),
        "gates": [{"subset": g[0], "gate": g[1], "verdict": g[2], "detail": g[3]}
                  for g in gates],
        "per_subset": {ds: {
            "pred_len": (recs[ds] or {}).get("split", {}).get("meta_prediction_length"),
            "n_series": (recs[ds] or {}).get("split", {}).get("n_series"),
            "n_series_info_moirai": C.SUBSETS[ds]["series_info"],
            "input_len_min_med_max": [(recs[ds] or {}).get("split", {}).get("input_len_min"),
                                      (recs[ds] or {}).get("split", {}).get("input_len_median"),
                                      (recs[ds] or {}).get("split", {}).get("input_len_max")],
            "autotune_p": (recs[ds] or {}).get("autotune", {}).get("accepted_p"),
            "visionts_mae": (recs[ds] or {}).get("result_csv", {}).get("metrics", {}).get("MAE[0.5]"),
            "visionts_mse": (recs[ds] or {}).get("result_csv", {}).get("metrics", {}).get("MSE[mean]"),
            "naive_n1_mae": C.load_json(C.naive_json(ds)).get("n1", {}).get("metrics", {}).get("MAE[0.5]")
            if os.path.isfile(C.naive_json(ds)) else None,
            "naive_n2_mae": C.load_json(C.naive_json(ds)).get("n2", {}).get("metrics", {}).get("MAE[0.5]")
            if os.path.isfile(C.naive_json(ds)) else None,
            "ratio_n1": C.load_json(C.naive_json(ds)).get("ratios", {}).get("visionts_over_n1")
            if os.path.isfile(C.naive_json(ds)) else None,
            "ratio_n2": C.load_json(C.naive_json(ds)).get("ratios", {}).get("visionts_over_n2")
            if os.path.isfile(C.naive_json(ds)) else None,
            "wall_sec_vision": (recs[ds] or {}).get("timing", {}).get("wall_sec"),
            "gpu_peak_mb_vision": (recs[ds] or {}).get("gpu_peak", {}).get("allocated_mb"),
        } for ds in C.SUBSETS},
    }
    C.merge_json(C.VERIFY_JSON, "monash", summary)
    for g in gates:
        print("[t37][A] %-28s %-22s %s %s" % (g[0], g[1], g[2], g[3]))
    print("[t37][A] ALL %s (%d/%d gates pass)"
          % ("PASS" if all_pass else "FAIL",
             summary["n_pass"], summary["n_gates"]))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
