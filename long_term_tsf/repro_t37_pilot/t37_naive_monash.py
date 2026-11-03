"""EXP-037 task A: Naive baselines on the EXACT same test windows as the
VisionTS run, using the same official harness objects (dataset split,
imputation, evaluator, metric list).

Definitions (user correction #4):
  N1 "Naive"       = last observed (post-imputation) value repeated over
                     the horizon. PRIMARY denominator: Woo et al. 2024
                     (MOIRAI) report the Monash "Naive" baseline from the
                     Godahewa et al. 2021 benchmark set, where naive is
                     last-value repetition. For FRED-MD (P=1) the paper's
                     own seasonality list contains only P=1, so N2 == N1.
  N2 "SeasonalNaive" = last season (period P, the subset's autotuned P
                     from the VisionTS run log) tiled over the horizon;
                     kept alongside as sensitivity.

Identical-window guarantees (code-level, same objects and functions):
  - same split: run.get_gluonts_test_dataset(<ds>) with the same cache the
    VisionTS runner generated (regenerate=False here; file shas already
    archived in monash_<ds>.json data_version);
  - same imputation: run.imputation_nan applied to the same test_data.input
    targets, exactly as run.evaluate does;
  - same truth/mask handling: forecasts wrapped as SampleForecast with the
    same start dates, then the same gluonts evaluate_forecasts() call with
    the same metric list as run.evaluate.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import t37_common as C


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--p", type=int, required=True,
                    help="autotuned P parsed from the VisionTS run record")
    args = ap.parse_args()
    ds = args.dataset
    p = args.p

    os.makedirs(C.LOG_DIR, exist_ok=True)
    os.makedirs(C.SAVE_A_BASE, exist_ok=True)
    mrec = C.load_json(C.monash_json(ds))
    assert mrec["pred_len_check"] == "OK" and mrec["p_check"] == "OK", \
        "refusing naive run: vision run gates %s|%s" % (mrec["pred_len_check"],
                                                        mrec["p_check"])

    os.chdir(C.EVAL_GLUONTS)
    # sys.path[0] is this package dir under script execution; make the
    # official harness modules the only candidates, and prove it.
    if C.EVAL_GLUONTS not in sys.path:
        sys.path.insert(0, C.EVAL_GLUONTS)
    import run
    import dataset as eval_dataset_mod
    assert os.path.samefile(run.__file__,
                            os.path.join(C.EVAL_GLUONTS, "run.py")), run.__file__
    assert os.path.samefile(eval_dataset_mod.__file__,
                            os.path.join(C.EVAL_GLUONTS, "dataset.py")), \
        eval_dataset_mod.__file__
    from gluonts.model.forecast import SampleForecast
    from gluonts.ev.metrics import MAE, MASE, MSE, ND, NRMSE, SMAPE
    from gluonts.model.evaluation import evaluate_forecasts

    pred_len = mrec["split"]["meta_prediction_length"]
    n_series = mrec["split"]["n_series"]

    # identical split + imputation as run.evaluate (lines 109-111)
    test_data, metadata = run.get_gluonts_test_dataset(ds)
    data_train = [run.imputation_nan(x["target"]) for x in test_data.input]
    data_test = [x["target"] for x in test_data.label]
    assert len(data_test[0]) == pred_len
    assert len(data_train) == n_series

    # N1: last imputed value repeated
    n1 = np.stack([np.full(pred_len, x[-1], dtype=np.float64) for x in data_train])
    # N2: last season tiled (np.resize tiles when len < pred_len)
    n2 = np.stack([np.resize(np.asarray(x[-p:], dtype=np.float64), pred_len)
                   for x in data_train])
    assert n1.shape == n2.shape == (n_series, pred_len)

    metrics = [MSE(), MAE(), SMAPE(), MASE(), ND(), NRMSE()]
    out = {"dataset": ds, "P": p, "pred_len": pred_len, "n_series": n_series,
           "n1_def": "last imputed value repeated (Monash/Woo2024 Naive; primary)",
           "n2_def": "last season (P=%d) tiled over horizon (sensitivity)" % p}
    t0 = time.time()
    for tag, values in (("n1", n1), ("n2", n2)):
        sample_forecasts = []
        for item, ts in zip(values, test_data.input):
            forecast_start_date = ts["start"] + len(ts["target"])
            sample_forecasts.append(
                SampleForecast(samples=np.reshape(item, (1, -1)),
                               start_date=forecast_start_date))
        df = evaluate_forecasts(sample_forecasts, test_data=test_data,
                                metrics=metrics)
        df.insert(loc=0, column="dataset", value=[ds])
        csv_path = C.naive_csv(ds)
        df.to_csv(csv_path, index=False)
        row = C.parse_csv_metrics(csv_path)[0]
        out[tag] = {"csv": csv_path, "metrics": row}
        log("%s MAE[0.5]=%.6f" % (tag, row["MAE[0.5]"]))
    out["timing"] = {"wall_sec": time.time() - t0}

    vm_mae = mrec["result_csv"]["metrics"]["MAE[0.5]"]
    out["ratios"] = {"visionts_over_n1": vm_mae / out["n1"]["metrics"]["MAE[0.5]"],
                     "visionts_over_n2": vm_mae / out["n2"]["metrics"]["MAE[0.5]"]}
    paper_naive = C.PAPER_T15[ds]["naive"]
    out["n1_vs_paper_naive"] = {
        "paper_naive_mae": paper_naive,
        "rel_diff": C.rel_diff(out["n1"]["metrics"]["MAE[0.5]"], paper_naive),
        "align_tol": C.N1_ALIGN_TOL,
        "ok": C.rel_diff(out["n1"]["metrics"]["MAE[0.5]"], paper_naive) <= C.N1_ALIGN_TOL,
    }
    out["finished"] = C.time_str()
    C.dump_json(C.naive_json(ds), out)
    log("N1 vs paper Naive rel_diff %.4f (ok=%s) | VisionTS/N1 %.4f"
        % (out["n1_vs_paper_naive"]["rel_diff"], out["n1_vs_paper_naive"]["ok"],
           out["ratios"]["visionts_over_n1"]))
    log("DONE")


if __name__ == "__main__":
    main()
