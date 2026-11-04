"""EXP-038 task A: Naive baselines on the EXACT same test windows as the
VisionTS run, using the same official harness objects (dataset split,
imputation, evaluator, metric list). Runs in the AUTHOR-pinned env
(python 3.8 compatible; no walrus, no 3.9+ syntax).

Definitions (unchanged from EXP-037, user-confirmed 2026-11-04):
  N1 "Naive"         = last observed (post-imputation) value repeated over
                       the horizon. PRIMARY denominator (Woo et al. 2024
                       report the Monash "Naive" from Godahewa et al. 2021,
                       where naive is last-value repetition).
  N2 "SeasonalNaive" = last season (period = the subset's autotuned P from
                       the VisionTS run record) tiled over the horizon;
                       kept alongside as sensitivity.

New for EXP-038 (user correction #2): N1 must round-match the paper's
printed Naive at the print's own precision (paper_round_match) — a HARD
gate for the two replay subsets (2825.67 / 8.26), flag-only elsewhere.
CIF code subsets have no direct paper row (the pooled row is verified in
verify_t38_monash.py G9).

Preconditions: the VisionTS runner json exists with no "error" key and the
autotuned P recorded; replay subsets additionally require replay_hard.pass.
The dataset cache is the one the runner generated (regenerate=False here);
file shas were archived in monash_<ds>.json data_version.
"""
import argparse
import inspect
import os
import sys
import time

import numpy as np

import t38_common as C


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    ds = args.dataset
    assert ds in C.MONASH_ORDER, "unknown subset %s" % ds

    os.makedirs(C.LOG_DIR, exist_ok=True)
    os.makedirs(C.SAVE_A_BASE, exist_ok=True)
    mrec = C.load_json(C.monash_json(ds))
    if "error" in mrec:
        raise SystemExit("[t38] refusing naive run: vision run errored for "
                         "%s" % ds)
    accepted = mrec.get("autotune", {}).get("accepted_p")
    if not accepted:
        raise SystemExit("[t38] refusing naive run: no accepted P in %s"
                         % C.monash_json(ds))
    if ds in C.REPLAY and not mrec.get("replay_hard", {}).get("pass"):
        raise SystemExit("[t38] refusing naive run: replay hard gate failed "
                         "for %s" % ds)
    p = int(accepted)

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

    import pandas as pd
    from gluonts.model.forecast import SampleForecast
    from gluonts.ev.metrics import MAE, MASE, MSE, ND, NRMSE, SMAPE
    from gluonts.model.evaluation import evaluate_forecasts

    # ---- gluonts 0.14.4 constructor/signature self-check ----------------
    _f0 = SampleForecast(samples=np.zeros((1, 4)),
                         start_date=pd.Period("2020-01-01", freq="D"))
    assert _f0.samples.shape == (1, 4)
    assert "test_data" in inspect.signature(evaluate_forecasts).parameters
    for _m in (MAE, MASE, MSE, ND, NRMSE, SMAPE):
        _m()

    pred_len = mrec["split"]["meta_prediction_length"]
    n_series = mrec["split"]["n_series"]

    # identical split + imputation as run.evaluate (:109-111)
    test_data, metadata = run.get_gluonts_test_dataset(ds)
    data_train = [run.imputation_nan(x["target"]) for x in test_data.input]
    data_test = [x["target"] for x in test_data.label]
    assert len(data_test[0]) == pred_len
    assert len(data_train) == n_series

    # N1: last imputed value repeated; N2: last season tiled (np.resize
    # tiles when len < pred_len)
    n1 = np.stack([np.full(pred_len, x[-1], dtype=np.float64)
                   for x in data_train])
    n2 = np.stack([np.resize(np.asarray(x[-p:], dtype=np.float64), pred_len)
                   for x in data_train])
    assert n1.shape == n2.shape == (n_series, pred_len)

    metrics = [MSE(), MAE(), SMAPE(), MASE(), ND(), NRMSE()]
    out = {"dataset": ds, "P": p, "pred_len": pred_len, "n_series": n_series,
           "env": mrec.get("env"),
           "start_type": str(type(next(iter(test_data.input))["start"])),
           "n1_def": "last imputed value repeated (Monash/Woo2024 Naive; "
                     "primary)",
           "n2_def": "last season (P=%d) tiled over horizon (sensitivity)" % p,
           "selfcheck": "SampleForecast(samples, start_date=Period) + "
                        "evaluate_forecasts(test_data=) OK"}
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
        csv_path = C.naive_n_csv(ds, tag)
        df.to_csv(csv_path, index=False)
        row = C.parse_csv_metrics(csv_path)[0]
        out[tag] = {"csv": csv_path, "metrics": row}
        log("%s MAE[0.5]=%.6f" % (tag, row["MAE[0.5]"]))
    out["timing"] = {"wall_sec": time.time() - t0}

    vm_mae = mrec["result_csv"]["metrics"]["MAE[0.5]"]
    out["ratios"] = {"visionts_over_n1": vm_mae / out["n1"]["metrics"]["MAE[0.5]"],
                     "visionts_over_n2": vm_mae / out["n2"]["metrics"]["MAE[0.5]"]}
    paper = C.PAPER_T15.get(ds)
    if paper is not None:
        n1_mae = out["n1"]["metrics"]["MAE[0.5]"]
        ok, quantum, want = C.paper_round_match(n1_mae, paper["naive"])
        rel = C.rel_diff(n1_mae, want)
        out["n1_vs_paper_naive"] = {
            "paper_naive": paper["naive"], "printed_float": want,
            "quantum": quantum, "round_match": ok, "rel_diff": rel,
            "align_ok": rel <= C.N1_ALIGN_TOL}
    else:
        out["n1_vs_paper_naive"] = {
            "paper_naive": None,
            "note": "no direct paper row (CIF pooled row handled in verify)"}
    out["finished"] = C.time_str()
    C.dump_json(C.naive_json(ds), out)

    if paper is not None:
        log("N1 round_match=%s (rel %.4f) | VisionTS/N1 %.4f"
            % (out["n1_vs_paper_naive"]["round_match"],
               out["n1_vs_paper_naive"]["rel_diff"],
               out["ratios"]["visionts_over_n1"]))
    else:
        log("N1 (no direct paper row) | VisionTS/N1 %.4f"
            % out["ratios"]["visionts_over_n1"])
    if ds in C.REPLAY:
        if not out["n1_vs_paper_naive"]["round_match"]:
            log("REPLAY N1 round-match FAIL")
            sys.exit(2)
    log("DONE")


if __name__ == "__main__":
    main()
