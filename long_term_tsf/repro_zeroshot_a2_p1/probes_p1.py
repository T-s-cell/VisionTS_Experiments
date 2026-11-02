"""EXP-032 per-group probes (run right after each P=1 inference).
Three acceptance assertions per group:
  A. P actually enters the imaging pipeline: the pre-interpolation 2D
     segmentation shape recorded on the real forward path equals
     (B*nvars, 1, 1, ctx) for P=1 (vs (B*nvars, 1, P_orig, ctx/P_orig)
     for the baseline period); batch count == ceil(N/32); exactly one
     update_config record with periodicity=1, pad_left=pad_right=0.
  B. test windows and truth align with EXP-001: true.npy bitwise equal,
     window count == closed-form expectation.
  C. predictions actually changed vs baseline: max|pred-P1 - pred-base| > 0.
On pass writes <run_dir>/PROBES_OK.txt with the evidence summary.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2p1_common as C


def check_group(ds, h):
    c = C.DATASETS[ds]
    rd = C.p1_dir(ds, h)
    bd = C.base_dir(ds, h)
    problems, evid = [], {}

    # --- shape_record.jsonl ---
    sl = os.path.join(rd, "shape_record.jsonl")
    recs = []
    with open(sl) as f:
        for line in f:
            recs.append(json.loads(line))
    ucs = [r for r in recs if r["type"] == "update_config"]
    segs = [r for r in recs if r["type"] == "segmentation"]
    if len(ucs) != 1:
        problems.append("update_config records=%d (want 1)" % len(ucs))
    else:
        u = ucs[0]
        evid["update_config"] = {k: u[k] for k in
                                 ("periodicity", "context_len", "pred_len",
                                  "pad_left", "pad_right", "num_patch_input",
                                  "scale_x", "input_resize_size",
                                  "output_resize_size")}
        if u["periodicity"] != 1 or u["pad_left"] != 0 or u["pad_right"] != 0:
            problems.append("update_config not P=1/no-pad: %s" % evid["update_config"])
    n_exp = C.expected_n_windows(ds, h)
    n_batches = math.ceil(n_exp / 32)
    if len(segs) != n_batches:
        problems.append("segmentation records=%d want ceil(%d/32)=%d"
                        % (len(segs), n_exp, n_batches))
    for i, r in enumerate(segs):
        f_, shape = r["f"], r["out_shape"]
        # (b*nvars, 1, f, p); f=1, p=ctx (pad_left=0)
        if f_ != 1 or len(shape) != 4 or shape[1] != 1 or shape[2] != 1 \
                or shape[3] != c["ctx"]:
            problems.append("seg[%d] f=%s shape=%s != (N,1,1,%d)"
                            % (i, f_, shape, c["ctx"]))
            break
    evid["segmentation"] = {
        "n_records": len(segs), "first_shape": segs[0]["out_shape"] if segs else None,
        "expected_f_p": [1, c["ctx"]],
        "baseline_period_f_p": [c["period"], c["ctx"] // c["period"]]}

    # --- arrays vs baseline ---
    p1_pred = np.load(os.path.join(rd, "results", "_", "pred.npy"))
    p1_true = np.load(os.path.join(rd, "results", "_", "true.npy"))
    b_pred = np.load(os.path.join(bd, "results", "_", "pred.npy"))
    b_true = np.load(os.path.join(bd, "results", "_", "true.npy"))
    if p1_true.shape != b_true.shape:
        problems.append("true shape differs: %s vs %s"
                        % (p1_true.shape, b_true.shape))
    else:
        if not np.array_equal(p1_true, b_true):
            problems.append("true.npy NOT bitwise equal to EXP-001 baseline")
        else:
            evid["truth_bitwise_equal"] = True
    if p1_pred.shape != b_pred.shape:
        problems.append("pred shape differs: %s vs %s"
                        % (p1_pred.shape, b_pred.shape))
    else:
        d = float(np.max(np.abs(p1_pred.astype(np.float64)
                                - b_pred.astype(np.float64))))
        evid["pred_max_abs_diff_vs_baseline"] = d
        if not d > 0.0:
            problems.append("pred identical to baseline — P=1 had no effect")

    # --- window identity ---
    if p1_pred.shape[0] != n_exp:
        problems.append("N=%d want %d" % (p1_pred.shape[0], n_exp))
    evid["windows"] = {"n": int(p1_pred.shape[0]), "expected": n_exp,
                       "first_pred_abs_row0": C.first_pred_abs_row0(ds),
                       "horizons": [h]}

    if problems:
        return problems, evid
    C.dump_json(os.path.join(rd, "probes_evidence.json"), evid)
    with open(os.path.join(rd, "PROBES_OK.txt"), "w") as f:
        f.write("PROBES OK %s_%d  seg_shape=%s (f=%d p=%d)  truth_bitwise=OK  "
                "pred_diff=%.6g  windows=%d\n"
                % (ds, h, evid["segmentation"]["first_shape"],
                   evid["segmentation"]["expected_f_p"][0],
                   evid["segmentation"]["expected_f_p"][1],
                   evid["pred_max_abs_diff_vs_baseline"], n_exp))
    return [], evid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs=2, metavar=("DS", "H"), action="append")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    groups = C.group_ids() if args.all else \
        [(ds, int(h)) for ds, h in args.group]
    nfail = 0
    for ds, h in groups:
        problems, evid = check_group(ds, h)
        if problems:
            nfail += 1
            print("[probes] FAIL %s_%d" % (ds, h))
            for p in problems:
                print("  -", p)
        else:
            print("[probes] OK %s_%d  seg=%s  truth_bitwise  pred_diff=%.6g"
                  % (ds, h, evid["segmentation"]["first_shape"],
                     evid["pred_max_abs_diff_vs_baseline"]))
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
