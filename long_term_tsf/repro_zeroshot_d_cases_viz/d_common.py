"""EXP-032 helpers: VisionTS Appendix D success/failure case visualization.
Case = ETTh1 / OT / L=2880 / H=96 / P=24. Reuses EXP-001 frozen zero-shot
arrays+weights and EXP-031 seasonal-baseline arrays and per-window npz.
Qualitative reproduction on NEW windows; win/loss judged against the two
seasonal baselines (paper Fig.14 judged against MOIRAI-Large instead).

Window selection (fixed rule, no fallback): align the three npz by
window_index / win_start / channel_names, rank by the UNROUNDED OT-column
win_var_mae, groups = VisionTS strictly better than BOTH baselines /
strictly worse than BOTH, sort by |vts - nearest baseline| then
window_index, take the LOWER MEDIAN at index (n-1)//2. Any mismatch with
the frozen expectations STOPS the run (investigate indexing/sort, never
substitute a different quantile).
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # long_term_tsf/
REPO = os.path.dirname(ROOT)
PKG = os.path.dirname(os.path.abspath(__file__))
SAVE_D = os.path.join(ROOT, "save_d_cases")
EXP001_BASE = os.path.join(ROOT, "save_repro_zeroshot", "ETTh1_96")
B4_BASE = os.path.join(ROOT, "save_b4_seasonal", "ETTh1_96")

DS = "ETTh1"
CTX, PRED_LEN, PERIOD = 2880, 96, 24
OT_IDX = 6
BATCH_SIZE = 32          # EXP-001 default (run.py L81)
FROZEN = {
    "success": {"n_cand": 2046, "win": 1480,
                "mae": {"visionts": 0.0938, "naive": 0.1334, "avg": 0.5072}},
    "failure": {"n_cand": 61, "win": 2663,
                "mae": {"visionts": 0.2378, "naive": 0.2028, "avg": 0.1858}},
}
WIN_START0 = 16 * 720    # 11520, absolute csv row of window 0's first step
HIST_SHOW = 336          # display span of recent history (standardized)

FP_FILES = [
    "models/VisionTS.py", "visionts/model.py", "visionts/models_mae.py",
    "visionts/util.py", "data_provider/data_loader.py",
    "data_provider/data_factory.py", "exp/exp_long_term_forecasting.py",
    "run.py", "utils/metrics.py",
]


def win_npz_paths():
    return {
        "visionts": os.path.join(B4_BASE, "win_visionts_ETTh1_96.npz"),
        "naive": os.path.join(B4_BASE, "naive", "win_ETTh1_96.npz"),
        "avg": os.path.join(B4_BASE, "avg", "win_ETTh1_96.npz"),
    }


def frozen_arrays_paths():
    return {
        "visionts": os.path.join(EXP001_BASE, "results", "_"),
        "naive": os.path.join(B4_BASE, "naive", "results", "_"),
        "avg": os.path.join(B4_BASE, "avg", "results", "_"),
    }


def load_win_mae():
    """Three npz -> dict method -> win_var_mae[:, OT] as float64 (values of
    the f32 array), after alignment checks on window_index/win_start/names."""
    paths = win_npz_paths()
    data = {m: np.load(p) for m, p in paths.items()}
    ref_idx = data["visionts"]["window_index"]
    ref_start = data["visionts"]["win_start"]
    ref_names = data["visionts"]["channel_names"]
    for m, z in data.items():
        assert np.array_equal(z["window_index"], ref_idx), \
            "%s window_index misaligned" % m
        assert np.array_equal(z["win_start"], ref_start), \
            "%s win_start misaligned" % m
        assert [str(x) for x in z["channel_names"]] == \
            [str(x) for x in ref_names], "%s channel_names misaligned" % m
    names = [str(x) for x in ref_names]
    assert names[OT_IDX] == "OT" and len(names) == 7, names
    n = ref_idx.shape[0]
    assert n == 2785 and np.array_equal(ref_idx, np.arange(n)), n
    assert ref_start[0] == WIN_START0
    return {m: z["win_var_mae"][:, OT_IDX].astype(np.float64)
            for m, z in data.items()}


def select_windows():
    """Fixed rule; returns dict case -> evidence. Raises SystemExit on any
    mismatch with FROZEN (counts, selected index, MAEs vs 4-dec table)."""
    mae = load_win_mae()
    v, nv, av = mae["visionts"], mae["naive"], mae["avg"]
    groups = {
        "success": ((v < nv) & (v < av), np.minimum(nv, av)),
        "failure": ((v > nv) & (v > av), np.maximum(nv, av)),
    }
    out, problems = {}, []
    for case, (gmask, nearest) in groups.items():
        cand = np.where(gmask)[0]                      # window indices
        n = cand.shape[0]
        diff = np.abs(v[cand] - nearest[cand])
        order = np.lexsort((cand, diff))               # diff asc, idx asc
        rank_pos = (n - 1) // 2                        # lower median, fixed
        win = int(cand[order[rank_pos]])
        exp = FROZEN[case]
        rec = {"case": case, "n_cand": int(n), "window_index": win,
               "rank_pos": int(rank_pos),
               "diff_to_nearest": float(diff[order[rank_pos]]),
               "mae_npz": {m: float(mae[m][win]) for m in mae},
               "win_start_row": int(WIN_START0 + win)}
        if n != exp["n_cand"]:
            problems.append("%s: n_cand=%d != frozen %d"
                            % (case, n, exp["n_cand"]))
        if win != exp["win"]:
            around = [(int(cand[order[i]]), float(diff[order[i]]))
                      for i in range(max(0, rank_pos - 3),
                                     min(n, rank_pos + 4))]
            problems.append("%s: selected win=%d != frozen %d; neighborhood "
                            "(win, diff_to_nearest)=%s"
                            % (case, win, exp["win"], around))
        for m, mval in exp["mae"].items():
            got = rec["mae_npz"][m]
            if abs(got - mval) > 5e-5:
                problems.append("%s %s mae npz=%.6f vs frozen %.4f"
                                % (case, m, got, mval))
        out[case] = rec
    if problems:
        for x in problems:
            print("[select_windows] FAIL:", x)
        raise SystemExit("[select_windows] frozen expectations mismatch; "
                         "check index alignment and sorting, no fallback.")
    out["_problems"] = []
    return out


def mae_f64_from_arrays(win):
    """Per-window OT MAE recomputed in f64 from the frozen pred/true arrays."""
    paths = frozen_arrays_paths()
    out = {}
    for m, rd in paths.items():
        pred = np.load(os.path.join(rd, "pred.npy"))[win, :, OT_IDX]
        true = np.load(os.path.join(rd, "true.npy"))[win, :, OT_IDX]
        err = pred.astype(np.float64) - true.astype(np.float64)
        out[m] = float(np.mean(np.abs(err)))
    return out


def timestamps(win):
    import pandas as pd
    csv = os.path.join(ROOT, "dataset", "ETT-small", "ETTh1.csv")
    dates = pd.read_csv(csv, usecols=["date"])
    start = WIN_START0 + win
    return {"start_row": int(start), "end_row": int(start + PRED_LEN - 1),
            "start_time": str(dates.iloc[start, 0]),
            "end_time": str(dates.iloc[start + PRED_LEN - 1, 0]),
            "hist_start_row": int(start - HIST_SHOW),
            "hist_start_time": str(dates.iloc[start - HIST_SHOW, 0])}


def batch_of(win):
    b = win // BATCH_SIZE
    return {"batch": int(b), "pos_in_batch": int(win - b * BATCH_SIZE),
            "batch_rows": [b * BATCH_SIZE, (b + 1) * BATCH_SIZE - 1]}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_md5():
    return {f: md5_of(os.path.join(ROOT, f)) for f in FP_FILES}


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
        f.write("\n")


def load_json(path):
    with open(path) as f:
        return json.load(f)
