"""EXP-041 step 1: sample 300 stratified Monash histories (seed 2021) from
the 29 paper Table-15 rows, using the SAME gluonts cache EXP-038 read
(author-pinned env; HOME=/dev_data/wlt/t37_author_home).

Quota: floor(300/29)=10 per paper row (290), remaining 10 to the rows with
the largest window capacity (ties by PAPER_T15_ORDER order).  Deficit
redistribution chain when a row's capacity cannot fill its quota (e.g.
short histories).  CIF_2016_POOLED draws from cif_2016_6 + cif_2016_12
with a deterministic 50/50 split (spillover recorded).

Notation: H = archived pred_rule.expected (cross-checked against the
gluonts metadata AND the EXP-038 archived split stats); P = archived
autotune accepted_p (fail-stop if absent); L_config = 300 if P==1 else
round(1000/P)*P; L_eff = min(L_config, N_in) with N_in = len(series) (the
official test input = series[:-H]); short histories are KEPT and flagged.

Candidate windows per series: stride-1 [s, s+L_eff), end <= N_in; selected
windows within a series are pairwise non-overlapping.  Draw per subset:
random-greedy first; on shortfall the subset's greedy picks are DISCARDED
and redrawn by pure stride-L_eff packing (which always achieves the
X.max_windows capacity that redistribution budgets against).  CIF spillover
flows to the sibling subset.  NaN windows are forward-filled with the
official imputation_nan (raw NaN counts recorded).

Outputs (save_t41_monash/):
  monash_manifest.json  subset/row/sample records (H/P/L_eff per sample)
  monash_windows.npz    win_<k> float64 imputed windows (global k order)
Run (author env): PYTHON_A sample_monash.py [--smoke N_PER_ROW]
"""
import argparse
import os
import sys

import numpy as np

_PKG = os.path.dirname(os.path.abspath(__file__))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import t41_common as X          # noqa: E402
import t38_common as T38        # noqa: E402


def load_subset_inputs(ds):
    """Official test inputs through get_gluonts_test_dataset (cache read,
    regenerate=False); returns (series list of float64 arrays, meta dict)."""
    os.chdir(T38.EVAL_GLUONTS)
    if T38.EVAL_GLUONTS not in sys.path:
        sys.path.insert(0, T38.EVAL_GLUONTS)
    from dataset import get_gluonts_test_dataset
    test_data, metadata = get_gluonts_test_dataset(ds, regenerate=False)
    inputs = list(test_data.input)
    series = [np.asarray(x["target"], dtype=np.float64) for x in inputs]
    lens = [len(s) for s in series]
    meta = {
        "meta_prediction_length": int(metadata.prediction_length),
        "meta_freq": str(metadata.freq),
        "n_series": len(series),
        "input_len_min": int(min(lens)),
        "input_len_median": float(np.median(lens)),
        "input_len_max": int(max(lens)),
        "item_ids": [str(x.get("item_id", "")) for x in inputs],
        "starts": [str(x["start"]) for x in inputs],
    }
    return series, meta


def subset_facts(ds):
    """H / P from the EXP-038 archive (fail-stop) + expected-P table."""
    p = X.monash_json(ds)
    assert os.path.isfile(p), "missing EXP-038 archive %s" % p
    rec = X.load_json(p)
    h = rec.get("pred_rule", {}).get("expected")
    assert isinstance(h, int) and h > 0, "%s: no archived H" % ds
    acc = rec.get("autotune", {}).get("accepted_p")
    assert isinstance(acc, int) and acc > 0, \
        "%s: no archived accepted_p (fail-stop)" % ds
    return {"H": h, "P": acc, "P_table": T38.EXPECTED_P[ds],
            "archive": p, "split": rec.get("split", {})}


def draw_subset(f, n_target, seed_parts):
    """Random-greedy over stride-1 candidates; on shortfall DISCARD the
    greedy picks and redraw by pure stride-L_eff packing — packing always
    achieves sum(max_windows(N_in, L_eff)) (the same capacity redistribution
    budgeted against), so min(n_target, capacity) is guaranteed."""
    rs = np.random.RandomState(X.derived_seed(*seed_parts))
    l_effs = f["l_eff"]
    cands = []
    for si, l_eff in enumerate(l_effs):
        n_in = len(f["series"][si])
        if l_eff <= 0:
            continue
        cands.extend((si, st) for st in range(0, n_in - l_eff + 1))
    rs.shuffle(cands)
    used, got = {}, []
    for si, st in cands:            # random-greedy pass
        if len(got) >= n_target:
            break
        lo, hi = st - l_effs[si] + 1, st + l_effs[si] - 1
        if any(not (hi < a0 or lo > a1) for a0, a1 in used.get(si, [])):
            continue
        used.setdefault(si, []).append((st, st + l_effs[si] - 1))
        got.append((si, st, "random_greedy"))
    if len(got) < n_target:
        used, got = {}, []          # greedy fragments series space: discard
        for si, l_eff in enumerate(l_effs):
            n_in = len(f["series"][si])
            if l_eff <= 0:
                continue
            for st in range(0, n_in - l_eff + 1, l_eff):
                if len(got) >= n_target:
                    break
                got.append((si, st, "packing"))
            if len(got) >= n_target:
                break
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="cap each row at N samples and only process the "
                         "first 3 rows (smoke end-to-end)")
    a = ap.parse_args()

    cache = os.path.join(os.path.expanduser("~"), ".gluonts")
    assert os.path.isdir(cache), \
        "gluonts cache missing at %s (run with HOME=/dev_data/wlt/t37_author_home)" % cache

    order = T38.PAPER_T15_ORDER
    rows = order[:3] if a.smoke else order

    # ---- per-subset facts + series ---------------------------------------
    subsets = {}
    for row in rows:
        for ds in X.row_subsets(row):
            if ds in subsets:
                continue
            facts = subset_facts(ds)
            series, meta = load_subset_inputs(ds)
            # identity evidence vs the EXP-038 archived split stats
            arch = facts["split"]
            for k in ("n_series", "input_len_min", "input_len_median",
                      "input_len_max"):
                assert arch and meta[k] == arch.get(k), \
                    "%s cache-vs-archive split mismatch on %s: %s vs %s" % (
                        ds, k, meta[k], arch.get(k))
            assert meta["meta_prediction_length"] == facts["H"], \
                "%s: metadata H %d != archived %d" % (
                    ds, meta["meta_prediction_length"], facts["H"])
            l_cfg = X.expected_context(facts["P"])
            facts.update({"l_config": l_cfg, "series": series, "meta": meta,
                          "l_eff": [min(l_cfg, len(s)) for s in series]})
            subsets[ds] = facts
            X.log("%s: H=%d P=%d L_cfg=%d n_series=%d N_in[%d..%d]"
                  % (ds, facts["H"], facts["P"], l_cfg, meta["n_series"],
                     meta["input_len_min"], meta["input_len_max"]))

    # ---- capacities / quota / redistribution ------------------------------
    def row_capacity(row):
        cap = 0
        for ds in X.row_subsets(row):
            f = subsets[ds]
            for si, l_eff in enumerate(f["l_eff"]):
                cap += X.max_windows(len(f["series"][si]), l_eff)
        return cap

    capacities = {row: row_capacity(row) for row in rows}
    if a.smoke:
        quota, surplus_log = {r: a.smoke for r in rows}, []
    else:
        quota, surplus_log = X.row_quota_plan(capacities)
    q_final, transfers, unmet = X.redistribute(quota, capacities)
    if unmet:
        X.log("UNMET after redistribution: %s" % unmet)
        raise SystemExit("[t41-sample] 300 samples unreachable: %s" % unmet)

    # ---- draw --------------------------------------------------------------
    samples = []
    windows = {}
    k_global = 0
    for row in rows:
        want = q_final[row]
        subsets_row = X.row_subsets(row)
        chosen = []
        n_first = want // 2 if row == "CIF_2016_POOLED" else want
        ds0 = subsets_row[0]
        got0 = draw_subset(subsets[ds0], n_first,
                           ("t41", "draw", row, ds0, X.SEED))
        chosen.extend((ds0, si, st, how) for si, st, how in got0)
        remaining = want - len(got0)
        if remaining and len(subsets_row) > 1:
            ds1 = subsets_row[1]
            X.log("%s/%s short by %d (spillover to %s)"
                  % (row, ds0, remaining, ds1))
            got1 = draw_subset(subsets[ds1], remaining,
                               ("t41", "draw", row, ds1, X.SEED))
            chosen.extend((ds1, si, st, how) for si, st, how in got1)
            remaining -= len(got1)
        if remaining:
            raise SystemExit("[t41-sample] row %s short by %d "
                             "(capacity budget violated)" % (row, remaining))
        for ds, si, st, how in chosen:
            f = subsets[ds]
            l_eff = f["l_eff"][si]
            raw = f["series"][si][st:st + l_eff]
            n_nan = int(np.isnan(raw).sum())
            win = X.imputation_nan(raw).astype(np.float64)
            assert np.isfinite(win).all()
            samples.append({
                "k": k_global, "row": row, "subset": ds, "series_idx": si,
                "item_id": f["meta"]["item_ids"][si],
                "start_ts": f["meta"]["starts"][si],
                "H": f["H"], "P": f["P"], "P_table": f["P_table"],
                "L_config": f["l_config"], "L_eff": int(l_eff),
                "window_start": int(st),
                "window_end": int(st + l_eff),      # exclusive, <= N_in
                "N_in": int(len(f["series"][si])),
                "short_history": bool(l_eff < f["l_config"]),
                "n_nan_raw_window": n_nan,
                "fill_method": how, "draw": "seed2021",
            })
            windows["win_%d" % k_global] = win
            k_global += 1
        X.log("row %s: %d samples" % (row, len(chosen)))

    if a.smoke:
        cap_note = "SMOKE (rows=%d, per-row<=%d)" % (len(rows), a.smoke)
    else:
        assert k_global == X.N_MONASH, "got %d != 300" % k_global
        cap_note = "FULL (300)"

    manifest = {
        "exp": "EXP-041 Monash sampling", "mode": cap_note,
        "seed": X.SEED, "started": X.time_str(),
        "notation": "H=forecast length, P=selected periodicity, "
                    "L_eff=actual input history (min(L_config, N-H))",
        "subsets": {ds: {kk: vv for kk, vv in f.items()
                         if kk not in ("series", "l_eff")}
                    for ds, f in subsets.items()},
        "capacities": {r: capacities[r] for r in rows},
        "quota_initial": quota, "quota_final": q_final,
        "surplus_log": surplus_log, "transfers": transfers,
        "samples": samples,
        "t38_archive_base": X.t38_logs_base(),
        "gluonts_cache": cache,
    }
    X.dump_json(X.MANIFEST_JSON, manifest)
    np.savez_compressed(X.WINDOWS_NPZ, **windows)
    X.log("wrote %s (%d samples) + %s" % (X.MANIFEST_JSON, len(samples),
                                          X.WINDOWS_NPZ))
    n_short = sum(1 for s in samples if s["short_history"])
    n_nan_w = sum(1 for s in samples if s["n_nan_raw_window"] > 0)
    X.log("summary: %d samples, short_history=%d, nan-windows=%d"
          % (len(samples), n_short, n_nan_w))


if __name__ == "__main__":
    main()
