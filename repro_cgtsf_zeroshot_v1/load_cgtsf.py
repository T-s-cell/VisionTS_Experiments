#!/usr/bin/env python3
"""Parse CGTSF CSVs into window tables. Identity = (dataset, Idx, Date);
CSV row order is the fixed window order everywhere. Hist feeds the model
(nvars=1, original units — the model's own instance norm handles
scaling); Pred is used only for verification (len) and scoring. All-zero
history windows are kept (sqrt(var+1e-5) keeps the forward finite).
Any NaN/Inf in the CSV data is a hard stop."""
import csv
import json

import numpy as np

import protocol as P


def iter_rows(ds):
    with open(P.csv_path(ds), newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        ci = {c: i for i, c in enumerate(hdr)}
        for c in ("Idx", "Date", "Hist", "Pred"):
            if c not in ci:
                raise SystemExit(f"{ds}: missing CSV column {c!r} in {hdr}")
        for row in r:
            yield int(row[ci["Idx"]]), row[ci["Date"]], \
                row[ci["Hist"]], row[ci["Pred"]]


def load_windows(ds):
    """All windows of one subset, in CSV row order, with full data gates."""
    cfg = P.SUBSETS[ds]
    C, Pr = cfg["context"], cfg["pred"]
    out, seen = [], set()
    for ix, date, hs, ps in iter_rows(ds):
        key = (ds, ix, date)
        if key in seen:
            raise SystemExit(f"{ds}: duplicate identity {key}")
        seen.add(key)
        h = np.asarray(json.loads(hs), dtype=np.float64)
        t = np.asarray(json.loads(ps), dtype=np.float64)
        if len(h) != C:
            raise SystemExit(f"{ds} {key}: len(Hist)={len(h)} != {C}")
        if len(t) != Pr:
            raise SystemExit(f"{ds} {key}: len(Pred)={len(t)} != {Pr}")
        if not (np.isfinite(h).all() and np.isfinite(t).all()):
            raise SystemExit(f"{ds} {key}: non-finite value in Hist/Pred")
        out.append({"ds": ds, "idx": ix, "date": date, "hist": h,
                    "target": t})
    if len(out) != P.EXPECTED_WINDOWS[ds]:
        raise SystemExit(f"{ds}: {len(out)} windows != "
                         f"{P.EXPECTED_WINDOWS[ds]}")
    n_seq = len({w["idx"] for w in out})
    if n_seq != cfg["n_seq"]:
        raise SystemExit(f"{ds}: {n_seq} sequences != {cfg['n_seq']}")
    return out


def zero_hist_positions(windows, seq_rows):
    """Positions (into `windows`) of all-zero-history windows among the
    given row indices of one sequence."""
    return [i for i in seq_rows if not windows[i]["hist"].any()]


def select_preflight(windows_by_ds):
    """Per sequence: first / middle / last window, plus the earliest
    all-zero-history window of every sequence that has one; dedup by
    position. Fixed order: PTF -> LEU -> MSPG, seq asc, then
    (first, zero, mid, last) rows asc. Must total exactly 230."""
    sel = []
    for ds in P.ORDER:
        wins = windows_by_ds[ds]
        by_seq = {}
        for i, w in enumerate(wins):
            by_seq.setdefault(w["idx"], []).append(i)
        picked = set()
        for ix in sorted(by_seq):
            rows = by_seq[ix]
            picked.update((rows[0], rows[len(rows) // 2], rows[-1]))
        for ix in sorted(by_seq):
            z = zero_hist_positions(wins, by_seq[ix])
            if z:
                picked.add(z[0])
        sel.append((ds, sorted(picked)))
    total = sum(len(p) for _, p in sel)
    if total != P.EXPECTED_PREFLIGHT_WINDOWS:
        raise SystemExit(f"preflight selection = {total} != "
                         f"{P.EXPECTED_PREFLIGHT_WINDOWS}")
    return sel
