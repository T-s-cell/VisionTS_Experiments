"""Server-side independent recheck of the validation P search.

Runs on the machine that stores the 136 validation arrays; the arrays are
NOT redistributed. For every combo (psearch/val__*.npz):
  - integrity vs the frozen protocol split: keys, count, window_index
    permutation, identity (s/orig_row/label_dates) + truth bitwise,
    finite predictions;
  - RECOMPUTE MSE/MAE from the arrays (status.json metrics are never
    trusted) and cross-check the ledger value + fingerprint;
then re-aggregate the per-domain per-P 4H equal-weight mean, re-derive the
argmin selection (tie -> smaller P) and require it to equal the frozen
selected_p.json.

Outputs (small files, next to the arrays):
  psearch_recheck.csv          136 rows
  psearch_recheck_report.json  totals, failures, selection comparison,
                               per-array path/bytes/sha256 manifest
With --ledger=<original accepted-run psearch/status.json>, the recomputed
MSE is additionally cross-checked against that ledger (metric-level
reproduction evidence; the original ledger itself is never used as the
result). Exit nonzero on any failure or selection mismatch.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import die, dump_json, load_json, \
    setup_deterministic_env, sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData
from repro_timemmd_common_v1.p_search import combo_fingerprint, \
    load_status, metrics, verify_combo_arrays

CSV_COLS = ["domain", "p", "h", "combo", "status", "windows", "mse",
            "mae", "mse_f64", "mae_f64", "ledger_mse", "ledger_abs_diff",
            "orig_mse", "orig_abs_diff", "fingerprint_ok", "identity_ok",
            "npz_bytes", "npz_path", "npz_sha256"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--ledger", default=None,
                    help="path to the ORIGINAL accepted-run "
                         "psearch/status.json; recomputed MSE is also "
                         "cross-checked against it")
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())
    ps = out_root / "psearch"
    status = load_status(out_root)
    orig = None
    orig_path = Path(args.ledger) if args.ledger else None
    if orig_path is not None:
        orig = load_json(orig_path)["combos"]
    sel = load_json(out_root / "selected_p.json")

    domcache = {}
    rows, manifest, failures = [], [], []
    t0 = time.time()
    for dom, p, h in P.all_psearch_combos():
        cid = P.combo_id(dom, p, h)
        row = {"domain": dom, "p": p, "h": h, "combo": cid,
               "status": "fail"}
        rows.append(row)
        npz_path = ps / P.val_npz_name(dom, p, h)
        if not npz_path.exists():
            failures.append("%s: npz missing" % cid)
            row["status"] = "missing"
            continue
        row["npz_path"] = str(npz_path)
        row["npz_bytes"] = npz_path.stat().st_size
        row["npz_sha256"] = sha256_of(npz_path)
        manifest.append({"combo": cid, "path": str(npz_path),
                         "bytes": row["npz_bytes"],
                         "sha256": row["npz_sha256"]})
        if dom not in domcache:
            domcache[dom] = DomainData(pinfo["protocol_dir"],
                                       pinfo["splits"], dom)
        sp = domcache[dom].load_split(h, "val")
        chk = verify_combo_arrays(npz_path, sp)
        row["identity_ok"] = bool(chk["ok"])
        if not chk["ok"]:
            failures.append("%s: %s" % (cid, chk["reason"]))
            row["status"] = "integrity_fail"
            continue
        row["windows"] = chk["windows"]
        arr = dict(np.load(npz_path))
        mse, mae, mse64, mae64 = metrics(arr["pred"], arr["truth"])
        row.update({"mse": mse, "mae": mae, "mse_f64": mse64,
                    "mae_f64": mae64})
        ent = status["combos"].get(cid, {})
        led = ent.get("mse")
        row["ledger_mse"] = led
        row["ledger_abs_diff"] = (abs(mse - led) if led is not None
                                  else None)
        fp_ok = ent.get("fingerprint") == combo_fingerprint(pinfo, dom,
                                                            p, h)
        row["fingerprint_ok"] = bool(fp_ok)
        ok = True
        if led is None:
            failures.append("%s: ledger has no mse" % cid)
            ok = False
        elif not row["ledger_abs_diff"] <= 1e-12 + 1e-9 * abs(led):
            failures.append("%s: ledger mse %.10f != recomputed %.10f"
                            % (cid, led, mse))
            ok = False
        if not fp_ok:
            failures.append("%s: ledger fingerprint mismatch" % cid)
            ok = False
        if orig is not None:
            om = orig.get(cid, {}).get("mse")
            row["orig_mse"] = om
            row["orig_abs_diff"] = (abs(mse - om) if om is not None
                                    else None)
            if om is None:
                failures.append("%s: original ledger missing mse" % cid)
                ok = False
            elif not row["orig_abs_diff"] <= 1e-12 + 1e-9 * abs(om):
                failures.append("%s: original ledger mse %.10f != "
                                "recomputed %.10f" % (cid, om, mse))
                ok = False
        row["status"] = "ok" if ok else "fail"

    by_p = {}
    for r in rows:
        if r["status"] == "ok":
            by_p.setdefault((r["domain"], r["p"]), []).append(r["mse"])
    selection, max_tab_diff = {}, 0.0
    for dom in P.ORDER:
        table = {}
        for p in P.TABLE[dom]["P"]:
            ms = by_p.get((dom, p), [])
            table[str(p)] = (float(np.mean(ms))
                             if len(ms) == len(P.TABLE[dom]["H"])
                             else None)
        cand = [(v, int(k)) for k, v in table.items() if v is not None]
        selection[dom] = {
            "selected": min(cand)[1]
            if len(cand) == len(P.TABLE[dom]["P"]) else None,
            "mean_val_mse_by_P": table}
    frozen = {d: sel["selection"][d]["selected"] for d in P.ORDER}
    derived = {d: selection[d]["selected"] for d in P.ORDER}
    selection_ok = (derived == frozen
                    and all(v is not None for v in derived.values()))
    if not selection_ok:
        failures.append("selection mismatch: derived %s vs frozen %s"
                        % (derived, frozen))
    for d in P.ORDER:
        for k, v in selection[d]["mean_val_mse_by_P"].items():
            fv = sel["selection"][d]["mean_val_mse_by_P"].get(k)
            if v is not None and fv is not None:
                max_tab_diff = max(max_tab_diff, abs(v - fv))

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    report = {
        "n_combos": len(P.all_psearch_combos()), "n_ok": n_ok,
        "n_fail": len(rows) - n_ok,
        "windows_total": int(sum(r.get("windows") or 0 for r in rows)),
        "selection_ok": selection_ok, "derived_selection": derived,
        "frozen_selection": frozen,
        "max_mean_table_abs_diff": max_tab_diff,
        "frozen_selected_p_sha256":
            sha256_of(out_root / "selected_p.json"),
        "orig_ledger_crosscheck": {
            "path": str(orig_path) if orig_path else None,
            "sha256": sha256_of(orig_path) if orig_path else None,
            "checked": bool(orig is not None),
            "n_matched": int(sum(1 for r in rows
                                 if r.get("orig_abs_diff") is not None
                                 and r["orig_abs_diff"]
                                 <= 1e-12 + 1e-9 * (r["orig_mse"] or 0)))
            if orig is not None else None,
        },
        "failures": failures, "all_pass": not failures,
        "elapsed_s": round(time.time() - t0, 3),
        "arrays_manifest": manifest,
        "note": "Validation arrays re-verified in place on the server "
                "that stores them; raw arrays are not redistributed.",
    }
    with open(out_root / "psearch_recheck.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    dump_json(out_root / "psearch_recheck_report.json", report)
    print("PSEARCH RECHECK: %d/%d ok, windows=%d, selection_ok=%s, "
          "max_table_diff=%.3e, failures=%d"
          % (n_ok, report["n_combos"], report["windows_total"],
             selection_ok, max_tab_diff, len(failures)))
    print("derived P:", derived)
    if failures:
        die("[FATAL] psearch recheck failed (%d): %s..."
            % (len(failures), failures[:5]))
    print("psearch recheck ALL PASS")


if __name__ == "__main__":
    main()
