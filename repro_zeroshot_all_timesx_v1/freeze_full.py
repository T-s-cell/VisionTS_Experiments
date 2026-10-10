#!/usr/bin/env python3
"""S1 freeze: enumerate ALL native TimesX windows and freeze the window manifest
plus past/target arrays. Assertions hard-stop on any mismatch.

Outputs (frozen/):
  inventory.json   {meta, windows:[flat-indexed rows]}
  vars.json        [{var_key, domain, group, n, ..., calib_std, ...}]
  frozen_arrays.npz past f8[T,96], target f8[T,12], d f8[T],
                     denom_source U-strings, flat_index i4, var_id i2
  freeze_report.json  assertion results + artifact hashes
"""
import argparse
import os
import sys
import time

import common as C
import protocol as P


def iso(dt):
    return dt.isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(C.FROZEN_DIR, exist_ok=True)
    for p in (C.INV_PATH, C.ARRAYS_PATH, C.VARS_PATH):
        if os.path.isfile(p) and not args.force:
            raise SystemExit(f"{p} already exists; pass --force to refreeze")

    from timesx_data import find_zip, iter_variables, parse_ts, pstdev

    zip_path = args.zip or find_zip(None)
    t0 = time.time()
    print(f"[freeze] zip={zip_path}")
    zsha = C.sha256_file(zip_path)
    if zsha != P.ZIP_SHA256:
        raise SystemExit(f"zip sha256 mismatch: {zsha} != {P.ZIP_SHA256}")

    splits = C.load_json(P.OLD_SPLITS)
    splits_sha = C.sha256_file(P.OLD_SPLITS)
    if not splits_sha.startswith(P.OLD_SPLITS_SHA256_PREFIX):
        raise SystemExit(f"old splits.json sha prefix mismatch: {splits_sha}")

    import numpy as np

    windows = []
    vars_out = []
    past_blocks, target_blocks = [], []
    d_blocks, dsrc_blocks = [], []
    flat = 0
    var_id = 0
    problems = []

    for v in iter_variables(zip_path):
        entry = splits["variables"].get(v.var_key)
        if entry is None:
            raise SystemExit(f"variable {v.var_key} missing from old splits.json")
        sids = [s.sample_id for s in v.samples]
        if len(set(sids)) != len(sids):
            raise SystemExit(f"{v.var_key}: duplicate sample_id after sort")

        part_ids = {
            "test": [r["sample_id"] for r in entry["test"]],
            "val": [r["sample_id"] for r in entry["val"]],
            "cross_boundary_removed": list(entry["cross_boundary_removed_ids"]),
            "train_region": list(entry["calib_remaining_ids"]),
        }
        seen = {}
        for k, ids in part_ids.items():
            for i in ids:
                if i in seen:
                    raise SystemExit(f"{v.var_key}: id {i} in both {seen[i]} and {k}")
                seen[i] = k
        if set(seen) != set(sids):
            missing = set(sids) - set(seen)
            extra = set(seen) - set(sids)
            raise SystemExit(f"{v.var_key}: partition mismatch missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")

        first = v.samples[0]
        calib = pstdev(first.past_val)
        sc = float(entry["calib_std"])
        if abs(calib - sc) > 1e-9 * max(1.0, abs(sc)):
            raise SystemExit(f"{v.var_key}: calib_std recompute {calib!r} != splits {sc!r}")

        for s in v.samples:
            if len(s.past_ts) != P.CONTEXT_LEN or len(s.future_ts) != P.PRED_LEN:
                raise SystemExit(f"{s.sample_id}: geom {len(s.past_ts)}->{len(s.future_ts)}")
            if parse_ts(s.past_ts[-1]) >= parse_ts(s.future_ts[0]):
                raise SystemExit(f"{s.sample_id}: past_end {s.past_ts[-1]} !< future_start {s.future_ts[0]}")
            pv = np.asarray(s.past_val, dtype=np.float64)
            fv = np.asarray(s.future_val, dtype=np.float64)
            if pv.shape != (P.CONTEXT_LEN,) or fv.shape != (P.PRED_LEN,):
                raise SystemExit(f"{s.sample_id}: value shape {pv.shape}/{fv.shape}")
            if not (np.isfinite(pv).all() and np.isfinite(fv).all()):
                raise SystemExit(f"{s.sample_id}: non-finite values")
            windows.append({
                "flat_index": flat,
                "var_id": var_id,
                "domain": v.domain,
                "group": v.group,
                "var_key": v.var_key,
                "sample_id": s.sample_id,
                "idx": s.idx,
                "date": s.date,
                "freq": s.freq,
                "split_in_old_protocol": seen[s.sample_id],
                "future_start": iso(s.future_start),
                "future_end": iso(s.future_end),
                "past_start": s.past_ts[0],
                "past_end": s.past_ts[-1],
                "win_std": float(pv.std(ddof=0)),
            })
            past_blocks.append(pv)
            target_blocks.append(fv)
            ws = float(pv.std(ddof=0))
            if ws >= P.EPS_STD:
                d_blocks.append(ws)
                dsrc_blocks.append("window")
            else:
                d_blocks.append(calib)
                dsrc_blocks.append("calib_std")
            flat += 1
        vars_out.append({
            "var_id": var_id,
            "var_key": v.var_key,
            "domain": v.domain,
            "group": v.group,
            "n": len(v.samples),
            "n_test": len(part_ids["test"]),
            "n_val": len(part_ids["val"]),
            "n_cross_boundary_removed": len(part_ids["cross_boundary_removed"]),
            "n_train_region": len(part_ids["train_region"]),
            "calib_std": calib,
            "calib_std_rule": "std(ddof=0) of the earliest native window's past96",
            "calib_source_sample_id": first.sample_id,
            "calib_matches_old_splits": True,
        })
        var_id += 1

    # global counts
    n_by_split = {}
    for k in ("test", "val", "cross_boundary_removed", "train_region"):
        n_by_split[k] = sum(1 for w in windows if w["split_in_old_protocol"] == k)
    checks = {
        "n_vars": (var_id, P.N_VARS),
        "total": (flat, P.TOTAL_WINDOWS),
        "test": (n_by_split["test"], P.N_TEST),
        "val": (n_by_split["val"], P.N_VAL),
        "cross_boundary_removed": (n_by_split["cross_boundary_removed"], P.N_CROSS_REMOVED),
        "train_region": (n_by_split["train_region"], P.N_TRAIN_REGION),
    }
    for k, (got, want) in checks.items():
        if got != want:
            raise SystemExit(f"count mismatch {k}: got {got} != frozen {want}")

    all_sids = [w["sample_id"] for w in windows]
    if len(set(all_sids)) != len(all_sids):
        raise SystemExit("duplicate sample_id across variables")

    # test/val rows must match old splits row fields exactly
    for w in windows:
        if w["split_in_old_protocol"] in ("test", "val"):
            r = next(r for r in splits["variables"][w["var_key"]][w["split_in_old_protocol"]]
                     if r["sample_id"] == w["sample_id"])
            if (str(r["idx"]) != str(w["idx"]) or r["date"] != w["date"]
                    or r["future_start"] != w["future_start"] or r["future_end"] != w["future_end"]):
                raise SystemExit(f"row field mismatch vs old splits: {w['sample_id']}")

    # zero-variance census
    zero_var = [w for w in windows if w["win_std"] < P.EPS_STD]
    print(f"[freeze] zero-variance windows: {len(zero_var)}")
    for w in zero_var:
        print(f"  - {w['var_key']} {w['sample_id']} (old split: {w['split_in_old_protocol']})")
    if len(zero_var) != P.ZERO_VAR_EXPECTED:
        raise SystemExit(
            f"zero-variance census {len(zero_var)} != expected {P.ZERO_VAR_EXPECTED}; "
            "data version drift suspected — stopping per protocol")

    past = np.stack(past_blocks)
    target = np.stack(target_blocks)
    assert past.shape == (P.TOTAL_WINDOWS, P.CONTEXT_LEN)
    assert target.shape == (P.TOTAL_WINDOWS, P.PRED_LEN)
    assert np.isfinite(past).all() and np.isfinite(target).all()

    domain_counts = {}
    for w in windows:
        domain_counts[w["domain"]] = domain_counts.get(w["domain"], 0) + 1

    meta = {
        "protocol_version": P.PROTOCOL_VERSION,
        "protocol_name": P.PROTOCOL_NAME,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "zip_path": zip_path,
        "zip_sha256": zsha,
        "old_splits_sha256": splits_sha,
        "sort_key": "variables in zip enumeration order; windows by (future_start, idx)",
        "identity": "(var_key, sample_id), sample_id is a string",
        "counts": {
            "vars": var_id, "windows": flat,
            "domains": len(domain_counts),
            "test": n_by_split["test"], "val": n_by_split["val"],
            "cross_boundary_removed": n_by_split["cross_boundary_removed"],
            "train_region": n_by_split["train_region"],
        },
        "zero_variance_windows": [
            {"sample_id": w["sample_id"], "var_key": w["var_key"],
             "old_split": w["split_in_old_protocol"]} for w in zero_var],
        "code_md5": {"new": C.new_code_md5(), "old_reused": C.code_md5_map(P.OLD_CODE_REUSED, C.OLD_DIR)},
        "env": C.env_info(),
    }

    inv = {"meta": meta, "windows": windows}
    C.dump_json(inv, C.INV_PATH)
    C.dump_json(vars_out, C.VARS_PATH)
    np.savez_compressed(C.ARRAYS_PATH, past=past, target=target,
                        d=np.asarray(d_blocks, dtype=np.float64),
                        denom_source=np.array(dsrc_blocks, dtype=np.str_),
                        flat_index=np.arange(flat, dtype=np.int32),
                        var_id=np.array([w["var_id"] for w in windows], dtype=np.int16))

    report = {
        "stage": "S1_freeze",
        "checks": {k: {"got": g, "expected": w, "pass": g == w} for k, (g, w) in checks.items()},
        "n_domains": len(domain_counts),
        "domain_window_counts": domain_counts,
        "zero_variance_census": meta["zero_variance_windows"],
        "artifacts": {
            "inventory.json": C.sha256_file(C.INV_PATH),
            "vars.json": C.sha256_file(C.VARS_PATH),
            "frozen_arrays.npz": C.sha256_file(C.ARRAYS_PATH),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    report["checks"]["past_end_lt_future_start"] = {"got": "all", "pass": True}
    report["checks"]["scoring_d_frozen"] = {
        "got": {"n_calib_fallback": sum(1 for s in dsrc_blocks if s == "calib_std"),
                "n_window": sum(1 for s in dsrc_blocks if s == "window")},
        "pass": sum(1 for s in dsrc_blocks if s == "calib_std") == P.ZERO_VAR_EXPECTED}
    C.dump_json(report, C.FREEZE_REPORT)
    C.mark_done("s1_freeze", {"artifacts": report["artifacts"],
                              "sha": {f"frozen/{k}": v for k, v in report["artifacts"].items()},
                              "input_guards": C.stage_input_guards("freeze", zip_path=zip_path)})
    print(f"[freeze] OK: {flat} windows / {var_id} vars / {len(domain_counts)} domains "
          f"in {report['elapsed_s']}s")
    print(f"[freeze] inventory={report['artifacts']['inventory.json'][:12]}… "
          f"arrays={report['artifacts']['frozen_arrays.npz'][:12]}…")


if __name__ == "__main__":
    main()
