#!/usr/bin/env python3
"""Stage prepare (spec sec 3, runs WITHOUT torch).

1. verify data/weights/core-source hashes against the frozen spec values;
2. rebuild each variable's timestamp-value series from the native JSON windows
   (dedup with value-consistency, contiguity + adjacency-evidence assertions);
3. domain-common time boundaries (per-var 70/10/20 candidates, domain min);
4. native train/val/test assignment + cross-boundary exclusion; dense train pool;
5. per-variable fallback std (first 96 training observations, non-degenerate);
6. freeze manifests/{protocol.json, split_manifest.json, split_counts.csv} +
   manifests/data_cache.npz; hard-fail unless every count equals the spec table;
7. isolation baseline snapshot (static sources/data/weights/frozen results).

Idempotent: re-running with unchanged inputs just re-verifies. --force rebuilds.
"""
import argparse
import bisect
import csv
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (AUDIT, BASELINE_JSON, CFG, CTX, DATASET_DIR, DOMAINS,
                    DOMAIN_IDX, DOMAIN_NAMES, MANIFESTS, PRED, PROTOCOL_JSON,
                    REPO_ROOT, SPLIT_COUNTS, SPLIT_MANIFEST, DATA_CACHE,
                    ZERO_DIR, code_fingerprint, md5_file, sha256_file,
                    timesx_data, u_d_s_d, zip_path, atomic_write_json)

EPS_STD = CFG["data"]["std_eps"]
CTX = CFG["data"]["ctx"]
PRED = CFG["data"]["pred"]
WIN = CTX + PRED


class Abort(Exception):
    pass


def pstdev_np(x):
    return float(np.std(x))


def rebuild_variable(v):
    """Return (ts_keys sorted list of datetime, values float64, samples located)."""
    ts_map = {}
    for s in v.samples:
        if len(s.past_ts) != CTX or len(s.future_ts) != PRED:
            raise Abort(f"{v.var_key}: sample {s.sample_id} has "
                        f"{len(s.past_ts)}/{len(s.future_ts)} steps, expected {CTX}/{PRED}")
        for t, val in list(zip(s.past_ts, s.past_val)) + list(zip(s.future_ts, s.future_val)):
            fv = float(val)
            if not np.isfinite(fv):
                raise Abort(f"{v.var_key}: non-finite value at {t} (sample {s.sample_id})")
            if t in ts_map:
                if ts_map[t] != fv:
                    raise Abort(f"{v.var_key}: conflicting value at {t}: "
                                f"{ts_map[t]} vs {fv}")
            else:
                ts_map[t] = fv
    keys = sorted(ts_map, key=timesx_data.parse_ts)
    vals = np.array([ts_map[k] for k in keys], dtype=np.float64)
    idx = {k: i for i, k in enumerate(keys)}
    located = []
    for s in v.samples:
        s0 = idx[s.past_ts[0]]
        e0 = idx[s.future_ts[-1]]
        if e0 - s0 + 1 != WIN:
            raise Abort(f"{v.var_key}: sample {s.sample_id} not contiguous in "
                        f"reconstructed series (s={s0}, e={e0})")
        if not (np.array_equal(vals[s0:s0 + CTX],
                               np.array([float(x) for x in s.past_val]))
                and np.array_equal(vals[s0 + CTX:s0 + WIN],
                                   np.array([float(x) for x in s.future_val]))):
            raise Abort(f"{v.var_key}: sample {s.sample_id} values differ from series")
        located.append((s, s0))
    cover = np.zeros(len(keys) - 1, dtype=bool)
    for _s, s0 in located:
        cover[s0:s0 + WIN - 1] = True
    if not cover.all():
        gaps = np.where(~cover)[0]
        raise Abort(f"{v.var_key}: {len(gaps)} adjacent observations not evidenced "
                    f"by any native 108-step fragment, e.g. {keys[gaps[0]]} -> "
                    f"{keys[gaps[0] + 1]}")
    return [timesx_data.parse_ts(k) for k in keys], vals, located


def assign_split(future_start, future_end, T_train, T_test):
    if future_end < T_train:
        return "train", None
    if future_start >= T_test:
        return "test", None
    if future_start >= T_train and future_end < T_test:
        return "val", None
    return "excluded", ("cross_train_val" if future_start < T_train
                        else "cross_val_test")


def verify_hashes():
    problems, skipped = [], []
    zp = zip_path()
    got = sha256_file(zp)
    if got != CFG["data"]["zip_sha256"]:
        problems.append(f"zip sha256 mismatch: {got}")
    ckpt = os.path.join(REPO_ROOT, CFG["model"]["ckpt_rel"])
    if os.path.isfile(ckpt):
        got = sha256_file(ckpt)
        if got != CFG["model"]["ckpt_sha256"]:
            problems.append(f"ckpt sha256 mismatch: {got}")
    else:
        skipped.append(f"ckpt not on this host: {ckpt}")
    for rel, want in CFG["model"]["core_md5"].items():
        p = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(p):
            skipped.append(f"core source not on this host: {p}")
            continue
        got = md5_file(p)
        if got != want:
            problems.append(f"{rel} md5 mismatch: {got} (expected {want})")
    return problems, skipped


def build_baseline():
    """Static scope hashed strictly; everything else outside our dir recorded
    (size+mtime only) as dynamic — co-tenant outputs change legitimately."""
    static = []

    def add(rel, base=REPO_ROOT):
        p = os.path.join(base, rel)
        if os.path.isfile(p):
            static.append(rel)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d != "__pycache__")
                for fn in sorted(files):
                    if fn.endswith(".pyc"):
                        continue
                    static.append(os.path.relpath(os.path.join(root, fn), base))

    add("visionts")
    # repro_fullshot: CODE only, per the frozen isolation scope — its
    # logs/checkpoints/results and live status files belong to the sibling
    # queue and are dynamic outputs that must never enter the hash diff.
    fs_root = os.path.join(REPO_ROOT, "repro_fullshot")
    for root, dirs, files in os.walk(fs_root):
        dirs[:] = sorted(d for d in dirs
                         if d not in ("logs", "checkpoints", "results",
                                      "__pycache__"))
        for fn in sorted(files):
            if fn.endswith((".py", ".sh")):
                static.append(
                    os.path.normpath(os.path.join(
                        os.path.relpath(root, REPO_ROOT), fn)))
    for pat_dir in ("long_term_tsf/data_provider", "long_term_tsf/exp",
                    "long_term_tsf/models", "long_term_tsf/utils"):
        add(pat_dir)
    for f in ("long_term_tsf/run.py", "long_term_tsf/run_allScript.py",
              "repro_zeroshot_ext/timesx_data.py", "repro_zeroshot_ext/metrics_util.py",
              "repro_zeroshot_ext/infer.py", "repro_zeroshot_ext/build_splits.py",
              "repro_zeroshot_ext/select_period.py", "repro_zeroshot_ext/evaluate_test.py",
              "repro_zeroshot_ext/aggregate.py", "repro_zeroshot_ext/record_fingerprint.py",
              "repro_zeroshot_ext/data_audit.py", "repro_zeroshot_ext/audit_ext.sh",
              "repro_zeroshot_ext/run_all.sh", "repro_zeroshot_ext/README.md",
              "repro_zeroshot_ext/splits.json", "repro_zeroshot_ext/P_selection.json",
              "repro_zeroshot_ext/P_selection_run1.json"):
        add(f)
    add("repro_zeroshot_ext/results")
    add("repro_fig7/outputs/sampling_lists")
    for f in ("README.md", "requirements.txt", "setup.py", "LICENSE", "demo.ipynb"):
        add(f)
    add(os.path.relpath(os.path.join(REPO_ROOT, CFG["model"]["ckpt_rel"])))
    snap = {}
    for rel in sorted(set(static)):
        p = os.path.join(REPO_ROOT, rel)
        if os.path.isfile(p):
            snap[rel] = {"size": os.path.getsize(p), "sha256": sha256_file(p)}
    dynamic = {}
    for root, dirs, files in os.walk(REPO_ROOT):
        rel_root = os.path.relpath(root, REPO_ROOT)
        if rel_root.split(os.sep)[0] in ("repro_ln_timesx_v1", "__pycache__") \
                or "__pycache__" in rel_root.split(os.sep):
            dirs[:] = []
            continue
        for fn in sorted(files):
            rel = os.path.relpath(os.path.join(root, fn), REPO_ROOT)
            if rel in snap:
                continue
            p = os.path.join(root, fn)
            dynamic[rel] = {"size": os.path.getsize(p),
                            "mtime": int(os.path.getmtime(p))}
    return {"static": snap, "dynamic_seen": dynamic}


def old_cache_ids(var_key):
    """Read-only peek at the old zero-shot cache for one variable (or None)."""
    import zipfile
    npz = os.path.join(ZERO_DIR, "cache", f"{var_key}__P1.npz")
    if not os.path.isfile(npz):
        return None
    with np.load(npz, allow_pickle=False) as d:
        return [str(x) for x in d["sample_ids"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(PROTOCOL_JSON) and os.path.exists(SPLIT_MANIFEST) \
            and os.path.exists(DATA_CACHE) and not args.force:
        print("[prepare] manifests exist; verifying hashes only (use --force to rebuild)")
        problems, skipped = verify_hashes()
        for s in skipped:
            print(f"[prepare] (skipped: {s})")
        if problems:
            print("[prepare] HASH PROBLEMS:", *problems, sep="\n  ")
            sys.exit(1)
        proto = json.load(open(PROTOCOL_JSON))
        if proto["data_cache_sha256"] != sha256_file(DATA_CACHE):
            print("[prepare] data_cache.npz sha256 drift vs protocol.json")
            sys.exit(1)
        print("[prepare] OK (existing manifests verified)")
        return

    print("[prepare] 1) verify hashes")
    problems, skipped = verify_hashes()
    for s in skipped:
        print(f"[prepare] (skipped on this host: {s})")
    if problems:
        print("[prepare] HASH PROBLEMS (spec App.B):", *problems, sep="\n  ")
        sys.exit(1)

    print("[prepare] 2) rebuild sequences + boundaries")
    zp = zip_path()
    domains_vars = {d: [] for d in DOMAIN_NAMES}
    per_var = {}
    for v in timesx_data.iter_variables(zp, load_values=True):
        if v.domain not in DOMAIN_IDX:
            raise Abort(f"unknown domain {v.domain} ({v.var_key})")
        keys_dt, vals, located = rebuild_variable(v)
        per_var[v.var_key] = (v, keys_dt, vals, located)
        domains_vars[v.domain].append(v.var_key)
    if len(per_var) != CFG["expected_totals"]["n_vars"]:
        raise Abort(f"expected {CFG['expected_totals']['n_vars']} variables, "
                    f"got {len(per_var)}")

    T_train_d, T_test_d = {}, {}
    for d, vkeys in domains_vars.items():
        if len(vkeys) != DOMAINS[DOMAIN_IDX[d]]["n_vars"]:
            raise Abort(f"domain {d}: {len(vkeys)} vars, expected "
                        f"{DOMAINS[DOMAIN_IDX[d]]['n_vars']}")
        ct, cts = [], []
        for vk in vkeys:
            _v, keys_dt, _vals, _loc = per_var[vk]
            n = len(keys_dt)
            ct.append(keys_dt[(7 * n) // 10])
            cts.append(keys_dt[n - n // 5])
        T_train_d[d] = min(ct)
        T_test_d[d] = min(cts)
        if not (T_train_d[d] < T_test_d[d]):
            raise Abort(f"domain {d}: T_train >= T_test")

    print("[prepare] 3) split assignment + dense pools")
    manifest = {"protocol_version": CFG["protocol_version"],
                "zip_sha256": CFG["data"]["zip_sha256"], "zip_path_used": zp,
                "domains": {d: {"T_train": T_train_d[d].isoformat(),
                                "T_test": T_test_d[d].isoformat()}
                            for d in DOMAIN_NAMES},
                "variables": {}}
    counts = {d: dict(vars_=0, native_train=0, dense=0, val=0, test=0, excluded=0)
              for d in DOMAIN_NAMES}
    totals = dict(native_train=0, dense=0, val=0, test=0, excluded=0)
    series_flat, series_off, series_len, var_keys_out = [], [], [], []
    series_pos = 0
    native_starts_flat, native_off = [], []
    all_sample_ids = set()
    for d in DOMAIN_NAMES:
        for vk in sorted(domains_vars[d]):
            v, keys_dt, vals, located = per_var[vk]
            n = len(keys_dt)
            n_train_obs = sum(1 for t in keys_dt if t < T_train_d[d])
            Ttr, Tte = T_train_d[d], T_test_d[d]
            if n_train_obs < 96:
                raise Abort(f"{vk}: only {n_train_obs} training observations (<96)")
            fb_std = pstdev_np(vals[:96])
            if fb_std < EPS_STD:
                raise Abort(f"{vk}: fallback std {fb_std} degenerate")
            native = {"train": [], "val": [], "test": [], "excluded": []}
            sample_start = {}
            starts_sorted = []
            for s, s0 in located:
                reason = None
                if s.future_end < Ttr:
                    split = "train"
                elif s.future_start >= Tte:
                    split = "test"
                elif s.future_start >= Ttr and s.future_end < Tte:
                    split = "val"
                else:
                    split = "excluded"
                    reason = ("cross_train_val" if s.future_start < Ttr
                              else "cross_val_test")
                sample_start[s.sample_id] = s0
                starts_sorted.append(s0)
                if split == "excluded":
                    native["excluded"].append({"sample_id": s.sample_id,
                                               "reason": reason})
                else:
                    native[split].append(s.sample_id)
                if s.sample_id in all_sample_ids:
                    raise Abort(f"duplicate sample_id {s.sample_id}")
                all_sample_ids.add(s.sample_id)
            starts_sorted.sort()
            dense_count = max(0, n_train_obs - (WIN - 1))
            links = {}
            if dense_count:
                link_set = {s0: s.sample_id for s, s0 in located}
                for s0 in range(dense_count):
                    if s0 in link_set and link_set[s0] in native["train"]:
                        links[f"{s0:07d}"] = link_set[s0]
            counts[d]["vars_"] += 1
            counts[d]["native_train"] += len(native["train"])
            counts[d]["dense"] += dense_count
            counts[d]["val"] += len(native["val"])
            counts[d]["test"] += len(native["test"])
            counts[d]["excluded"] += len(native["excluded"])
            totals["native_train"] += len(native["train"])
            totals["dense"] += dense_count
            totals["val"] += len(native["val"])
            totals["test"] += len(native["test"])
            totals["excluded"] += len(native["excluded"])
            for k in ("train", "val", "test"):
                if not native[k]:
                    raise Abort(f"{vk}: empty {k} pool")
            manifest["variables"][vk] = {
                "group": v.group, "domain": v.domain, "frequency": (
                    "daily" if v.group in timesx_data.DAILY_GROUPS else "weekly"),
                "n": n, "n_train_obs": n_train_obs,
                "T_train": Ttr.isoformat(), "T_test": Tte.isoformat(),
                "fallback_std": fb_std,
                "native": native, "sample_start_idx": sample_start,
                "dense_count": dense_count,
                "dense_native_links": links,
            }
            series_off.append(series_pos)
            series_len.append(n)
            series_pos += n
            series_flat.append(vals)
            var_keys_out.append(vk)
            native_off.append(len(native_starts_flat))
            native_starts_flat.extend(starts_sorted)

    print("[prepare] 4) count checks vs spec table")
    et = CFG["expected_totals"]
    for k, want in (("native_train", et["native_train"]), ("dense", et["dense_train"]),
                    ("val", et["val"]), ("test", et["test"]),
                    ("excluded", et["excluded"])):
        if totals[k] != want:
            raise Abort(f"total {k}: {totals[k]} != expected {want}")
    for row in CFG["expected_domain_table"]:
        d = row["domain"]
        c = counts[d]
        for key, want in (("native_train", row["native_train"]),
                          ("dense", row["dense"]), ("val", row["val"]),
                          ("test", row["test"])):
            if c[key] != want:
                raise Abort(f"domain {d} {key}: {c[key]} != expected {want}")

    with open(SPLIT_COUNTS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "n_vars", "native_train", "dense", "val", "test",
                    "excluded", "U_d", "S_d"])
        for row in CFG["expected_domain_table"]:
            d = row["domain"]
            c = counts[d]
            u, s = u_d_s_d(c["dense"])
            w.writerow([d, c["vars_"], c["native_train"], c["dense"], c["val"],
                        c["test"], c["excluded"], u, s])
        u_sum = sum(u_d_s_d(counts[d]["dense"])[0] for d in DOMAIN_NAMES)
        w.writerow(["TOTAL", et["n_vars"], totals["native_train"],
                    totals["dense"], totals["val"], totals["test"],
                    totals["excluded"], u_sum, ""])

    print("[prepare] 5) data cache")
    np.savez_compressed(
        DATA_CACHE,
        **{"var_keys": np.array(var_keys_out),
           "series": np.concatenate(series_flat),
           "series_off": np.array(series_off, dtype=np.int64),
           "series_len": np.array(series_len, dtype=np.int64),
           "native_starts": np.array(native_starts_flat, dtype=np.int64),
           "native_off": np.array(native_off, dtype=np.int64)})
    # write-then-read-back self-check: the cache must reproduce the built
    # per-var series bit-for-bit through (series_off, series_len)
    with np.load(DATA_CACHE, allow_pickle=False) as zc:
        c_keys = [str(k) for k in zc["var_keys"]]
        c_ser = zc["series"]
        c_off = zc["series_off"]
        c_len = zc["series_len"]
        assert c_keys == var_keys_out, "cache key order drift"
        for i in (0, len(c_keys) // 2, len(c_keys) - 1):
            o, n = int(c_off[i]), int(c_len[i])
            if not np.array_equal(c_ser[o:o + n], series_flat[i]):
                raise Abort(f"cache self-check failed for {var_keys_out[i]}")
    print(f"[prepare] cache self-check ok ({series_pos} scalars, "
          f"{len(var_keys_out)} vars)")

    print("[prepare] 6) old-cache cross-check")
    cache_status = {"available_vars": 0, "missing_vars": [], "ids_total": 0,
                    "ids_missing_in_manifest": []}
    for vk in var_keys_out:
        ids = old_cache_ids(vk)
        if ids is None:
            cache_status["missing_vars"].append(vk)
            continue
        cache_status["available_vars"] += 1
        cache_status["ids_total"] += len(ids)
        for sid in ids:
            if sid not in manifest["variables"][vk]["sample_start_idx"]:
                cache_status["ids_missing_in_manifest"].append(sid)
    manifest["old_cache"] = cache_status

    manifest["counts"] = {d: {k: v for k, v in counts[d].items() if k != "vars_"}
                          for d in DOMAIN_NAMES}
    manifest["totals"] = totals
    atomic_write_json(manifest, SPLIT_MANIFEST)

    print("[prepare] 7) protocol + baseline snapshot")
    ckpt_p = os.path.join(REPO_ROOT, CFG["model"]["ckpt_rel"])
    proto = {"protocol_version": CFG["protocol_version"],
             "config": CFG,
             "zip_path_used": zp,
             "zip_sha256": sha256_file(zp),
             "ckpt_sha256": sha256_file(ckpt_p) if os.path.isfile(ckpt_p) else None,
             "ckpt_verified": os.path.isfile(ckpt_p),
             "core_md5": {rel: md5_file(os.path.join(REPO_ROOT, rel))
                          for rel in CFG["model"]["core_md5"]
                          if os.path.isfile(os.path.join(REPO_ROOT, rel))},
             "data_cache_sha256": sha256_file(DATA_CACHE),
             "split_manifest_sha256": None,
             "code_fingerprint": code_fingerprint(),
             "environment": {"python": sys.version.split()[0],
                             "numpy": np.__version__},
             "created": dt.datetime.now().isoformat(timespec="seconds"),
             "rng_families": CFG["sampling"]["rng_families"],
             "totals": totals,
             "U_d": {d: u_d_s_d(counts[d]["dense"])[0] for d in DOMAIN_NAMES}}
    atomic_write_json(proto, PROTOCOL_JSON)
    proto["split_manifest_sha256"] = sha256_file(SPLIT_MANIFEST)
    atomic_write_json(proto, PROTOCOL_JSON)

    baseline = build_baseline()
    atomic_write_json(baseline, BASELINE_JSON)
    print(f"[prepare] baseline: {len(baseline['static'])} static files hashed, "
          f"{len(baseline['dynamic_seen'])} dynamic files listed")

    print(f"[prepare] DONE: totals {totals}; U_d sum={sum(proto['U_d'].values())}")


if __name__ == "__main__":
    try:
        main()
    except Abort as e:
        print(f"[prepare] ABORT: {e}")
        sys.exit(2)
