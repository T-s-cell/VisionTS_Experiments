#!/usr/bin/env python3
"""Stage audit (final): isolation + integrity verification.

1. Baseline bidirectional diff — static scope (source/data/weights/frozen
   results) re-hashed vs audit/baseline.json; anything outside our dir is
   dynamic and only existence-diffed (co-tenant writes are legitimate).
2. Hash chain — zip / data_cache / ckpt / code fingerprint re-hashed vs
   manifests/protocol.json; every selection.json selected ckpt re-hashed.
3. 285 runs — SUCCESS markers, 11 epoch jsons, jsonl update budgets, local
   best-epoch recompute, selection recompute comparison.
Writes audit/audit_report.json; exit 1 on any violation.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (AUDIT, BASELINE_JSON, CHECKPOINTS, CFG, DATA_CACHE,
                    DOMAIN_NAMES, LOGS, METHODS_TRAIN, PREFLIGHT_JSON,
                    PROTOCOL_JSON, RESULTS, SEEDS, atomic_write_json,
                    code_fingerprint, environment_snapshot, expected_ln_names,
                    run_id, sha256_file, u_d_s_d, zip_path)
from trainer import ExperimentData, select_epochs
from freeze_selection import epoch_val_paths, jsonl_budget_ok


def diff_baseline():
    import prepare_data
    base = json.load(open(BASELINE_JSON))
    now = prepare_data.build_baseline()
    st_was, st_now = base["static"], now["static"]
    changed = [r for r in st_was if r in st_now
               and st_was[r]["sha256"] != st_now[r]["sha256"]]
    missing = [r for r in st_was if r not in st_now]
    added = [r for r in st_now if r not in st_was]
    dy_was, dy_now = set(base.get("dynamic_seen", {})), set(now["dynamic_seen"])
    dyn = {"new": sorted(dy_now - dy_was), "removed": sorted(dy_was - dy_now)}
    ok = not (changed or missing or added)
    return {"ok": ok, "static_changed": changed, "static_missing": missing,
            "static_added": added,
            "dynamic_new_n": len(dyn["new"]), "dynamic_new": dyn["new"][:20],
            "dynamic_removed_n": len(dyn["removed"]),
            "dynamic_removed": dyn["removed"][:20],
            "n_static": len(st_now)}


def hash_chain():
    from common import SPLIT_MANIFEST, actual_fingerprints
    proto = json.load(open(PROTOCOL_JSON))
    out = {}
    checks = []
    act = actual_fingerprints()
    got = sha256_file(zip_path())
    out["zip"] = {"expected": proto["zip_sha256"], "got": got}
    checks.append(("zip", got == proto["zip_sha256"]))
    got = sha256_file(DATA_CACHE)
    out["data_cache"] = {"expected": proto["data_cache_sha256"], "got": got}
    checks.append(("data_cache", got == proto["data_cache_sha256"]))
    got = sha256_file(SPLIT_MANIFEST)
    out["split_manifest"] = {"expected": proto.get("split_manifest_sha256"),
                             "got": got}
    checks.append(("split_manifest",
                   got == proto.get("split_manifest_sha256")))
    fp = code_fingerprint()
    out["code_fingerprint"] = {"expected": proto["code_fingerprint"], "got": fp}
    checks.append(("code_fingerprint", fp == proto["code_fingerprint"]))
    if proto.get("ckpt_sha256"):
        from common import REPO_ROOT
        ckpt = os.path.join(REPO_ROOT, CFG["model"]["ckpt_rel"])
        if os.path.isfile(ckpt):
            got = sha256_file(ckpt)
            out["ckpt"] = {"expected": proto["ckpt_sha256"], "got": got}
            checks.append(("ckpt", got == proto["ckpt_sha256"]))
        else:
            checks.append(("ckpt", False))
            out["ckpt"] = {"expected": proto["ckpt_sha256"], "got": "missing"}
    # preflight report must be bound to these same fingerprints
    if os.path.isfile(PREFLIGHT_JSON):
        pre = json.load(open(PREFLIGHT_JSON))
        bound = pre.get("fingerprints", {})
        pb = [("preflight.code", act["code"] == bound.get("code")),
              ("preflight.data_cache",
               act["data_cache"] == bound.get("data_cache")),
              ("preflight.split_manifest",
               act["split_manifest"] == bound.get("split_manifest"))]
        out["preflight_binding"] = {k: v for k, v in pb}
        checks.extend(pb)
    out["ok"] = all(v for _k, v in checks)
    out["failures"] = [k for k, v in checks if not v]
    return out


def verify_aggregate_outputs():
    """Hard checks on the aggregated tables (only if aggregate has run)."""
    p_overall = os.path.join(RESULTS, "overall.csv")
    if not os.path.isfile(p_overall):
        return {"skipped": "aggregate outputs not present yet"}
    problems = []
    import csv as _csv
    import math
    with open(p_overall) as f:
        rows = list(_csv.DictReader(f))
    methods = [r["method"] for r in rows]
    if methods[:2] != ["Z0", "N0"]:
        problems.append(f"overall.csv first rows {methods[:2]}")
    if sorted(methods[2:]) != sorted(METHODS_TRAIN):
        problems.append(f"overall.csv methods {methods[2:]}")
    for r in rows:
        for k in ("MSE", "MAE"):
            v = float(r[k].split("+/-")[0])
            if not math.isfinite(v):
                problems.append(f"non-finite {k} for {r['method']}")
    p_var = os.path.join(RESULTS, "test_variable_level.csv")
    with open(p_var) as f:
        vrows = list(_csv.DictReader(f))
    for m in ("Z0", "N0") + tuple(METHODS_TRAIN):
        n = sum(1 for r in vrows if r["method"] == m)
        if n != 190:
            problems.append(f"{m}: {n} var rows, expected 190")
    p_dom = os.path.join(RESULTS, "domain_summary.csv")
    with open(p_dom) as f:
        drows = list(_csv.DictReader(f))
    for m in ("Z0", "N0") + tuple(METHODS_TRAIN):
        doms = {r["domain"] for r in drows if r["method"] == m}
        if doms != set(DOMAIN_NAMES):
            problems.append(f"{m}: domain coverage {len(doms)}/19")
    return {"problems": problems, "ok": not problems}


def verify_runs(data, sel):
    problems, n_ok = [], 0
    sel_per_run = sel["per_run"] if sel else {}
    for d in DOMAIN_NAMES:
        U_d, S_d = u_d_s_d(data.domain_dense_total(d))
        for s in SEEDS:
            for m in METHODS_TRAIN:
                rid = run_id(m, d, s)
                succ_p = os.path.join(CHECKPOINTS, rid, "SUCCESS.json")
                if not os.path.isfile(succ_p):
                    problems.append(f"{rid}: no SUCCESS marker")
                    continue
                succ = json.load(open(succ_p))
                if succ.get("status") != "success":
                    problems.append(f"{rid}: status {succ.get('status')}")
                    continue
                vals = {}
                bad = False
                for e in range(11):
                    p = epoch_val_paths(rid)[e]
                    if not os.path.isfile(p):
                        problems.append(f"{rid}: missing ep{e:02d}.val.json")
                        bad = True
                        break
                    with open(p) as f:
                        vals[e] = json.load(f)["val_mse"]
                if bad:
                    continue
                ok, detail = jsonl_budget_ok(rid, U_d, S_d)
                if not ok:
                    problems.append(f"{rid}: budget check failed ({detail})")
                best = select_epochs(vals, fallback=False)
                if succ.get("local_best_epoch") != best["epoch"]:
                    problems.append(
                        f"{rid}: trainer pick {succ.get('local_best_epoch')} "
                        f"!= recompute {best['epoch']}")
                entry = sel_per_run.get(rid)
                if entry:
                    if entry["best_epoch"] != best["epoch"]:
                        problems.append(
                            f"{rid}: selection best_epoch {entry['best_epoch']} "
                            f"!= recompute {best['epoch']}")
                    ck = os.path.join(CHECKPOINTS, rid,
                                      f"ep{entry['best_epoch']:02d}.ln.pt")
                    if not os.path.isfile(ck):
                        problems.append(f"{rid}: selected ckpt missing")
                    elif sha256_file(ck) != entry["selected_ckpt_sha256"]:
                        problems.append(f"{rid}: selected ckpt hash drift")
                n_ok += 1
    return n_ok, problems


def main():
    problems = []
    rep = {"finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "environment": environment_snapshot()}

    if not os.path.isfile(BASELINE_JSON):
        problems.append("baseline.json missing — run prepare first")
        rep["baseline_diff"] = {"ok": False, "error": "missing"}
    else:
        rep["baseline_diff"] = diff_baseline()
        if not rep["baseline_diff"]["ok"]:
            problems.append("baseline diff: static files changed/added/removed")
        print(f"[audit] baseline diff: "
              f"changed={len(rep['baseline_diff']['static_changed'])} "
              f"added={len(rep['baseline_diff']['static_added'])} "
              f"missing={len(rep['baseline_diff']['static_missing'])}")

    rep["hash_chain"] = hash_chain()
    if not rep["hash_chain"]["ok"]:
        problems.append(f"hash chain failures: {rep['hash_chain']['failures']}")
    print(f"[audit] hash chain ok={rep['hash_chain']['ok']} "
          f"{rep['hash_chain'].get('failures', [])}")

    rep["aggregate_outputs"] = verify_aggregate_outputs()
    if rep["aggregate_outputs"].get("problems"):
        problems.extend(
            f"aggregate: {p}" for p in rep["aggregate_outputs"]["problems"])
        print(f"[audit] aggregate outputs: {len(problems)} problem(s)")

    sel_path = os.path.join(RESULTS, "selection.json")
    sel = json.load(open(sel_path)) if os.path.isfile(sel_path) else None
    data = ExperimentData()
    n_ok, run_problems = verify_runs(data, sel)
    rep["runs"] = {"n_expected": 285, "n_verified_ok": n_ok,
                   "problems": run_problems[:50], "n_problems": len(run_problems)}
    problems.extend(run_problems)
    print(f"[audit] runs verified: {n_ok}/285, "
          f"problems={len(run_problems)}")

    if sel:
        mism = [rid for rid, r in sel["per_run"].items()
                if r["method"] not in METHODS_TRAIN
                or r["seed"] not in SEEDS or r["domain"] not in DOMAIN_NAMES]
        if len(sel["per_run"]) != 285 or mism:
            problems.append("selection.json composition invalid")
    else:
        problems.append("selection.json missing — freeze_selection not run")

    if os.path.isfile(PREFLIGHT_JSON):
        pre = json.load(open(PREFLIGHT_JSON))
        rep["preflight_all_pass"] = pre.get("all_pass")
        if not pre.get("all_pass"):
            problems.append("preflight report shows all_pass=false")
    else:
        problems.append("preflight report missing")

    # artifact presence (informational)
    pred_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "predictions")
    rep["artifacts"] = {
        "selection": sel is not None,
        "overall_csv": os.path.isfile(os.path.join(RESULTS, "overall.csv")),
        "report_md": os.path.isfile(os.path.join(RESULTS, "report.md")),
        "n_pred_npz": sum(1 for f in os.listdir(pred_dir)
                          if f.endswith("__test.npz")) if os.path.isdir(
                              pred_dir) else 0}
    ln_n = len(expected_ln_names())
    rep["ln_tensor_count"] = ln_n
    if ln_n != 84:
        problems.append(f"LN name list has {ln_n} tensors, expected 84")

    rep["all_ok"] = not problems
    rep["violations"] = problems
    atomic_write_json(rep, os.path.join(AUDIT, "audit_report.json"))
    print(f"[audit] {'ALL OK' if rep['all_ok'] else 'VIOLATIONS'} — "
          f"{len(problems)} problem(s); report -> audit/audit_report.json")
    for p in problems[:20]:
        print("  -", p)
    sys.exit(0 if rep["all_ok"] else 1)


if __name__ == "__main__":
    main()
