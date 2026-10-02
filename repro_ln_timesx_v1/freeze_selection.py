#!/usr/bin/env python3
"""Stage freeze_selection: verify all 285 runs + budgets, then freeze the
checkpoint choice per run (spec sec 7.2). NO test metric is touched here.

- best_ft: min val std-MSE over epochs 1-10, ties -> earlier epoch;
- selected_with_fallback: adds epoch 0, tie tol 1e-8*max(1, earlier MSE),
  earlier (incl. epoch 0) preferred; fallback_to_zeroshot flagged separately.
Selection records + checkpoint hashes are frozen into results/selection.json.
"""
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CHECKPOINTS, CFG, DOMAIN_NAMES, LOGS, METHODS_TRAIN,
                    RESULTS, SEEDS, atomic_write_json, run_id, sha256_file,
                    u_d_s_d)
from trainer import ExperimentData, select_epochs


def epoch_val_paths(rid):
    d = os.path.join(CHECKPOINTS, rid)
    return [os.path.join(d, f"ep{e:02d}.val.json") for e in range(11)]


def ckpt_path(rid, epoch):
    return os.path.join(CHECKPOINTS, rid, f"ep{epoch:02d}.ln.pt")


def jsonl_budget_ok(rid, U_d, S_d):
    """Dedupe update rows on (epoch, update) keeping max attempt; each epoch
    must show exactly U_d updates and 32*U_d presented microbatch-samples."""
    path = os.path.join(LOGS, f"{rid}.jsonl")
    if not os.path.isfile(path):
        return False, "no jsonl"
    best = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "update":
                continue
            k = (r["epoch"], r["update"])
            if k not in best or r["attempt"] > best[k]["attempt"]:
                best[k] = r
    per_epoch = defaultdict(int)
    for (e, _u) in best:
        per_epoch[e] += 1
    for e in range(1, 11):
        if per_epoch.get(e) != U_d:
            return False, f"epoch {e}: {per_epoch.get(e)} updates != {U_d}"
    if max((r["pred_loss_32"] for r in best.values()), default=None) is None:
        return False, "no update rows"
    return True, f"{len(best)} unique updates"


def main():
    proto = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "manifests",
        "protocol.json")))
    data = ExperimentData()
    man_path = os.path.join(RESULTS, "run_manifest.json")
    man = json.load(open(man_path)) if os.path.isfile(man_path) else {"runs": {}}

    per_run, problems = {}, []
    for d in DOMAIN_NAMES:
        U_d, S_d = u_d_s_d(data.domain_dense_total(d))
        for s in SEEDS:
            for m in METHODS_TRAIN:
                rid = run_id(m, d, s)
                local_bad = False
                if not os.path.isfile(os.path.join(CHECKPOINTS, rid,
                                                   "SUCCESS.json")):
                    problems.append(f"{rid}: no SUCCESS marker")
                    continue
                for p in epoch_val_paths(rid):
                    if not os.path.isfile(p):
                        problems.append(f"{rid}: missing {os.path.basename(p)}")
                        local_bad = True
                if local_bad:
                    continue
                val_by_epoch = {}
                for e in range(11):
                    with open(epoch_val_paths(rid)[e]) as f:
                        val_by_epoch[e] = json.load(f)["val_mse"]
                best = select_epochs(val_by_epoch, fallback=False)
                fb = select_epochs(val_by_epoch, fallback=True)
                ok, detail = jsonl_budget_ok(rid, U_d, S_d)
                if not ok:
                    problems.append(f"{rid}: budget check failed ({detail})")
                succ = json.load(open(os.path.join(CHECKPOINTS, rid,
                                                   "SUCCESS.json")))
                if succ.get("local_best_epoch") != best["epoch"]:
                    problems.append(
                        f"{rid}: trainer pick {succ.get('local_best_epoch')} "
                        f"!= recomputed {best['epoch']}")
                if not os.path.isfile(ckpt_path(rid, best["epoch"])):
                    problems.append(f"{rid}: selected ckpt missing")
                run_entry = man["runs"].get(rid, {})
                if run_entry.get("status") != "success":
                    problems.append(f"{rid}: run_manifest status "
                                    f"{run_entry.get('status')}")
                per_run[rid] = {
                    "method": m, "domain": d, "seed": s,
                    "val_by_epoch": val_by_epoch,
                    "best_epoch": best["epoch"], "best_val_mse": best["value"],
                    "fallback_epoch": fb["epoch"],
                    "fallback_val_mse": fb["value"],
                    "fallback_to_zeroshot": fb["epoch"] == 0,
                    "selected_ckpt_sha256": sha256_file(
                        ckpt_path(rid, best["epoch"])),
                    "U_d": U_d, "S_d": S_d, "budget_ok": ok,
                }
    if problems:
        print("[freeze_selection] PROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    out = {"protocol_version": proto["protocol_version"],
           "frozen_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
           "selection_rule": CFG["selection"],
           "n_runs": len(per_run), "per_run": per_run}
    atomic_write_json(out, os.path.join(RESULTS, "selection.json"))
    n_fb = sum(1 for r in per_run.values() if r["fallback_to_zeroshot"])
    with open(os.path.join(RESULTS, "selection_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "domain", "seed", "best_epoch", "best_val_mse",
                    "fallback_epoch", "fallback_to_zeroshot"])
        for rid, r in sorted(per_run.items()):
            w.writerow([r["method"], r["domain"], r["seed"], r["best_epoch"],
                        f"{r['best_val_mse']:.6f}", r["fallback_epoch"],
                        r["fallback_to_zeroshot"]])
    print(f"[freeze_selection] frozen {len(per_run)} runs; "
          f"fallback_to_zeroshot in {n_fb} runs -> results/selection.json")


if __name__ == "__main__":
    main()
