"""36-group LN full-shot training driver (EXP-028) with tier-1 guards,
fingerprint-gated resume, and quarantine-on-drift.

Requires out_root/preflight/preflight_report.json with all_pass=true
(live-checked, never sha-pinned — wall-clock content). Guards pin protocol
pins, weights sha, code md5s, repo commit, EXP-027 selected_p sha, LN
training config, and expected LN inventory. Skip rule: state==done AND
group fingerprint unchanged AND ckpt/meta/val artifacts verify; anything
else is quarantined to quarantine/<gid>/ and retrained FROM ORIGINAL
WEIGHTS + FIXED SEED (a partial retrain is never counted as continuing
this run's budget accounting; steps are asserted per group anyway).

Final gates: 36/36 done, sum(steps) == 236, sum(n_train) == 55,836.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

from repro_timemmd_common_ln_v1 import protocol as P
from repro_timemmd_common_ln_v1.common import core_guards, die, dump_json, \
    guards_equal, load_json, setup_deterministic_env, sha256_of
from repro_timemmd_common_ln_v1.ln_train import group_fingerprint, \
    train_config, train_one_group


def group_artifacts_ok(out_dir, dom, h, want_val):
    npz = out_dir / P.ckpt_npz_name(dom, h)
    meta_p = out_dir / P.ckpt_meta_name(dom, h)
    val_p = out_dir / P.val_npz_name(dom, h)
    if not (npz.exists() and meta_p.exists() and val_p.exists()):
        return False
    try:
        meta = load_json(meta_p)
        if meta.get("gid") != P.group_id(dom, h):
            return False
        if meta.get("npz_sha256") != sha256_of(npz):
            return False
        if meta.get("steps") != P.expected_steps(
                int(meta.get("n_train_windows", -1))):
            return False
        arr = np.load(str(val_p))
        for k in ("pred", "truth", "window_index"):
            if k not in arr.files:
                return False
        if arr["pred"].shape[0] != want_val:
            return False
        if not np.array_equal(np.sort(arr["window_index"]),
                              np.arange(want_val)):
            return False
    except Exception:
        return False
    return True


def quarantine_group(out_dir, dom, h):
    gid = P.group_id(dom, h)
    qdir = out_dir / "quarantine" / gid
    qdir.mkdir(parents=True, exist_ok=True)
    for f in (out_dir / P.ckpt_npz_name(dom, h),
              out_dir / P.ckpt_meta_name(dom, h),
              out_dir / P.val_npz_name(dom, h),
              out_dir / P.train_log_name(dom, h)):
        if f.exists():
            dst = qdir / f.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(f), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--groups", default=None,
                    help="comma-separated Domain_H ids (debug reruns)")
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_dir = Path(args.out_root or P.out_root())
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = load_json(out_dir / "preflight" / "preflight_report.json") \
        if (out_dir / "preflight" / "preflight_report.json").exists() \
        else None
    if pf is None or not pf.get("all_pass") or pf.get("errors"):
        die("[TIER-1] preflight report missing or not all_pass; "
            "run preflight first")
    microbatch_final = int(pf.get("b5", {}).get("microbatch_final",
                                                P.MICROBATCH))

    # EXP-027 selected_p.json copied in + sha pinned (source of FIXED_P)
    sel_src = P.exp027_out() / "selected_p.json"
    if not sel_src.exists():
        die("[TIER-1] EXP-027 selected_p.json missing: %s" % sel_src)
    sel_sha = sha256_of(sel_src)
    if sel_sha != P.EXP027_SELECTED_P_SHA256:
        die("[TIER-1] EXP-027 selected_p sha mismatch: %s" % sel_sha)
    sel_dst = out_dir / "selected_p.json"
    if not sel_dst.exists() or sha256_of(sel_dst) != sel_sha:
        shutil.copyfile(str(sel_src), str(sel_dst))
        if sha256_of(sel_dst) != sel_sha:
            die("[TIER-1] selected_p copy failed")

    guards = core_guards(pinfo, P.BATCH, sel_sha, None)
    guards["ln"] = {
        "microbatch_final": microbatch_final,
        "lr": P.LN_LR, "betas": list(P.LN_BETAS), "eps": P.LN_EPS,
        "weight_decay": P.LN_WD, "epochs": P.EPOCHS,
        "seed": P.SEED, "shuffle_seed": P.SEED,
        "expected_ln_tensors": P.EXPECTED_LN_TENSORS,
        "expected_ln_scalars": P.EXPECTED_LN_SCALARS,
        "exp027_selected_p_sha256": P.EXP027_SELECTED_P_SHA256,
        "exp027_overall_mse": P.EXP027_OVERALL_MSE,
        "exp027_overall_mae": P.EXP027_OVERALL_MAE,
        "fixed_p": P.FIXED_P,
        "train_windows_total": P.TRAIN_WINDOWS_TOTAL,
        "train_steps_total": P.TRAIN_STEPS_TOTAL,
    }
    gpath = out_dir / "guards.json"
    if gpath.exists():
        ok, diff = guards_equal(load_json(gpath), guards)
        if not ok:
            die("[TIER-1] guards mismatch (%s); refusing to train. "
                "Delete guards.json only after human review."
                % ",".join(diff))
    else:
        dump_json(gpath, guards)

    splits = pinfo["splits"]
    groups = P.all_groups()
    if args.groups:
        want = set(args.groups.split(","))
        groups = [g for g in groups if P.group_id(*g) in want]

    status_path = out_dir / "status.json"
    status = load_json(status_path) if status_path.exists() else \
        {"groups": {}}

    n_train_sum = 0
    steps_sum = 0
    for dom, h in groups:
        gid = P.group_id(dom, h)
        want_steps = P.expected_steps(
            int(splits[dom]["window_counts"][str(h)]["train"]))
        want_val = int(splits[dom]["window_counts"][str(h)]["val"])
        tcfg = train_config(microbatch_final)
        fp = group_fingerprint(pinfo, dom, h, tcfg)
        ent = status["groups"].get(gid, {})
        if ent.get("state") == "done" and ent.get("fingerprint") == fp \
                and group_artifacts_ok(out_dir, dom, h, want_val):
            print("SKIP %s (done, fingerprint+artifacts ok)" % gid,
                  flush=True)
            n_train_sum += int(ent.get("n_train_windows", 0))
            steps_sum += int(ent.get("steps", 0))
            continue
        if any((out_dir / f(dom, h)).exists() for f in
               (P.ckpt_npz_name, P.ckpt_meta_name, P.val_npz_name)) \
                or (out_dir / P.train_log_name(dom, h)).exists():
            quarantine_group(out_dir, dom, h)
            print("QUARANTINED stale %s" % gid, flush=True)
        status["groups"][gid] = {"state": "running", "fingerprint": fp}
        dump_json(status_path, status)
        res = train_one_group(pinfo, dom, h, out_dir,
                              microbatch_final=microbatch_final)
        if res["steps"] != want_steps:
            die("[TIER-1] %s steps %d != %d" % (gid, res["steps"],
                                                want_steps))
        log_path = out_dir / P.train_log_name(dom, h)
        res["fingerprint"] = fp
        dump_json(log_path, res)
        status["groups"][gid] = {
            "state": "done", "fingerprint": fp, "steps": res["steps"],
            "n_train_windows": res["n_train_windows"],
            "final_train_loss": res["final_train_loss"],
            "mean_train_loss": res["mean_train_loss"],
            "val_mse": res["val_mse"], "val_mae": res["val_mae"],
            "ckpt_npz_sha256": res["ckpt_npz_sha256"],
            "packed_sha256": res["packed_sha256"],
            "unchanged_ln": res["freeze_check"]["unchanged_ln"],
            "elapsed_s": res["elapsed_s"], "peak_mem_mb": res["peak_mem_mb"],
        }
        n_train_sum += res["n_train_windows"]
        steps_sum += res["steps"]
        dump_json(status_path, status)
        print("DONE %s steps=%d val_mse=%.6f (%.1fs)"
              % (gid, res["steps"], res["val_mse"], res["elapsed_s"]),
              flush=True)

    done = sum(1 for g in status["groups"].values()
               if g.get("state") == "done")
    print("RUN_TRAIN EXIT: done=%d/%d steps=%d train_windows=%d"
          % (done, len(groups), steps_sum, n_train_sum))
    if len(groups) == 36:
        if done != 36:
            die("[TIER-1] not all groups done")
        if steps_sum != P.TRAIN_STEPS_TOTAL:
            die("[TIER-1] total steps %d != %d"
                % (steps_sum, P.TRAIN_STEPS_TOTAL))
        if n_train_sum != P.TRAIN_WINDOWS_TOTAL:
            die("[TIER-1] total train windows %d != %d"
                % (n_train_sum, P.TRAIN_WINDOWS_TOTAL))
        print("TRAIN TOTALS OK: 36 groups, 236 steps, 55,836 windows")


if __name__ == "__main__":
    main()
