#!/usr/bin/env python3
"""S2 verify-cache: read-only validation of the old P=1 cache against the frozen
full-window manifest, plus fingerprint binding of the old cache manifest to the
old splits.json (never skipped). Produces reuse_plan.json (verdicts here are
"reuse_candidate"; S3 preflight may flip variables to "reinfer").
"""
import argparse
import json
import os
import time

import numpy as np

import common as C
import protocol as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.isfile(C.REUSE_PLAN) and not args.force:
        raise SystemExit(f"{C.REUSE_PLAN} exists; pass --force to redo")

    t0 = time.time()
    inv = C.load_json(C.INV_PATH)
    vars_meta = {v["var_key"]: v for v in C.load_json(C.VARS_PATH)}
    var_rows = inv["windows"]  # informational count only
    rows_by_var = {}
    for w in inv["windows"]:
        rows_by_var.setdefault(w["var_key"], {})[w["sample_id"]] = w

    import record_fingerprint as RF  # read-only helpers from the old experiment

    old_manifest = C.load_json(os.path.join(P.OLD_CACHE, "manifest.json"))
    fp = old_manifest.get("fingerprint", {})

    # --- binding checks (amendment: old splits_sha256 MUST be verified, not skipped)
    binding = {}
    binding["zip_sha256_ok"] = fp.get("zip_sha256") == P.ZIP_SHA256
    binding["ckpt_sha256_ok"] = fp.get("ckpt_sha256") == P.CKPT_SHA256
    binding["splits_sha256_ok"] = fp.get("splits_sha256") == C.sha256_file(P.OLD_SPLITS)
    binding["splits_sha256"] = fp.get("splits_sha256")
    binding["run_config_ok"] = {
        k: fp.get("run_config", {}).get(k) for k in
        ("arch", "finetune_type", "context_len", "pred_len", "norm_const_r",
         "align_const_c", "interpolation", "precision", "tf32")}
    # key sets follow the OLD manifest's own definitions (record_fingerprint.py):
    # forward = {timesx_data, infer, record_fingerprint}; selection = {select_period, metrics_util}
    binding["code_md5_forward_current"] = C.code_md5_map(
        ["timesx_data.py", "infer.py", "record_fingerprint.py"], P.OLD_DIR)
    binding["code_md5_forward_match"] = all(
        fp.get("code_md5_forward", {}).get(k) == v
        for k, v in binding["code_md5_forward_current"].items())
    binding["code_md5_selection_match"] = all(
        fp.get("code_md5_selection", {}).get(k) == v for k, v in
        C.code_md5_map(["select_period.py", "metrics_util.py"], P.OLD_DIR).items())
    binding["env"] = fp.get("env")
    splits_calib = {vk: float(e["calib_std"]) for vk, e in
                    C.load_json(P.OLD_SPLITS)["variables"].items()}
    hard_fails = [k for k in ("zip_sha256_ok", "ckpt_sha256_ok", "splits_sha256_ok",
                              "code_md5_forward_match", "code_md5_selection_match") if not binding[k]]
    if hard_fails:
        raise SystemExit(f"old cache manifest binding FAILED: {hard_fails}")
    if binding["run_config_ok"] != {"arch": "mae_base", "finetune_type": "none",
                                    "context_len": 96, "pred_len": 12, "norm_const_r": 0.4,
                                    "align_const_c": 0.4, "interpolation": "bilinear",
                                    "precision": "fp32", "tf32": False}:
        raise SystemExit(f"old cache run_config mismatch: {binding['run_config_ok']}")
    for vk, sc in splits_calib.items():
        rc = float(vars_meta[vk]["calib_std"])
        if abs(rc - sc) > 1e-12 * max(1.0, abs(sc)):
            raise SystemExit(f"{vk}: recomputed calib_std {rc!r} vs splits {sc!r} beyond 1ulp tolerance")

    # --- entries key set
    entries = old_manifest.get("entries", {})
    by_p = {"P1": 0, "P5": 0, "P52": 0}
    for k in entries:
        for suf in by_p:
            if k.endswith(f"__{suf}"):
                by_p[suf] += 1
    if by_p["P1"] != P.N_VARS:
        raise SystemExit(f"P1 entries {by_p['P1']} != {P.N_VARS} vars")

    arrays = np.load(C.ARRAYS_PATH)
    past_all, target_all = arrays["past"], arrays["target"]

    variables = {}
    issues_all = []
    reuse_windows = 0
    for vk in sorted(vars_meta):
        key = f"{vk}__P1"
        ent = entries.get(key)
        issues = []
        if ent is None:
            variables[vk] = {"verdict": "pending", "issues": ["missing P1 entry"]}
            continue
        npz_path = os.path.join(P.OLD_CACHE, ent["file"])
        if not os.path.isfile(npz_path):
            variables[vk] = {"verdict": "pending", "file": ent["file"], "issues": ["npz missing"]}
            continue
        with np.load(npz_path, allow_pickle=False) as z:
            keys = set(z.files)
            if keys != P.OLD_NPZ_KEYS:
                issues.append(f"key set {sorted(keys)}")
                variables[vk] = {"verdict": "pending", "file": ent["file"], "issues": issues}
                continue
            sids = [str(s) for s in z["sample_ids"]]
            pred = z["pred"]
            true = z["true"]
            mu = z["mu"]
            win_std = z["win_std"]
            calib = z["calib_std"]
            geom = json.loads(str(z["geom"]))

        frozen = rows_by_var[vk]
        # cache scope = old test+val rows only
        old_scope = {sid for sid, w in frozen.items()
                     if w["split_in_old_protocol"] in ("test", "val")}
        if len(sids) != len(set(sids)):
            issues.append("duplicate ids in npz")
        if set(sids) != old_scope:
            issues.append(f"id set mismatch (npz {len(sids)} vs frozen {len(old_scope)})")
        if ent.get("n") != len(sids):
            issues.append(f"manifest n {ent.get('n')} != npz rows {len(sids)}")
        if pred.shape != (len(sids), P.PRED_LEN) or not np.isfinite(pred).all():
            issues.append(f"pred shape/finiteness {pred.shape}")
        if true.shape != (len(sids), P.PRED_LEN):
            issues.append(f"true shape {true.shape}")

        # per-window checks by id
        bad_target = bad_std = bad_mu = bad_calib = 0
        for i, sid in enumerate(sids):
            w = frozen.get(sid)
            if w is None:
                issues.append(f"id {sid} not in frozen manifest")
                break
            if not np.array_equal(true[i], target_all[w["flat_index"]]):
                bad_target += 1
            ws = win_std[i]
            if not (ws == w["win_std"] or abs(float(ws) - w["win_std"]) <= 1e-12 * max(1.0, abs(w["win_std"]))):
                bad_std += 1
            pv = past_all[w["flat_index"]]
            if not (float(mu[i]) == float(pv.mean())):
                bad_mu += 1
            # npz calib was filled from splits.json -> compare to that value exactly;
            # our recomputed vars.json calib may differ by <=1 ulp (checked below)
            if float(calib[i]) != splits_calib[vk]:
                bad_calib += 1
        if bad_target:
            issues.append(f"target mismatch x{bad_target}")
        if bad_std:
            issues.append(f"win_std mismatch x{bad_std}")
        if bad_mu:
            issues.append(f"mu mismatch x{bad_mu}")
        if bad_calib:
            issues.append(f"calib_std mismatch x{bad_calib}")
        want_geom = {"pad_left": 0, "pad_right": 0, "num_patch_input": 4}
        if any(geom.get(k) != v for k, v in want_geom.items()):
            issues.append(f"geom {geom}")

        verdict = "reuse_candidate" if not issues else "pending"
        if verdict == "reuse_candidate":
            reuse_windows += len(sids)
        else:
            issues_all.extend(f"{vk}: {i}" for i in issues)
        variables[vk] = {
            "verdict": verdict,
            "n": len(sids),
            "file": ent["file"],
            "sha256": C.sha256_file(npz_path),
            "manifest_geom": ent.get("geom"),
            "issues": issues,
        }

    n_pending = sum(1 for v in variables.values() if v["verdict"] != "reuse_candidate")
    plan = {
        "stage": "S2_verify_cache",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "old_cache_dir": P.OLD_CACHE,
        "binding": binding,
        "entries_check": {"total": len(entries), **by_p},
        "variables": variables,
        "totals": {
            "reuse_candidate_vars": P.N_VARS - n_pending,
            "pending_vars": n_pending,
            "reuse_candidate_windows": reuse_windows,
            "expected_reuse_windows": P.N_CACHE_P1_EXPECT,
        },
        "issues": issues_all,
    }
    if reuse_windows != P.N_CACHE_P1_EXPECT:
        print(f"[verify-cache] WARNING: reuse windows {reuse_windows} != expected "
              f"{P.N_CACHE_P1_EXPECT}; {n_pending} vars pending — see issues")
    C.dump_json(plan, C.REUSE_PLAN)
    C.mark_done("s2_verify_cache", {"reuse_candidate_windows": reuse_windows,
                                    "pending_vars": n_pending,
                                    "sha": {"reuse_plan.json": C.sha256_file(C.REUSE_PLAN)},
                                    "input_guards": C.stage_input_guards("verify")})
    print(f"[verify-cache] OK: {plan['totals']['reuse_candidate_vars']}/190 vars reuse_candidate, "
          f"{reuse_windows} windows, {n_pending} pending ({time.time()-t0:.1f}s)")
    if issues_all:
        print("issues:")
        for i in issues_all[:20]:
            print("  -", i)


if __name__ == "__main__":
    main()
