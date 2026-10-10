#!/usr/bin/env python3
"""S4 infer-fill: forward every window in the fill set (all windows minus those
covered by finally-reusable cached variables — SET DIFFERENCE) and store
per-variable P=1 npz under cache_all/.

Resume (amendment): a manifest entry is only skipped when the manifest
fingerprint matches the current state AND the entry file exists with the full
key set AND its recorded sha256 matches AND n and sample_ids match the wanted
list. Any mismatch => that variable is re-inferred.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

import common as C
import protocol as P


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def disable_tf32():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def load_model(device):
    import record_fingerprint as RF  # old module, read-only
    from visionts.model import VisionTS
    set_seeds(P.RUN_CONFIG["seed"])
    disable_tf32()
    ckpt = RF.ckpt_path(None)
    sha = C.sha256_file(ckpt)
    assert sha == P.CKPT_SHA256, f"ckpt sha mismatch {sha}"
    model = VisionTS(arch=P.RUN_CONFIG["arch"], finetune_type=P.RUN_CONFIG["finetune_type"],
                     ckpt_dir=os.path.dirname(ckpt), load_ckpt=True)
    model.update_config(context_len=P.RUN_CONFIG["context_len"],
                        pred_len=P.RUN_CONFIG["pred_len"],
                        periodicity=P.RUN_CONFIG["periodicity"],
                        norm_const=P.RUN_CONFIG["norm_const_r"],
                        align_const=P.RUN_CONFIG["align_const_c"],
                        interpolation=P.RUN_CONFIG["interpolation"])
    geom_tuple = (model.pad_left, model.pad_right, model.num_patch_input)
    assert geom_tuple == P.EXPECTED_GEOM_P1, f"geom {geom_tuple} != {P.EXPECTED_GEOM_P1}"
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def forward_rows(model, past_rows, device, batch_size):
    """past_rows: list of float64 arrays (96,). Returns float64 preds (n,12)."""
    import torch
    preds = np.zeros((len(past_rows), P.PRED_LEN), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(past_rows), batch_size):
            chunk = past_rows[i:i + batch_size]
            x = torch.from_numpy(np.stack(chunk)).float().unsqueeze(-1).to(device)
            assert x.shape[1] == P.CONTEXT_LEN
            y = model(x)
            preds[i:i + len(chunk)] = y.squeeze(-1).double().cpu().numpy()
    return preds


def reusable_vars():
    """Reuse VERDICTS from reuse_final.json (post-preflight). NOTE: reuse_final
    carries verdicts only — cache file names/hashes live in reuse_plan.json."""
    path = os.path.join(C.HERE, "reuse_final.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} missing — run preflight first")
    final = C.load_json(path)
    return {vk for vk, v in final["variables"].items() if v["verdict"] == "reuse"}, final


def cache_covered_sids(reuse_vars, plan):
    """Sample_ids actually present in the reusable old-cache npz files (the
    authoritative covered set — NOT 'all windows of reusable variables').
    `plan` MUST be reuse_plan.json (S2), whose entries carry file+sha256."""
    covered = set()
    for vk in sorted(reuse_vars):
        try:
            v = plan["variables"][vk]
        except KeyError:
            raise SystemExit(f"{vk}: reusable in reuse_final but missing from reuse_plan")
        if "file" not in v or "sha256" not in v:
            raise SystemExit(f"{vk}: reuse_plan entry lacks file/sha256 — re-run S2")
        with np.load(os.path.join(P.OLD_CACHE, v["file"]), allow_pickle=False) as z:
            covered.update(str(s) for s in z["sample_ids"])
    return covered


def compute_fill(inv, covered_sids):
    """Fill set = ALL windows − actually-covered cached windows (set difference)."""
    return [w for w in inv["windows"] if w["sample_id"] not in covered_sids]


def wanted_by_var(fill):
    by_var = {}
    for w in fill:
        by_var.setdefault(w["var_key"], []).append(w)
    return by_var


def validate_entry(npz_path, recorded_sha, n_want, want_ids):
    if not os.path.isfile(npz_path):
        return False, "file missing"
    with np.load(npz_path, allow_pickle=False) as z:
        if set(z.files) != P.NEW_NPZ_KEYS:
            return False, f"key set {sorted(z.files)}"
        if C.sha256_file(npz_path) != recorded_sha:
            return False, "sha256 mismatch"
        sids = [str(s) for s in z["sample_ids"]]
        if sids != want_ids:
            return False, "sample_ids mismatch"
        if z["pred"].shape != (n_want, P.PRED_LEN) or not np.isfinite(z["pred"]).all():
            return False, "pred invalid"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=P.RUN_CONFIG["batch_size"])
    ap.add_argument("--only-var", default=None, help="substring filter (smoke)")
    ap.add_argument("--out-dir", default=C.CACHE_DIR)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inv = C.load_json(C.INV_PATH)
    reuse_vars, final_plan = reusable_vars()

    # S3->S4 binding: the preflight verdict is only valid for the exact
    # data/model/config/code/env state it was produced under.
    fp = C.forward_fingerprint()
    binding = final_plan.get("forward_binding")
    if binding is None:
        raise SystemExit("reuse_final.json has no forward_binding — preflight predates the "
                         "binding fix; re-run preflight (--run)")
    if binding != fp:
        diff = [k for k in sorted(set(binding) | set(fp)) if binding.get(k) != fp.get(k)]
        raise SystemExit("reuse_final forward_binding mismatch on keys "
                         f"{diff} — model/config/code/data/env changed since S3; "
                         "re-run preflight before filling")

    # cache file names + hashes come from reuse_plan.json (S2), NOT reuse_final
    plan = C.load_json(C.REUSE_PLAN)
    missing = sorted(reuse_vars - set(plan["variables"]))
    if missing:
        raise SystemExit(f"{len(missing)} reuse_final vars absent from reuse_plan: "
                         f"{missing[:5]} — re-run S2/S3")
    covered = cache_covered_sids(reuse_vars, plan)
    fill = compute_fill(inv, covered)
    if len(fill) != P.TOTAL_WINDOWS - len(covered):
        raise SystemExit("fill set arithmetic broken: "
                         f"|fill|={len(fill)} != {P.TOTAL_WINDOWS}-|covered|={len(covered)}")
    expected_fill = final_plan["totals"]["fill_windows"]
    if len(fill) != expected_fill:
        raise SystemExit(f"fill set {len(fill)} != reuse_final.fill_windows {expected_fill}")
    by_var = wanted_by_var(fill)
    if args.only_var:
        by_var = {k: v for k, v in by_var.items() if args.only_var in k}
    n_fill = sum(len(v) for v in by_var.values())
    print(f"[fill] reusable vars={len(reuse_vars)} covered(cached)={len(covered)} "
          f"fill windows={n_fill} across {len(by_var)} vars "
          f"(expected fill={P.N_FILL_EXPECT} when all 6,413 cache windows reused, "
          "no smoke filter)")

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = {"fingerprint": fp, "entries": {}}
    if os.path.isfile(manifest_path) and not args.force:
        old = C.load_json(manifest_path)
        if old.get("fingerprint") == fp:
            manifest["entries"] = old.get("entries", {})
        else:
            raise SystemExit(
                "cache_all/manifest.json fingerprint mismatch (data/ckpt/config/code/"
                "window-manifest changed) — refusing to mix; pass --force to restart cache_all")

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda unavailable, falling back to cpu", file=sys.stderr)
        args.device = "cpu"
    model = load_model(args.device)

    arrays = np.load(C.ARRAYS_PATH)
    past_all, target_all = arrays["past"], arrays["target"]
    flat_of = {w["sample_id"]: w["flat_index"] for w in inv["windows"]}
    var_meta = {v["var_key"]: v for v in C.load_json(C.VARS_PATH)}

    t0 = time.time()
    n_fwd = 0
    entries = manifest["entries"]
    for vk in sorted(by_var):
        rows = sorted(by_var[vk], key=lambda w: w["flat_index"])
        want_ids = [w["sample_id"] for w in rows]
        key = f"{vk}__P1"
        npz_path = os.path.join(args.out_dir, f"{key}.npz")
        ent = entries.get(key)
        if ent:
            ok, why = validate_entry(npz_path, ent.get("sha256"), len(want_ids), want_ids)
            if ok:
                print(f"[fill] {vk}: resume skip ({ent.get('n')} rows)")
                continue
            print(f"[fill] {vk}: invalid entry ({why}) — re-inferring")
        past_rows = [past_all[flat_of[sid]] for sid in want_ids]
        preds = forward_rows(model, past_rows, args.device, args.batch_size)
        mu = np.array([float(past_all[flat_of[sid]].mean()) for sid in want_ids])
        wstd = np.array([float(past_all[flat_of[sid]].std(ddof=0)) for sid in want_ids])
        calib = np.full(len(want_ids), float(var_meta[vk]["calib_std"]), dtype=np.float64)
        trues = np.stack([target_all[flat_of[sid]] for sid in want_ids])
        geom = {"pad_left": model.pad_left, "pad_right": model.pad_right,
                "num_patch_input": model.num_patch_input,
                "num_patch": model.num_patch, "mask_ratio": model.mask_ratio}
        np.savez_compressed(npz_path, sample_ids=np.array(want_ids), pred=preds,
                            true=trues, mu=mu, win_std=wstd, calib_std=calib,
                            geom=np.array(json.dumps(geom)))
        entries[key] = {"n": len(want_ids), "geom": geom,
                        "file": os.path.basename(npz_path),
                        "sha256": C.sha256_file(npz_path)}
        manifest["entries"] = entries
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        n_fwd += len(want_ids)
        print(f"[fill] {vk}: {len(want_ids)} windows ({time.time()-t0:.0f}s)")

    print(f"[fill] done: forwards_executed={n_fwd}, entries={len(entries)}, "
          f"{time.time()-t0:.1f}s -> {args.out_dir}")
    if not args.only_var:
        C.mark_done("s4_infer_fill", {"fill_windows": n_fill,
                                      "manifest_sha256": C.sha256_file(manifest_path),
                                      "sha": {os.path.relpath(manifest_path, C.HERE):
                                              C.sha256_file(manifest_path)},
                                      "input_guards": C.stage_input_guards("fill")})


if __name__ == "__main__":
    main()
