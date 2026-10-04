#!/usr/bin/env python3
"""Stage 01 — GPU feature extraction (EXP-009).

Artifacts under features/:
  Z0__imagenet.npz            1,000 reference images through the ORIGINAL
                              ImageNet-pretrained MAE-base (encoded once,
                              reused across all 19 domains)
  Z0__timesx.npz              ALL 2,474 native test windows (past-96 only)
                              through Z0; per-domain subsets are slices of it
  F0__<domain>__s2021__imagenet.npz   Ed(I): same 1,000 images re-encoded by
                              the domain's fine-tuned encoder (new weights,
                              must re-encode)
  F0__<domain>__s2021__timesx.npz     that domain's test windows

Per-model verification (first TimesX batch of every model):
  - predictions bitwise-equal with instrumentation off vs on
  - captured vision_model.norm output == direct forward_encoder on the same
    captured image with the same deterministic noise (bitwise)
  - kept-token order == argsort(noise)[:56] and set == leftmost 4 columns
Identities are asserted on every batch for every artifact.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "exp009_common", os.path.join(HERE, "common.py"))
E9 = importlib.util.module_from_spec(_spec)
sys.modules["exp009_common"] = E9
_spec.loader.exec_module(E9)

LOG_PATH = os.path.join(E9.LOGS, "extract.jsonl")
BASELINE_PATH = os.path.join(E9.AUDIT, "baseline.json")


def log(row):
    E9.append_jsonl(LOG_PATH, {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **row})
    print(f"[extract] {json.dumps(row, ensure_ascii=False)}", flush=True)


def build_baseline():
    """Isolation snapshot of every static input EXP-009 reads (04 re-diffs)."""
    entries = {}
    for name, p in E9.REUSED_SOURCES:
        entries[f"reused/{name}"] = E9.sha256_file(p)
    vt = os.path.join(E9.REPO_ROOT, "visionts")
    for fn in sorted(os.listdir(vt)):
        if fn.endswith(".py"):
            entries[f"visionts/{fn}"] = E9.sha256_file(os.path.join(vt, fn))
    entries["ckpt/mae_visualize_vit_base.pth"] = E9.sha256_file(
        os.path.join(E9.REPO_ROOT, "ckpt", "mae_visualize_vit_base.pth"))
    for p in (E9.SPLIT_MANIFEST,
              os.path.join(E9.LN_DIR, "manifests", "protocol.json"),
              os.path.join(E9.LN_DIR, "manifests", "split_counts.csv"),
              E9.DATA_CACHE, E9.SELECTION_JSON, E9.DOMAIN_SUMMARY_CSV,
              os.path.join(E9.LN_DIR, "results", "overall.csv"),
              E9.FIG7_LIST_JSON):
        entries[f"ln_inputs/{os.path.basename(p)}"] = E9.sha256_file(p)
    for fn in sorted(os.listdir(E9.LN_PREDICTIONS)):
        if fn.endswith(".npz"):
            entries[f"ln_predictions/{fn}"] = E9.sha256_file(
                os.path.join(E9.LN_PREDICTIONS, fn))
    for root, _dirs, files in os.walk(E9.HERE):
        rel_root = os.path.relpath(root, E9.HERE)
        if rel_root.split(os.sep)[0] in ("features", "coords", "figures",
                                         "manifests", "audit", "logs",
                                         "__pycache__"):
            continue
        for fn in sorted(files):
            if fn.endswith((".py", ".sh", ".yaml")):
                rel = os.path.normpath(os.path.join(rel_root, fn))
                entries[f"exp009_code/{rel}"] = E9.sha256_file(
                    os.path.join(root, fn))
    E9.atomic_write_json(
        {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "entries": entries},
        BASELINE_PATH)
    log({"event": "baseline", "n_entries": len(entries)})


def load_cache_index():
    if os.path.isfile(E9.CACHE_INDEX):
        with open(E9.CACHE_INDEX) as f:
            return json.load(f)
    return {}


def save_cache_index(idx):
    E9.atomic_write_json(idx, E9.CACHE_INDEX)


def artifact_up_to_date(idx, name, kind, weight_sha):
    path = os.path.join(E9.FEATURES, name)
    if not os.path.isfile(path):
        return False
    ent = idx.get(name)
    if not ent:
        return False
    key, _comps = E9.feature_cache_key(kind, weight_sha)
    return ent.get("key") == key and ent.get("sha256") == E9.sha256_file(path)


def record_artifact(idx, name, kind, weight_sha, arrays):
    path = os.path.join(E9.FEATURES, name)
    sha = E9.sha256_file(path)
    key, comps = E9.feature_cache_key(kind, weight_sha)
    idx[name] = {"kind": kind, "sha256": sha, "key": key,
                 "key_components": comps,
                 "shape": {k: list(v.shape) for k, v in arrays.items()},
                 "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_cache_index(idx)


# ------------------------------------------------------------------ torch part
def instrument(model, fv):
    cap = {"image": None, "feat_norm": None}
    h1 = model.vision_model.register_forward_pre_hook(
        lambda m, inp: cap.__setitem__("image", inp[0]))
    h2 = model.vision_model.norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("feat_norm", o))
    box = {}
    orig = model.vision_model.forward_encoder

    def wrapped(img, mask_ratio, noise=None, **kw):
        out, mbool, ids = orig(img, mask_ratio, noise=noise, **kw)
        box["ids_restore"] = ids
        return out, mbool, ids

    model.vision_model.forward_encoder = wrapped
    return cap, box, (h1, h2), orig


def remove_instrumentation(model, handles, orig):
    for h in handles:
        h.remove()
    model.vision_model.forward_encoder = orig


def assert_keep_order(model, fv, ids_restore, batch, device):
    import torch
    noise = fv.noise_mask(model, batch, device)
    keep = torch.argsort(ids_restore, dim=1)[:, :E9.N_VISIBLE]
    keep_expected = torch.argsort(noise, dim=1)[:, :E9.N_VISIBLE]
    assert torch.equal(keep, keep_expected), "kept-token order drift"
    assert set(keep[0].tolist()) == set(fv.expected_kept_set()), \
        "kept set != leftmost 4 columns (56 patches)"


def per_model_equivalence_check(model, fv, xt, device, tag):
    """First-TimesX-batch verification for one model (bitwise)."""
    import torch
    was = model.training
    model.eval()
    with torch.no_grad():
        pred_plain = model(xt)
    cap, box, handles, orig = instrument(model, fv)
    try:
        with torch.no_grad():
            pred_instr = model(xt)
        assert torch.equal(pred_plain, pred_instr), \
            f"{tag}: instrumentation changed predictions"
        feat = cap["feat_norm"]
        img = cap["image"]
        prod_ids = box["ids_restore"]
        assert feat.shape[1:] == (57, 768), f"{tag}: feat shape {feat.shape}"
        noise = fv.noise_mask(model, img.shape[0], device)
        lat_direct, _m, ids_direct = model.vision_model.forward_encoder(
            img, model.mask_ratio, noise=noise)
        assert torch.equal(lat_direct, feat), \
            f"{tag}: captured norm output != direct forward_encoder"
        assert torch.equal(prod_ids, ids_direct), \
            f"{tag}: production ids_restore != direct ids_restore"
        assert_keep_order(model, fv, ids_direct, img.shape[0], device)
        flat = feat.detach().float().reshape(img.shape[0], -1).cpu().numpy()
        assert flat.shape[1] == E9.FLAT_DIM and np.isfinite(flat).all()
    finally:
        remove_instrumentation(model, handles, orig)
        model.train(was)
    log({"event": "equivalence_check", "tag": tag, "batch": int(xt.shape[0]),
         "pred_bitwise_equal": True, "hook_equals_direct": True})


def encode_imagenet(model, fv, files, device, tag, idx, weight_sha):
    """1,000 images via direct forward_encoder, deterministic mask, list order."""
    import torch
    from PIL import Image
    tfm = fv.imagenet_transform()
    name = f"{tag}__imagenet.npz"
    if artifact_up_to_date(idx, name, f"{tag}__imagenet", weight_sha):
        log({"event": "skip", "artifact": name})
        return
    feats, t0 = [], time.time()
    with torch.no_grad():
        for i0 in range(0, len(files), E9.BATCH):
            chunk = files[i0:i0 + E9.BATCH]
            imgs = torch.stack([
                tfm(Image.open(os.path.join(E9.IMAGENET_VAL_DIR,
                                            fn)).convert("RGB"))
                for fn in chunk]).to(device)
            noise = fv.noise_mask(model, imgs.shape[0], device)
            lat, _m, ids = model.vision_model.forward_encoder(
                imgs, model.mask_ratio, noise=noise)
            assert_keep_order(model, fv, ids, imgs.shape[0], device)
            f = lat.detach().float().reshape(imgs.shape[0], -1).cpu().numpy()
            assert f.shape[1] == E9.FLAT_DIM and np.isfinite(f).all()
            feats.append(f.astype(np.float32))
    arr = np.concatenate(feats)
    assert arr.shape == (E9.IMAGENET_N, E9.FLAT_DIM)
    np.savez_compressed(os.path.join(E9.FEATURES, name),
                        feats=arr, filenames=np.array(files),
                        model=np.array(tag))
    record_artifact(idx, name, f"{tag}__imagenet", weight_sha,
                    {"feats": arr, "filenames": np.array(files)})
    log({"event": "artifact", "artifact": name, "n": int(arr.shape[0]),
         "secs": round(time.time() - t0, 1)})


def encode_timesx(model, fv, rows, device, tag, idx, weight_sha):
    """rows = [(domain, var_key, sample_id, x)] canonical order; real
    model.forward path."""
    import torch
    name = f"{tag}__timesx.npz"
    if artifact_up_to_date(idx, name, f"{tag}__timesx", weight_sha):
        log({"event": "skip", "artifact": name})
        return
    feats, vks, sids, doms, t0 = [], [], [], [], time.time()
    was = model.training
    model.eval()
    checked = False
    with torch.no_grad():
        for i0 in range(0, len(rows), E9.BATCH):
            chunk = rows[i0:i0 + E9.BATCH]
            xt = torch.tensor(np.stack([r[3] for r in chunk]),
                              dtype=torch.float32,
                              device=device).unsqueeze(-1)
            if not checked:
                per_model_equivalence_check(model, fv, xt, device, tag)
                checked = True
            cap, box, handles, orig = instrument(model, fv)
            try:
                model(xt)
                feat = cap["feat_norm"]
                assert_keep_order(model, fv, box["ids_restore"],
                                  xt.shape[0], device)
            finally:
                remove_instrumentation(model, handles, orig)
            f = feat.detach().float().reshape(xt.shape[0], -1).cpu().numpy()
            assert f.shape[1] == E9.FLAT_DIM and np.isfinite(f).all()
            feats.append(f.astype(np.float32))
            doms.extend(r[0] for r in chunk)
            vks.extend(r[1] for r in chunk)
            sids.extend(r[2] for r in chunk)
    model.train(was)
    arr = np.concatenate(feats)
    sids_a, vks_a, doms_a = np.array(sids), np.array(vks), np.array(doms)
    assert arr.shape == (len(rows), E9.FLAT_DIM)
    np.savez_compressed(os.path.join(E9.FEATURES, name),
                        feats=arr, sample_ids=sids_a, var_keys=vks_a,
                        domains=doms_a, model=np.array(tag))
    record_artifact(idx, name, f"{tag}__timesx", weight_sha,
                    {"feats": arr, "sample_ids": sids_a, "var_keys": vks_a})
    log({"event": "artifact", "artifact": name, "n": int(arr.shape[0]),
         "secs": round(time.time() - t0, 1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", default=None,
                    help="comma list for F0 (default: all 19; Z0 artifacts "
                         "are always full)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    E9.pause_if_requested("extract")
    free = E9.gpu_free_mib()
    need = int(E9.CFG["resources"]["gpu_gate_min_free_gib"]) * 1024
    if free < need:
        print(f"[extract] GPU free {free} MiB < required {need} MiB — abort",
              flush=True)
        sys.exit(4)
    log({"event": "gpu_gate", "free_mib": free, "need_mib": need})

    if not os.path.isfile(BASELINE_PATH):
        build_baseline()

    import torch
    fv = E9.ltfs_stack()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    assert device.startswith("cuda"), \
        "EXP-009 runs its GPU stage on cuda only (never CPU fallback)"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    files = E9.imagenet_list()
    assert os.path.isdir(E9.IMAGENET_VAL_DIR), E9.IMAGENET_VAL_DIR
    rows_all = E9.test_rows_all_domains()
    by_domain = {d: [(d, vk, sid, x) for dom, vk, sid, x in rows_all
                     if dom == d]
                 for d in E9.DOMAIN_NAMES}

    idx = load_cache_index()
    mae_sha = E9.input_hashes()["mae_base_ckpt"]

    # ---- Z0 (once): original ImageNet-pretrained MAE-base ----
    model = E9.lnc.load_model(device, finetune_type="none")
    model.eval()
    encode_imagenet(model, fv, files, device, "Z0", idx, mae_sha)
    encode_timesx(model, fv, rows_all, device, "Z0", idx, mae_sha)
    del model
    torch.cuda.empty_cache()

    # ---- F0 per domain (best_ft, seed 2021) ----
    domains = (args.domains.split(",") if args.domains else E9.DOMAIN_NAMES)
    assert set(domains) <= set(E9.DOMAIN_NAMES), domains
    ckpt_rows = []
    if os.path.isfile(E9.CKPT_TABLE):
        with open(E9.CKPT_TABLE) as f:
            ckpt_rows = json.load(f)
    done_domains = {r["domain"] for r in ckpt_rows}
    for d in domains:
        if d in done_domains:
            log({"event": "skip_domain", "domain": d})
            continue
        model = E9.lnc.load_model(device, finetune_type="ln")
        bind_rec = E9.bind_and_load_f0(model, d)
        model.eval()
        ln_sha = bind_rec["ln_pt_sha256"]
        encode_imagenet(model, fv, files, device,
                        f"F0__{d}__s{E9.F0_SEED}", idx, ln_sha)
        encode_timesx(model, fv, by_domain[d], device,
                      f"F0__{d}__s{E9.F0_SEED}", idx, ln_sha)
        ckpt_rows = [r for r in ckpt_rows if r["domain"] != d]
        ckpt_rows.append(bind_rec)
        E9.atomic_write_json(ckpt_rows, E9.CKPT_TABLE)
        del model
        torch.cuda.empty_cache()
        log({"event": "domain_done", "domain": d})

    log({"event": "extract_complete", "domains": len(set(domains) | done_domains)})


if __name__ == "__main__":
    main()
