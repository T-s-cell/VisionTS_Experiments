#!/usr/bin/env python3
"""Stage 03: extract 768-d encoder features.

- TimesX: features for ALL 1677 test windows are extracted ONCE through the
  production forward with runtime hooks. Batch composition replicates infer.py
  exactly (all val+test rows of a variable in stored cache order, sequential
  bs=64 chunks) so the same forwards reproduce the cached predictions within
  the declared rtol and features correspond to those predictions. The 117
  plotting windows are later taken as an ID-based subset of the SAME features.
- ImageNet: the SAME 2000 frozen files passed twice through forward_encoder:
  masked (production noise: [1,196] buffer expanded to [B,196]) and no-mask
  (mask_ratio=0, explicit arange noise -> natural token order).
"""
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

from common import (BATCH, DIRS, attach_capture, device_auto, ensure_dirs,
                    load_lists, load_model, noise_arange, noise_mask, parse_args,
                    pooled, pred_compare)

ZERO = os.path.dirname(HERE)
PRED_RTOL = 1e-4


def main():
    args = parse_args(__doc__)
    device = args.device or device_auto()
    ensure_dirs()
    model, ckpt = load_model(device)
    sel, allv, man = load_lists()

    # Frozen cache order = infer.py composition: ALL val+test rows of a
    # variable in stored npz order, forwarded in sequential bs=64 chunks.
    # Features must come from those same forwards (fp32 CUDA kernel selection
    # is batch-size dependent; features must match the cached predictions).
    from timesx_data import find_zip, iter_variables
    want = sorted({r["var_key"] for r in allv})
    order = {}
    for vk in want:
        d = np.load(os.path.join(ZERO, "cache", f"{vk}__P1.npz"), allow_pickle=False)
        order[vk] = [str(s) for s in d["sample_ids"]]
    need = {}
    for vk, ids in order.items():
        need.setdefault(vk, set()).update(ids)
    past = {}
    for v in iter_variables(find_zip(None)):
        wl = need.get(v.var_key)
        if not wl:
            continue
        for s in v.samples:
            if s.sample_id in wl:
                past[(v.var_key, s.sample_id)] = np.asarray(s.past_val, dtype=np.float32)
    n_rows = sum(len(v) for v in order.values())
    assert len(past) == n_rows, f"missing past values: {len(past)}/{n_rows}"

    # ---- TimesX: infer.py batch composition, production forward + hooks ----
    cap, handles = attach_capture(model)
    feats, vks, sids, worst = [], [], [], 0.0
    t0 = time.time()
    for vk in want:
        d = np.load(os.path.join(ZERO, "cache", f"{vk}__P1.npz"), allow_pickle=False)
        ids = order[vk]
        sp = [str(s) for s in d["splits"]]
        for i0 in range(0, len(ids), BATCH):
            chunk = ids[i0:i0 + BATCH]
            xc = torch.tensor(np.stack([past[(vk, s)] for s in chunk]),
                              dtype=torch.float32, device=device).unsqueeze(-1)
            with torch.no_grad():
                y = model(xc)
            feat = pooled(cap["feat_norm"]).detach().cpu().numpy()
            yc = y[:, :, 0].cpu().numpy()
            for k, gid in enumerate(chunk):
                gi = i0 + k
                _, rel = pred_compare(yc[k], d["pred"][gi])
                worst = max(worst, rel)
                assert rel <= PRED_RTOL, f"{vk} {gid}: rel={rel:.3e} exceeds rtol"
                if sp[gi] == "test":
                    feats.append(feat[k])
                    vks.append(vk)
                    sids.append(gid)
    for h in handles:
        h.remove()
    feat_all = np.stack(feats, 0).astype(np.float32)
    assert feat_all.shape == (len(allv), 768), feat_all.shape
    assert len(set(sids)) == len(sids) == 1677
    np.savez_compressed(os.path.join(DIRS["features"], "timesx_all.npz"),
                        feat=feat_all, var_key=np.array(vks), sample_id=np.array(sids))
    print(f"timesx_all: {feat_all.shape}, worst cache-pred rel={worst:.2e}, "
          f"{time.time() - t0:.1f}s")

    # ---- 117-window subset: by sample_id, from the SAME feature array ----
    sel_ids = {r["sample_id"] for r in sel}
    assert sel_ids <= set(sids)
    pos = {sid: i for i, sid in enumerate(sids)}
    idx117 = sorted(pos[r["sample_id"]] for r in sel)
    assert len(idx117) == 117 and len(set(idx117)) == 117
    by_row = {pos[r["sample_id"]]: r for r in sel}
    np.savez_compressed(os.path.join(DIRS["features"], "timesx_sel117.npz"),
                        feat=feat_all[idx117],
                        row_id=np.array(idx117, dtype=np.int64),
                        var_key=np.array([by_row[i]["var_key"] for i in idx117]),
                        sample_id=np.array([sids[i] for i in idx117]),
                        group=np.array([by_row[i]["sel_group"] for i in idx117]))

    # ---- ImageNet: one image load, two forward_encoder passes per batch ----
    from PIL import Image
    from common import imagenet_transform
    tf = imagenet_transform()
    files = man["files"]
    fm, fn = [], []
    t0 = time.time()
    for i in range(0, len(files), BATCH):
        chunk = files[i:i + BATCH]
        imgs = torch.stack(
            [tf(Image.open(os.path.join(man["val_dir"], f)).convert("RGB"))
             for f in chunk]).to(device)
        with torch.no_grad():
            out_m, _, _ = model.vision_model.forward_encoder(
                imgs, model.mask_ratio, noise=noise_mask(model, imgs.shape[0], device))
            out_n, _, _ = model.vision_model.forward_encoder(
                imgs, 0.0, noise=noise_arange(imgs.shape[0], device))
        fm.append(pooled(out_m).cpu().numpy())
        fn.append(pooled(out_n).cpu().numpy())
    fm = np.concatenate(fm, 0).astype(np.float32)
    fn = np.concatenate(fn, 0).astype(np.float32)
    assert fm.shape == fn.shape == (len(files), 768)
    # key "fname" (not "file"): collides with savez_compressed()'s own arg
    np.savez_compressed(os.path.join(DIRS["features"], "imagenet_masked.npz"),
                        feat=fm, fname=np.array(files))
    np.savez_compressed(os.path.join(DIRS["features"], "imagenet_nomask.npz"),
                        feat=fn, fname=np.array(files))
    print(f"imagenet: masked {fm.shape}, nomask {fn.shape}, {time.time() - t0:.1f}s")

    with open(os.path.join(DIRS["features"], "meta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "source", "masked", "id", "var_key"])
        for i, f_ in enumerate(files):
            w.writerow([i, "imagenet", 1, f_, ""])
        for i, f_ in enumerate(files):
            w.writerow([2000 + i, "imagenet", 0, f_, ""])
        for i, sid in enumerate(sids):
            w.writerow([4000 + i, "timesx", 1, sid, vks[i]])

    with open(os.path.join(DIRS["logs"], "extract_env.json"), "w") as f:
        json.dump({"device": device, "torch": torch.__version__,
                   "ckpt_sha256_prefix": ckpt[:16], "batch": BATCH,
                   "pred_rtol_declared": PRED_RTOL, "worst_cache_pred_rel": worst,
                   "protocol": {
                       "feat": "encoder final-norm output, CLS excluded, mean over 56 visible tokens (768-d)",
                       "masked_noise": "model.mask [1,196] expanded to [B,196] (production path)",
                       "nomask_noise": "explicit arange(196), mask_ratio=0 (natural token order)"}},
                  f, indent=1)
    print("features written ->", DIRS["features"])


if __name__ == "__main__":
    main()
