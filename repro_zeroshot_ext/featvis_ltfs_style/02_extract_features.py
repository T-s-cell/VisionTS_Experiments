#!/usr/bin/env python3
"""Stage 02: extract features through the production forward (eta GPU).

- TimesX: the 18 selected variables' FULL val+test rows in frozen cache order,
  bs=64 chunks (tails included), through model(x) with runtime hooks — the same
  forwards re-verify every prediction against the frozen cache.
- ImageNet: the LTFS 1000 files in their original list order, direct
  forward_encoder, bs=64 chunks (tail included), production noise (mask buffer
  expanded to [B,196]).
- Features from the SAME encoder outputs: flatten [N,43776] (CLS kept, token
  order preserved) and pooled [N,768] (CLS excluded, 56-patch mean).
- EVERY batch — TimesX and ImageNet, full or tail — asserts the visible-token
  ids_keep ORDER (elementwise, not just the set) against the archived order.
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

from common import (BATCH, DIRS, FLAT_DIM, N_VISIBLE, PRED_RTOL, attach_capture,
                    capture_ids_restore, device_auto, ensure_dirs, flat,
                    ids_keep_from_restore, imagenet_transform,
                    load_fig7_imagenet_json, load_model, noise_mask, parse_args,
                    pooled, pred_compare, timesx_windows)

ZERO = os.path.dirname(HERE)


def main():
    args = parse_args(__doc__)
    device = args.device or device_auto()
    ensure_dirs()
    model, ckpt = load_model(device)
    box = capture_ids_restore(model)
    win = timesx_windows()

    ref = None

    def assert_order(ids_restore, ctx):
        nonlocal ref
        ik = ids_keep_from_restore(ids_restore, N_VISIBLE)
        if ref is None:
            ref = [int(i) for i in ik[0].tolist()]
        r = torch.tensor([ref], device=device)
        assert bool((ik == r).all()), f"{ctx}: visible-token ORDER changed"

    # ---- TimesX: cache-order forwards, every chunk order-asserted ----
    want = sorted({r["var_key"] for r in win})
    order = {}
    for vk in want:
        d = np.load(os.path.join(ZERO, "cache", f"{vk}__P1.npz"), allow_pickle=False)
        order[vk] = [str(s) for s in d["sample_ids"]]
    need = {}
    for vk, ids in order.items():
        need.setdefault(vk, set()).update(ids)
    past = {}
    from timesx_data import find_zip, iter_variables
    for v in iter_variables(find_zip(None)):
        wl = need.get(v.var_key)
        if not wl:
            continue
        for s in v.samples:
            if s.sample_id in wl:
                past[(v.var_key, s.sample_id)] = np.asarray(
                    s.past_val, dtype=np.float32)
    n_rows = sum(len(v) for v in order.values())
    assert len(past) == n_rows, f"missing past values: {len(past)}/{n_rows}"

    cap, handles = attach_capture(model)
    feats = {}   # (vk, sid) -> (flat_row, pooled_row)
    worst = 0.0
    t0 = time.time()
    n_fwd = 0
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
            assert_order(box["ids_restore"], f"timesx/{vk}@{i0}+{len(chunk)}")
            fl = flat(cap["feat_norm"]).cpu().numpy()
            pl = pooled(cap["feat_norm"]).cpu().numpy()
            yc = y[:, :, 0].cpu().numpy()
            for k, gid in enumerate(chunk):
                gi = i0 + k
                _, rel = pred_compare(yc[k], d["pred"][gi])
                worst = max(worst, rel)
                assert rel <= PRED_RTOL, f"{vk} {gid}: rel={rel:.3e} exceeds rtol"
                if sp[gi] == "test":
                    feats[(vk, gid)] = (fl[k], pl[k])
            n_fwd += len(chunk)
    for h in handles:
        h.remove()
    assert len(feats) == 117, f"expected 117 test windows, got {len(feats)}"
    assert {(w["var_key"], w["sample_id"]) for w in win} == set(feats)
    print(f"timesx: {n_fwd} rows forwarded, 117 kept, worst cache-pred "
          f"rel={worst:.2e}, {time.time() - t0:.1f}s")

    # ---- ImageNet: LTFS 1000 in original order, every chunk order-asserted ----
    man = json.load(open(os.path.join(DIRS["lists"], "imagenet_list.json")))
    files = [str(s) for s in man["filenames"]]
    files_chk, sha_list, _ = load_fig7_imagenet_json()
    assert files == files_chk, "imagenet_list.json drifted from the frozen source"
    val_dir = man["val_dir"]
    from PIL import Image
    tf = imagenet_transform()
    fm, pm = [], []
    t0 = time.time()
    for i in range(0, len(files), BATCH):
        chunk = files[i:i + BATCH]
        imgs = torch.stack(
            [tf(Image.open(os.path.join(val_dir, f)).convert("RGB"))
             for f in chunk]).to(device)
        with torch.no_grad():
            out, _, ids_restore = model.vision_model.forward_encoder(
                imgs, model.mask_ratio, noise=noise_mask(model, imgs.shape[0],
                                                         device))
        assert_order(ids_restore, f"imagenet@{i}+{len(chunk)}")
        fm.append(flat(out).cpu().numpy())
        pm.append(pooled(out).cpu().numpy())
    fm = np.concatenate(fm, 0).astype(np.float32)
    pm = np.concatenate(pm, 0).astype(np.float32)
    print(f"imagenet: {fm.shape[0]} files, {time.time() - t0:.1f}s")

    # ---- assemble: rows 0-999 imagenet, 1000-1116 timesx (list order) ----
    src = ["imagenet"] * 1000 + ["timesx"] * 117
    rid = list(range(1000)) + [1000 + j for j in range(117)]
    ids_col = list(files) + [w["sample_id"] for w in win]
    vk_col = [""] * 1000 + [w["var_key"] for w in win]
    grp_col = [""] * 1000 + [w["sel_group"] for w in win]
    flt = np.concatenate([fm] + [feats[(w["var_key"], w["sample_id"])][0][None]
                                 for w in win], 0).astype(np.float32)
    pol = np.concatenate([pm] + [feats[(w["var_key"], w["sample_id"])][1][None]
                                 for w in win], 0).astype(np.float32)
    assert flt.shape == (1117, FLAT_DIM) and pol.shape == (1117, 768)
    assert np.isfinite(flt).all() and np.isfinite(pol).all()
    assert len(set(ids_col)) == 1117

    np.savez_compressed(os.path.join(DIRS["features"], "flatten_1117.npz"),
                        feat=flt, source=np.array(src), id=np.array(ids_col),
                        var_key=np.array(vk_col), sel_group=np.array(grp_col))
    np.savez_compressed(os.path.join(DIRS["features"], "pooled_1117.npz"),
                        feat=pol, source=np.array(src), id=np.array(ids_col),
                        var_key=np.array(vk_col), sel_group=np.array(grp_col))
    with open(os.path.join(DIRS["features"], "meta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "source", "masked", "id", "var_key", "sel_group"])
        for j in range(1117):
            w.writerow([rid[j], src[j], 1, ids_col[j], vk_col[j], grp_col[j]])

    import sklearn
    with open(os.path.join(DIRS["logs"], "extract_env.json"), "w") as f:
        json.dump({"device": device, "torch": torch.__version__,
                   "numpy": np.__version__, "sklearn": sklearn.__version__,
                   "ckpt_sha256_prefix": ckpt[:16], "batch": BATCH,
                   "pred_rtol_declared": PRED_RTOL, "worst_cache_pred_rel": worst,
                   "rows_forwarded_timesx": n_fwd,
                   "token_order_ids_keep": ref,
                   "imagenet_list_sha256_prefix": sha_list[:16],
                   "protocol": {
                       "flatten": "encoder final-norm output [B,57,768] -> "
                                  "flatten [B,43776]; CLS kept; no pooling/L2/PCA",
                       "pooled": "same tokens, CLS excluded, 56-patch mean -> 768-d",
                       "order_check": "ids_keep ORDER asserted elementwise for "
                                      "EVERY batch (timesx+imagenet, tails incl.)"}},
                  f, indent=1)
    print("features written ->", DIRS["features"])


if __name__ == "__main__":
    main()
