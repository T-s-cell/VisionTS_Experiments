#!/usr/bin/env python3
"""Stage 01: sanity checks BEFORE any feature is used (eta, same env as stage-1).

  S1 mask buffer geometry: [1,196], sum=140, kept set = leftmost 4 columns
     of each of the 14 rows.
  S2 token ORDER under production noise (direct forward_encoder, B=4):
     per-sample ids_keep must be ELEMENTWISE identical across the batch
     (not just the same set), equal to the order derived from the noise
     (argsort(noise)[:56]), and its set must cover exactly the leftmost-4
     columns. The order array is archived (logs/token_order.json). The order
     itself is not asserted to be row-major; core sort logic is untouched.
  S3 image captured by the production-forward hook == independently built
     image (torch.equal).
  S4 hook flatten [B,43776] == independent forward_encoder flatten
     (torch.equal).
  S5 prediction spot-check on 2 full variables (infer.py order, bs=64
     chunks incl. the tail chunk): within the declared rtol, per-chunk
     ids_keep order asserted against the archived order; max abs AND
     scale-normalized error recorded.
  S6 same device/batch/config repeat forward: bitwise equal (pred+flatten).
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

from common import (DIRS, FLAT_DIM, N_TOKENS, N_VISIBLE, PRED_RTOL,
                    attach_capture, build_ts_image, capture_ids_restore,
                    device_auto, ensure_dirs, expected_kept_set, flat,
                    ids_keep_from_restore, keep_order_from_noise, load_model,
                    noise_mask, parse_args, pred_compare, timesx_windows)

results = {}


def check(name, ok, detail=""):
    results[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main():
    args = parse_args(__doc__)
    device = args.device or device_auto()
    ensure_dirs()
    model, ckpt = load_model(device)

    exp = expected_kept_set()

    # S1 mask buffer
    m = model.mask
    check("S1a_mask_shape", tuple(m.shape) == (1, 196), str(tuple(m.shape)))
    check("S1b_mask_sum140", float(m.sum()) == 140.0, f"sum={float(m.sum())}")
    kept_buf = {int(i) for i in range(196) if m[0, i] == 0}
    check("S1c_kept_set", kept_buf == exp,
          f"buffer kept set == leftmost-4-columns: {kept_buf == exp}")

    # S2 token ORDER under production noise
    B = 4
    img = torch.randn(B, 3, 224, 224, device=device)
    with torch.no_grad():
        out, mbool, ids_restore = model.vision_model.forward_encoder(
            img, model.mask_ratio, noise=noise_mask(model, B, device))
    check("S2a_norm_shape", tuple(out.shape) == (B, N_TOKENS, 768),
          str(tuple(out.shape)))
    ids_keep = ids_keep_from_restore(ids_restore, N_VISIBLE)
    same_order = bool((ids_keep == ids_keep[0:1]).all())
    check("S2b_order_elementwise_same_across_batch", same_order)
    noise_order = keep_order_from_noise(model, B, device)
    check("S2c_order_equals_noise_derived",
          bool((ids_keep == noise_order).all()))
    kept = {int(i) for i in ids_keep[0].tolist()}
    check("S2d_kept_set", kept == exp, f"n_unique={len(kept)}")
    with open(os.path.join(DIRS["logs"], "token_order.json"), "w") as f:
        json.dump({"ids_keep_order": [int(i) for i in ids_keep[0].tolist()],
                   "note": "archived order used for flatten; identical across "
                           "samples and both data types (asserted per batch in 02)"},
                  f, indent=1)

    # S3/S4 hook vs independent build on 6 real windows
    win = timesx_windows()
    picks = []
    seen = set()
    for r in win:
        if r["var_key"] not in seen:
            seen.add(r["var_key"])
            picks.append(r)
        if len(picks) == 6:
            break
    from timesx_data import find_zip, iter_variables
    want = {r["sample_id"] for r in picks}
    past = {}
    for v in iter_variables(find_zip(None)):
        for s in v.samples:
            if s.sample_id in want:
                past[s.sample_id] = np.asarray(s.past_val, dtype=np.float32)
    assert len(past) == len(picks)
    x = torch.tensor(np.stack([past[r["sample_id"]] for r in picks]),
                     dtype=torch.float32, device=device).unsqueeze(-1)

    box = capture_ids_restore(model)
    cap, handles = attach_capture(model)
    with torch.no_grad():
        y = model(x)
    img_hook = cap["image"].detach().clone()
    flat_hook = flat(cap["feat_norm"]).detach().clone()
    for h in handles:
        h.remove()

    img_ind = build_ts_image(model, x)
    check("S3_independent_image_equals_hook", bool(torch.equal(img_ind, img_hook)),
          f"max|d|={float((img_ind - img_hook).abs().max()):.3e}")

    with torch.no_grad():
        out_ind, _, _ = model.vision_model.forward_encoder(
            img_ind, model.mask_ratio, noise=noise_mask(model, x.shape[0], device))
    flat_ind = flat(out_ind)
    check("S4_independent_flatten_equals_hook",
          bool(torch.equal(flat_ind, flat_hook)) and flat_hook.shape[1] == FLAT_DIM,
          f"max|d|={float((flat_ind - flat_hook).abs().max()):.3e}, "
          f"dim={flat_hook.shape[1]}")
    ids_keep3 = ids_keep_from_restore(box["ids_restore"], N_VISIBLE)
    check("S4b_forward_ids_keep_order_matches_archive",
          bool((ids_keep3 == torch.tensor([ids_keep[0].tolist()],
               device=device)).all()))

    # S5 prediction spot-check: 2 full variables, infer order, bs=64 chunks
    vks2 = sorted({r["var_key"] for r in win})[:2]
    ok, details = True, []
    zero = os.path.dirname(HERE)
    for vk in vks2:
        d = np.load(os.path.join(zero, "cache", f"{vk}__P1.npz"), allow_pickle=False)
        ids = [str(s) for s in d["sample_ids"]]
        p_past = {}
        for v in iter_variables(find_zip(None)):
            if v.var_key == vk:
                for s in v.samples:
                    if s.sample_id in set(ids):
                        p_past[s.sample_id] = np.asarray(s.past_val, dtype=np.float32)
                break
        assert len(p_past) == len(ids)
        worst_rel, worst_abs = 0.0, 0.0
        for i0 in range(0, len(ids), 64):
            chunk = ids[i0:i0 + 64]
            xc = torch.tensor(np.stack([p_past[s] for s in chunk]),
                              dtype=torch.float32, device=device).unsqueeze(-1)
            with torch.no_grad():
                yc = model(xc)[:, :, 0].cpu().numpy()
            ik = ids_keep_from_restore(box["ids_restore"], N_VISIBLE)
            ok_order = bool((ik == torch.tensor([ids_keep[0].tolist()],
                                                 device=device)).all())
            assert ok_order, f"{vk}: chunk@{i0} token order changed"
            for k, s in enumerate(chunk):
                ma, rel = pred_compare(yc[k], d["pred"][ids.index(s)])
                worst_rel, worst_abs = max(worst_rel, rel), max(worst_abs, ma)
        ok &= (worst_rel <= PRED_RTOL)
        details.append(f"{vk[:28]}: {len(ids)} rows max_abs={worst_abs:.3e} "
                       f"rel={worst_rel:.2e}")
    check("S5_pred_spotcheck_within_declared_rtol", ok,
          f"rtol<={PRED_RTOL}; " + "; ".join(details))

    # S6 repeat forward bitwise
    with torch.no_grad():
        y2 = model(x)
    flat2 = flat(cap["feat_norm"]).detach().clone()
    check("S6_repeat_forward_bitwise",
          bool(torch.equal(y, y2)) and bool(torch.equal(flat_hook, flat2)))

    with open(os.path.join(DIRS["logs"], "sanity_report.json"), "w") as f:
        json.dump({"device": device, "ckpt_sha256_prefix": ckpt[:16],
                   "pred_rtol_declared": PRED_RTOL, "checks": results}, f, indent=1)
    n_fail = sum(1 for v in results.values() if not v["pass"])
    print(f"sanity: {len(results) - n_fail}/{len(results)} PASS")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
