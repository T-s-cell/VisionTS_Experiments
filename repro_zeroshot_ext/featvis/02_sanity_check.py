#!/usr/bin/env python3
"""Stage 02: sanity checks BEFORE any feature is used (run on eta, same
device/env as stage-1 cache).

Checks:
  S1 mask buffer geometry: [1,196], sum=140, ratio=5/7, kept set = leftmost 4
     columns of each of the 14 rows.
  S2 forward_encoder with production noise (mask expanded to [B,196]):
     per-sample 56 unique kept ids; kept-id SET covers exactly the leftmost-4
     columns of all 14 rows (set equality; output ORDER is not asserted).
     Encoder token order (from ids_restore) is a permutation of the kept set.
  S3 no-mask control with explicit arange noise, mask_ratio=0: all 196 kept,
     natural token order.
  S4 independent image build == image captured by hook in the production
     forward (torch.equal; mismatch -> investigate, no tolerance loosening).
  S5 pooled hook features == independent forward_encoder features on the
     independently built image (torch.equal).
  S6 predictions vs frozen stage-1 cache: declared tolerance rtol<=1e-4
     relative to per-row pred scale; report max abs AND scale-normalized error.
     Batch-size differences allowed; exceeding tolerance FAILS (never auto-
     loosened).
  S7 determinism: same device/batch/config run twice -> bitwise equal.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

from common import (DIRS, N_VISIBLE, PRED_RTOL, attach_capture, build_ts_image,
                    device_auto, ensure_dirs, expected_kept_set,
                    ids_keep_from_restore, load_model, noise_arange, noise_mask,
                    parse_args, pooled, pred_compare)

SAMPLES = [  # (var_key, window offset within test rows) -- >=2 per group
    ("traffic__tour_de_france_96_12_4_10events", 0),
    ("shopping__air_conditioner_96_12_4_10events", 0),
    ("CropsAndStaples__rice_usd_cwt_96_12_12_10events", 0),
    ("arts__food_wine_festivals_96_12_4_10events", 0),
    ("economy__cost_of_living_96_12_4_10events", 0),
    ("science__nobel_prize_96_12_4_10events", 0),
]

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
    check("S1c_ratio", abs(model.mask_ratio - 5 / 7) < 1e-6,
          f"{model.mask_ratio} (fp32 buffer mean; 5/7 differs at fp32 eps)")
    kept_buf = {int(i) for i in range(196) if m[0, i] == 0}
    check("S1d_kept_set", kept_buf == exp,
          f"buffer kept set == leftmost-4-columns: {kept_buf == exp}")

    # S2/S3 masking behaviour via forward_encoder
    B = 3
    img = torch.randn(B, 3, 224, 224, device=device)
    with torch.no_grad():
        out, mbool, ids_restore = model.vision_model.forward_encoder(
            img, model.mask_ratio, noise=noise_mask(model, B, device))
    check("S2a_norm_shape", tuple(out.shape) == (B, 1 + N_VISIBLE, 768),
          str(tuple(out.shape)))
    ok_set, ok_uniq, ok_perm = True, True, True
    enc_order = ids_keep_from_restore(ids_restore, N_VISIBLE)
    for b in range(B):
        kept = {int(i) for i in range(196) if mbool[b, i] == 0}
        ok_set &= (kept == exp)
        ok_uniq &= (len(enc_order[b].tolist()) == len(set(enc_order[b].tolist()))
                    == N_VISIBLE)
        ok_perm &= (set(enc_order[b].tolist()) == kept)
    check("S2b_kept_set_per_sample", ok_set)
    check("S2c_56_unique_ids", ok_uniq)
    check("S2d_encoder_order_is_permutation_of_kept", ok_perm,
          "(order itself not asserted)")

    with torch.no_grad():
        out0, mbool0, ids_restore0 = model.vision_model.forward_encoder(
            img, 0.0, noise=noise_arange(B, device))
    enc_order0 = ids_keep_from_restore(ids_restore0, 196)
    check("S3a_nomask_all_kept", bool((mbool0 == 0).all()))
    check("S3b_nomask_natural_order",
          bool((enc_order0 == torch.arange(196, device=device).unsqueeze(0)).all()))

    # S4-S7 on real windows
    zero = os.path.dirname(HERE)
    rows = []
    for vk, off in SAMPLES:
        d = np.load(os.path.join(zero, "cache", f"{vk}__P1.npz"), allow_pickle=False)
        ids = [str(s) for s in d["sample_ids"]]
        sp = [str(s) for s in d["splits"]]
        test_ids = [i for i, s in zip(ids, sp) if s == "test"]
        sid = test_ids[off]
        i = ids.index(sid)
        rows.append((vk, sid, d["pred"][i]))

    # rebuild inputs from the zip (past values), mirroring infer.py
    by_id = {}
    from timesx_data import find_zip, iter_variables
    want = {sid for _, sid, _ in rows}
    for v in iter_variables(find_zip(None)):
        for s in v.samples:
            if s.sample_id in want:
                by_id[s.sample_id] = np.asarray(s.past_val, dtype=np.float32)
    assert len(by_id) == len(rows), "missing past values"
    x = torch.tensor(np.stack([by_id[sid] for _, sid, _ in rows]),
                     dtype=torch.float32, device=device).unsqueeze(-1)

    cap, handles = attach_capture(model)
    with torch.no_grad():
        y = model(x)
    img_hook = cap["image"].detach().clone()
    feat_hook = pooled(cap["feat_norm"]).detach().clone()

    img_ind = build_ts_image(model, x)
    check("S4_independent_image_equals_hook",
          bool(torch.equal(img_ind, img_hook)),
          f"max|d|={float((img_ind - img_hook).abs().max()):.3e}")

    with torch.no_grad():
        out_ind, _, _ = model.vision_model.forward_encoder(
            img_ind, model.mask_ratio, noise=noise_mask(model, x.shape[0], device))
    feat_ind = pooled(out_ind)
    check("S5_independent_features_equals_hook",
          bool(torch.equal(feat_ind, feat_hook)),
          f"max|d|={float((feat_ind - feat_hook).abs().max()):.3e}")

    ok, details = True, []
    solo_diag = []
    for j, (vk, sid, pred_ref) in enumerate(rows):
        max_abs, rel = pred_compare(y[j].squeeze(-1).cpu().numpy(), pred_ref)
        ok &= (rel <= PRED_RTOL)
        details.append(f"{vk[:24]}../batch6: max_abs={max_abs:.3e} rel={rel:.2e}")
        with torch.no_grad():
            y1 = model(x[j:j + 1])
        s_abs, s_rel = pred_compare(y1[0, :, 0].cpu().numpy(), pred_ref)
        solo_diag.append(f"{vk[:24]}../solo: max_abs={s_abs:.3e} rel={s_rel:.2e}")
    check("S6a_pred_vs_cache_batch_within_declared_rtol", ok,
          f"rtol<={PRED_RTOL}; " + "; ".join(details))

    # S6b diagnostic only (NOT acceptance): solo-vs-cache agreement per window.
    # First eta run showed rel~2e-4 at B=1; root cause was cudnn TF32
    # (default-on) hitting the patch-embed conv, which infer.py disables.
    # After matching that switch, solo B=1 agrees at ~1e-7. Extraction (03)
    # still replicates the cache batch composition (infer.py val+test order,
    # bs=64 chunks) as belt-and-braces.
    check("S6b_solo_batch_dependence_diagnostic", True, "; ".join(solo_diag)
          + " | root cause (resolved): cudnn TF32 default-on vs infer.py "
          "TF32-off; after parity solo agrees ~1e-7")

    # S6c acceptance: single variable, infer.py order + bs=64 chunking
    # (exact stage-1 batch composition) -> expect near-bitwise agreement.
    vk0, _, _ = rows[0]
    d0 = np.load(os.path.join(zero, "cache", f"{vk0}__P1.npz"), allow_pickle=False)
    ids0 = [str(s) for s in d0["sample_ids"]]
    want0 = set(ids0)
    past0 = {}
    from timesx_data import iter_variables
    for v in iter_variables(find_zip(None)):
        if v.var_key == vk0:
            for s in v.samples:
                if s.sample_id in want0:
                    past0[s.sample_id] = np.asarray(s.past_val, dtype=np.float32)
            break
    assert len(past0) == len(ids0)
    okc, worst_rel = True, 0.0
    with torch.no_grad():
        for i0 in range(0, len(ids0), 64):
            chunk = ids0[i0:i0 + 64]
            xc = torch.tensor(np.stack([past0[s] for s in chunk]),
                              dtype=torch.float32, device=device).unsqueeze(-1)
            yc = model(xc)[:, :, 0].cpu().numpy()
            for k, s in enumerate(chunk):
                _, rel = pred_compare(yc[k], d0["pred"][ids0.index(s)])
                worst_rel = max(worst_rel, rel)
                okc &= (rel <= PRED_RTOL)
    check("S6c_infer_order_B64_replication_within_rtol", okc,
          f"{vk0}: {len(ids0)} rows, worst rel={worst_rel:.2e} (rtol<={PRED_RTOL})")

    with torch.no_grad():
        y2 = model(x)
    feat2 = pooled(cap["feat_norm"]).detach().clone()
    check("S7_repeat_forward_bitwise",
          bool(torch.equal(y, y2)) and bool(torch.equal(feat_hook, feat2)))

    for h in handles:
        h.remove()

    with open(os.path.join(DIRS["logs"], "sanity_report.json"), "w") as f:
        json.dump({"device": device, "ckpt_sha256_prefix": ckpt[:16],
                   "pred_rtol_declared": PRED_RTOL, "checks": results}, f, indent=1)
    n_fail = sum(1 for v in results.values() if not v["pass"])
    print(f"sanity: {len(results) - n_fail}/{len(results)} PASS")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
