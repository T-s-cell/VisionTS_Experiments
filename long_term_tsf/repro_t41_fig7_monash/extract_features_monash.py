"""EXP-041 step 2: extract 54528-dim MAE-encoder features for the 300
Monash windows (visionts env, GPU, frozen ckpt).

Hard gates, in order:
  1. frozen EXP-002 features_1900x54528.npy file sha == pin (BEFORE any
     model work);
  2. frozen MAE ckpt file sha == pin;
  3. mask_ref.json ids_keep_order == left-5-column row-major ids and
     len == 70.

Geometry is configured PER GROUP (subset, L_eff, H, P) through
VisionTS.update_config — model.mask / model.mask_ratio are NEVER used:
the mask comes from a FIXED reference noise tensor built from mask_ref
ids_keep_order (0 for the 70 visible ids, 1 elsewhere) passed explicitly
to forward_encoder with the pinned mask_ratio.  Every batch asserts
argsort(noise)[:, :70] == mask_ref ids_keep_order — decoupled from the
per-sample geometry (num_patch_input varies 3-5 across Monash subsets).

Outputs (save_t41_monash/): features_monash_300x54528.npy (float32),
groups_monash.npy (subset per row), input_hashes_monash.json,
extract_env.json.
"""
import os
import sys
import time

import numpy as np
import torch

_PKG = os.path.dirname(os.path.abspath(__file__))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import t41_common as X          # noqa: E402

# EXP-002 mechanics (load_model / set_seeds / build_ts_image), zero reimplementation
sys.path.insert(0, X.fig7_base())
import common as F7             # noqa: E402


def main():
    os.makedirs(X.SAVE_BASE, exist_ok=True)

    # ---- gate 1: frozen features file sha, before anything else ----------
    fp = X.frozen_features_path()
    assert os.path.isfile(fp), "frozen features missing: %s" % fp
    t0 = time.time()
    sha = X.sha256_of(fp)
    assert sha == X.FROZEN_FEATURES_SHA256, \
        "FROZEN FEATURES SHA DRIFT: %s != %s" % (sha, X.FROZEN_FEATURES_SHA256)
    X.log("gate1 frozen features sha OK (%.1f s)" % (time.time() - t0))

    # ---- gate 2/3: ckpt sha + mask_ref -----------------------------------
    ckpt = os.path.join(X.REPO_ROOT, "ckpt", "mae_visualize_vit_base.pth")
    assert os.path.isfile(ckpt), "ckpt missing: %s" % ckpt
    ck_sha = X.sha256_of(ckpt)
    assert ck_sha == X.CKPT_SHA256, "MAE ckpt sha drift: %s" % ck_sha
    ref = X.load_json(X.mask_ref_path())
    ids_ref = ref["ids_keep_order"]
    assert len(ids_ref) == X.NUM_VISIBLE
    assert ids_ref == sorted(ids_ref) and len(set(ids_ref)) == X.NUM_VISIBLE
    assert all((i // 14) < 14 and (i % 14) < 5 for i in ids_ref), \
        "mask_ref ids are not the left-5-column row-major ids"
    X.log("gate2/3 ckpt + mask_ref OK (mask_ratio pin %.16f)"
          % ref["mask_ratio"])
    assert abs(ref["mask_ratio"] - X.MASK_RATIO_PIN) < 1e-12

    manifest = X.load_json(X.MANIFEST_JSON)
    samples = manifest["samples"]
    smoke = str(manifest.get("mode", "")).startswith("SMOKE")
    assert len(samples) == (X.N_MONASH if not smoke else len(samples)), \
        "manifest has %d samples (mode %s)" % (
            len(samples), manifest.get("mode"))
    wins = np.load(X.WINDOWS_NPZ)
    assert set(wins.files) == {"win_%d" % s["k"] for s in samples}

    # ---- input hashes (FG3 drift audit) ----------------------------------
    input_hashes = {}
    for s in samples:
        w = np.ascontiguousarray(wins["win_%d" % s["k"]], dtype=np.float64)
        input_hashes["win_%d" % s["k"]] = X.sha256_of_bytes(w.tobytes())
    X.dump_json(X.INPUT_HASHES_JSON, input_hashes)

    # ---- model + fixed reference noise ------------------------------------
    F7.set_seeds()
    assert torch.cuda.is_available(), "GPU required"
    device = "cuda"
    model = F7.load_model()
    model.eval().to(device)
    ref_noise = torch.ones(1, X.TOTAL_PATCHES, dtype=torch.float32)
    ref_noise[0, torch.tensor(ids_ref, dtype=torch.long)] = 0.0
    ref_ids = torch.tensor([ids_ref], dtype=torch.long)

    groups = {}
    for s in samples:
        groups.setdefault((s["subset"], s["L_eff"], s["H"], s["P"]), []).append(s)
    X.log("%d samples in %d geometry groups" % (len(samples), len(groups)))

    feats = np.zeros((len(samples), X.FEATURE_DIM), dtype=np.float32)
    row_of = {s["k"]: i for i, s in enumerate(samples)}
    n_cfg_calls = 0
    for (ds, l_eff, h, p), ss in sorted(groups.items(), key=lambda kv: kv[1][0]["k"]):
        model.update_config(context_len=l_eff, pred_len=h, periodicity=p,
                            norm_const=F7.NORM_CONST, align_const=F7.ALIGN_CONST,
                            interpolation="bilinear")
        n_cfg_calls += 1
        npi = model.num_patch_input
        for b0 in range(0, len(ss), F7.BATCH):
            chunk = ss[b0:b0 + F7.BATCH]
            x = torch.stack([
                torch.as_tensor(np.ascontiguousarray(
                    wins["win_%d" % s["k"]]), dtype=torch.float32)
                for s in chunk]).unsqueeze(-1).to(device)   # (B, L_eff, 1)
            image = F7.build_ts_image(model, x)
            noise = ref_noise.repeat(image.shape[0], 1).to(device)
            ik = torch.argsort(noise, dim=1)[:, :X.NUM_VISIBLE]
            assert torch.equal(ik, ref_ids.to(device).expand_as(ik)), \
                "%s batch %d: fixed-reference mask drift" % (ds, b0)
            with torch.no_grad():
                out, _, _ = model.vision_model.forward_encoder(
                    image, X.MASK_RATIO_PIN, noise)
            flat = out.flatten(1).to(torch.float32).cpu().numpy()
            assert flat.shape == (len(chunk), X.FEATURE_DIM)
            assert np.isfinite(flat).all(), "%s batch %d: non-finite" % (ds, b0)
            for j, s in enumerate(chunk):
                feats[row_of[s["k"]]] = flat[j]
        X.log("%s (L=%d H=%d P=%d npi=%d): %d done"
              % (ds, l_eff, h, p, npi, len(ss)))

    assert np.isfinite(feats).all()
    np.save(X.FEATURES_MONASH, feats)
    np.save(X.GROUPS_MONASH, np.array([s["subset"] for s in samples]))
    env = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "manifest_mode": manifest.get("mode"),
        "n_samples": len(samples),
        "ckpt_path": ckpt, "ckpt_sha256": ck_sha,
        "frozen_features_sha256": sha,
        "mask_ratio": X.MASK_RATIO_PIN,
        "feature_dim": X.FEATURE_DIM, "seed": X.SEED,
        "n_update_config_calls": n_cfg_calls,
        "mask_source": "fixed reference noise from mask_ref ids_keep_order "
                       "(decoupled from per-sample geometry)",
        "note": "Monash rows [1900,2200); unified left-5-column visible mask "
                "is a VISUALIZATION setting, different from the prediction-"
                "time geometry (which uses update_config's own mask).",
    }
    X.dump_json(X.EXTRACT_ENV_JSON, env)
    X.log("DONE: %s" % X.FEATURES_MONASH)


if __name__ == "__main__":
    main()
