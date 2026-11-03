"""EXP-036 step 2: extract 1,900x54528 encoder features for one condition.

Replicates repro_fig7/extract_features.py verbatim (same archived sampling
lists, batch=32, FP32, eval, no_grad, per-batch ids_keep + finite asserts)
with one delta: the encoder may carry the 50 spliced encoder-LN tensors of
an EXP-003 checkpoint. Inputs (images / mask noise) are identical across
conditions by construction; every batch input tensor is SHA-256-recorded so
verify_t36_features can prove it. Writes:
  save_t36_tsne/features_{cond}.npy          (float32, 1900x54528)
  save_t36_tsne/features_{cond}_sha256.txt
  save_t36_tsne/input_hashes_{cond}.json     (per-group per-batch input + noise)
"""
import argparse
import hashlib
import os
import sys
import time

import t36_common as C


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def tensor_sha(t):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=C.COND_LIST)
    args = ap.parse_args()
    cond = args.cond

    import numpy as np
    import torch
    from PIL import Image

    fig7, model = C.load_vision_model()
    splice_info = {"cond": cond, "source": None, "keys_equal_src": 0}
    if cond != "frozen":
        ds = C.COND_SOURCE[cond]
        src = C.load_exp003_visionts_dict(ds)
        evidence = C.splice_encoder_ln(model, src)
        splice_info = {"cond": cond, "source": ds,
                       "ckpt_sha256": C.sha256_of(os.path.join(
                           C.exp003_job(ds), "checkpoints", "_", "checkpoint.pth")),
                       "keys_equal_src": len(evidence)}
        log("spliced %d encoder-LN keys from EXP-003 %s checkpoint" % (len(evidence), ds))
    model.eval().to("cuda")

    # archived row order is the backbone of this experiment: re-verify now
    C.check_meta_rows()
    C.check_groups_order()

    common = fig7
    common.set_seeds()
    assert torch.cuda.is_available(), "GPU required for extraction"
    device = "cuda"
    tfm = common.imagenet_transform()

    lists = {g: C.load_sampling_list(g) for g in C.GROUPS}
    common.configure(model, "ETTm1")
    ref_ids_keep = torch.argsort(common.get_noise(model, 1), dim=1)[:, :70]
    ids_keep_list = ref_ids_keep[0].tolist()
    mask_ref = C.load_json(C.ORIG_MASK_REF)
    assert ids_keep_list == mask_ref["ids_keep_order"], "ids_keep != archived mask_ref"

    input_hashes = {"noise_sha256": tensor_sha(common.get_noise(model, 1)), "groups": {}}

    feats = np.zeros((1900, C.FEATURE_DIM), dtype=np.float32)
    pos = 0
    for g in C.GROUPS:
        if g in common.TS_CONFIGS:
            common.configure(model, g)
        entries = lists[g]["filenames"] if g == "ImageNet" else lists[g]["samples"]
        log("%s: %d samples, num_patch_input=%d, mask_ratio=%.6f"
            % (g, len(entries), model.num_patch_input, model.mask_ratio))
        gh = []
        for s in range(0, len(entries), common.BATCH):
            batch = entries[s:s + common.BATCH]
            if g == "ImageNet":
                imgs = []
                for fn in batch:
                    img = Image.open(os.path.join(common.IMAGENET_VAL_DIR, fn)).convert("RGB")
                    imgs.append(tfm(img))
                image = torch.stack(imgs).to(device)
            else:
                x = common_batch_ts(common, g, batch, device)
                image = common.build_ts_image(model, x)
            gh.append(tensor_sha(image))
            with torch.no_grad():
                out, _, _ = model.vision_model.forward_encoder(
                    image, model.mask_ratio, common.get_noise(model, image.shape[0]))
            ik = torch.argsort(common.get_noise(model, image.shape[0]), dim=1)[:, :70]
            assert torch.equal(ik, ref_ids_keep.expand_as(ik)), "%s batch %d: ids_keep drift" % (g, s)
            flat = out.flatten(1).to(torch.float32).cpu().numpy()
            assert np.isfinite(flat).all(), "%s batch %d: non-finite features" % (g, s)
            feats[pos:pos + len(batch)] = flat
            pos += len(batch)
            if (s // common.BATCH) % 10 == 0:
                log("%s: %d / batch %d-%d" % (g, pos, s, s + len(batch)))
        input_hashes["groups"][g] = {"n_batches": len(gh), "batch_sha256": gh}
        log("%s: DONE (%d total)" % (g, pos))

    assert pos == 1900
    os.makedirs(C.SAVE_BASE, exist_ok=True)
    np.save(C.cond_features(cond), feats)
    sha = C.sha256_of(C.cond_features(cond))
    with open(C.cond_features_sha(cond), "w") as f:
        f.write(sha + "\n")
    C.dump_json(C.cond_inputs_json(cond), input_hashes)
    data = C.load_json(C.VERIFY_JSON) if os.path.isfile(C.VERIFY_JSON) else {}
    data.setdefault("extract", {})[cond] = {
        **splice_info,
        "features_sha256": sha,
        "feature_shape": list(feats.shape),
        "noise_sha256": input_hashes["noise_sha256"],
        "ids_keep_match_mask_ref": True,
    }
    C.dump_json(C.VERIFY_JSON, data)
    log("ALL DONE: %s -> %s (sha256 %s)" % (cond, C.cond_features(cond), sha))


def common_batch_ts(common, name, entries, device):
    """Verbatim of repro_fig7/extract_features.py ts_batch()."""
    import numpy as np
    import torch
    ds = common.build_dataset(name)
    seqs = []
    for e in entries:
        if common.TS_CONFIGS[name]["data"] == "custom":
            seq = ds[e["k"]][0]
        else:
            seq = ds[e["window_idx"]][0][:, e["channel_idx"]:e["channel_idx"] + 1]
        seqs.append(torch.as_tensor(np.asarray(seq), dtype=torch.float32))
    return torch.stack(seqs).to(device)


if __name__ == "__main__":
    main()
