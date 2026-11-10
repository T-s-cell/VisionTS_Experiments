#!/usr/bin/env python
"""EXP-042 step-0 finetune-scope audit (run inside preflight).

On a REAL visionts.model.VisionTS instance per ft type (bias/mlp/attn),
with the EXP-003 archived construction config (vm_arch / vm_ckpt /
vm_pretrained=1):

  (a) ACTUAL trainable set  = {n for n,p in vision_model.named_parameters()
                                if p.requires_grad}
  (b) PREDICATE re-derivation from the official rules
      (visionts/model.py:46-57) applied to the same name space
  (c) HARD assert actual == predicate

Recorded per ft: trainable names/count/numel / n_params_total / percent,
encoder+decoder coverage statement (blocks.* vs decoder_blocks.*),
frozen-key list (pos_embed / cls_token / mask_token / decoder_pos_embed
...), and for 'bias' the EXPLICIT patch_embed.proj.bias listing
(Conv2d bias — the scope is wider than the paper's BitFit text; the
official-code scope is adopted, difference recorded, never widened or
narrowed by us).

Also writes logs/frozen_digest.json: per-key {sha256, dtype, shape} of
the frozen ckpt state_dict (bare key names, ck["model"]) — shared by the
driver and verify; asserts the ckpt key set == the model parameter-name
set, which pins the name space all three parties use.

Output: logs/scope_audit_t42.json + logs/frozen_digest.json
(through t42_common.SCOPE_JSON / FROZEN_DIGEST_JSON).
"""
import os

import t42_common as X
import t38_common as C


def main():
    os.makedirs(X.LOG_DIR, exist_ok=True)
    os.chdir(C.ROOT)  # long_term_tsf — EXP-003 CWD contract (vm_ckpt relative)

    cfg_args = C.load_json(os.path.join(C.exp003_job("ETTh1", 96),
                                        "args.json"))
    assert cfg_args.get("vm_pretrained") == 1, \
        "EXP-003 vm_pretrained must be 1 (original pretrained weights)"
    arch = cfg_args["vm_arch"]
    ckpt_dir = cfg_args["vm_ckpt"]
    expected_ckpt = os.path.abspath(os.path.join(
        C.ROOT, "..", "ckpt", "mae_visualize_vit_base.pth"))
    resolved_ckpt = os.path.normpath(os.path.join(
        C.ROOT, ckpt_dir, "mae_visualize_vit_base.pth"))
    assert os.path.samefile(resolved_ckpt, expected_ckpt), \
        "EXP-003 vm_ckpt %r resolves elsewhere: %s" % (ckpt_dir,
                                                       resolved_ckpt)
    assert C.sha256_of(expected_ckpt) == C.CKPT_SHA256
    assert os.path.getsize(expected_ckpt) == C.CKPT_BYTES

    import torch
    import visionts.model as vmodel
    assert os.path.samefile(vmodel.__file__,
                            os.path.join(C.REPO_ROOT, "visionts",
                                         "model.py")), vmodel.__file__

    def predicate(ft, n):
        # verbatim semantics of visionts/model.py:46-57 (full excluded —
        # not a t42 arm)
        if "ln" == ft:
            return "norm" in n
        if "bias" == ft:
            return "bias" in n
        if "none" == ft:
            return False
        if "mlp" in ft:
            return ".mlp." in n
        if "attn" in ft:
            return ".attn." in n
        raise ValueError("unexpected finetune_type %r" % ft)

    out = {"started": C.time_str(), "arch": arch,
           "ckpt_path": expected_ckpt,
           "ckpt_sha256": C.CKPT_SHA256, "ckpt_bytes": C.CKPT_BYTES,
           "rules_source": "visionts/model.py:46-57 (verbatim predicates)",
           "per_ft": {}}

    for ft in X.FT_TYPES:
        model = vmodel.VisionTS(arch=arch, finetune_type=ft,
                                ckpt_dir=ckpt_dir, load_ckpt=True)
        named = [(n, p) for n, p in
                 model.vision_model.named_parameters()]
        actual = sorted(n for n, p in named if p.requires_grad)
        preds = sorted(n for n, _ in named if predicate(ft, n))
        assert actual == preds, \
            "scope mismatch for %s: actual-vs-predicate diff %s" % (
                ft, sorted(set(actual) ^ set(preds)))
        trainable_numel = int(sum(p.numel() for n, p in named
                                  if p.requires_grad))
        total_numel = int(sum(p.numel() for _, p in named))
        enc = [n for n in actual if n.startswith("blocks.")]
        dec = [n for n in actual if n.startswith("decoder_blocks.")]
        other = [n for n in actual
                 if not n.startswith(("blocks.", "decoder_blocks."))]
        frozen = sorted(n for n, p in named if not p.requires_grad)
        all_names = {n for n, _ in named}
        ent = {
            "trainable_names": actual,
            "trainable_count": len(actual),
            "trainable_numel": trainable_numel,
            "n_params_total": total_numel,
            "trainable_pct": round(100.0 * trainable_numel / total_numel, 4),
            "encoder_blocks_trainable": len(enc),
            "decoder_blocks_trainable": len(dec),
            "other_trainable": other,
            "coverage_statement":
                "encoder blocks (%d names) AND decoder blocks (%d names) "
                "both covered" % (len(enc), len(dec))
                if enc and dec else
                "WARNING: encoder/decoder coverage incomplete (%d/%d)"
                % (len(enc), len(dec)),
            "frozen_names": frozen,
            "frozen_count": len(frozen),
            "predicate_set": preds,
            "predicate_match": True,
        }
        if ft == "bias":
            assert "patch_embed.proj.bias" in actual, \
                "bias arm must include patch_embed.proj.bias (Conv2d bias)"
            ent["patch_embed_proj_bias_included"] = True
            ent["layernorm_bias_count"] = sum(
                1 for n in actual if "norm" in n and n.endswith("bias"))
            ent["linear_bias_count"] = len(actual) - \
                ent["layernorm_bias_count"]
        out["per_ft"][ft] = ent
        print("[t42-scope] %s: %d trainable / %d total (%.4f%%), enc %d + "
              "dec %d" % (ft, len(actual), total_numel, ent["trainable_pct"],
                          len(enc), len(dec)))
        last_names = all_names
        del model

    # distinctness sanity: the three trainable sets must pairwise differ
    sets = {ft: set(out["per_ft"][ft]["trainable_names"])
            for ft in X.FT_TYPES}
    assert len({frozenset(v) for v in sets.values()}) == 3, \
        "trainable sets not pairwise distinct across ft types"

    # frozen digest shared by driver/verify (bare ck["model"] key names)
    ck = torch.load(expected_ckpt, map_location="cpu")
    sd = ck["model"]
    digest = {"ckpt_path": expected_ckpt, "ckpt_sha256": C.CKPT_SHA256,
              "n_keys": len(sd), "keys": {}}
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        digest["keys"][k] = {
            "sha256": C.sha256_bytes(t.numpy().tobytes()),
            "dtype": str(t.dtype), "shape": list(t.shape)}
    model_names = last_names   # pinned from the last constructed instance
    assert model_names == set(sd.keys()), \
        "ckpt key set != model named_parameters set (name-space pin)"
    C.dump_json(X.FROZEN_DIGEST_JSON, digest)
    out["frozen_digest"] = {"n_keys": len(sd),
                            "path": X.FROZEN_DIGEST_JSON}
    out["finished"] = C.time_str()
    C.dump_json(X.SCOPE_JSON, out)
    print("[t42-scope] wrote %s and %s" % (X.SCOPE_JSON,
                                           X.FROZEN_DIGEST_JSON))


if __name__ == "__main__":
    main()
