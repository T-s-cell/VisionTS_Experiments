"""Single choke point for model construction, inference, and LN freezing.

adapter.configure / adapter.forward_batch is the inference path (identical
semantics to EXP-027). adapter.forward_batch_grad is the training path:
same forward, WITHOUT inference_mode so LayerNorm gradients flow. The two
are asserted equivalent in preflight B2 (grad path vs this package AND vs
the frozen EXP-027 entry via cross-package import).
"""
import torch

from repro_timemmd_common_ln_v1 import protocol as P
from repro_timemmd_common_ln_v1.common import model_state_digest


def build_visionts(ckpt_dir=None, load_ckpt=True):
    """Frozen VisionTS: finetune_type='none', eval mode, strict local ckpt
    load only (no download; weights sha is verified in verify_frozen)."""
    from visionts.model import VisionTS
    ckpt_dir = str(ckpt_dir or P.ckpt_dir())
    if load_ckpt:
        import os
        want = os.path.join(ckpt_dir, P.WEIGHTS_FILENAME)
        if not os.path.isfile(want):
            raise SystemExit("[FATAL] MAE ckpt missing: %s" % want)
    model = VisionTS(arch=P.ARCH, finetune_type=P.FINETUNE_TYPE,
                     ckpt_dir=ckpt_dir, load_ckpt=load_ckpt)
    for prm in model.vision_model.parameters():
        if prm.requires_grad:
            raise SystemExit("[FATAL] non-frozen parameter found")
    model.eval()
    return model


def configure(model, L, H, periodicity):
    model.update_config(context_len=L, pred_len=H, periodicity=periodicity,
                        norm_const=P.NORM_CONST, align_const=P.ALIGN_CONST,
                        interpolation=P.INTERPOLATION)
    assert model.context_len == L and model.pred_len == H
    assert model.periodicity == periodicity
    return model


@torch.inference_mode()
def forward_batch(model, x):
    """x: [B, L, 1] float32 (already on device) -> [B, H, 1] float32."""
    y = model(x, export_image=False, fp64=P.FP64)
    assert y.shape[0] == x.shape[0] and y.shape[2] == 1
    assert y.shape[1] == model.pred_len
    assert y.dtype == torch.float32
    return y


def forward_batch_grad(model, x):
    """Differentiable twin of forward_batch (NO inference_mode). Used ONLY
    by ln_train / preflight B2/B3. Equivalence with forward_batch is gated
    in preflight B2 across all 36 configured geometries."""
    y = model(x, export_image=False, fp64=P.FP64)
    assert y.shape[0] == x.shape[0] and y.shape[2] == 1
    assert y.shape[1] == model.pred_len
    assert y.dtype == torch.float32
    return y


def ln_param_names(model):
    """Exact trainable-LN inventory: every LayerNorm module's weight+bias.
    Must be exactly EXPECTED_LN_TENSORS tensors / EXPECTED_LN_SCALARS
    scalars (42 LayerNorms: 24 encoder block norms + final encoder norm +
    16 decoder block norms + final decoder norm)."""
    names = []
    for mod_name, mod in model.named_modules():
        if mod.__class__.__name__ == "LayerNorm":
            for pname in ("weight", "bias"):
                full = "%s.%s" % (mod_name, pname)
                if dict(model.named_parameters()).get(full) is not None:
                    names.append(full)
    return sorted(names)


def freeze_to_ln(model):
    """requires_grad=False everywhere, then True ONLY on the 84 LN tensors.
    Hard asserts: exact name set, tensor count 84, scalar count 55,808.
    Must be called AFTER configure (mask buffer is rebuilt by config, and
    freezing order does not interact with buffers, but the LN inventory is
    configuration-independent so this is safe in either order)."""
    for prm in model.parameters():
        prm.requires_grad_(False)
    names = ln_param_names(model)
    if len(names) != P.EXPECTED_LN_TENSORS:
        raise SystemExit("[FATAL] LN tensor inventory %d != %d"
                         % (len(names), P.EXPECTED_LN_TENSORS))
    ps = dict(model.named_parameters())
    scalars = 0
    for n in names:
        t = ps[n]
        if t.ndim != 1:
            raise SystemExit("[FATAL] LN tensor %s not 1-D" % n)
        scalars += t.numel()
        t.requires_grad_(True)
    if scalars != P.EXPECTED_LN_SCALARS:
        raise SystemExit("[FATAL] LN scalar inventory %d != %d"
                         % (scalars, P.EXPECTED_LN_SCALARS))
    for n, prm in model.named_parameters():
        want = n in set(names)
        if prm.requires_grad != want:
            raise SystemExit("[FATAL] requires_grad mismatch on %s" % n)
    return names


def predict_split(model, X, device, batch=P.BATCH):
    """Sequential-slicing batch inference over [N, L] float32 array.
    Returns pred [N, H, 1] float32 on CPU."""
    from repro_timemmd_common_ln_v1.data_timemmd import infer_batches
    n = X.shape[0]
    out = torch.empty((n, model.pred_len, 1), dtype=torch.float32,
                      device="cpu", pin_memory=False)
    for lo, hi in infer_batches(n, batch):
        xb = torch.from_numpy(X[lo:hi]).to(device).unsqueeze(-1)
        yb = forward_batch(model, xb)
        out[lo:hi] = yb.to("cpu", non_blocking=False)
    return out.numpy()


def digest_or_none(model):
    try:
        return model_state_digest(model)
    except Exception:
        return None
