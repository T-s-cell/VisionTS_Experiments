"""Single choke point for model construction and configured inference.

adapter.configure / adapter.forward_batch is the ONLY path used to call
VisionTS in p_search / run_infer / replay / preflight A3. State isolation
is asserted on every configure.
"""
import torch

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import model_state_digest


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


def predict_split(model, X, device, batch=P.BATCH):
    """Sequential-slicing batch inference over [N, L] float32 array.
    Returns pred [N, H, 1] float32 on CPU."""
    from repro_timemmd_common_v1.data_timemmd import infer_batches
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
