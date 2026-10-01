"""Single-batch VRAM probes for the heaviest full-shot configs (pre-declared).

Probes Weather/96 (99 encoder tokens = max) and ETTm1/96 (71 tokens,
representative of ECL/Traffic/ETTm): one real train batch (256, L, 1)
forward + backward on the LN-tunable model from original MAE weights,
reports torch.cuda.max_memory_allocated() per probe.

Threshold (pre-declared): GPU free memory at launch (torch.cuda.mem_get_info,
measured before anything is allocated) minus 4GB safety margin. Any probe
peak above the threshold -> exit nonzero, queue must not start.

Run with CWD=long_term_tsf.
"""
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.getcwd())

import torch

from batch_check import CONFIGS, build_loader
from visionts import VisionTS

PROBES = [('Weather', 96), ('ETTm1', 96)]
MARGIN_BYTES = 4 * 1024 ** 3


def main():
    free0, total0 = torch.cuda.mem_get_info()
    threshold = free0 - MARGIN_BYTES
    print(f'GPU {torch.cuda.get_device_name(0)}: free at launch {free0/1024**3:.2f} GiB '
          f'/ total {total0/1024**3:.2f} GiB; threshold (free-4GiB) {threshold/1024**3:.2f} GiB',
          flush=True)

    device = 'cuda'
    model = VisionTS(arch='mae_base', finetune_type='ln', load_ckpt=True,
                     ckpt_dir='../ckpt/' + os.sep)
    model = model.to(device)
    model.train()

    failed = []
    for name, pred in PROBES:
        cfg = CONFIGS[name]
        model.update_config(context_len=cfg['seq_len'], pred_len=pred,
                            periodicity=cfg['periodicity'], norm_const=cfg['r'],
                            align_const=cfg['c'], interpolation='bilinear')
        ds, loader = build_loader(cfg, pred, 'train')
        xb, yb, _, _ = next(iter(loader))
        xb = xb.float().to(device)
        yb = yb.float().to(device)
        torch.cuda.reset_peak_memory_stats()
        out = model(xb)
        assert tuple(out.shape) == (256, pred, 1), tuple(out.shape)
        loss = torch.nn.functional.mse_loss(out, yb[:, -pred:, :])
        loss.backward()
        peak = torch.cuda.max_memory_allocated()
        model.zero_grad(set_to_none=True)
        ok = peak < threshold
        ok = ok and bool(ok)
        line = (f'[{"PASS" if ok else "FAIL"}] {name}/{pred}: tokens={model.num_patch_input} '
                f'peak={peak/1024**3:.2f} GiB (threshold {threshold/1024**3:.2f} GiB)')
        print(line, flush=True)
        if not ok:
            failed.append(f'{name}/{pred}')
        del xb, yb, out, loss
        torch.cuda.empty_cache()

    if failed:
        print(f'PROBE FAILED: {failed} — do not start the queue', flush=True)
        return 1
    print('PROBE ALL PASSED', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
