"""Diagnostic per-iter timing breakdown (not part of the reproduction).

Times data fetch / forward-only / forward+backward on ETTh1/96 with the real
train loader, reports TF32 flags, and re-times with TF32 enabled (potential
speedup only — NOT applied to the reproduction without explicit approval).
Run with CWD=long_term_tsf.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.getcwd())

import torch

from batch_check import CONFIGS, build_loader
from visionts import VisionTS


def bench(model, it, pred, n, backward):
    t_data = t_all = 0.0
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        s = time.time()
        xb, yb, _, _ = next(it)
        xb = xb.float().cuda()
        yb = yb.float().cuda()
        torch.cuda.synchronize()
        t_data += time.time() - s
        out = model(xb)
        loss = torch.nn.functional.mse_loss(out, yb[:, -pred:, :])
        if backward:
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    t_all = time.time() - t0
    return t_all / n, t_data / n


def main():
    pred = 96
    cfg = CONFIGS['ETTh1']
    print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
    print('tf32 matmul:', torch.backends.cuda.matmul.allow_tf32,
          'tf32 cudnn:', torch.backends.cudnn.allow_tf32, flush=True)
    print('cpu count:', os.cpu_count(), 'torch threads:', torch.get_num_threads(), flush=True)

    model = VisionTS(arch='mae_base', finetune_type='ln', load_ckpt=True,
                     ckpt_dir='../ckpt/' + os.sep)
    model.update_config(context_len=cfg['seq_len'], pred_len=pred,
                        periodicity=cfg['periodicity'], norm_const=cfg['r'],
                        align_const=cfg['c'], interpolation='bilinear')
    model = model.to('cuda').train()

    ds, loader = build_loader(cfg, pred, 'train')
    it = iter(loader)
    for _ in range(3):
        xb, yb, _, _ = next(it)
        xb = xb.float().cuda()
        yb = yb.float().cuda()
        out = model(xb)
        torch.nn.functional.mse_loss(out, yb[:, -pred:, :]).backward()
        model.zero_grad(set_to_none=True)

    t, td = bench(model, it, pred, 10, backward=False)
    print(f'forward-only: {t:.3f} s/iter (data {td:.3f})', flush=True)
    t, td = bench(model, it, pred, 10, backward=True)
    print(f'fwd+bwd:      {t:.3f} s/iter (data {td:.3f})', flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t, td = bench(model, it, pred, 10, backward=True)
    print(f'fwd+bwd TF32: {t:.3f} s/iter (data {td:.3f})  [potential only]', flush=True)


if __name__ == '__main__':
    main()
