"""Minimal LN fine-tuning check (must pass before any training run).

Asserts, for arch=mae_base with finetune_type='ln':
  1. trainable set == exactly the 84 LayerNorm tensors (names listed) / 55,808 scalars;
  2. on a real forward/backward, every trainable tensor gets a grad (>=1 nonzero)
     and every frozen tensor gets grad=None;
  3. one Adam step changes LN weights (max|delta|>0) while sampled frozen params
     (qkv/proj weight, cls_token, pos_embed, decoder pos_embed) stay bit-identical.

Run with CWD=long_term_tsf on the GPU server. Writes ln_check_report.txt.
"""
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.getcwd())

import torch

from visionts import VisionTS

ARCH = 'mae_base'
CKPT_DIR = '../ckpt/' + os.sep


def expected_trainable():
    names = set()
    for i in range(12):
        names.add(f'vision_model.blocks.{i}.norm1.weight')
        names.add(f'vision_model.blocks.{i}.norm1.bias')
        names.add(f'vision_model.blocks.{i}.norm2.weight')
        names.add(f'vision_model.blocks.{i}.norm2.bias')
    names.add('vision_model.norm.weight')
    names.add('vision_model.norm.bias')
    for i in range(8):
        names.add(f'vision_model.decoder_blocks.{i}.norm1.weight')
        names.add(f'vision_model.decoder_blocks.{i}.norm1.bias')
        names.add(f'vision_model.decoder_blocks.{i}.norm2.weight')
        names.add(f'vision_model.decoder_blocks.{i}.norm2.bias')
    names.add('vision_model.decoder_norm.weight')
    names.add('vision_model.decoder_norm.bias')
    return names


def check(name, ok, detail=''):
    line = f'[{"PASS" if ok else "FAIL"}] {name}' + (f' | {detail}' if detail else '')
    print(line, flush=True)
    if not ok:
        raise AssertionError(line)


def main():
    torch.manual_seed(2021)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VisionTS(arch=ARCH, finetune_type='ln', load_ckpt=True, ckpt_dir=CKPT_DIR)
    model.update_config(context_len=1152, pred_len=96, periodicity=24,
                        norm_const=0.4, align_const=0.4, interpolation='bilinear')
    model = model.to(device)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    expected = expected_trainable()
    n_scalars = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    check('trainable set == expected 84 LN tensors', trainable == expected,
          f'got {len(trainable)} tensors, expected {len(expected)}; '
          f'missing={sorted(expected - trainable)[:5]} extra={sorted(trainable - expected)[:5]}')
    check('trainable scalars == 55808', n_scalars == 55808, str(n_scalars))

    check('all non-LN params frozen', all(
        not p.requires_grad for n, p in model.named_parameters() if n not in expected))

    # real batch forward/backward
    model.train()
    x = torch.randn(8, 1152, 1, device=device)
    target = torch.randn(8, 96, 1, device=device)
    out = model(x)
    check('forward output shape', tuple(out.shape) == (8, 96, 1), str(tuple(out.shape)))
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    bad_train = [n for n, p in model.named_parameters()
                 if p.requires_grad and (p.grad is None or not bool(p.grad.any()))]
    bad_frozen = [n for n, p in model.named_parameters()
                  if not p.requires_grad and p.grad is not None]
    check('all 84 trainable grads present & nonzero', not bad_train, f'bad={bad_train[:5]}')
    check('all frozen grads None', not bad_frozen, f'bad={bad_frozen[:5]}')

    # one Adam step on trainable params only
    watched_frozen = ['vision_model.blocks.0.attn.qkv.weight',
                      'vision_model.blocks.0.attn.proj.weight',
                      'vision_model.blocks.0.mlp.fc1.weight',
                      'vision_model.cls_token',
                      'vision_model.pos_embed',
                      'vision_model.decoder_pos_embed',
                      'vision_model.patch_embed.proj.weight']
    names = dict(model.named_parameters())
    ln_before = {n: names[n].detach().clone() for n in expected}
    before = {n: names[n].detach().clone() for n in watched_frozen}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    opt.step()
    ln_moved = max((ln_before[n] - names[n].detach()).abs().max().item() for n in expected)
    unchanged = {n: bool(torch.equal(before[n], names[n].detach())) for n in watched_frozen}
    check('LN weights changed after step', ln_moved > 0, f'max|delta|={ln_moved:.3e}')
    check('sampled frozen params bit-identical', all(unchanged.values()), str(unchanged))

    report = ['LN CHECK ALL PASSED',
              f'trainable_tensors={len(trainable)} trainable_scalars={n_scalars}',
              '--- 84 trainable parameter names ---']
    report += sorted(trainable)
    out_path = '../repro_fullshot/ln_check_report.txt'
    with open(out_path, 'w') as f:
        f.write('\n'.join(report) + '\n')
    print('report ->', out_path, flush=True)


if __name__ == '__main__':
    main()
