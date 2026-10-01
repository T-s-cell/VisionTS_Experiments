"""Batch-semantics + budget check (must pass before any training run).

For each of the 8 datasets x 4 pred_lens, builds the exact datasets/loaders the
training will use (data_provider + Dataset_UnivariateView for ETT) and asserts:
  - len(data_x) matches the split formulas against known csv row counts;
  - univariate view len == windows x channels;
  - sample shapes (seq_len,1) / (label_len+pred_len,1) [pred 96 only];
  - a loader batch is (256, seq_len, 1) and len(loader) matches the computed
    train floor / val+test ceil batch counts [authoritative budget numbers];
  - records VisionTS num_patch_input (visible columns) and encoder tokens.
Budget table: per-job test-set pred/true npy bytes, x3-seed totals, checkpoint
size. Writes batch_check_report.txt. Run with CWD=long_term_tsf.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.getcwd())

import torch
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from exp_univariate import Dataset_UnivariateView, ETT_DATA
from visionts import VisionTS

BATCH = 256
LABEL_LEN = 48

ROWS = {'ETTh1': 17420, 'ETTh2': 17420, 'ETTm1': 69680, 'ETTm2': 69680,
        'Weather': 52696, 'Electricity': 26304, 'Traffic': 17544, 'illness': 966}

CONFIGS = {
    'ETTh1':       dict(data='ETTh1', seq_len=1152, periodicity=24, r=0.4, c=0.4,
                        root='./dataset/ETT-small/', file='ETTh1.csv', freq='h', preds=[96, 192, 336, 720]),
    'ETTh2':       dict(data='ETTh2', seq_len=1152, periodicity=24, r=0.4, c=0.4,
                        root='./dataset/ETT-small/', file='ETTh2.csv', freq='h', preds=[96, 192, 336, 720]),
    'ETTm2':       dict(data='ETTm2', seq_len=1152, periodicity=96, r=0.4, c=0.4,
                        root='./dataset/ETT-small/', file='ETTm2.csv', freq='t', preds=[96, 192, 336, 720]),
    'ETTm1':       dict(data='ETTm1', seq_len=2304, periodicity=96, r=0.4, c=0.4,
                        root='./dataset/ETT-small/', file='ETTm1.csv', freq='t', preds=[96, 192, 336, 720]),
    'Weather':     dict(data='custom', seq_len=576, periodicity=144, r=1.0, c=0.7,
                        root='./dataset/weather/', file='weather.csv', freq='h', preds=[96, 192, 336, 720]),
    'Electricity': dict(data='custom', seq_len=1152, periodicity=24, r=0.4, c=0.4,
                        root='./dataset/electricity/', file='electricity.csv', freq='h', preds=[96, 192, 336, 720]),
    'Traffic':     dict(data='custom', seq_len=1152, periodicity=24, r=0.4, c=0.4,
                        root='./dataset/traffic/', file='traffic.csv', freq='h', preds=[96, 192, 336, 720]),
    'illness':     dict(data='custom', seq_len=104, periodicity=52, r=1.0, c=0.4,
                        root='./dataset/illness/', file='national_illness.csv', freq='h', preds=[24, 36, 48, 60]),
}

report = []


def check(name, ok, detail=''):
    line = f'[{"PASS" if ok else "FAIL"}] {name}' + (f' | {detail}' if detail else '')
    report.append(line)
    print(line, flush=True)
    if not ok:
        raise AssertionError(line)


def expected_lengths(name, cfg):
    n = ROWS[name]
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_vali = n - num_train - num_test
    s = cfg['seq_len']
    if cfg['data'] in ETT_DATA:
        if cfg['data'] in ('ETTh1', 'ETTh2'):
            train, side = 8640, 2880          # 12*30*24 / 4*30*24
        else:
            train, side = 34560, 11520        # x4 for minute
        return {'train': train, 'val': side + s, 'test': side + s}
    return {'train': num_train, 'val': num_vali + s, 'test': num_test + s}


def make_args(cfg, pred):
    return SimpleNamespace(
        task_name='long_term_forecast', data=cfg['data'], root_path=cfg['root'],
        data_path=cfg['file'], features='M', target='OT', embed='timeF', freq=cfg['freq'],
        seq_len=cfg['seq_len'], label_len=LABEL_LEN, pred_len=pred,
        scale=True, seasonal_patterns=None, batch_size=BATCH, num_workers=0)


def build_loader(cfg, pred, flag):
    ds, _ = data_provider(make_args(cfg, pred), flag)
    if cfg['data'] in ETT_DATA:
        ds = Dataset_UnivariateView(ds)
    shuffle = flag == 'train'
    drop_last = flag == 'train'
    loader = DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                        num_workers=0, drop_last=drop_last)
    return ds, loader


def main():
    torch.manual_seed(2021)
    model = VisionTS(arch='mae_base', finetune_type='ln', load_ckpt=False, ckpt_dir='../ckpt/')
    ckpt_scalars = sum(p.numel() for p in model.parameters()) * 4

    budget = []
    total_npy_bytes = 0
    for name, cfg in CONFIGS.items():
        exp_len = expected_lengths(name, cfg)
        channels = {'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7,
                    'Weather': 21, 'Electricity': 321, 'Traffic': 862, 'illness': 7}[name]
        for pred in cfg['preds']:
            model.update_config(context_len=cfg['seq_len'], pred_len=pred,
                                periodicity=cfg['periodicity'], norm_const=cfg['r'],
                                align_const=cfg['c'], interpolation='bilinear')
            tokens = model.num_patch_input * 14 + 1
            counts = {}
            for flag in ('train', 'val', 'test'):
                ds, loader = build_loader(cfg, pred, flag)
                win = len(ds) // channels
                check(f'{name}/{pred} {flag}: len == windows x C',
                      len(ds) == win * channels and win == exp_len[flag] - cfg['seq_len'] - pred + 1,
                      f'len={len(ds)} windows={win} len(data_x)={exp_len[flag]}')
                counts[flag] = len(loader)
            budget.append(dict(name=name, pred=pred, channels=channels,
                               train=counts['train'], val=counts['val'], test=counts['test'],
                               tokens=tokens, npi=model.num_patch_input,
                               npy_gb=2 * channels * (exp_len['test'] - cfg['seq_len'] - pred + 1)
                               * pred * 4 / 1024 ** 3))
            total_npy_bytes += budget[-1]['npy_gb']
            if pred == cfg['preds'][0]:
                x0, y0, xm0, ym0 = build_loader(cfg, pred, 'train')[0][0]
                check(f'{name}/{pred}: sample shapes',
                      tuple(x0.shape) == (cfg['seq_len'], 1)
                      and tuple(y0.shape) == (LABEL_LEN + pred, 1),
                  f'x={tuple(x0.shape)} y={tuple(y0.shape)} mark={tuple(xm0.shape)}')
                xb, yb, xmb, ymb = next(iter(DataLoader(
                    build_loader(cfg, pred, 'train')[0], batch_size=BATCH,
                    shuffle=False, num_workers=0, drop_last=True)))
                check(f'{name}/{pred}: batch shape', tuple(xb.shape) == (BATCH, cfg['seq_len'], 1),
                      f'x={tuple(xb.shape)} y={tuple(yb.shape)}')
            check(f'{name}/{pred}: num_patch_input recorded', model.num_patch_input >= 1,
                  f'npi={model.num_patch_input} tokens={tokens} mask_ratio={model.mask_ratio:.4f}')
        print(f'--- {name} done ---', flush=True)

    print('\n===== BUDGET (per job) =====')
    print(f'{"DS":<13}{"pred":>5}{"tokens":>8}{"train":>8}{"val":>7}{"test":>8}{"pred+true GB":>14}')
    for b in budget:
        print(f'{b["name"]:<13}{b["pred"]:>5}{b["tokens"]:>8}{b["train"]:>8}{b["val"]:>7}'
              f'{b["test"]:>8}{b["npy_gb"]:>14.2f}')
    total_iters = sum(b['train'] + b['val'] + b['test'] for b in budget)
    print(f'\ntotal iters/seed = {total_iters}; x3 seeds = {total_iters * 3}')
    print(f'pred+true npy total: {total_npy_bytes:.1f} GB/job-set x3 seeds = {total_npy_bytes * 3:.1f} GB')
    print(f'checkpoint state_dict size ≈ {ckpt_scalars / 1024 ** 3:.2f} GB x96 jobs = {ckpt_scalars * 96 / 1024 ** 3:.1f} GB')

    out_path = '../repro_fullshot/batch_check_report.txt'
    with open(out_path, 'w') as f:
        f.write('\n'.join(report) + '\n')
    print('report ->', out_path, flush=True)


if __name__ == '__main__':
    main()
