"""VisionTS LN full-shot single-job trainer.

Mirror of long_term_tsf/run.py with exactly three deltas:
  1. --seed argument (paper repeats runs 3x without disclosing seeds; we use
     2021/2022/2023 — supplementary). Seeding happens right after parsing.
  2. Exp_Univariate is used instead of Exp_Long_Term_Forecast (batch=256
     univariate samples for every dataset; complete fixed val; element-weighted
     val MSE; explicit trainable-param Adam).
  3. After train() and BEFORE test(): verify checkpoint.pth + valid_loss.json
     exist and load_state_dict(strict=True) succeeds in-process (the official
     train->test(test=0) path evaluates in-memory weights and would silently
     test the last epoch if the best-checkpoint reload had failed).

Job bookkeeping (fingerprint / resume / archive / SUCCESS) is handled here so
run_all.sh stays a thin serial loop. Exit codes: 0 done-or-skip, 3 fingerprint
mismatch on a SUCCESS job (queue must abort), other nonzero = failure.

Run with CWD=long_term_tsf (root_path values are './dataset/...').
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.getcwd(), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp_univariate import Exp_Univariate  # noqa: E402
from utils.print_args import print_args  # noqa: E402

FINGERPRINT_FILES = [
    'repro_fullshot/run_train.py',
    'repro_fullshot/exp_univariate.py',
    'long_term_tsf/exp/exp_long_term_forecasting.py',
    'long_term_tsf/exp/exp_basic.py',
    'long_term_tsf/models/VisionTS.py',
    'long_term_tsf/data_provider/data_factory.py',
    'long_term_tsf/data_provider/data_loader.py',
    'long_term_tsf/utils/tools.py',
    'long_term_tsf/utils/metrics.py',
    'visionts/model.py',
    'visionts/models_mae.py',
    'visionts/util.py',
]

# mirrors visionts.model.MAE_ARCH (arch -> ckpt filename)
MAE_CKPT_FILE = {'mae_base': 'mae_visualize_vit_base.pth',
                 'mae_large': 'mae_visualize_vit_large.pth',
                 'mae_huge': 'mae_visualize_vit_huge.pth'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def make_fingerprint(args):
    repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    csv_path = os.path.join(args.root_path, args.data_path)
    ckpt_path = os.path.join(args.vm_ckpt, MAE_CKPT_FILE[args.vm_arch])
    code_md5 = {}
    for rel in FINGERPRINT_FILES:
        p = os.path.join(repo_root, rel)
        code_md5[rel] = md5(p) if os.path.isfile(p) else 'MISSING:' + p
    return {
        'args': {k: v for k, v in vars(args).items()},
        'seed': args.seed,
        'data_sha256': {csv_path: sha256(csv_path)},
        'ckpt_sha256': {ckpt_path: sha256(ckpt_path)},
        'code_md5': code_md5,
        'env': {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        },
    }


def main():
    parser = argparse.ArgumentParser(description='VisionTS LN full-shot repro (mirror of run.py)')
    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--model', type=str, required=True, default='VisionTS')
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)
    # model define (unused by VisionTS, kept for run.py parity)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--channel_independence', type=int, default=0)
    # optimization
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    # de-stationary projector params (run.py parity)
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)
    # VisionTS
    parser.add_argument('--vm_pretrained', type=int, default=1)
    parser.add_argument('--vm_ckpt', type=str, default='./ckpt/')
    parser.add_argument('--vm_arch', type=str, default='mae_base')
    parser.add_argument('--ft_type', type=str, default='ln')
    parser.add_argument('--periodicity', type=int, default=0)
    parser.add_argument('--interpolation', type=str, default='bilinear')
    parser.add_argument('--norm_const', type=float, default=0.4)
    parser.add_argument('--align_const', type=float, default=0.4)
    # repro delta 1: seed
    parser.add_argument('--seed', type=int, default=2021)
    # repro bookkeeping
    parser.add_argument('--archive_dir', type=str,
                        default=None, help='where failed (no SUCCESS) job dirs are moved to before a rerun')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Args in experiment:')
    print_args(args)

    if args.is_training:
        assert args.save_dir != '.', 'save_dir must be an explicit job directory'
        setting = '_'
        ckpt_dir = os.path.join(args.save_dir, args.checkpoints, setting)
        result_dir = os.path.join(args.save_dir, 'results', setting)
        success_path = os.path.join(args.save_dir, 'SUCCESS.txt')
        fp_path = os.path.join(args.save_dir, 'fingerprint.json')
        args_path = os.path.join(args.save_dir, 'args.json')

        fingerprint = make_fingerprint(args)

        if os.path.isfile(success_path):
            with open(fp_path) as f:
                stored = json.load(f)
            if stored == fingerprint:
                print(f'[SKIP] {args.save_dir} already SUCCESS with matching fingerprint')
                return 0
            print(f'[ABORT] {args.save_dir} SUCCESS exists but fingerprint mismatch. '
                  f'Diff details: args/seed/code/data changed. Old job kept intact; '
                  f'use a new save_dir (versioned rerun) or resolve manually.')
            return 3

        if os.path.isdir(args.save_dir) and os.listdir(args.save_dir):
            archive_root = args.archive_dir or os.path.join(os.path.dirname(args.save_dir.rstrip('/')), '_failed')
            os.makedirs(archive_root, exist_ok=True)
            target = os.path.join(archive_root, os.path.basename(args.save_dir.rstrip('/')) + '_' + time.strftime('%Y%m%d_%H%M%S'))
            shutil.move(args.save_dir, target)
            print(f'[ARCHIVE] previous unfinished job dir moved to {target}')

        os.makedirs(args.save_dir, exist_ok=True)
        with open(args_path, 'w') as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        with open(fp_path, 'w') as f:
            json.dump(fingerprint, f, indent=2, ensure_ascii=False)

        for ii in range(args.itr):
            exp = Exp_Univariate(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            # repro delta 3: verify best checkpoint in-process before testing
            ckpt_file = os.path.join(ckpt_dir, 'checkpoint.pth')
            valid_file = os.path.join(ckpt_dir, 'valid_loss.json')
            assert os.path.isfile(ckpt_file), f'checkpoint missing after train(): {ckpt_file}'
            assert os.path.isfile(valid_file), f'valid_loss.json missing after train(): {valid_file}'
            state = torch.load(ckpt_file, map_location='cpu')
            exp.model.load_state_dict(state)
            with open(valid_file) as f:
                valid = json.load(f)
            assert 'best_valid_loss' in valid and 'best_valid_epoch' in valid, f'bad valid_loss.json: {valid}'
            print(f'[CHECKPOINT VERIFIED] {ckpt_file} | valid={valid}')

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.use_gpu and torch.cuda.is_available():
                print('Peak GPU memory (this process): {:.2f} GB'.format(torch.cuda.max_memory_allocated() / 1024 ** 3))
            torch.cuda.empty_cache()

        metrics = np.load(os.path.join(result_dir, 'metrics.npy'))
        assert np.isfinite(metrics).all(), f'non-finite metrics: {metrics}'
        with open(fp_path) as f:
            fingerprint = json.load(f)
        with open(success_path, 'w') as f:
            json.dump({
                'job': os.path.basename(args.save_dir.rstrip('/')),
                'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
                'seed': args.seed,
                'mse': float(metrics[1]),
                'mae': float(metrics[0]),
                'metrics_mae_mse_rmse_mape_mspe_valloss_valepoch': metrics.tolist(),
                'fingerprint': fingerprint,
            }, f, indent=2, ensure_ascii=False)
        print(f'[SUCCESS] {args.save_dir} mse={metrics[1]:.4f} mae={metrics[0]:.4f}')
        return 0

    else:
        # test-only path (mirror of run.py; not used by the queue)
        ii = 0
        setting = '_'
        exp = Exp_Univariate(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu and torch.cuda.is_available():
            print('Peak GPU memory (this process): {:.2f} GB'.format(torch.cuda.max_memory_allocated() / 1024 ** 3))
        torch.cuda.empty_cache()
        return 0


if __name__ == '__main__':
    sys.exit(main())
