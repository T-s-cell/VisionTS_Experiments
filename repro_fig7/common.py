"""Shared constants and helpers for the Figure 7 partial reproduction.

Import side effect: puts BOTH the repo root (for the `visionts` package, which
uses relative imports) and `long_term_tsf/` (for the `data_provider` package,
which uses absolute imports) on sys.path, independent of CWD.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
LTSF_ROOT = os.path.join(REPO_ROOT, 'long_term_tsf')
for _p in (REPO_ROOT, LTSF_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import numpy as np
import torch
import torch.nn.functional as F
import einops

SEED = 2021
PRED_LEN = 96
NORM_CONST = 0.4
ALIGN_CONST = 0.4
LABEL_LEN = 48
BATCH = 32
N_IMAGENET = 1000
N_TS = 300
FEATURE_DIM = (1 + 70) * 768  # (cls + 70 visible patches) x embed_dim = 54528
GROUPS = ['ImageNet', 'ETTm1', 'Weather', 'Electricity']

TS_CONFIGS = {
    'ETTm1': dict(
        data='ETTm1', seq_len=2304, periodicity=96,
        root_path='./dataset/ETT-small/', data_path='ETTm1.csv',
        border1=46080 - 2304,
    ),
    'Weather': dict(
        data='custom', seq_len=4032, periodicity=144,
        root_path='./dataset/weather/', data_path='weather.csv',
        border1=None,
    ),
    'Electricity': dict(
        data='custom', seq_len=2880, periodicity=24,
        root_path='./dataset/electricity/', data_path='electricity.csv',
        border1=None,
    ),
}

IMAGENET_VAL_DIR = os.environ.get('IMAGENET_VAL_DIR', '/dev_data/wlt/data/ImageNet/val')
CKPT_DIR = os.path.join(REPO_ROOT, 'ckpt')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
LIST_DIR = os.path.join(OUT_DIR, 'sampling_lists')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPLEMENTARY = {
    'pred_len': PRED_LEN,
    'note': 'pred_len=96, reusing zero-shot configs for Fig.7, and the resulting left-5-column mask '
            'are supplementary settings of this reproduction, NOT specified by the paper.',
}


def set_seeds():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_model():
    from visionts.model import VisionTS
    model = VisionTS(arch='mae_base', ckpt_dir=CKPT_DIR + os.sep, load_ckpt=True)
    return model


def configure(model, name):
    cfg = TS_CONFIGS[name]
    model.update_config(context_len=cfg['seq_len'], pred_len=PRED_LEN, periodicity=cfg['periodicity'],
                        norm_const=NORM_CONST, align_const=ALIGN_CONST, interpolation='bilinear')
    assert model.num_patch_input == 5, f'{name}: num_patch_input={model.num_patch_input}, expected 5'
    return cfg


def get_noise(model, n):
    return einops.repeat(model.mask, '1 l -> n l', n=n)


def build_ts_image(model, x):
    """Replicates VisionTS.forward steps 1-4 (visionts/model.py:107-128) exactly.

    Normalization divisor is sqrt(var(x_enc, unbiased=False) + 1e-5) / norm_const,
    matching the production code (1e-5 added to the variance inside the sqrt).
    x: (B, L, 1) float tensor on the target device. Returns (B*n, 3, 224, 224).
    """
    means = x.mean(1, keepdim=True).detach()
    x_enc = x - means
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    stdev = stdev / model.norm_const
    x_enc = x_enc / stdev
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')
    x_pad = F.pad(x_enc, (model.pad_left, 0), mode='replicate')
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=model.periodicity)
    x_resize = model.input_resize(x_2d)
    masked = torch.zeros((x_2d.shape[0], 1, model.image_size, model.num_patch_output * model.patch_size),
                         device=x_2d.device, dtype=x_2d.dtype)
    x_cat = torch.cat([x_resize, masked], dim=-1)
    image_input = einops.repeat(x_cat, 'b 1 h w -> b c h w', c=3)
    return image_input


def build_dataset(name):
    """Instantiate the test dataset through the same factory code path as the audited zero-shot runs."""
    from types import SimpleNamespace
    from data_provider.data_factory import data_provider
    cfg = TS_CONFIGS[name]
    args = SimpleNamespace(
        task_name='long_term_forecast',
        data=cfg['data'], root_path=cfg['root_path'], data_path=cfg['data_path'],
        features='M', target='OT', embed='timeF', freq='h',
        seq_len=cfg['seq_len'], label_len=LABEL_LEN, pred_len=PRED_LEN,
        scale=True, seasonal_patterns=None, batch_size=BATCH, num_workers=0,
    )
    ds, _ = data_provider(args, 'test')
    return ds


def variable_names(name):
    """Feature column names in post-reorder order (matching channel_index semantics)."""
    import pandas as pd
    cfg = TS_CONFIGS[name]
    df = pd.read_csv(os.path.join(LTSF_ROOT, cfg['root_path'], cfg['data_path']), nrows=1)
    if cfg['data'] == 'ETTm1':
        return list(df.columns[1:])
    date_col, target = 'date', 'OT'
    feats = [c for c in df.columns if c not in (date_col, target)]
    return feats + [target]


def custom_border1(name):
    """border1 of the test split for Dataset_Custom: len - int(len*0.2) - seq_len."""
    import pandas as pd
    cfg = TS_CONFIGS[name]
    df = pd.read_csv(os.path.join(LTSF_ROOT, cfg['root_path'], cfg['data_path']), usecols=['date'])
    total = len(df)
    num_test = int(total * 0.2)
    return total - num_test - cfg['seq_len']


def border1(name):
    cfg = TS_CONFIGS[name]
    return cfg['border1'] if cfg['border1'] is not None else custom_border1(name)


def timestamps_for(name, window_idx):
    """Real start/end timestamps (csv date column) of a window in the test split."""
    import pandas as pd
    cfg = TS_CONFIGS[name]
    dates = pd.read_csv(os.path.join(LTSF_ROOT, cfg['root_path'], cfg['data_path']),
                        usecols=['date'])['date']
    b1 = border1(name)
    s = b1 + window_idx
    e = s + cfg['seq_len'] - 1
    return str(dates.iloc[s]), str(dates.iloc[e]), s, e


def imagenet_transform():
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_sampling_list(group):
    with open(os.path.join(LIST_DIR, f'{group}.json')) as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
