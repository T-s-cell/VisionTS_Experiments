"""EXP-046:Moirai-1.0-R-Base 冻结推理(uni2ts 官方)。

- 逐变量单序列;历史 = 标准化窗全长 L(恰好 context_length,无隐式截断);
- patch_size=32 固定(本轮预设,尚未验证是否最优),num_samples=200,点预测=逐时刻中位数;
- 随机性:random/numpy/torch/cuda 种子全显式固定;批大小、样本顺序(item_id↔起点)记录;
- 耗时分记录:module 初始化 / 各数据集首批(含预热)/ 各数据集推理;GPU 峰值显存;
- 单样本失败不可发生在此层(批量推理);有限性逐样本断言,失败 → 非零退出(管线级错误)。
"""
import csv
import json
import os
import random
import sys
import time

import numpy as np

SEED = 2021
PATCH_SIZE = 32
NUM_SAMPLES = 200
BATCH_SIZE = 64

H = 96


def main(repo_root, save_dir, smoke=False):
    random.seed(SEED)
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    import pandas as pd
    from gluonts.dataset.common import ListDataset

    from t46_common import DS_CFG, DS_ORDER, load_csv, sample_list_path, std_all, train_scaler
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    import inspect
    sig = str(inspect.signature(MoiraiForecast.__init__))
    assert 'num_samples' in sig, 'unexpected uni2ts API: ' + sig

    os.makedirs(save_dir, exist_ok=True)
    samples = list(csv.DictReader(open(sample_list_path())))
    t0 = time.time()
    module = MoiraiModule.from_pretrained('Salesforce/moirai-1.0-R-base')
    t_init = time.time() - t0
    print('module init %.1fs' % t_init)

    meta = dict(seed=SEED, patch_size=PATCH_SIZE, num_samples=NUM_SAMPLES,
                batch_size=BATCH_SIZE, weights='Salesforce/moirai-1.0-R-base',
                init_seconds=t_init, per_ds={}, moirai_signature=sig,
                torch=torch.__version__, cuda_seed_fixed=True)

    if smoke:
        samples = [s for s in samples if s['ds'] in ('ETTh1', 'ETTm1')]
        groups = ['ETTh1', 'ETTm1']
    else:
        groups = list(DS_ORDER)

    for ds in groups:
        cfg = DS_CFG[ds]
        ss = [s for s in samples if s['ds'] == ds]
        if smoke:
            assert len(ss) >= 8, (ds, len(ss))
            ss = ss[:8]
        data = load_csv(os.path.join(repo_root, cfg['csv']))
        mean, scale = train_scaler(data, cfg['K'])
        std = std_all(data, mean, scale)

        entries, true = [], []
        for s in ss:
            start_row, ch = int(s['start_row']), int(s['channel'])
            x = std[start_row - cfg['L']:start_row, ch]
            assert len(x) == cfg['L']
            entries.append(dict(target=x.astype(np.float32),
                                start=pd.Period('2020-01-01', freq=cfg['freq']),
                                feat_static_cat=[0], item_id=s['sample_id']))
            true.append(std[start_row:start_row + H, ch])
        gds = ListDataset(entries, freq=cfg['freq'])

        model = MoiraiForecast(module=module, prediction_length=H, context_length=cfg['L'],
                               patch_size=PATCH_SIZE, num_samples=NUM_SAMPLES, target_dim=1,
                               feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        predictor = model.create_predictor(batch_size=BATCH_SIZE)
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        preds = {}
        n_done = 0
        for fcst in predictor.predict(gds):
            arr = np.asarray(fcst.samples)  # (num_samples, H)
            assert arr.shape == (NUM_SAMPLES, H), arr.shape
            point = np.median(arr, axis=0)
            assert np.isfinite(point).all(), fcst.item_id
            preds[int(fcst.item_id)] = point
            n_done += 1
            if n_done == 1:
                meta['per_ds'].setdefault(ds, {})['first_batch_seconds'] = round(time.time() - t1, 2)
        elapsed = time.time() - t1
        assert n_done == len(ss), (n_done, len(ss))
        assert set(preds) == {int(s['sample_id']) for s in ss}
        pred = np.stack([preds[int(s['sample_id'])] for s in ss]).astype(np.float32)
        true = np.stack(true).astype(np.float32)
        assert np.isfinite(pred).all()
        peak_gib = torch.cuda.max_memory_allocated() / 2**30

        suffix = '_smoke' if smoke else ''
        np.savez_compressed(os.path.join(save_dir, f'arr_moirai{suffix}_{ds}.npz'),
                            pred=pred, true=true,
                            item_ids=np.array([int(s['sample_id']) for s in ss], dtype=np.int64),
                            start_rows=np.array([int(s['start_row']) for s in ss], dtype=np.int64))
        meta['per_ds'][ds].update(n=len(ss), infer_seconds=round(elapsed, 2),
                                  seconds_per_sample=round(elapsed / len(ss), 3),
                                  gpu_peak_gib=round(peak_gib, 3),
                                  sample_order=[int(s['sample_id']) for s in ss])
        print(ds, {k: v for k, v in meta['per_ds'][ds].items() if k != 'sample_order'})

    suffix = '_smoke' if smoke else ''
    with open(os.path.join(save_dir, f'moirai_meta{suffix}.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('moirai done, meta written')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'save_t46_histdiag',
         smoke='--smoke' in sys.argv)
