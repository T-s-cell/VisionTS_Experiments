"""EXP-047 推理:Moirai(t46diag env,GPU)。

配对随机数控制(用户修正 2026-11-19):**按条件分别推理** — 每个 cond_id 内按
ds 顺序、该条件固定基准窗口序(batch64)推理,**每次实际推理前重置全部种子
2021** → 同 ds 的 A/B 条件共享同一随机流状态;结果按 item_id=sid_t46 映射回
manifest 行序。δ=0 一致性由冒烟两次实际推理验证(不复制预测)。
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(repo_root, save_dir, smoke=False):
    with open(os.path.join(SCRIPT_DIR, 'manifest_t47.csv')) as f:
        man = list(csv.DictReader(f))
    if smoke:
        from t47_common import smoke_subset
        man = smoke_subset(man)
    assert len(man) % 16 == 0

    import pandas as pd
    import torch
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from t47_common import M_MULTS, apply_variant, cond_params  # 先于 t46_common(其导入会挂 t46 路径)
    from t46_common import DS_CFG, DS_ORDER

    import inspect
    sig = str(inspect.signature(MoiraiForecast.__init__))
    assert 'num_samples' in sig, sig

    os.makedirs(save_dir, exist_ok=True)
    t0 = time.time()
    module = MoiraiModule.from_pretrained('Salesforce/moirai-1.0-R-base')
    print('module init %.1fs' % (time.time() - t0))

    n_total = len(man)
    pred = np.full((7168, 96), np.nan, dtype=np.float32)
    true_p = np.full((7168, 96), np.nan, dtype=np.float32)
    per = {}
    for cid in range(16):
        g, di, mi, dmult, mop = cond_params(cid)
        for ds in DS_ORDER:
            rows = [r for r in man if r['ds'] == ds and int(r['cond_id']) == cid]
            if not rows:
                continue
            cfg = DS_CFG[ds]
            L, P = int(rows[0]['L']), int(rows[0]['P'])
            bz = np.load(os.path.join(save_dir, 'baseline_%s.npz' % ds))
            x_f64, y_f64 = bz['x'], bz['y']
            sid_row = {int(s): i for i, s in enumerate(bz['sid_t46'])}
            set_seeds(SEED)  # 每次实际推理前重置(条件×ds 粒度)
            entries, trues = [], []
            for r in rows:
                w = sid_row[int(r['sid_t46'])]
                m_steps = 0 if mi is None else M_MULTS[mi] * P
                x = apply_variant(x_f64[w], float(r['delta']), m_steps, g)
                assert len(x) == L
                entries.append(dict(target=x.astype(np.float32),
                                    start=pd.Period('2020-01-01', freq=cfg['freq']),
                                    feat_static_cat=[0], item_id=int(r['sid_t46'])))
                trues.append(y_f64[w] + float(r['delta']))
            gds = ListDataset(entries, freq=cfg['freq'])
            model = MoiraiForecast(module=module, prediction_length=H, context_length=L,
                                   patch_size=PATCH_SIZE, num_samples=NUM_SAMPLES,
                                   target_dim=1, feat_dynamic_real_dim=0,
                                   past_feat_dynamic_real_dim=0)
            predictor = model.create_predictor(batch_size=BATCH_SIZE)
            t1 = time.time()
            preds, n_done = {}, 0
            for fcst in predictor.predict(gds):
                arr = np.asarray(fcst.samples)
                assert arr.shape == (NUM_SAMPLES, H), arr.shape
                point = np.median(arr, axis=0)
                assert np.isfinite(point).all(), fcst.item_id
                preds[int(fcst.item_id)] = point
                n_done += 1
            assert n_done == len(rows), (cid, ds, n_done, len(rows))
            elapsed = time.time() - t1
            key = 'cond%d_%s' % (cid, ds)
            per[key] = dict(group=g, n=len(rows), seconds=round(elapsed, 2),
                             seconds_per_sample=round(elapsed / len(rows), 3))
            for k, r in enumerate(rows):
                rg = int(r['row_global'])
                pred[rg] = preds[int(r['sid_t46'])].astype(np.float32)
                true_p[rg] = trues[k].astype(np.float32)
            print(cid, g, ds, per[key])
    rg = np.array([int(r['row_global']) for r in man], dtype=np.int64)
    assert np.isfinite(pred[rg]).all()
    item_ids = np.array([int(r['sid_t46']) for r in man], dtype=np.int64)
    cond_ids = np.array([int(r['cond_id']) for r in man], dtype=np.int64)
    suffix = '_smoke' if smoke else ''
    np.savez_compressed(os.path.join(save_dir, 'pred_moirai_t47%s.npz' % suffix),
                        pred=pred[rg], true_p=true_p[rg], item_ids=item_ids,
                        cond_ids=cond_ids, row_globals=rg)
    meta = dict(seed=SEED, patch_size=PATCH_SIZE, num_samples=NUM_SAMPLES,
                batch_size=BATCH_SIZE, seed_reset_granularity='cond_x_ds',
                per_cond_ds=per, moirai_signature=sig, torch=torch.__version__)
    with open(os.path.join(save_dir, 'moirai_meta_t47%s.json' % suffix), 'w') as f:
        json.dump(meta, f, indent=1)
    print('moirai done: %d rows' % len(man))


M_MULT_SIGN = {0: 1, 1: 2, 2: 4}  # m_idx → m/P

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else os.path.join(SCRIPT_DIR, 'save_t47_levelshift'), smoke='--smoke' in sys.argv)
