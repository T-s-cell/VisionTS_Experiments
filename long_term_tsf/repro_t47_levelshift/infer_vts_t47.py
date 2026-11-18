"""EXP-047 推理:VisionTS(visionts env,GPU)。

- 输入:manifest 行序(448×16),单变量 [bs,L,1] batch32,构造在线派生
  (baseline_<ds>.npz f64 → 构造 f64 → f32);
- 模型:mae_base,norm_const=0.4 / align_const=0.4,seeds 2021,ckpt sha 预核;
- 输出:pred_vts_t47.npz(pred/true_p f32 (7168,96)+item_ids/cond_ids/row_globals)+ meta;
- A 组等变性事后核验:pred_A vs t46 arr pred+δ(记录分布,不作门禁)。
"""
import json
import os
import sys
import time

import numpy as np
import torch

from t47_common import (CKPT_SHA, DS_CFG, DS_ORDER, N_CONDS, SCRIPT_DIR, T46_DIR,
                        apply_variant, manifest_path, smoke_subset)
import random


def set_seeds(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vm(repo_root, L, P):
    vs_root = os.path.dirname(os.path.abspath(repo_root))
    ckpt_dir = os.path.join(vs_root, 'ckpt')
    from t46_common import sha256_file
    sha = sha256_file(os.path.join(ckpt_dir, 'mae_visualize_vit_base.pth'))
    assert sha == CKPT_SHA, sha
    sys.path.insert(0, vs_root)
    from visionts import VisionTS
    vm = VisionTS(arch='mae_base', finetune_type='ln', load_ckpt=True, ckpt_dir=ckpt_dir)
    vm.update_config(context_len=L, pred_len=96, periodicity=P, interpolation='bilinear',
                     norm_const=0.4, align_const=0.4)
    return vm.cuda().eval()


def predict(vm, xs_f64):
    """(n,L) f64 → (n,96) f32;batch 32。"""
    outs = []
    for i in range(0, len(xs_f64), 32):
        xt = torch.tensor(xs_f64[i:i + 32], dtype=torch.float32).cuda().unsqueeze(-1)
        with torch.no_grad():
            out = vm.forward(xt)
        outs.append(out.squeeze(-1).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def main(repo_root, save_dir, smoke=False):
    import csv
    with open(manifest_path()) as f:
        man = list(csv.DictReader(f))
    assert len(man) == 7168
    if smoke:
        man = smoke_subset(man)

    set_seeds(2021)
    pred = np.full((7168, 96), np.nan, dtype=np.float32)
    true_p = np.full((7168, 96), np.nan, dtype=np.float32)
    per_ds = {}
    a_dev = []          # A 组等变性偏差样本
    for ds in DS_ORDER:
        t0 = time.time()
        rows = [r for r in man if r['ds'] == ds]
        L, P = int(rows[0]['L']), int(rows[0]['P'])
        vm = load_vm(repo_root, L, P)
        bz = np.load(os.path.join(save_dir, 'baseline_%s.npz' % ds))
        x_f64, y_f64, sid_arr = bz['x'], bz['y'], bz['sid_t46']
        sid_row = {int(s): i for i, s in enumerate(sid_arr)}
        xs = np.empty((len(rows), L), dtype=np.float64)
        ys = np.empty((len(rows), 96), dtype=np.float64)
        for k, r in enumerate(rows):
            w = sid_row[int(r['sid_t46'])]
            xs[k] = apply_variant(x_f64[w], float(r['delta']), int(r['m_steps']), r['group'])
            ys[k] = y_f64[w] + float(r['delta'])
        # 行序即 manifest 序(ds 块连续,window-major then cond),batch 32
        pred_rows = predict(vm, xs)
        idx0 = int(rows[0]['row_global'])
        pred[idx0:idx0 + len(rows)] = pred_rows
        true_p[idx0:idx0 + len(rows)] = ys.astype(np.float32)

        # A 组等变性(事后核验,非门禁):pred_A vs t46 pred + δ
        t46 = np.load(os.path.join(T46_DIR, 'save_t46_histdiag', 'arr_vts_t46.npz'))
        t46_pred = t46['%s_pred' % ds]
        sid_pos = {int(s): i for i, s in enumerate(sid_arr)}
        for k, r in enumerate(rows):
            if r['group'] == 'A':
                j = sid_pos[int(r['sid_t46'])]
                a_dev.append(float(np.max(np.abs(
                    pred_rows[k].astype(np.float64) - (t46_pred[j].astype(np.float64) + float(r['delta']))))))
        per_ds[ds] = dict(n=len(rows), seconds=round(time.time() - t0, 1))
        del vm
        torch.cuda.empty_cache()

    rows_meta = man
    item_ids = np.array([int(r['sid_t46']) for r in rows_meta], dtype=np.int64)
    cond_ids = np.array([int(r['cond_id']) for r in rows_meta], dtype=np.int64)
    row_globals = np.array([int(r['row_global']) for r in rows_meta], dtype=np.int64)
    suffix = '_smoke' if smoke else ''
    assert np.isfinite(pred[row_globals]).all()
    np.savez_compressed(os.path.join(save_dir, 'pred_vts_t47%s.npz' % suffix),
                        pred=pred[row_globals], true_p=true_p[row_globals], item_ids=item_ids,
                        cond_ids=cond_ids, row_globals=row_globals)
    a_dev = np.array(a_dev)
    meta = dict(seed=2021, ckpt_sha_ok=True, batch=32, per_ds=per_ds,
                a_equivariance=dict(n=int(len(a_dev)), max=float(a_dev.max()),
                                    mean=float(a_dev.mean()), p95=float(np.quantile(a_dev, 0.95)),
                                    note='post-hoc check only, not a gate'))
    with open(os.path.join(save_dir, 'meta_vts_t47%s.json' % suffix), 'w') as f:
        json.dump(meta, f, indent=1)
    print('vts done: %d rows; per_ds=%s; A-equiv max=%.3e' % (len(man), per_ds, a_dev.max()))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else os.path.join(SCRIPT_DIR, 'save_t47_levelshift'),
         smoke='--smoke' in sys.argv)
