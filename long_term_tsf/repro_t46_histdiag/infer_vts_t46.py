"""EXP-046:VisionTS 臂 = 切片复用 EXP-043 val 臂 arr_val.npz(零推理)。

对全部 1792 样本核验:arr_index = start−12K、变量 c、真值 = csv 标准化 f64 重算
(容差 1e-5,f32 链路参考 maxdiff ~2e-7)。任何对齐/真值错 → 非零退出。
"""
import csv
import json
import os
import sys

import numpy as np

from t46_common import (DS_CFG, DS_ORDER, H, TRUTH_TOL as TOL, load_csv,
                        sample_list_path, std_all, train_scaler)


def main(repo_root, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    samples = list(csv.DictReader(open(sample_list_path())))
    report = {}
    out = {}
    for ds in DS_ORDER:
        cfg = DS_CFG[ds]
        ss = [s for s in samples if s['ds'] == ds]
        assert len(ss) == 448
        z = np.load(os.path.join(repo_root, 'save_t43_period_val_val', cfg['group'], 'arr_val.npz'))
        assert sorted(z.files) == ['pred', 'true'], z.files
        pred_all, true_all = z['pred'], z['true']
        assert pred_all.dtype == np.float32 and true_all.dtype == np.float32
        n_expect = 4 * cfg['K'] - H + 1
        assert pred_all.shape == true_all.shape == (n_expect, H, 7), pred_all.shape

        data = load_csv(os.path.join(repo_root, cfg['csv']))
        mean, scale = train_scaler(data, cfg['K'])
        std = std_all(data, mean, scale)

        pred = np.empty((448, H), np.float32)
        true = np.empty((448, H), np.float32)
        max_diff = 0.0
        for j, s in enumerate(ss):
            i, ch = int(s['arr_index']), int(s['channel'])
            assert int(s['start_row']) == 12 * cfg['K'] + i
            pred[j] = pred_all[i, :, ch]
            true[j] = true_all[i, :, ch]
            ref = std[int(s['start_row']):int(s['start_row']) + H, ch]
            max_diff = max(max_diff, float(np.abs(true[j].astype(np.float64) - ref).max()))
        assert np.isfinite(pred).all() and np.isfinite(true).all()
        status = 'PASS' if max_diff <= TOL else 'FAIL'
        report[ds] = dict(n=448, group=cfg['group'], truth_max_diff=max_diff,
                          tol=TOL, status=status)
        out[ds] = (pred, true)
        print(ds, report[ds])

    arrays = {}
    for ds in DS_ORDER:
        arrays[f'{ds}_pred'] = out[ds][0]
        arrays[f'{ds}_true'] = out[ds][1]
    np.savez_compressed(os.path.join(save_dir, 'arr_vts_t46.npz'), **arrays)
    with open(os.path.join(save_dir, 'verify_vts_t46.json'), 'w') as f:
        json.dump(dict(all_ok=all(r['status'] == 'PASS' for r in report.values()),
                       n_total=1792, per_ds=report), f, indent=1)
    if not all(r['status'] == 'PASS' for r in report.values()):
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'save_t46_histdiag')
