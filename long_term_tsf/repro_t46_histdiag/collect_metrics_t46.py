"""EXP-046:逐样本 f64 MSE/MAE 汇总(真值 = csv 标准化 f64 重算)。

- 全部 1792 样本核对:预测起点/变量/历史长度/item_id↔start_row↔arr_index;
- 各方法数组内 true 与 f64 重算 maxdiff > TRUTH_TOL → 管线级错误,非零退出;
- ok 判定:vts=infer_vts 全量核验通过即 ok;moirai=批量断言通过即 ok;
  autoets/autoarima=数组 status=='ok';失败样本指标记 NaN,绝不静默删除;
- --smoke:moirai/sfdx 读 *_smoke 数组(ETTh1/ETTm1 各前 8 样本),vts 列取自全量数组。
"""
import csv
import json
import os
import sys

import numpy as np

from t46_common import (DS_CFG, DS_ORDER, H, TRUTH_TOL, load_csv,
                        sample_list_path, std_all, train_scaler)

METHODS = ['vts', 'moirai', 'autoets', 'autoarima']
SMOKE_DS = ('ETTh1', 'ETTm1')
FIELDS = ['sample_id', 'ds', 'seg', 'channel', 'var', 'start_row', 'arr_index']
for m in METHODS:
    FIELDS += [f'ok_{m}', f'mse_{m}', f'mae_{m}']
FIELDS += ['status_autoets', 'status_autoarima']


def per_ds_rows(repo_root):
    """{ds: [sample dict 行序]}(全量 sample list 内按 ds 切出)。"""
    rows = {}
    for s in csv.DictReader(open(sample_list_path())):
        rows.setdefault(s['ds'], []).append(s)
    return rows


def main(repo_root, save_dir, smoke=False):
    with open(os.path.join(save_dir, 'verify_vts_t46.json')) as f:
        assert json.load(f)['all_ok'] is True, 'infer_vts verification failed; refusing to collect'

    suffix = '_smoke' if smoke else ''
    all_rows = per_ds_rows(repo_root)
    ds_list = [ds for ds in DS_ORDER if ds in all_rows]
    cache = {}
    out_rows = []
    meta = dict(smoke=smoke, truth_tol=TRUTH_TOL, per_ds={})

    for ds in ds_list:
        cfg = DS_CFG[ds]
        if ds not in cache:
            data = load_csv(os.path.join(repo_root, cfg['csv']))
            assert data.shape[0] == cfg['csv_rows'], (ds, data.shape)
            mean, scale = train_scaler(data, cfg['K'])
            cache[ds] = std_all(data, mean, scale)
        std = cache[ds]

        full = all_rows[ds]
        pos = {int(s['sample_id']): j for j, s in enumerate(full)}
        ss = full[:8] if (smoke and ds in SMOKE_DS) else full
        idx = np.array([pos[int(s['sample_id'])] for s in ss], dtype=np.int64)
        n = len(ss)

        true64 = np.stack([std[int(s['start_row']):int(s['start_row']) + H, int(s['channel'])]
                           for s in ss])  # (n,96) f64
        for s in ss:
            start, L = int(s['start_row']), cfg['L']
            assert 12 * cfg['K'] <= start <= 16 * cfg['K'] - H, s
            assert start - L >= 0 and int(s['arr_index']) == start - 12 * cfg['K'], s
            assert int(s['L']) == L and int(s['P']) == cfg['P'], s

        suffix_m = '_smoke' if smoke else ''
        zv = np.load(os.path.join(save_dir, 'arr_vts_t46.npz'))
        zm = np.load(os.path.join(save_dir, f'arr_moirai{suffix_m}_{ds}.npz'))
        assert sorted(zm.files) == ['item_ids', 'pred', 'start_rows', 'true'], zm.files
        assert list(zm['item_ids']) == [int(s['sample_id']) for s in full], (ds, 'moirai item_ids')
        assert list(zm['start_rows']) == [int(s['start_row']) for s in full], (ds, 'moirai start_rows')
        sfdx_arr = {}
        for name in ('AutoETS', 'AutoARIMA'):
            zs = np.load(os.path.join(save_dir, f'{name}{suffix_m}_{ds}.npz'))
            assert sorted(zs.files) == ['item_ids', 'pred', 'status', 'true'], zs.files
            assert list(zs['item_ids']) == [int(s['sample_id']) for s in full], (ds, name, 'item_ids')
            sfdx_arr[name.lower()] = zs

        arrs = {'vts': (zv[f'{ds}_pred'], zv[f'{ds}_true'], None),
                'moirai': (zm['pred'], zm['true'], None),
                'autoets': (sfdx_arr['autoets']['pred'], sfdx_arr['autoets']['true'],
                            sfdx_arr['autoets']['status']),
                'autoarima': (sfdx_arr['autoarima']['pred'], sfdx_arr['autoarima']['true'],
                              sfdx_arr['autoarima']['status'])}

        per_method = {}
        results = {}  # m -> (ok(bool n), mse(f64 n), mae, status list)
        for m in METHODS:
            pred_all, true_all, status = arrs[m]
            pred = pred_all[idx].astype(np.float64)
            tstored = true_all[idx].astype(np.float64)
            assert pred.shape == true64.shape == (n, H), (m, ds, pred.shape)
            diff = float(np.abs(tstored - true64).max())
            assert diff <= TRUTH_TOL, (m, ds, diff)
            if m in ('vts', 'moirai'):
                ok = np.ones(n, bool)
                status_col = ['ok'] * n
            else:
                ok = np.asarray(status) == 'ok'
                status_col = [str(x) for x in status]
            if ok.any():
                assert np.isfinite(pred[ok]).all(), (m, ds, 'nonfinite in ok preds')
            mse = np.where(ok, ((pred - true64) ** 2).mean(1), np.nan)
            mae = np.where(ok, np.abs(pred - true64).mean(1), np.nan)
            results[m] = (ok, mse, mae, status_col)
            per_method[m] = dict(n=n, n_ok=int(ok.sum()), truth_max_diff=diff,
                                 mse_ok_mean=float(np.nanmean(mse)) if ok.any() else None,
                                 mae_ok_mean=float(np.nanmean(mae)) if ok.any() else None)
        meta['per_ds'][ds] = dict(n=n, methods=per_method)

        for j, s in enumerate(ss):
            row = [s['sample_id'], ds, s['seg'], s['channel'], s['var'],
                   s['start_row'], s['arr_index']]
            for m in METHODS:
                ok, mse, mae, _ = results[m]
                row += [int(ok[j]), '%.12g' % mse[j], '%.12g' % mae[j]]
            row += [results['autoets'][3][j], results['autoarima'][3][j]]
            out_rows.append(row)
        print(ds, {m: per_method[m]['n_ok'] for m in METHODS},
              'n=%d' % n)

    path = os.path.join(save_dir, f'per_sample_metrics{suffix}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(out_rows)
    with open(os.path.join(save_dir, f'collect_meta{suffix}.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('metrics rows=%d -> %s' % (len(out_rows), path))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'save_t46_histdiag',
         smoke='--smoke' in sys.argv)
