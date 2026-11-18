"""EXP-047:逐条件 f64 MSE/MAE 汇总(真值 y' = baseline y + δ,f64 重算)。

- 主对照 vts/moirai/autoets 全 7168 行;辅助臂 autoarima 448 行(ETTh1/ETTh2 rank≤1);
- npz true_p(f32)vs f64 重算 maxdiff > 1e-5 → 管线级错误,非零退出;
- ok 判定:vts/moirai=推理脚本全量断言通过即 ok;autoets/autoarima=status=='ok';
  失败样本指标记 NaN,绝不静默删除;
- 长表 per_sample_metrics_t47.csv:每行每方法 ok/mse/mae(f64,%.12g)。
"""
import csv
import json
import os
import sys

import numpy as np

from t47_common import (H, METHODS, TRUTH_TOL, cond_params, manifest_path)

FIELDS = ['row_global', 'sid_t46', 'ds', 'var', 'channel', 'cond_id', 'group',
          'delta_idx', 'm_idx', 'delta_mult', 'delta', 'm_steps', 'm_over_P']
for m in METHODS:
    FIELDS += [f'ok_{m}', f'mse_{m}', f'mae_{m}']
FIELDS += ['status_autoets', 'status_autoarima']


def main(repo_root, save_dir):
    del repo_root  # y' 全部由 baseline npz 重算,不需要 csv
    with open(manifest_path()) as f:
        man = list(csv.DictReader(f))
    assert len(man) == 7168

    arrs = {}
    for m in METHODS:
        z = np.load(os.path.join(save_dir, 'pred_%s_t47.npz' % m))
        for k in ('pred', 'true_p', 'item_ids', 'cond_ids', 'row_globals'):
            assert k in z.files, (m, k)
        arrs[m] = z
        assert np.array_equal(z['cond_ids'],
                              np.array([int(r['cond_id']) for r in man if
                                        (m != 'autoarima' or (r['ds'] in ('ETTh1', 'ETTh2')
                                                              and int(r['rank']) <= 1))],
                                       dtype=np.int64)), m
    assert all(np.array_equal(arrs['vts']['row_globals'], arrs[m]['row_globals'])
               for m in ('moirai', 'autoets'))

    rg_arr = arrs['vts']['row_globals']
    assert np.array_equal(rg_arr, np.array([int(r['row_global']) for r in man], dtype=np.int64))
    pos = {int(rg): j for j, rg in enumerate(rg_arr)}  # row_global -> vts/moirai/autoets 行号
    pos_a = {int(rg): j for j, rg in enumerate(arrs['autoarima']['row_globals'])}

    base_cache = {}
    out_rows = []
    meta = dict(truth_tol=TRUTH_TOL, per_method={})
    true_diff_max = 0.0
    counts = {m: dict(n=0, ok=0) for m in METHODS}

    for r in man:
        rg = int(r['row_global'])
        ds, sid = r['ds'], int(r['sid_t46'])
        if ds not in base_cache:
            base_cache[ds] = np.load(os.path.join(save_dir, 'baseline_%s.npz' % ds))
        bz = base_cache[ds]
        sid_row = {int(s): i for i, s in enumerate(bz['sid_t46'])}
        delta = float(r['delta'])
        yp64 = bz['y'][sid_row[sid]] + delta                      # (96,) f64
        assert np.isfinite(yp64).all()
        g, di, mi, dmult, mop = cond_params(int(r['cond_id']))
        assert g == r['group'] and di == int(r['delta_idx'])
        assert (mi is None if r['m_idx'] == '' else mi == int(r['m_idx']))
        assert r['delta_mult'] == '' or abs(float(r['delta_mult']) - dmult) < 1e-12

        row = [rg, sid, ds, r['var'], r['channel'], int(r['cond_id']), g, di,
               '' if mi is None else mi, '%.12g' % dmult, '%.12g' % delta,
               int(r['m_steps']), r['m_over_P']]
        for m in METHODS:
            if m == 'autoarima' and rg not in pos_a:
                row += ['', '', '']
                continue
            z = arrs[m]
            j = pos[rg] if m != 'autoarima' else pos_a[rg]
            tdiff = float(np.abs(z['true_p'][j].astype(np.float64) - yp64).max())
            true_diff_max = max(true_diff_max, tdiff)
            if tdiff > TRUTH_TOL:
                raise SystemExit('truth mismatch: %s rg=%d maxdiff=%.3e' % (m, rg, tdiff))
            pred = z['pred'][j].astype(np.float64)
            if m in ('vts', 'moirai'):
                ok = True
            else:
                ok = str(z['status'][j]) == 'ok'
                assert ok or not np.isfinite(pred).any(), (m, rg)
            if ok:
                assert np.isfinite(pred).all(), (m, rg)
                mse = float(((pred - yp64) ** 2).mean())
                mae = float(np.abs(pred - yp64).mean())
                counts[m]['ok'] += 1
            else:
                mse = mae = float('nan')
            counts[m]['n'] += 1
            row += [int(ok), '%.12g' % mse, '%.12g' % mae]
        row += [str(arrs['autoets']['status'][pos[rg]])]
        row += [str(arrs['autoarima']['status'][pos_a[rg]]) if rg in pos_a else '']
        out_rows.append(row)

    path = os.path.join(save_dir, 'per_sample_metrics_t47.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(out_rows)
    meta['per_method'] = counts
    meta['true_diff_max'] = true_diff_max
    with open(os.path.join(save_dir, 'collect_meta_t47.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('metrics rows=%d -> %s; true_diff_max=%.2e; ok=%s'
          % (len(out_rows), path, true_diff_max, {m: counts[m]['ok'] for m in METHODS}))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_t47_levelshift'))
