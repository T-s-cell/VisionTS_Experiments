"""EXP-047h aggregate:主表/附表/方法表(全部基于逐配对差值,禁止两组中位数之差)。

用法:$PY aggregate_t47h.py <OUT输出目录>
输入:pairs_horizon_t47h.csv / method_diff_horizon_t47h.csv(compute 产物)。
- table_main_t47h.csv       主表:(pair, ds, m/P, band) → D_diff/gap_A/gap_B/cost_diff/
                            cost_level_diff/cost_centered_diff 的 mean/median/pos_rate
                            + n_base/n_pairs/n_cand/coverage;
- table_var_t47h.csv        附表1:同口径加 ds×var(仍按 ds 分列,不跨数据集合并);
- table_direction_t47h.csv  附表2:同口径加 δ 方向(down/up,按 ds 分列);
- table_amplitude_t47h.csv  附表3:同口径加 |δ|/s 幅度(0.5/1.0,按 ds 分列);
- table_method_t47h.csv     方法表:(method, ds, m/P, band) → D/cost 统计 + mse_A/B、
                            level_B/centered_B 均值(问题③背景)。
正值约定:D_diff>0 = VTS 牵引更强;cost_diff>0 = 代价更大;gap_B>0 = 误差更大。
full 行单列汇报,不作第五时段解释。
"""
import csv
import json
import os
import sys

import numpy as np

from t47h_common import DIFF_H, PAIRS_H, read_csv_dicts

QUANT_DIFF = ['D_diff', 'gap_A', 'gap_B', 'cost_diff', 'cost_level_diff', 'cost_centered_diff']
QUANT_METHOD = ['D', 'cost', 'mse_A', 'mse_B', 'level_B', 'centered_B']


def g12(x):
    return '%.12g' % x if x == x else 'nan'


def stats(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return None
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                pos_rate=float((v > 0).mean()))


def render(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def stat_cells(d, q, cols3=True):
    s = stats(d[q])
    if s is None:
        return ['nan', 'nan', 'nan'] if cols3 else ['nan']
    return [g12(s['mean']), g12(s['median']), '%.4f' % s['pos_rate']] if cols3 else [g12(s['mean'])]


def diff_table(rows, keyfn, keycols, fname, out_dir):
    """配对差聚合表:keycols 维分组 + m/P + band;统计 6 个差值量。"""
    acc, cand, sids = {}, {}, {}
    for r in rows:
        k = keyfn(r)
        cand[k] = cand.get(k, 0) + 1
        sids.setdefault(k, set())
        if r['_ok']:
            sids[k].add(r['sid'])
            acc.setdefault(k, {q: [] for q in QUANT_DIFF})
            for q in QUANT_DIFF:
                if r[q] == r[q]:
                    acc[k][q].append(r[q])
    header = (list(keycols) + ['m_over_P', 'band']
              + [q + s for q in QUANT_DIFF for s in ('_mean', '_median', '_pos_rate')]
              + ['n_base', 'n_pairs', 'n_cand', 'coverage'])
    out = []
    for k in sorted(acc, key=lambda t: tuple(str(x) for x in t)):
        d = acc[k]
        n_pairs = len(d['D_diff'])
        line = list(k)
        for q in QUANT_DIFF:
            line += stat_cells(d, q)
        line += [len(sids[k]), n_pairs, cand[k], '%.4f' % (n_pairs / cand[k])]
        out.append(line)
    render(os.path.join(out_dir, fname), header, out)
    print('[aggregate] %s: %d rows' % (fname, len(out)))


def main(out_dir):
    rows = []
    for r in read_csv_dicts(os.path.join(out_dir, DIFF_H)):
        rec = dict(pair=r['pair'], ds=r['ds'], var=r['var'],
                   direction='down' if float(r['delta_mult']) < 0 else 'up',
                   amplitude='%.1f' % abs(float(r['delta_mult'])),
                   m_over_P=int(float(r['m_over_P'])), band=r['band'], _ok=r['ok'] == '1',
                   sid=int(r['sid_t46']))
        for q in QUANT_DIFF:
            rec[q] = float(r[q])
        rows.append(rec)

    diff_table(rows, lambda r: (r['pair'], r['ds'], r['m_over_P'], r['band']),
               ['pair', 'ds'], 'table_main_t47h.csv', out_dir)
    diff_table(rows, lambda r: (r['pair'], r['ds'], r['var'], r['m_over_P'], r['band']),
               ['pair', 'ds', 'var'], 'table_var_t47h.csv', out_dir)
    diff_table(rows, lambda r: (r['pair'], r['ds'], r['direction'], r['m_over_P'], r['band']),
               ['pair', 'ds', 'direction'], 'table_direction_t47h.csv', out_dir)
    diff_table(rows, lambda r: (r['pair'], r['ds'], r['amplitude'], r['m_over_P'], r['band']),
               ['pair', 'ds', 'amplitude'], 'table_amplitude_t47h.csv', out_dir)

    # ---- 方法表(逐配对量,口径同冻结 table_D_cost_t47.csv)----
    acc, cand, sids = {}, {}, {}
    for r in read_csv_dicts(os.path.join(out_dir, PAIRS_H)):
        ok = r['ok_A'] == '1' and r['ok_B'] == '1'
        k = (r['method'], r['ds'], int(float(r['m_over_P'])), r['band'])
        cand[k] = cand.get(k, 0) + 1
        sids.setdefault(k, set())
        if ok:
            sids[k].add(int(r['sid_t46']))
            acc.setdefault(k, {q: [] for q in QUANT_METHOD})
            for q in QUANT_METHOD:
                v = float(r[q])
                if v == v:
                    acc[k][q].append(v)
    header = (['method', 'ds', 'm_over_P', 'band']
              + [q + s for q in ('D', 'cost') for s in ('_mean', '_median', '_pos_rate')]
              + ['mseA_mean', 'mseB_mean', 'levelB_mean', 'centeredB_mean']
              + ['n_base', 'n_pairs', 'n_cand', 'coverage'])
    out = []
    for k in sorted(acc, key=lambda t: tuple(str(x) for x in t)):
        d = acc[k]
        n_pairs = len(d['D'])
        line = list(k)
        for q in ('D', 'cost'):
            line += stat_cells(d, q)
        for q in ('mse_A', 'mse_B', 'level_B', 'centered_B'):
            line += stat_cells(d, q, cols3=False)
        line += [len(sids[k]), n_pairs, cand[k], '%.4f' % (n_pairs / cand[k])]
        out.append(line)
    render(os.path.join(out_dir, 'table_method_t47h.csv'), header, out)
    print('[aggregate] table_method_t47h.csv: %d rows' % len(out))

    with open(os.path.join(out_dir, 'aggregate_meta_t47h.json'), 'w') as f:
        json.dump({'n_diff_rows': len(rows),
                   'tables': ['table_main_t47h.csv', 'table_var_t47h.csv',
                              'table_direction_t47h.csv', 'table_amplitude_t47h.csv',
                              'table_method_t47h.csv']}, f, indent=1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
