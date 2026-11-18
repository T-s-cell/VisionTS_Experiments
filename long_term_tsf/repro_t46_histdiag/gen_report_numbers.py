"""EXP-046:报告数字自动生成(唯一数字来源,report_t46.md 引用其输出)。

读 save_t46_histdiag/ 各表 → report_numbers_t46.json(报告引用) +
report_tables_t46.md(markdown 片段,报告逐字引用)。禁止手抄数字。
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

DS_ORDER = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
CHANNELS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
METHODS = ['vts', 'moirai', 'autoets', 'autoarima']
PAIRS = [('vts', 'moirai'), ('vts', 'autoets'), ('vts', 'autoarima')]
FEATURES = ['period_corr', 'recent_old', 'trend', 'vol', 'hist_std']


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    return None if x == '' else float(x)


def fmt(x, nd=4):
    return '' if x is None else ('%.*f' % (nd, x))


def main(save_dir):
    out = {}
    tmd = []

    # ---- overall ----
    ov = rows_of(os.path.join(save_dir, 'overall_t46.csv'))
    ov_idx = {(r['ds'], r['method']): r for r in ov}
    out['overall'] = [dict(ds=r['ds'], method=r['method'], coverage=r['coverage'],
                           mse_own_ok=fnum(r['mse_own_ok']), mse_common4=fnum(r['mse_common4']),
                           n_common4=int(r['n_common4'])) for r in ov]
    lines = ['| ds | ' + ' | '.join(m + ' (own-ok / common4) ' for m in METHODS) + ' |',
             '|---|' + '---|' * len(METHODS)]
    for ds in DS_ORDER + ['ALL']:
        cells = []
        for m in METHODS:
            r = ov_idx[(ds, m)]
            cells.append('%s / %s' % (fmt(fnum(r['mse_own_ok'])), fmt(fnum(r['mse_common4']))))
        lines.append('| %s | %s |' % (ds, ' | '.join(cells)))
    tmd += ['## overall(MSE:own-ok / common4)'] + lines + ['']

    # ---- direction_summary + by_var ----
    dsum = rows_of(os.path.join(save_dir, 'direction_summary_t46.csv'))
    out['direction_summary'] = [dict(feature=r['feature'], pair=r['pair'],
                                     n_cells=int(r['n_cells_valid']),
                                     n_high_gt_low=int(r['n_high_gt_low']),
                                     n_low_gt_high=int(r['n_low_gt_high']),
                                     n_monotone=int(r['n_monotone_nondecreasing']),
                                     median_delta=fnum(r['median_rel_gap_high_minus_low']))
                                for r in dsum]
    dvar = rows_of(os.path.join(save_dir, 'direction_by_var_t46.csv'))
    out['direction_by_var'] = [dict(feature=r['feature'], pair=r['pair'], var=r['var'],
                                    n_ds=int(r['n_ds_valid']),
                                    n_ds_pos=int(r['n_ds_high_gt_low']),
                                    n_ds_neg=int(r['n_ds_low_gt_high']),
                                    median_delta=fnum(r['median_delta_across_ds']))
                               for r in dvar]
    dvar_idx = {(r['feature'], r['pair'], r['var']): r for r in dvar}
    per_var_stats = {}
    for feat in FEATURES:
        per_var_stats[feat] = {}
        for mv, mb in PAIRS:
            pair = f'{mv}_vs_{mb}'
            meds = [fnum(dvar_idx[(feat, pair, v)]['median_delta_across_ds']) for v in CHANNELS
                    if (feat, pair, v) in dvar_idx]
            pos = sum(1 for x in meds if x is not None and x > 0)
            neg = sum(1 for x in meds if x is not None and x < 0)
            per_var_stats[feat][pair] = dict(
                n_var=len(meds), n_var_median_pos=pos, n_var_median_neg=neg,
                medians={v: fnum(dvar_idx[(feat, pair, v)]['median_delta_across_ds'])
                         for v in CHANNELS if (feat, pair, v) in dvar_idx})
    out['per_var_stats'] = per_var_stats
    lines = ['| feature | pair | ' + ' | '.join(CHANNELS) + ' | 中位为正变量数 |',
             '|---|---|' + '---|' * (len(CHANNELS) + 1)]
    for feat in FEATURES:
        for mv, mb in PAIRS:
            pair = f'{mv}_vs_{mb}'
            st = per_var_stats[feat][pair]
            cells = [fmt(st['medians'].get(v), 3) if st['medians'].get(v) is not None else ''
                     for v in CHANNELS]
            lines.append('| %s | %s | %s | %d/%d |' % (feat, pair, ' | '.join(cells),
                                                       st['n_var_median_pos'], st['n_var']))
    tmd += ['## direction by var(口径:每 ds 取 high−low,再跨 ds 取中位;正 = VTS 相对弱势线索)'] + lines + ['']

    # ---- amplitude ----
    asum = rows_of(os.path.join(save_dir, 'amplitude_strat_summary_t46.csv'))
    out['amplitude_summary'] = [dict(feature=r['feature'], pair=r['pair'],
                                     hist_tertile=r['hist_tertile'], n_cells=int(r['n_cells']),
                                     n_pos=int(r['n_delta_pos']), n_neg=int(r['n_delta_neg']),
                                     median_delta=fnum(r['median_delta'])) for r in asum]
    lines = ['| feature | pair | hist_tertile | cells | delta>0 | delta<0 | median delta |',
             '|---|---|---|---|---|---|---|']
    for r in asum:
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (
            r['feature'], r['pair'], r['hist_tertile'], r['n_cells'], r['n_delta_pos'],
            r['n_delta_neg'], fmt(fnum(r['median_delta']), 3)))
    tmd += ['## amplitude-stratified(hist_std 三分位内,特征中位切分)'] + lines + ['']

    # ---- coverage ----
    cov = {}
    for r in ov:
        cov[(r['ds'], r['method'])] = dict(coverage=r['coverage'], n_ok=int(r['n_ok']), n=int(r['n']))
    out['coverage'] = [dict(ds=r['ds'], method=r['method'], **cov[(r['ds'], r['method'])]) for r in ov]

    # ---- joint_hard ----
    jh = rows_of(os.path.join(save_dir, 'joint_hard_t46.csv'))
    jhs = rows_of(os.path.join(save_dir, 'joint_hard_summary_t46.csv'))
    out['joint_hard'] = dict(n=len(jh), by_ds=dict(Counter(r['ds'] for r in jh)),
                             by_var=dict(Counter(r['var'] for r in jh)),
                             summary=[dict(group=r['group'], pair=r['pair'], n=int(r['n']),
                                           rel_gap=fnum(r['rel_gap'])) for r in jhs])

    # ---- cases ----
    cases = rows_of(os.path.join(save_dir, 'cases_t46.csv'))
    ameta = json.load(open(os.path.join(save_dir, 'analyze_meta_t46.json')))
    out['cases'] = dict(n=len(cases),
                        worse_candidates=ameta['case_candidates'],
                        rows=[dict(case=r['case'], ds=r['ds'], var=r['var'],
                                   sample_id=int(r['sample_id']), margin=fnum(r['margin']),
                                   mse_vts=fnum(r['mse_vts'])) for r in cases])
    lines = ['| case | ds | var | sid | margin | mse_vts |', '|---|---|---|---|---|---|']
    for r in cases:
        lines.append('| %s | %s | %s | %s | %s | %s |' % (
            r['case'], r['ds'], r['var'], r['sample_id'], r['margin'], fmt(fnum(r['mse_vts']))))
    tmd += ['## cases(规则:mse4[0]=VTS;每类每 ds margin 前 2)'] + lines + ['']

    with open(os.path.join(save_dir, 'report_numbers_t46.json'), 'w') as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(save_dir, 'report_tables_t46.md'), 'w') as f:
        f.write('\n'.join(tmd) + '\n')
    print('report numbers -> report_numbers_t46.json / report_tables_t46.md')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'save_t46_histdiag')
