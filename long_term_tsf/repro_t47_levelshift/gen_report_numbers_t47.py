"""EXP-047:报告数字自动生成(report_numbers_t47.json + report_tables_t47.md)。

一切写入报告的数字都出自本脚本(禁止手抄)。来源:
- table_D_cost_t47.csv / table_method_diff_t47.csv / table_D_common_m.csv(问题①②与"是否比对照更强")
- table_s6_t47.csv(问题③:稳定 vs 变化后持续;含 hist_std 三分位分层)
- collect_meta_t47.json(覆盖率/ok 数)、sfdx_meta_t47.json(ledger 事件)
- meta_vts_t47.json(A 组等变性事后核验)、smoke_report_t47.json(冒烟实测差值)
"""
import csv
import json
import os
import sys


def read_csv_dicts(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main(repo_root, save_dir):
    del repo_root
    num = {}

    # 问题①②:D / cost 总表(overall × method × m)
    num['D_cost_overall'] = read_csv_dicts(os.path.join(save_dir, 'table_D_cost_t47.csv'))
    num['D_common_m'] = read_csv_dicts(os.path.join(save_dir, 'table_D_common_m.csv'))
    num['method_diff'] = read_csv_dicts(os.path.join(save_dir, 'table_method_diff_t47.csv'))

    def pick(scope, key, m, mp, tab):
        for r in tab:
            if (r.get('scope') == scope and r.get('key') == key and r.get('method') == m
                    and float(r.get('m_over_P', 'nan')) == float(mp)):
                return r
        return None

    headline = {}
    for m in ('vts', 'moirai', 'autoets', 'autoarima'):
        headline[m] = {str(mp): pick('overall', 'all', m, mp, num['D_cost_overall'])
                       for mp in (1, 2, 4)}
    num['headline'] = headline

    dd_head = {}
    for pair in ('vts_vs_moirai', 'vts_vs_autoets', 'vts_vs_autoarima'):
        dd_head[pair] = {}
        for r in num['method_diff']:
            if r['scope'] == 'overall' and r['pair'] == pair:
                dd_head[pair][r['m_over_P']] = r
    num['method_diff_overall'] = dd_head

    # 问题③:§6 类别表(scope ∈ all / ds / 类别 / 类别×三分位;method 维)
    num['s6_table'] = read_csv_dicts(os.path.join(save_dir, 'table_s6_t47.csv'))

    # 覆盖率 / 事件
    with open(os.path.join(save_dir, 'collect_meta_t47.json')) as f:
        num['coverage'] = json.load(f)['per_method']
    sfdx_meta_p = os.path.join(save_dir, 'sfdx_meta_t47.json')
    if os.path.exists(sfdx_meta_p):
        with open(sfdx_meta_p) as f:
            sm = json.load(f)
        num['sfdx'] = {m: {k: v for k, v in d.items() if k != 'row_globals'}
                       for m, d in sm.get('per_method', {}).items()}
    vts_meta_p = os.path.join(save_dir, 'meta_vts_t47.json')
    if os.path.exists(vts_meta_p):
        with open(vts_meta_p) as f:
            num['vts_a_equivariance'] = json.load(f).get('a_equivariance')
    smoke_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smoke_report_t47.json')
    if os.path.exists(smoke_p):
        with open(smoke_p) as f:
            num['smoke'] = json.load(f)
    analysis_meta_p = os.path.join(save_dir, 'analysis_meta_t47.json')
    if os.path.exists(analysis_meta_p):
        with open(analysis_meta_p) as f:
            num['analysis_meta'] = json.load(f)

    with open(os.path.join(save_dir, 'report_numbers_t47.json'), 'w') as f:
        json.dump(num, f, indent=1)

    # ---- markdown 表(全部由上面数字渲染) ----
    L = []
    L.append('# EXP-047 report tables (auto-generated; do not edit by hand)\n')

    L.append('## D / cost overall (per method x m/P)\n')
    L.append('| method | m/P | n | D_mean | D_median | D>0 rate | cost_mean | cost_median |')
    L.append('|---|---|---|---|---|---|---|---|')
    for m in ('vts', 'moirai', 'autoets', 'autoarima'):
        for mp in (1, 2, 4):
            r = headline[m].get(str(mp))
            if r:
                L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
                    m, r['m_over_P'], r['n'], r['D_mean'], r['D_median'], r['D_pos_rate'],
                    r['cost_mean'], r['cost_median']))
    L.append('')

    L.append('## D on common-windows-across-m subset (question 2)\n')
    L.append('| method | m/P | n_common | D_mean | D_median | D>0 rate | cost_mean | cost_median |')
    L.append('|---|---|---|---|---|---|---|---|')
    for r in num['D_common_m']:
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['method'], r['m_over_P'], r['n_common_windows'], r['D_mean'], r['D_median'],
            r['D_pos_rate'], r['cost_mean'], r['cost_median']))
    L.append('')

    L.append('## Method difference on common valid windows (is VTS stronger?)\n')
    L.append('| pair | m/P | n | Ddiff_mean | Ddiff_median | costdiff_mean | costdiff_median |')
    L.append('|---|---|---|---|---|---|---|')
    for pair, d in dd_head.items():
        for mp in sorted(d, key=float):
            r = d[mp]
            L.append('| %s | %s | %s | %s | %s | %s | %s |' % (
                pair, r['m_over_P'], r['n'], r['Ddiff_mean'], r['Ddiff_median'],
                r['costdiff_mean'], r['costdiff_median']))
    L.append('')

    L.append('## Section-6 recheck: MSE by class x method (t46 predictions reused)\n')
    L.append('| method | scope | key | n_ok | mse_mean | mse_median |')
    L.append('|---|---|---|---|---|---|')
    for r in num['s6_table']:
        L.append('| %s | %s | %s | %s | %s | %s |' % (
            r['method'], r['scope'], ' '.join(str(r.get(k, '')) for k in ('key1', 'key2')),
            r['n_ok'], r['mse_mean'], r['mse_median']))
    L.append('')

    L.append('## Coverage\n')
    L.append('| method | n | ok |')
    L.append('|---|---|---|')
    for m, d in num['coverage'].items():
        L.append('| %s | %s | %s |' % (m, d['n'], d['ok']))
    L.append('')
    if 'vts_a_equivariance' in num:
        L.append('## VisionTS A-group equivariance (post-hoc, not a gate)\n')
        L.append('```json\n%s\n```\n' % json.dumps(num['vts_a_equivariance'], indent=1))
    if 'smoke' in num:
        L.append('## Smoke gates (measured diffs)\n')
        for part, doc in num['smoke'].get('parts', {}).items():
            if isinstance(doc, dict) and 'checks' in doc:
                for name, c in doc['checks'].items():
                    detail = {k: v for k, v in c.items() if k not in ('name', 'pass', 'rows')}
                    if 'rows' in c:
                        detail['n_rows'] = len(c['rows'])
                        detail['rows'] = 'see smoke_report_t47.json'
                    L.append('- %s / %s: pass=%s detail=%s' % (part, name, c['pass'], detail))
        L.append('')

    with open(os.path.join(save_dir, 'report_tables_t47.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('report numbers -> report_numbers_t47.json / report_tables_t47.md '
          '(s6 rows=%d)' % len(num['s6_table']))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_t47_levelshift'))
