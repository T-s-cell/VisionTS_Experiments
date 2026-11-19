"""EXP-047h 报告数字自动生成(report_numbers_t47h.json + report_t47h.md 一页结论)。

一切写入报告的数字都出自本脚本(禁止手抄)。来源:五张汇总表 + 逐配对 CSV
(pairs_horizon/method_diff_horizon)。五问数字全部程序计算;结论措辞遵守协议纪律
(A 组只称参照条件;centered_error 不解释为旧形状依赖;四种解读区分;不归因 MAE)。

用法:$PY gen_report_numbers_t47h.py <OUT输出目录>
"""
import csv
import json
import os
import sys

from t47h_common import PAIRS_H, read_csv_dicts

BANDS5 = ['b1', 'b2', 'b3', 'b4', 'full']
DSS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
MPS = ['1', '2', '4']


def main(out_dir):
    num = {}
    t_main = read_csv_dicts(os.path.join(out_dir, 'table_main_t47h.csv'))
    t_dir = read_csv_dicts(os.path.join(out_dir, 'table_direction_t47h.csv'))
    t_amp = read_csv_dicts(os.path.join(out_dir, 'table_amplitude_t47h.csv'))
    t_var = read_csv_dicts(os.path.join(out_dir, 'table_var_t47h.csv'))
    t_meth = read_csv_dicts(os.path.join(out_dir, 'table_method_t47h.csv'))
    meta = json.load(open(os.path.join(out_dir, 'compute_meta_t47h.json')))
    ver = json.load(open(os.path.join(out_dir, 'verify_t47h.json')))
    num['meta'] = {'n_pairs_rows': meta['n_pairs_rows'], 'n_diff_rows': meta['n_diff_rows'],
                   'max_true_p_diff': meta['max_true_p_diff'],
                   'ok_pairs_per_method': meta['n_pairs_per_method'],
                   'verify_all_pass': ver['all_pass']}
    tm = {}
    for r in t_main:
        tm[(r['pair'], r['ds'], r['m_over_P'], r['band'])] = r

    def cell(pair, ds, m, band, col):
        return float(tm[(pair, ds, m, band)][col])

    # ---- 背景表:逐方法 D/cost 按 band(vts/moirai,主比较)----
    meth = {}
    for r in t_meth:
        if r['method'] in ('vts', 'moirai', 'autoets', 'autoarima'):
            meth[(r['method'], r['ds'], r['m_over_P'], r['band'])] = r
    num['method_D_by_band'] = {
        '%s_%s_m%s' % (m, ds, mp): {b: {'D_median': float(meth[(m, ds, mp, b)]['D_median']),
                                       'cost_median': float(meth[(m, ds, mp, b)]['cost_median'])}
                                    for b in BANDS5}
        for m in ('vts', 'moirai') for ds in DSS for mp in MPS}

    # ---- Q1:早期 b1 vs full 结论不同(主比较 vts_vs_moirai)----
    q1 = []
    for ds in DSS:
        for mp in MPS:
            for q, pr in (('D_diff_median', 'D_diff_pos_rate'), ('cost_diff_median', 'cost_diff_pos_rate')):
                b1v, fv = cell('vts_vs_moirai', ds, mp, 'b1', q), cell('vts_vs_moirai', ds, mp, 'full', q)
                b1p, fp = cell('vts_vs_moirai', ds, mp, 'b1', pr), cell('vts_vs_moirai', ds, mp, 'full', pr)
                flip = (b1v > 0) != (fv > 0)
                if flip or abs(b1v) > 2 * max(abs(fv), 1e-6):
                    q1.append({'ds': ds, 'm': mp, 'q': q, 'b1_median': b1v, 'full_median': fv,
                               'b1_pos_rate': b1p, 'full_pos_rate': fp, 'sign_flip': flip})
    num['q1_early_vs_full'] = q1

    # ---- Q2:牵引更强与代价更大是否同时(逐格符号组合,b1–b4+full,60 格)----
    combo = {'both_pos': 0, 'pull_only': 0, 'cost_only': 0, 'both_neg': 0}
    cost_only_cells, both_pos_keys = [], []
    for ds in DSS:
        for mp in MPS:
            for b in BANDS5:
                dd = cell('vts_vs_moirai', ds, mp, b, 'D_diff_median')
                cd = cell('vts_vs_moirai', ds, mp, b, 'cost_diff_median')
                k = ('both_pos' if dd > 0 and cd > 0 else
                     'pull_only' if dd > 0 else 'cost_only' if cd > 0 else 'both_neg')
                combo[k] += 1
                if k == 'cost_only':
                    cost_only_cells.append('%s m=%s %s (Dd=%.3f cd=%.3f)' % (ds, mp, b, dd, cd))
                if k == 'both_pos':
                    both_pos_keys.append('%s|%s|%s' % (ds, mp, b))
    num['q2_sign_combo'] = {'counts': combo, 'cost_only_cells': cost_only_cells,
                            'note': 'both_pos=牵引更强且代价更大同时;cost_only=牵引不更强'
                                    '但代价更大(不能解释为更依赖旧水平);按格点中位数判定,'
                                    '属组统计,另见 q2_joint_pos_rate 的逐配对联合比例'}

    # ---- Q3:代价增量主要来自水平项还是去均值后误差 ----
    # 分解对象 = 方法间代价差 cost_diff;逐配对恒等式 cost_diff = cost_level_diff +
    # cost_centered_diff ⇒ 均值严格可加、中位数不可加,故分解示例一律用同配对集均值。
    n_tot = n_lvl_med = n_lvl_mean = 0
    for ds in DSS:
        for mp in MPS:
            for b in BANDS5:
                n_tot += 1
                if abs(cell('vts_vs_moirai', ds, mp, b, 'cost_level_diff_median')) >= \
                        abs(cell('vts_vs_moirai', ds, mp, b, 'cost_centered_diff_median')):
                    n_lvl_med += 1
                if abs(cell('vts_vs_moirai', ds, mp, b, 'cost_level_diff_mean')) >= \
                        abs(cell('vts_vs_moirai', ds, mp, b, 'cost_centered_diff_mean')):
                    n_lvl_mean += 1
    mean_ex = {q: cell('vts_vs_moirai', 'ETTh2', '1', 'full', q + '_mean')
               for q in ('cost_diff', 'cost_level_diff', 'cost_centered_diff')}
    num['q3_level_vs_centered'] = {
        'n_cells': n_tot, 'level_dominant_cells_median': n_lvl_med,
        'level_dominant_cells_mean': n_lvl_mean, 'mean_example_ETTh2_m1_full': mean_ex,
        'note': '分解对象=方法间代价差;逐配对恒等式 ⇒ 均值可加、中位数不可加;'
                'centered_error 不能解释为旧形状依赖'}

    # ---- Q4:代价更大的配对子集中,VTS 最终误差是否仍低于 Moirai(逐配对条件统计)----
    # 同一遍循环兼算 Q2 的逐配对联合比例(D_diff>0 且 cost_diff>0 的窗口占比)。
    dh = read_csv_dicts(os.path.join(out_dir, 'method_diff_horizon_t47h.csv'))
    q4 = {}
    joint = {}
    for r in dh:
        if r['pair'] != 'vts_vs_moirai' or r['ok'] != '1':
            continue
        dd, cd, gb = float(r['D_diff']), float(r['cost_diff']), float(r['gap_B'])
        if dd != dd or cd != cd or gb != gb:
            continue
        key = (r['ds'], r['m_over_P'], r['band'])
        d = q4.setdefault(key, {'n_cost_pos': 0, 'n_cost_pos_gapB_neg': 0, 'gapB_med_in_costpos': []})
        j = joint.setdefault(key, [0, 0])
        j[1] += 1
        if dd > 0 and cd > 0:
            j[0] += 1
        if cd > 0:
            d['n_cost_pos'] += 1
            d['gapB_med_in_costpos'].append(gb)
            if gb < 0:
                d['n_cost_pos_gapB_neg'] += 1
    num['q2_joint_pos_rate'] = {'%s|%s|%s' % k: v[0] / v[1] for k, v in sorted(joint.items())}
    q4_out = {}
    for key, d in q4.items():
        n = d['n_cost_pos']
        vals = sorted(d['gapB_med_in_costpos'])
        med = (vals[(n - 1) // 2] + vals[n // 2]) / 2.0 if n else None
        q4_out['%s|%s|%s' % key] = {
            'n_pairs_costpos': n,
            'frac_gapB_neg': (d['n_cost_pos_gapB_neg'] / n) if n else None,
            'gapB_median_in_costpos': med}
    num['q4_conditional'] = q4_out
    num['q4_headline'] = {
        'ETTh2_m1_full': q4_out.get('ETTh2|1|full'), 'ETTm1_m1_full': q4_out.get('ETTm1|1|full'),
        'ETTh1_m1_b1': q4_out.get('ETTh1|1|b1'),
        'note': 'gap_B<0 = VTS 最终误差低于对照;cost_diff>0 子集 = 冲突代价增量更大的配对'}

    # ---- Q5:方向/幅度/变量一致性 + 反例 ----
    q5_dir = {}
    for r in t_dir:
        if r['pair'] != 'vts_vs_moirai' or r['band'] != 'b1' or r['m_over_P'] != '1':
            continue
        q5_dir['%s|%s' % (r['ds'], r['direction'])] = float(r['D_diff_median'])
    q5_amp = {}
    for r in t_amp:
        if r['pair'] != 'vts_vs_moirai' or r['band'] != 'b1' or r['m_over_P'] != '1':
            continue
        q5_amp['%s|%s' % (r['ds'], r['amplitude'])] = float(r['D_diff_median'])
    var_neg = {}
    for r in t_var:
        if r['pair'] != 'vts_vs_moirai' or r['band'] != 'b1' or r['m_over_P'] != '1':
            continue
        if float(r['D_diff_median']) < 0:
            var_neg.setdefault(r['ds'], []).append(r['var'])
    tv = {}
    for r in t_main:
        if r['pair'] == 'vts_vs_autoets' and r['band'] == 'full':
            tv['%s|m%s' % (r['ds'], r['m_over_P'])] = {
                'D_diff_median': float(r['D_diff_median']), 'D_diff_pos_rate': float(r['D_diff_pos_rate'])}
    num['q5'] = {'direction_b1_m1_Ddiff_median': q5_dir, 'amplitude_b1_m1_Ddiff_median': q5_amp,
                 'var_neg_b1_m1': var_neg, 'vts_vs_autoets_full': tv}

    # ---- AutoARIMA 辅助臂(仅原子集 28 基准窗,单独汇报)----
    ar = {}
    for (m, ds, mp, b), r in meth.items():
        if m == 'autoarima':
            ar['%s|m%s|%s' % (ds, mp, b)] = {'D_median': float(r['D_median']),
                                             'cost_median': float(r['cost_median']),
                                             'n_pairs': int(r['n_pairs'])}
    num['autoarima_aux'] = ar

    with open(os.path.join(out_dir, 'report_numbers_t47h.json'), 'w') as f:
        json.dump(num, f, indent=1)

    # ---- 一页结论(全部数字由上面 num 注入)----
    L = []
    L.append('# EXP-047h 一页结论:预测距离 × 误差分解(零推理,探索性)\n')
    L.append('输入 = EXP-047 v2 冻结产物(13 文件 SHA 见 input_shas_t47h.json;'
             'verify V1–V7 %s,max_true_p_diff=%.2e)。区间 = 同一 H=96 预测的切片'
             '(b1=[0,24)…b4=[72,96);ETTh 每 band=1P、ETTm 每 band=0.25P);'
             'full=[0,96) 仅核对/单列,不作第五时段。配对 n:vts/moirai/autoets 各 5376、'
             'autoarima 336;汇总一律逐配对差值,均值/中位/正比例;线索级,无显著性检验。\n'
             % ('ALL PASS' if ver['all_pass'] else 'FAIL', meta['max_true_p_diff']))

    L.append('## 背景事实(两方法各自,主表 method)\n')
    r = num['method_D_by_band']
    L.append('- D(向旧水平牵引)随预测距离**增大**而非减小:ETTh1 m=1 VTS b1→b4 '
             '%.3f→%.3f,Moirai %.3f→%.3f(后者增长更快);ETTh1 m=4 VTS %.3f→%.3f。'
             % (r['vts_ETTh1_m1']['b1']['D_median'], r['vts_ETTh1_m1']['b4']['D_median'],
                r['moirai_ETTh1_m1']['b1']['D_median'], r['moirai_ETTh1_m1']['b4']['D_median'],
                r['vts_ETTh1_m4']['b1']['D_median'], r['vts_ETTh1_m4']['b4']['D_median']))
    dd13 = {ds: [cell('vts_vs_moirai', ds, '1', b, 'D_diff_median') for b in ('b1', 'b3', 'b4')]
            for ds in DSS}
    L.append('- 时段分解的回答与 full-96 平均不同,且**须按 ds 限定**(m=1 D_diff 中位 '
             'b1→b3→b4,逐 ds):%s。"早期(b1)VTS 牵引更强"见于 ETTh1/ETTm1/ETTm2;'
             '"后期(b3/b4)转为 Moirai 牵引更强"仅 ETTh1(b4 转负)与 ETTm2(b4 转负);'
             '**ETTm1 后期仍为正(b4 %+.3f,VTS 牵引更强)**,ETTh2 全程为负。\n'
             % (';'.join('%s %+.3f→%+.3f→%+.3f' % (ds, dd13[ds][0], dd13[ds][1], dd13[ds][2])
                         for ds in DSS), dd13['ETTm1'][2]))

    L.append('## 五问(逐配对差值,正值 = VTS 牵引更强/代价更大/误差更大)\n')
    L.append('**Q1 哪些 ds×m 的早期(b1)与全程(full)结论不同**\n')
    for e in [x for x in q1 if x['q'] == 'D_diff_median']:
        L.append('- %s m=%s D_diff:b1 中位 %+.3f(正比例 %.0f%%)→ full %+.3f(%.0f%%)%s'
                 % (e['ds'], e['m'], e['b1_median'], 100 * e['b1_pos_rate'], e['full_median'],
                    100 * e['full_pos_rate'], ';**符号翻转**' if e['sign_flip'] else ''))
    etth2m1 = [x for x in q1 if x['ds'] == 'ETTh2' and x['m'] == '1' and x['q'] == 'D_diff_median']
    if not etth2m1:
        L.append('- ETTh2 m=1 无翻转:b1 与 full 均为 Moirai 牵引更强(Dd_med b1 %+.3f / full %+.3f)'
                 % (cell('vts_vs_moirai', 'ETTh2', '1', 'b1', 'D_diff_median'),
                    cell('vts_vs_moirai', 'ETTh2', '1', 'full', 'D_diff_median')))
    cd_q1 = [x for x in q1 if x['q'] == 'cost_diff_median' and x['sign_flip']]
    cd_sub = [x for x in cd_q1 if abs(x['b1_median']) >= 0.005]
    L.append('- cost_diff 符号翻转 %d 格(全程转负/消失):%s;其中 b1 中位量级可见'
             '(≥0.005)的仅 %s——多数格 b1 值接近 0,翻转本身为线索级。'
             % (len(cd_q1),
                ';'.join('%s m=%s(b1 %+.4f→full %+.4f)'
                         % (e['ds'], e['m'], e['b1_median'], e['full_median']) for e in cd_q1) or '无',
                ';'.join('%s m=%s(b1 %+.4f)' % (e['ds'], e['m'], e['b1_median']) for e in cd_sub) or '无'))
    L.append('- 均值与中位可反向:ETTm1 m=1 b1 的 cost_diff 中位 %+.4f、均值 %+.4f'
             '(少量大幅负值配对拉低均值)——正/负解读以中位+均值并报为准,全表两列均有。'
             % (cell('vts_vs_moirai', 'ETTm1', '1', 'b1', 'cost_diff_median'),
                cell('vts_vs_moirai', 'ETTm1', '1', 'b1', 'cost_diff_mean')))
    L.append('')
    L.append('**Q2 牵引更强与代价更大是否同时**\n')
    c = num['q2_sign_combo']['counts']
    n_m1 = sum(1 for ds in DSS for b in BANDS5
               if cell('vts_vs_moirai', ds, '1', b, 'D_diff_median') > 0
               and cell('vts_vs_moirai', ds, '1', b, 'cost_diff_median') > 0)
    n_m2_etth2 = sum(1 for b in BANDS5
                     if cell('vts_vs_moirai', 'ETTh2', '2', b, 'D_diff_median') > 0
                     and cell('vts_vs_moirai', 'ETTh2', '2', b, 'cost_diff_median') > 0)
    L.append('- 60 格(4ds×3m×5band)符号组合:同时正 %d、仅牵引正 %d、仅代价正 %d、同负 %d。'
             % (c['both_pos'], c['pull_only'], c['cost_only'], c['both_neg']))
    L.append('- 同时正 %d 格的分布:m=1 占 %d/20,ETTh2 m=2 占 %d/5;其余 m≥2 格'
             '(ETTh1/ETTm1/ETTm2)无一同时正(D_diff 中位全负,即 Moirai 牵引更强)。'
             % (c['both_pos'], n_m1, n_m2_etth2))
    L.append('- **关键反例(仅代价正,共 %d 格,含 ETTh2 m=1 全部 5 格)**:%s → '
             '牵引不更强但代价更大,**不能解释为更依赖旧水平**。'
             % (c['cost_only'], ';'.join(num['q2_sign_combo']['cost_only_cells'][:4])))
    jp = num['q2_joint_pos_rate']
    frs = sorted((jp[k], k) for k in both_pos_keys if k in jp)
    j_ex = jp.get('ETTm2|1|b1')
    L.append('- 组统计≠窗口同时成立:"同时正 %d 格"按**格点中位数**判定;逐配对两差值'
             '(D_diff 与 cost_diff)同时>0 的窗口占比在这 %d 格仅 %.1f%%–%.1f%%、'
             '%d/%d 格过半(最低 %s %.1f%%;如 ETTm2 m=1 b1 %.1f%%)——两中位为正'
             '不代表多数窗口两指标同时为正。'
             % (c['both_pos'], len(frs), 100 * frs[0][0], 100 * frs[-1][0],
                sum(1 for f, _ in frs if f > 0.5), len(frs),
                frs[0][1], 100 * frs[0][0], 100 * j_ex if j_ex is not None else float('nan')))
    L.append('')
    L.append('**Q3 误差增量主要是水平项还是去均值后误差**\n')
    q3 = num['q3_level_vs_centered']
    ex = q3['mean_example_ETTh2_m1_full']
    L.append('- 分解对象是**方法间代价差** cost_diff = (MSE_B−MSE_A)_vts − (MSE_B−MSE_A)_moirai。'
             '逐配对恒等式 cost_diff = cost_level_diff + cost_centered_diff ⇒ **均值严格可加'
             '(中位数不可加,不得按中位相加)**。以同配对集均值分解,例 ETTh2 m=1 full:'
             'cost_diff 均值 %.5f = 水平项 %.5f + 去均值项 %.5f。'
             % (ex['cost_diff'], ex['cost_level_diff'], ex['cost_centered_diff']))
    L.append('- 主导项计数(按格点):|cost_level_diff| ≥ |cost_centered_diff| 的格点'
             '均值口径 %d/%d、中位数口径 %d/%d(中位数口径仅作稳健性描述,不参与相加)。'
             % (q3['level_dominant_cells_mean'], q3['n_cells'],
                q3['level_dominant_cells_median'], q3['n_cells']))
    L.append('- 即历史冲突的误差增量主要是**预测整体水平偏移(bias²)**;'
             'centered_error 不解释为旧形状依赖。\n')
    L.append('**Q4 冲突代价增量更大的配对,VTS 最终误差是否仍低于 Moirai(条件逐配对)**\n')
    for k, lab in (('ETTh1|1|b1', 'ETTh1 m=1 b1'), ('ETTm1|1|full', 'ETTm1 m=1 full'),
                   ('ETTh2|1|full', 'ETTh2 m=1 full')):
        d = q4_out.get(k)
        if d and d['n_pairs_costpos']:
            L.append('- %s:cost_diff>0 的 %d 对中,gap_B<0 仅占 %.0f%%,子集内 gap_B 中位 %+.4f'
                     '(正 = VTS 最终误差也更高)。'
                     % (lab, d['n_pairs_costpos'], 100 * d['frac_gapB_neg'], d['gapB_median_in_costpos']))
    g22 = cell('vts_vs_moirai', 'ETTh2', '2', 'full', 'gap_B_median')
    g21 = cell('vts_vs_moirai', 'ETTh2', '1', 'full', 'gap_B_median')
    L.append('- **口径区分**:无条件(full)gap_B 中位 ETTh1 %+.4f / ETTm1 %+.4f / ETTm2 %+.4f'
             ' / ETTh2 %+.4f——前三 ds VTS 最终误差仍低于 Moirai(VTS 的整体优势来自参照条件 A '
             '的大幅领先);ETTh2 无条件也正。'
             % (cell('vts_vs_moirai', 'ETTh1', '1', 'full', 'gap_B_median'),
                cell('vts_vs_moirai', 'ETTm1', '1', 'full', 'gap_B_median'),
                cell('vts_vs_moirai', 'ETTm2', '1', 'full', 'gap_B_median'), g21))
    h2m1 = [q4_out['ETTh2|1|%s' % b] for b in BANDS5]
    f_h1b1, f_m1f = q4_out['ETTh1|1|b1'], q4_out['ETTm1|1|full']
    f_m2f, f_m4f = q4_out['ETTm1|2|full'], q4_out['ETTm1|4|full']
    L.append('- 但条件统计**跨格方向不一,不能概括为单一结论**:ETTh2 m=1 各 band 的 '
             'cost_diff>0 子集 gap_B<0 仅 %.0f–%.0f%%;上述 ETTh1 m=1 b1 %.0f%%、'
             'ETTm1 m=1 full %.0f%%——这些格多数配对 VTS 最终误差更高。**反例:'
             'ETTm1 m=2 full %.1f%%、ETTm1 m=4 full %.1f%%**(子集 gap_B 中位 %+.4f / %+.4f)'
             '——多数配对 VTS 最终误差仍更低;ETTh2 m=1/m=2 无条件 gap_B 中位即为正'
             '(full %+.4f / %+.4f)。60 格逐格数值全列 report_numbers_t47h.json q4_conditional。'
             % (100 * min(h['frac_gapB_neg'] for h in h2m1),
                100 * max(h['frac_gapB_neg'] for h in h2m1),
                100 * f_h1b1['frac_gapB_neg'], 100 * f_m1f['frac_gapB_neg'],
                100 * f_m2f['frac_gapB_neg'], 100 * f_m4f['frac_gapB_neg'],
                f_m2f['gapB_median_in_costpos'], f_m4f['gapB_median_in_costpos'], g21, g22))
    L.append('')
    L.append('**Q5 方向 / 幅度 / 变量一致性 + 反例**\n')
    d5 = num['q5']['direction_b1_m1_Ddiff_median']
    a5 = num['q5']['amplitude_b1_m1_Ddiff_median']
    L.append('- 方向:δ 向下时 m=1 b1 的 D_diff 更大,**四个 ds 一致**(%s)。ETTh2 两方向'
             '均为负(Moirai 牵引更强),但向下 %+.3f > 向上 %+.3f,仍为“向下更大”。'
             % (';'.join('%s down %+.3f vs up %+.3f' % (ds, d5['%s|down' % ds], d5['%s|up' % ds])
                         for ds in ('ETTh1', 'ETTm1', 'ETTm2')),
                d5['ETTh2|down'], d5['ETTh2|up']))
    L.append('- 幅度:|δ|/s=1.0 比 0.5 的 D_diff 更大(%s);ETTh2 两档均为负(Moirai 牵引更强),'
             '且 1.0 档绝对值更小(%+.3f vs %+.3f,按绝对值反向)。'
             % (';'.join('%s 1.0 %+.3f vs 0.5 %+.3f' % (ds, a5['%s|1.0' % ds], a5['%s|0.5' % ds])
                         for ds in ('ETTh1', 'ETTm1', 'ETTm2')),
                a5['ETTh2|1.0'], a5['ETTh2|0.5']))
    vn = num['q5']['var_neg_b1_m1']
    L.append('- 变量级反例(m=1 b1 D_diff 中位<0 的 var):%s。'
             % (';'.join('%s: %s' % (ds, ','.join(v)) for ds, v in sorted(vn.items())) or '无'))
    ae = num['q5']['vts_vs_autoets_full']
    L.append('- 对照 AutoETS:VTS 牵引几乎处处更强(full D_diff 正比例 %s,中位 %s)'
             '(辅助比较);其 b1 早期值见主表。'
             % ('/'.join('%.2f' % (ae['%s|m1' % ds]['D_diff_pos_rate']) for ds in DSS),
                '/'.join('%+.3f' % (ae['%s|m1' % ds]['D_diff_median']) for ds in DSS)))
    ar1 = num['autoarima_aux']
    L.append('- AutoARIMA(仅原子集,ETTh1/h2 各 14 基准窗,单独汇报):m=1 D 中位 '
             'ETTh1 b1 %+.3f→b4 %+.3f、ETTh2 b1 %+.3f→b4 %+.3f(n=%s/%s)。'
             % (ar1['ETTh1|m1|b1']['D_median'], ar1['ETTh1|m1|b4']['D_median'],
                ar1['ETTh2|m1|b1']['D_median'], ar1['ETTh2|m1|b4']['D_median'],
                ar1['ETTh1|m1|full']['n_pairs'], ar1['ETTh2|m1|full']['n_pairs']))
    L.append('\n## 解读纪律\n')
    L.append('- 上述均为线索级:构造变体非独立样本,无显著性检验;A 组只称**参照条件**;'
             '不归因 MAE;不写独立验证结论。')
    L.append('- 四种解读对应:①m=1 早期同时正 + ETTh1 m=1 的 b1→full 符号翻转 + '
             'ETTm1/ETTm2 m=1 早期幅度大幅衰减(b4 转负或约为 b1 的 1/3)= '
             '“**全程平均掩盖早期差异/原有优势缩小**”的支持线索;②cost_only 组合'
             '(含 ETTh2 m=1 全部 5 格)= “**牵引不更强但代价更大,不能解释为更依赖旧水平**”;'
             '③ETTh2 的条件性线索(该 ds 上 Moirai 占优;m=2 时反而是 VTS 牵引更强且代价更大'
             '且误差更大);④其余 m≥2 无稳定组合 = “**无稳定差异,不支持‘全程平均掩盖问题’'
             '作为普遍结论**”。')

    with open(os.path.join(out_dir, 'report_t47h.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('report numbers -> report_numbers_t47h.json / report_t47h.md (q1=%d, q4=%d cells)'
          % (len(q1), len(q4_out)))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
