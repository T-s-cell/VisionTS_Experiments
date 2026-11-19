"""EXP-047 门禁(G1–G12),仿 t46 十门禁 + v2 修订。任何 FAIL → 汇总 verify_t47.json + 退出码 1。

G1  基准清单重放:subprocess 重跑 select_t47.py → sha 与 baseline_list_sha_t47.txt 一致
G2  manifest 重放 + 构造恒等式:重跑 build_variants_t47.py → manifest sha 不变;
    s↔features 448 全量 ≤1e-8;抽样 64 行在线重建 A/B 逐位恒等
    (v2 修订:按 cond_id 分层,每条件 4 行,覆盖全部 16 条件 = 全部 δ×m×组;
    原 stride 抽样步长 112 ≡ 0 (mod 16),实际只覆盖 cond_id=0,已修正)
G3  真值:各方法 npz true_p vs baseline y+δ(f64)maxdiff ≤1e-5
G4  覆盖率:主方法各 7168 行;autoarima 448 行(ETTh1/ETTh2 rank≤1);
    ledger 行数与 ok 计数 ↔ npz status 一致
G5  有限性:vts/moirai pred 全有限;sfdx ok 行有限、失败行全 NaN;true_p 全有限
G6  id 对齐:item_ids/cond_ids/row_globals ↔ manifest 逐行一致;rg = w_global*16+cond_id
G7  指标重算:CSV mse/mae vs npz pred + f64 真值 ≤1e-9
G8  D/cost/ΔD/Δcost 重算 ≤1e-9(pairs_t47.csv / method_diff_t47.csv)
G9  §6 复核:a/b/c/类别 1792 全量重算一致(值 ≤1e-9)
G10 图真值:fig_data_t47.npz 真值 vs f64 重算恒等、vs npz true_p ≤1e-5、
    各方法 pred 数组与 npz 逐位一致
G11 汇总表一致性(v2 新增):四张表(table_D_cost / table_method_diff / table_D_common_m /
    table_s6,含 hist_std 三分位与 class_common)从原始 CSV 独立重算,
    n/mean/median/pos_rate ≤1e-9;表头列数 = 每数据行列数(v1 曾出现 s6 表 7 列表头
    6 列数据、method_diff 分组塌缩,本门禁防回归)
G12 案例图一致性(v2 新增):cases_t47.csv 的 (ds,sid,δ,m,D) ↔ pairs 有效行 ≤1e-9;
    fig_data case meta(sid/di/mi/rgAB/D_vts)↔ manifest rg 公式与 B 行 m_steps 逐项一致;
    案例数 = 4(每 ds 1 例,选样含 m 维,标题与绘制条件同源)

用法:python verify_t47.py <repo_root> [save_dir]
"""
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from t47_common import (M_MULTS, METHODS, S_TOL, SCRIPT_DIR, T46_DIR, apply_variant,
                        baseline_list_path, manifest_path)

TRUE_TOL = 1e-5
RECOMP_TOL = 1e-9
DS_ALL = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')


def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def read_csv_dicts(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main(repo_root, save_dir):
    man = read_csv_dicts(manifest_path())
    assert len(man) == 7168
    base = {ds: np.load(os.path.join(save_dir, 'baseline_%s.npz' % ds)) for ds in DS_ALL}
    sid_row = {ds: {int(s): i for i, s in enumerate(z['sid_t46'])} for ds, z in base.items()}
    arrs = {m: np.load(os.path.join(save_dir, 'pred_%s_t47.npz' % m)) for m in METHODS}
    pos = {m: {int(rg): j for j, rg in enumerate(arrs[m]['row_globals'])} for m in METHODS}
    bl = read_csv_dicts(baseline_list_path())
    w_by_sid = {int(r['sid_t46']): int(r['w_global']) for r in bl}
    ledger = read_csv_dicts(os.path.join(save_dir, 'ledger_t47.csv'))

    def truth64(row):
        return base[row['ds']]['y'][sid_row[row['ds']][int(row['sid_t46'])]] + float(row['delta'])

    gates = {}

    def ok(g, cond, detail=''):
        gates[g] = dict(pass_=bool(cond), detail=str(detail)[:300])
        print('[%s] %s %s' % (g, 'PASS' if cond else 'FAIL', '' if cond else detail))

    # G1 基准清单重放
    sha_before = sha256(baseline_list_path())
    r = subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, 'select_t47.py')],
                       capture_output=True, text=True)
    ok('G1', r.returncode == 0 and sha256(baseline_list_path()) == sha_before
       == open(os.path.join(SCRIPT_DIR, 'baseline_list_sha_t47.txt')).read().strip(),
       'rc=%d sha=%s' % (r.returncode, sha256(baseline_list_path())[:16]))

    # G2 manifest 重放 + 构造恒等式
    msha_before = sha256(manifest_path())
    r = subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, 'build_variants_t47.py'),
                        repo_root, save_dir], capture_output=True, text=True)
    g2 = r.returncode == 0 and sha256(manifest_path()) == msha_before
    detail = 'rc=%d sha=%s' % (r.returncode, sha256(manifest_path())[:16])
    if g2:
        feats = {int(x['sample_id']): x for x in
                 read_csv_dicts(os.path.join(T46_DIR, 'features_t46.csv'))}
        for ds, z in base.items():
            for i, sid in enumerate(z['sid_t46']):
                d = abs(float(z['s'][i]) - float(feats[int(sid)]['hist_std']))
                if d > S_TOL:
                    g2, detail = False, 's mismatch ds=%s sid=%d diff=%.2e' % (ds, sid, d)
    if g2:
        by_cid = {}
        for i, r_ in enumerate(man):
            by_cid.setdefault(int(r_['cond_id']), []).append(i)
        sample_rows = []
        for cid in sorted(by_cid):
            glist = by_cid[cid]
            step = max(1, len(glist) // 4)
            sample_rows.extend(glist[::step][:4])
        assert len(sample_rows) == 64 and len({man[k]['cond_id'] for k in sample_rows}) == 16
        for k in sample_rows:  # 64 行 = 16 条件 × 各 4 行,覆盖全部 δ×m×组
            r_ = man[k]
            x = base[r_['ds']]['x'][sid_row[r_['ds']][int(r_['sid_t46'])]]
            delta, g_, m_steps = float(r_['delta']), r_['group'], int(r_['m_steps'] or 0)
            xv = apply_variant(x, delta, m_steps, g_)
            if g_ == 'B':
                if not (np.array_equal(xv[-m_steps:], (x + delta)[-m_steps:])
                        and xv[-1] == x[-1] + delta
                        and np.array_equal(xv[:-m_steps], x[:-m_steps])):
                    g2, detail = False, 'B identity fail row %d' % k
                    break
            elif not np.array_equal(xv, x + delta):
                g2, detail = False, 'A identity fail row %d' % k
                break
    ok('G2', g2, detail)

    # G3–G6 数组层(逐方法)
    exp_rg = np.array([int(r['row_global']) for r in man], dtype=np.int64)
    exp_sid = np.array([int(r['sid_t46']) for r in man], dtype=np.int64)
    exp_cid = np.array([int(r['cond_id']) for r in man], dtype=np.int64)
    g3 = g4 = g5 = g6 = True
    det = {}
    for m in METHODS:
        z = arrs[m]
        mask = (np.array([(r['ds'] in ('ETTh1', 'ETTh2') and int(r['rank']) <= 1)
                          for r in man]) if m == 'autoarima' else np.ones(7168, bool))
        if not (len(z['row_globals']) == int(mask.sum())
                and np.array_equal(z['row_globals'], exp_rg[mask])
                and np.array_equal(z['item_ids'], exp_sid[mask])
                and np.array_equal(z['cond_ids'], exp_cid[mask])):
            g6, det['G6_%s' % m] = False, 'id misalignment'
        formula_ok = all(
            int(rj) == w_by_sid[int(man[int(rj)]['sid_t46'])] * 16 + int(man[int(rj)]['cond_id'])
            for rj in z['row_globals'])
        if not formula_ok:
            g6, det['G6_%s_formula' % m] = False, 'rg != w*16+cid'
        dmax = 0.0
        for j, rj in enumerate(z['row_globals']):
            yp = truth64(man[int(rj)])
            dmax = max(dmax, float(np.abs(z['true_p'][j].astype(np.float64) - yp).max()))
            pred = z['pred'][j]
            if m in ('vts', 'moirai'):
                if not np.isfinite(pred).all():
                    g5, det['G5_%s' % m] = False, 'nonfinite rg=%d' % int(rj)
            else:
                st = str(z['status'][j])
                if st == 'ok' and not np.isfinite(pred).all():
                    g5, det['G5_%s' % m] = False, 'ok but nonfinite rg=%d' % int(rj)
                if st != 'ok' and np.isfinite(pred).any():
                    g5, det['G5_%s' % m] = False, 'fail but finite rg=%d' % int(rj)
        if not np.isfinite(z['true_p']).all():
            g5, det['G5_truep_%s' % m] = False, 'true_p nonfinite'
        if dmax > TRUE_TOL:
            g3, det['G3_%s' % m] = False, 'maxdiff=%.3e' % dmax
        det['G3_%s_maxdiff' % m] = '%.2e' % dmax
        if m in ('autoets', 'autoarima'):
            name = 'AutoETS' if m == 'autoets' else 'AutoARIMA'
            led_m = [r for r in ledger if r['method'] == name]
            arr_ok = sum(1 for s in z['status'] if str(s) == 'ok')
            if len(led_m) != int(mask.sum()):
                g4, det['G4_%s' % m] = False, 'ledger %d vs n %d' % (len(led_m), int(mask.sum()))
            elif sum(1 for r in led_m if r['status'] == 'ok') != arr_ok:
                g4, det['G4_%s_counts' % m] = False, 'ledger ok != npz ok'
    if len(ledger) != 7168 + 448:
        g4, det['G4_ledger'] = False, 'ledger rows=%d expect 7616' % len(ledger)
    ok('G3', g3, det.get('G3', 'true ok'))
    ok('G4', g4, det.get('G4', 'coverage ok'))
    ok('G5', g5, det.get('G5', 'finite ok'))
    ok('G6', g6, det.get('G6', 'ids ok'))

    # G7 指标重算
    met = read_csv_dicts(os.path.join(save_dir, 'per_sample_metrics_t47.csv'))
    worst = 0.0
    if len(met) == 7168:
        for r in met:
            rg = int(r['row_global'])
            yp = truth64(man[rg])
            for m in METHODS:
                if rg not in pos[m] or r['ok_%s' % m] != '1':
                    continue
                p = arrs[m]['pred'][pos[m][rg]].astype(np.float64)
                worst = max(worst, abs(float(((p - yp) ** 2).mean()) - float(r['mse_%s' % m])),
                            abs(float(np.abs(p - yp).mean()) - float(r['mae_%s' % m])))
    g7 = worst <= RECOMP_TOL and len(met) == 7168
    ok('G7', g7, 'max recompute diff=%.3e' % worst)

    # G8 D/cost/ΔD/Δcost 重算
    pairs = read_csv_dicts(os.path.join(save_dir, 'pairs_t47.csv'))
    diffs = read_csv_dicts(os.path.join(save_dir, 'method_diff_t47.csv'))
    worst8 = 0.0
    for r in pairs:
        m = r['method']
        rg_a, rg_b = int(r['rg_A']), int(r['rg_B'])
        if rg_a not in pos[m] or rg_b not in pos[m] or r['ok_A'] != '1' or r['ok_B'] != '1':
            continue
        delta = float(man[rg_a]['delta'])
        assert delta == float(man[rg_b]['delta'])
        pa = arrs[m]['pred'][pos[m][rg_a]].astype(np.float64)
        pb = arrs[m]['pred'][pos[m][rg_b]].astype(np.float64)
        D = float((pa - pb).mean() / delta)
        cost = float(((pb - truth64(man[rg_b])) ** 2).mean()
                     - ((pa - truth64(man[rg_a])) ** 2).mean())
        worst8 = max(worst8, abs(D - float(r['D'])), abs(cost - float(r['cost'])))
    pp = {(r['method'], int(r['sid_t46']), int(r['delta_idx']), int(r['m_idx'])): r
          for r in pairs}
    for r in diffs:
        mv, mc = r['pair'].split('_vs_')
        rv, rc = pp.get((mv, int(r['sid_t46']), int(r['delta_idx']), int(r['m_idx']))), \
            pp.get((mc, int(r['sid_t46']), int(r['delta_idx']), int(r['m_idx'])))
        if rv is None or rc is None:
            continue
        if min(rv['ok_A'], rv['ok_B'], rc['ok_A'], rc['ok_B']) != '1':
            continue
        dd = float(rv['D']) - float(rc['D'])
        dc = float(rv['cost']) - float(rc['cost'])
        worst8 = max(worst8, abs(dd - float(r['D_diff'])), abs(dc - float(r['cost_diff'])))
    g8 = len(pairs) > 0 and len(diffs) > 0 and worst8 <= RECOMP_TOL
    ok('G8', g8, 'max recompute diff=%.3e' % worst8)

    # G9 §6 复核(1792 全量重算 a/b/c/类别)
    from t46_common import DS_CFG, H, load_csv, sample_list_path, std_all, train_scaler
    samples = {int(r['sample_id']): r for r in read_csv_dicts(sample_list_path())}
    s6 = read_csv_dicts(os.path.join(save_dir, 's6_recheck_t47.csv'))
    g9 = len(s6) == 1792
    cache = {}
    worst9 = 0.0
    cls_bad = 0
    for r in s6:
        s = samples[int(r['sample_id'])]
        ds, cfg = s['ds'], DS_CFG[s['ds']]
        if ds not in cache:
            data = load_csv(os.path.join(repo_root, cfg['csv']))
            mean, scale = train_scaler(data, cfg['K'])
            cache[ds] = std_all(data, mean, scale)
        std = cache[ds]
        start, ch, L, P = int(s['start_row']), int(s['channel']), cfg['L'], cfg['P']
        x = std[start - L:start, ch]
        a = float(x[L - 4 * P:L - P].mean())
        b = float(x[L - P:L].mean())
        c = float(std[start:start + H, ch].mean())
        sv = float(r['s'])
        for name, v in (('a', a), ('b', b), ('c', c)):
            worst9 = max(worst9, abs(v - float(r[name])))
        stable = abs(b - a) <= 0.1 * sv and abs(c - b) <= 0.1 * sv
        changed = abs(b - a) >= 0.5 * sv and abs(c - b) <= 0.5 * abs(b - a)
        cls = 'stable' if stable else ('changed_persist' if changed else 'other')
        if cls != r['class']:
            cls_bad += 1
    if worst9 > RECOMP_TOL or cls_bad:
        g9 = False
    ok('G9', g9, 'value diff=%.3e class_bad=%d' % (worst9, cls_bad))

    # G10 图真值
    figd = np.load(os.path.join(save_dir, 'fig_data_t47.npz'))
    g10 = True
    worst10 = 0.0
    ncase = sum(1 for k in figd.files if k.endswith('_true'))
    for i in range(ncase):
        rg_a, rg_b = figd['case%d_rgAB' % i]
        yp = truth64(man[int(rg_a)])
        if not np.array_equal(figd['case%d_true' % i], yp):
            g10 = False
        worst10 = max(worst10, float(np.abs(figd['case%d_truep_vts_f32' % i].astype(np.float64)
                                            - yp).max()))
        for m in ('vts', 'moirai', 'autoets'):
            for tag, rg in (('A', int(rg_a)), ('B', int(rg_b))):
                if not np.array_equal(figd['case%d_pred%s_%s' % (i, tag, m)],
                                      arrs[m]['pred'][pos[m][rg]]):
                    g10 = False
    if worst10 > TRUE_TOL:
        g10 = False
    ok('G10', g10, 'cases=%d truep diff=%.3e' % (ncase, worst10))

    # G11 汇总表一致性(从原始 CSV 独立重算;表头列数 = 每数据行列数)
    g11 = True
    d11 = []

    def stats_v(vals):
        a = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
        return (int(len(a)), float(a.mean()), float(np.median(a)), float((a > 0).mean()))

    def scope_keys_v(r):
        direction = 'down' if float(r['delta_mult']) < 0 else 'up'
        return {'overall': 'all', 'ds': r['ds'], 'var': r['var'], 'direction': direction,
                'amplitude': '%.1f' % abs(float(r['delta_mult']))}

    def tbl_check(name, path, fields, rows_expect):
        nonlocal g11
        bad = None
        try:
            with open(path) as f:
                rd = list(csv.reader(f))
        except OSError as e:
            bad, rd = str(e), []
        if bad is None:
            if not rd or rd[0] != fields:
                bad = 'header=%s' % (rd[0] if rd else 'empty')
            elif any(len(r) != len(fields) for r in rd[1:]):
                bad = 'row width != header'
            elif len(rd) - 1 != len(rows_expect):
                bad = 'n_rows %d != %d' % (len(rd) - 1, len(rows_expect))
            else:
                for got, want in zip(rd[1:], rows_expect):
                    for gv, wv in zip(got, want):
                        if isinstance(wv, float):
                            if abs(float(gv) - wv) > RECOMP_TOL:
                                bad = '%s!=%s' % (gv, wv)
                                break
                        elif isinstance(wv, int):
                            if int(gv) != wv:
                                bad = '%s!=%s' % (gv, wv)
                                break
                        elif gv != str(wv):
                            bad = '%s!=%s' % (gv, wv)
                            break
                    if bad:
                        break
        if bad:
            g11 = False
        d11.append('%s=%s' % (name, bad or 'ok'))

    def agg_v(rows, val_of, id_of, cost_of):
        accd, accc = {}, {}
        for r in rows:
            for scope, key in scope_keys_v(r).items():
                k = (scope, key, id_of(r), float(r['m_over_P']))
                accd.setdefault(k, []).append(val_of(r))
                accc.setdefault(k, []).append(cost_of(r))
        out = []
        for k in sorted(accd):
            dv, cv = stats_v(accd[k]), stats_v(accc[k])
            out.append([k[0], k[1], k[2], k[3], dv[0], dv[1], dv[2], dv[3], cv[1], cv[2]])
        return out

    tbl_check('D_cost', os.path.join(save_dir, 'table_D_cost_t47.csv'),
              ['scope', 'key', 'method', 'm_over_P', 'n', 'D_mean', 'D_median', 'D_pos_rate',
               'cost_mean', 'cost_median'],
              agg_v(pairs, lambda r: float(r['D']), lambda r: r['method'],
                    lambda r: float(r['cost'])))
    tbl_check('method_diff', os.path.join(save_dir, 'table_method_diff_t47.csv'),
              ['scope', 'key', 'pair', 'm_over_P', 'n', 'Ddiff_mean', 'Ddiff_median',
               'pos_rate', 'costdiff_mean', 'costdiff_median'],
              agg_v(diffs, lambda r: float(r['D_diff']), lambda r: r['pair'],
                    lambda r: float(r['cost_diff'])))
    have = {}
    for r in pairs:
        if float(r['D']) == float(r['D']):
            have.setdefault((r['method'], int(r['sid_t46']), int(r['delta_idx']),
                             int(r['m_idx'])), (float(r['D']), float(r['cost'])))
    common_expect = []
    for m in METHODS:
        ks = sorted({k[1:] for k in have if k[0] == m and k[3] == 0})
        keep = [k for k in ks if all((m, k[0], k[1], mi) in have for mi in range(3))]
        for mi in range(3):
            dv = stats_v([have[(m, k[0], k[1], mi)][0] for k in keep])
            cv = stats_v([have[(m, k[0], k[1], mi)][1] for k in keep])
            common_expect.append([m, M_MULTS[mi], dv[0], dv[1], dv[2], dv[3], cv[1], cv[2]])
    tbl_check('D_common_m', os.path.join(save_dir, 'table_D_common_m.csv'),
              ['method', 'm_over_P', 'n_common_windows', 'D_mean', 'D_median', 'D_pos_rate',
               'cost_mean', 'cost_median'], common_expect)
    from t46_common import DS_ORDER as DS_ORDER_V
    featsv = {int(x['sample_id']): x for x in read_csv_dicts(
        os.path.join(T46_DIR, 'features_t46.csv'))}
    s6r = read_csv_dicts(os.path.join(save_dir, 's6_recheck_t47.csv'))
    tertv = {}
    for ds in DS_ORDER_V:
        for v_ in dict.fromkeys(r['var'] for r in s6r if r['ds'] == ds):
            grp = [r for r in s6r if r['ds'] == ds and r['var'] == v_
                   and featsv[int(r['sample_id'])]['near_constant'] == 'False']
            vals = [float(r['s']) for r in grp]
            q1, q2 = float(np.quantile(vals, 1 / 3)), float(np.quantile(vals, 2 / 3))
            for r in grp:
                sv = float(r['s'])
                tertv[int(r['sample_id'])] = ('low' if sv <= q1 else
                                              ('mid' if sv <= q2 else 'high'))
    accv = {}
    for r in s6r:
        keys = (('overall', '', ''), ('ds', r['ds'], ''), ('class', r['class'], ''),
                ('class_tert', r['class'], tertv[int(r['sample_id'])]))
        vals = {}
        for m in METHODS:
            if r['ok_%s' % m] == '1':
                mse = float(r['mse_%s' % m])
                if np.isfinite(mse):
                    vals[m] = mse
                    for key in keys:
                        accv.setdefault((m,) + key, []).append(mse)
        if len(vals) == len(METHODS):
            for m, mse in vals.items():
                accv.setdefault((m, 'class_common', r['class'], ''), []).append(mse)
    s6_expect = [[k[0], k[1], k[2], k[3], len(accv[k]), float(np.mean(accv[k])),
                  float(np.median(accv[k]))] for k in sorted(accv)]
    tbl_check('s6', os.path.join(save_dir, 'table_s6_t47.csv'),
              ['method', 'scope', 'key1', 'key2', 'n_ok', 'mse_mean', 'mse_median'], s6_expect)
    ok('G11', g11, '; '.join(d11)[:300])

    # G12 案例图一致性(cases ↔ pairs ↔ fig_data meta ↔ manifest)
    g12 = True
    d12 = []
    with open(os.path.join(save_dir, 'cases_t47.csv')) as f:
        cases = list(csv.DictReader(f))
    man_rg = {(int(r['sid_t46']), int(r['cond_id'])): int(r['row_global']) for r in man}
    if len(cases) != 4:
        g12 = False
        d12.append('n_cases=%d expect 4' % len(cases))
    for i, c in enumerate(cases):
        sid, di, mi = int(c['sid_t46']), int(c['delta_idx']), int(c['m_idx'])
        pv = pp.get(('vts', sid, di, mi))
        bad = None
        if pv is None or pv['D'] != pv['D'] or \
                abs(float(c['D_vts']) - float(pv['D'])) > RECOMP_TOL:
            bad = 'D vs pairs'
        elif not (int(pv['ok_A']) == 1 and int(pv['ok_B']) == 1):
            bad = 'pair not ok'
        else:
            rg_a, rg_b = man_rg[(sid, di)], man_rg[(sid, 4 + 3 * di + mi)]
            rb = man[rg_b]
            if int(rb['m_steps']) != M_MULTS[mi] * int(rb['P']):
                bad = 'm_steps vs manifest'
            elif not np.array_equal(figd['case%d_meta' % i],
                                    np.array([sid, di, mi, rg_a, rg_b], dtype=np.int64)):
                bad = 'meta vs manifest'
            elif not np.array_equal(figd['case%d_rgAB' % i],
                                    np.array([rg_a, rg_b], dtype=np.int64)):
                bad = 'rgAB vs manifest'
            elif abs(float(figd['case%d_D_vts' % i]) - float(c['D_vts'])) > 0.0:
                bad = 'fig D vs cases'
        if bad:
            g12 = False
            d12.append('case%d(%s sid%d) %s' % (i, c['ds'], sid, bad))
    ok('G12', g12, '; '.join(d12)[:300] if d12 else '4 cases ok')

    all_pass = all(v['pass_'] for v in gates.values())
    with open(os.path.join(save_dir, 'verify_t47.json'), 'w') as f:
        json.dump(dict(all_pass=all_pass, gates=gates), f, indent=1)
    print('verify_t47: %s (%d gates)' % ('ALL PASS' if all_pass else 'FAIL', len(gates)))
    return all_pass


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    main(args[0], args[1] if len(args) > 1
         else os.path.join(SCRIPT_DIR, 'save_t47_levelshift'))
