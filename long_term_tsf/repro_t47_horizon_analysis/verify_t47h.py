"""EXP-047h verify:V1–V7(fail 即停;容差 atol=1e-9、rtol=1e-10,实测最大差入 verify_t47h.json)。

用法:$PY verify_t47h.py <T47输入根> <OUT输出目录>
V1 对齐:npz row_globals/item_ids/cond_ids ↔ manifest;true_p vs 重建真值 ≤1e-5;
        δ ↔ delta_mult·s;冻结 pairs rg_A/rg_B ↔ manifest 条件表;ok 覆盖率计数。
V2 复现冻结:full-96 的 D/cost ↔ 冻结 pairs_t47.csv;mse ↔ per_sample_metrics;
        full D_diff/cost_diff ↔ 冻结 method_diff_t47.csv(冻结无行处须 nan)。
V3 逐配对 full-96 MSE/D/cost = 四个 24 步区间算术平均。
V4 full level/centered 独立直接重算(非四段平均,信息栏记录二者差);分解恒等式
        MSE=level+centered、cost=cost_level+cost_centered 逐区间。
V5 cost_diff = gap_B − gap_A 逐配对逐区间;gap_A/gap_B ↔ pairs_horizon mse 差。
V6 汇总表(5 张)从逐配对 CSV 独立复算;fig_data ↔ CSV 重算(图与表同源)。
V7 输入 13 文件 SHA256 与 input_shas_t47h.json 一致(compute 时记录);代码块变化仅告警。
verify 数值一律用独立实现(dot/sum 路径)重算,不调用 compute/aggregate 函数。
"""
import json
import os
import sys

import numpy as np

from t47h_common import (BANDS, BAND_IDS, CONTRASTS, DIFF_H, H, INPUT_FILES, INTERVALS,
                         METHODS, PAIRS_H, SAVE_SUB, TOL_ATOL, TOL_RTOL, TRUE_TOL,
                         input_path, load_manifest, pair_index, read_csv_dicts, sha256_file,
                         tol_diff)

QUANT_DIFF = ['D_diff', 'gap_A', 'gap_B', 'cost_diff', 'cost_level_diff', 'cost_centered_diff']
FAILS = []


def fail(msg):
    FAILS.append(msg)
    print('[verify][FAIL] %s' % msg)


def wrjson(doc, out_dir):
    doc['tolerances'] = {'atol': TOL_ATOL, 'rtol': TOL_RTOL, 'true_tol': TRUE_TOL}
    doc['all_pass'] = not FAILS
    doc['fails'] = FAILS
    with open(os.path.join(out_dir, 'verify_t47h.json'), 'w') as f:
        json.dump(doc, f, indent=1)
    print('[verify] all_pass=%s (%d fails) -> verify_t47h.json' % (doc['all_pass'], len(FAILS)))


def load_horizon(out_dir):
    ph = {}
    for r in read_csv_dicts(os.path.join(out_dir, PAIRS_H)):
        ph[(r['method'], int(r['sid_t46']), int(r['delta_idx']), int(r['m_idx']), r['band'])] = r
    dh = {}
    for r in read_csv_dicts(os.path.join(out_dir, DIFF_H)):
        dh[(r['pair'], int(r['sid_t46']), int(r['delta_idx']), int(r['m_idx']), r['band'])] = r
    return ph, dh


def recompute(pa, pb, yp, lo, hi, delta):
    """独立实现:dot/sum 路径(与 compute 的 np.mean 区分)。"""
    ea, eb = pa[lo:hi] - yp[lo:hi], pb[lo:hi] - yp[lo:hi]
    n = hi - lo
    mse_a = float(np.dot(ea, ea) / n)
    mse_b = float(np.dot(eb, eb) / n)
    d = float(np.sum(pa[lo:hi] - pb[lo:hi]) / n) / delta
    bias_a = float(np.sum(ea) / n)
    bias_b = float(np.sum(eb) / n)
    return dict(mse_A=mse_a, mse_B=mse_b, D=d, cost=mse_b - mse_a,
                bias_A=bias_a, bias_B=bias_b,
                level_A=bias_a * bias_a, level_B=bias_b * bias_b,
                centered_A=float(np.dot(ea - bias_a, ea - bias_a) / n),
                centered_B=float(np.dot(eb - bias_b, eb - bias_b) / n))


def main(repo_root, out_dir):
    doc = {}
    # ---------- V7 先做(SHA 现状)----------
    with open(os.path.join(out_dir, 'input_shas_t47h.json')) as f:
        shas = json.load(f)
    now_in = {rel: sha256_file(input_path(repo_root, rel)) for rel in INPUT_FILES}
    bad = [rel for rel in INPUT_FILES if now_in[rel] != shas['inputs'][rel]]
    code_now = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for fn in sorted(os.listdir(script_dir)):
        if fn.endswith('.py') or fn == 'protocol_t47h.md':
            h = sha256_file(os.path.join(script_dir, fn))
            code_now[fn] = h
            if fn in shas['code'] and shas['code'][fn] != h:
                fail('V7 code changed after compute: %s' % fn)
    if bad:
        for rel in bad:
            fail('V7 input sha mismatch: %s' % rel)
    doc['v7'] = {'ok': not any(f.startswith('V7') for f in FAILS),
                 'inputs_ok': not bad, 'n_inputs': len(INPUT_FILES),
                 'code_changed': [fn for fn in code_now
                                  if fn in shas['code'] and shas['code'][fn] != code_now[fn]]}
    print('[verify] V7 inputs %d/%d match' % (len(INPUT_FILES) - len(bad), len(INPUT_FILES)))

    # ---------- 装载 ----------
    man = load_manifest(repo_root)
    pidx = pair_index(man)
    meta_by_rg = {int(r['row_global']): r for r in man}
    base, w_of = {}, {}
    for ds in dict.fromkeys(p['ds'] for p in pidx.values()):
        bz = np.load(input_path(repo_root, '%s/baseline_%s.npz' % (SAVE_SUB, ds)))
        base[ds] = bz
        w_of[ds] = {int(s): i for i, s in enumerate(bz['sid_t46'])}
    preds, rg_pos, true_p_all = {}, {}, {}
    for m in METHODS:
        z = np.load(input_path(repo_root, '%s/pred_%s_t47.npz' % (SAVE_SUB, m)))
        rg_pos[m] = {int(rg): k for k, rg in enumerate(z['row_globals'])}
        preds[m] = z['pred'].astype(np.float64)
        true_p_all[m] = z['true_p'].astype(np.float64)
    met = {int(r['row_global']): r for r in read_csv_dicts(
        input_path(repo_root, '%s/per_sample_metrics_t47.csv' % SAVE_SUB))}
    frozen_pairs = {(r['method'], int(r['sid_t46']), int(r['delta_idx']), int(r['m_idx'])): r
                    for r in read_csv_dicts(input_path(repo_root, '%s/pairs_t47.csv' % SAVE_SUB))}
    frozen_diff = {(r['pair'], int(r['sid_t46']), int(r['delta_idx']), int(r['m_idx'])): r
                   for r in read_csv_dicts(input_path(repo_root, '%s/method_diff_t47.csv' % SAVE_SUB))}
    ph, dh = load_horizon(out_dir)

    def isfin(s):
        v = float(s)
        return v == v

    # ---------- V1 对齐(npz 内建序逐行)----------
    max_true, max_dres, n_align = 0.0, 0.0, {}
    ok_pairs = {m: 0 for m in METHODS}
    for m in METHODS:
        z = np.load(input_path(repo_root, '%s/pred_%s_t47.npz' % (SAVE_SUB, m)))
        cnt = 0
        for k, rg in enumerate(z['row_globals']):
            r = meta_by_rg[int(rg)]
            if int(z['item_ids'][k]) != int(r['sid_t46']) or int(z['cond_ids'][k]) != int(r['cond_id']):
                fail('V1 id mismatch %s rg=%d' % (m, int(rg)))
                break
            w = w_of[r['ds']][int(r['sid_t46'])]
            yp = base[r['ds']]['y'][w] + float(r['delta'])
            d = float(np.abs(yp - true_p_all[m][k]).max())
            max_true = max(max_true, d)
            if tol_diff(float(r['delta']), float(r['delta_mult']) * float(base[r['ds']]['s'][w])) > 0:
                fail('V1 delta!=mult*s %s rg=%d' % (m, int(rg)))
            max_dres = max(max_dres, abs(float(r['delta'])
                                         - float(r['delta_mult']) * float(base[r['ds']]['s'][w])))
            if d > TRUE_TOL:
                fail('V1 true_p diff %.3e > %.0e at %s rg=%d' % (d, TRUE_TOL, m, int(rg)))
            cnt += 1
        n_align[m] = cnt
    for (m, sid, di, mi), fr in frozen_pairs.items():
        p = pidx[(sid, di, mi)]
        if int(fr['rg_A']) != p['rg_a'] or int(fr['rg_B']) != p['rg_b']:
            fail('V1 frozen rg mismatch %s %d %d %d' % (m, sid, di, mi))
        if fr['ok_A'] == '1' and fr['ok_B'] == '1':
            ok_pairs[m] += 1
    if max_true > TRUE_TOL:
        fail('V1 max_true_p_diff %.3e' % max_true)
    doc['v1'] = {'ok': not any(f.startswith('V1') for f in FAILS),
                 'n_rows_per_method': n_align, 'max_true_p_diff': max_true,
                 'max_delta_resid': max_dres, 'ok_pairs_per_method': ok_pairs,
                 'n_pairs_per_method': {m: sum(1 for k in frozen_pairs if k[0] == m)
                                        for m in METHODS}}
    print('[verify] V1 ok=%s max_true=%.2e max_dres=%.2e ok_pairs=%s'
          % (doc['v1']['ok'], max_true, max_dres, ok_pairs))

    # ---------- V2/V3/V4/V5:独立重算全部配对×区间 ----------
    maxes = {'v2_D': 0.0, 'v2_cost': 0.0, 'v2_mse': 0.0, 'v2_ddiff': 0.0, 'v3': 0.0,
             'v4_ident': 0.0, 'v4_direct': 0.0, 'v5': 0.0, 'v5_gap': 0.0}
    n2 = n2d = n3 = n4 = n5 = 0
    n2miss = 0
    full_vs_avg = 0.0
    yp_cache = {}

    def yp_of(p):
        key = (p['ds'], p['sid'])
        if key not in yp_cache:
            yp_cache[key] = {}
        d = yp_cache[key]
        if p['di'] not in d:
            w = w_of[p['ds']][p['sid']]
            d[p['di']] = base[p['ds']]['y'][w] + p['delta']
        return d[p['di']]

    for (sid, di, mi), p in sorted(pidx.items()):
        yp = yp_of(p)
        for m in METHODS:
            fr = frozen_pairs.get((m, sid, di, mi))
            if fr is None:
                continue
            ok = fr['ok_A'] == '1' and fr['ok_B'] == '1'
            pa = preds[m][rg_pos[m][p['rg_a']]]
            pb = preds[m][rg_pos[m][p['rg_b']]]
            rec = {}
            for bname, lo, hi in INTERVALS:
                rec[bname] = recompute(pa, pb, yp, lo, hi, p['delta']) if ok else None
                row = ph.get((m, sid, di, mi, bname))
                if row is None:
                    fail('V2 missing horizon row %s %d %d %d %s' % (m, sid, di, mi, bname))
                    continue
                if not ok:
                    for q in ('mse_A', 'mse_B', 'D', 'cost', 'level_A', 'level_B',
                              'centered_A', 'centered_B', 'bias_A', 'bias_B',
                              'cost_level', 'cost_centered'):
                        if isfin(row[q]):
                            fail('V2 nan expected %s %d %d %d %s %s' % (m, sid, di, mi, bname, q))
                    continue
                for q, v in rec[bname].items():
                    if not isfin(row[q]):
                        fail('V2 finite expected %s %s %s' % (m, bname, q))
                        continue
                    dd = tol_diff(v, float(row[q]))
                    if dd > 0:
                        fail('V2 recompute vs CSV %s %s %s diff' % (m, bname, q))
                # 分解恒等式(逐区间,含 full):MSE = level + centered
                for side in ('A', 'B'):
                    ident = rec[bname]['mse_%s' % side] - (rec[bname]['level_%s' % side]
                                                           + rec[bname]['centered_%s' % side])
                    maxes['v4_ident'] = max(maxes['v4_ident'], abs(ident))
                icd = rec[bname]['cost'] - (rec[bname]['level_B'] - rec[bname]['level_A']
                                            + rec[bname]['centered_B'] - rec[bname]['centered_A'])
                maxes['v4_ident'] = max(maxes['v4_ident'], abs(ident), abs(icd))
                if abs(ident) > TOL_ATOL or abs(icd) > TOL_ATOL:
                    fail('V4 identity broken %s %d %d %d %s' % (m, sid, di, mi, bname))
            if ok:
                # V2:mse ↔ per_sample_metrics(full)
                for rg, side in ((p['rg_a'], 'A'), (p['rg_b'], 'B')):
                    mv = float(met[rg]['mse_%s' % m])
                    dd = tol_diff(rec['full']['mse_%s' % side], mv)
                    maxes['v2_mse'] = max(maxes['v2_mse'], abs(rec['full']['mse_%s' % side] - mv))
                    if dd > 0:
                        fail('V2 mse vs per_sample %s rg=%d' % (m, rg))
                # V2:D/cost ↔ 冻结 pairs(finite 处)
                fq = float(fr['D'])
                if fq == fq:
                    dd = tol_diff(rec['full']['D'], fq)
                    maxes['v2_D'] = max(maxes['v2_D'], abs(rec['full']['D'] - fq))
                    if dd > 0:
                        fail('V2 D vs frozen %s %d %d %d' % (m, sid, di, mi))
                fc = float(fr['cost'])
                if fc == fc:
                    dd = tol_diff(rec['full']['cost'], fc)
                    maxes['v2_cost'] = max(maxes['v2_cost'], abs(rec['full']['cost'] - fc))
                    if dd > 0:
                        fail('V2 cost vs frozen %s %d %d %d' % (m, sid, di, mi))
                # V3:full = 四 band 算术平均
                for q in ('mse_A', 'mse_B', 'D', 'cost'):
                    avg4 = sum(rec[b[0]][q] for b in BANDS) / 4.0
                    maxes['v3'] = max(maxes['v3'], abs(rec['full'][q] - avg4))
                    if tol_diff(rec['full'][q], avg4) > 0:
                        fail('V3 full != band-avg %s %d %d %d %s' % (m, sid, di, mi, q))
                # V4 信息量:full level 直接重算 vs 四 band 平均(应不同,仅记录)
                lv_direct = rec['full']['level_B']
                lv_avg = sum(rec[b[0]]['level_B'] for b in BANDS) / 4.0
                full_vs_avg = max(full_vs_avg, abs(lv_direct - lv_avg))
                n3 += 1
            n2 += 1
        # V2:diff ↔ 冻结 method_diff(对照对)
        for (mv, mc) in CONTRASTS:
            pair = '%s_vs_%s' % (mv, mc)
            frv = frozen_pairs.get((mv, sid, di, mi))
            frc = frozen_pairs.get((mc, sid, di, mi))
            fd = frozen_diff.get((pair, sid, di, mi))
            row_f = dh.get((pair, sid, di, mi, 'full'))
            if frc is None:
                continue  # 对照方法子集外(AutoARIMA 范围):不要求 horizon diff 行
            if row_f is None:
                fail('V2 missing horizon diff row %s %d %d %d' % (pair, sid, di, mi))
                continue
            if fd is None:
                n2miss += 1
                for q in QUANT_DIFF:
                    if isfin(row_f[q]):
                        fail('V2 nan expected diff %s %d %d %d %s' % (pair, sid, di, mi, q))
                continue
            okboth = (frv['ok_A'] == '1' and frv['ok_B'] == '1'
                      and frc['ok_A'] == '1' and frc['ok_B'] == '1')
            if not okboth:
                continue
            rv = recompute(preds[mv][rg_pos[mv][p['rg_a']]], preds[mv][rg_pos[mv][p['rg_b']]],
                           yp, 0, H, p['delta'])
            rc = recompute(preds[mc][rg_pos[mc][p['rg_a']]], preds[mc][rg_pos[mc][p['rg_b']]],
                           yp, 0, H, p['delta'])
            exp = {'D_diff': rv['D'] - rc['D'], 'gap_A': rv['mse_A'] - rc['mse_A'],
                   'gap_B': rv['mse_B'] - rc['mse_B'],
                   'cost_diff': (rv['mse_B'] - rv['mse_A']) - (rc['mse_B'] - rc['mse_A']),
                   'cost_level_diff': (rv['level_B'] - rv['level_A']) - (rc['level_B'] - rc['level_A']),
                   'cost_centered_diff': (rv['centered_B'] - rv['centered_A']) - (rc['centered_B'] - rc['centered_A'])}
            for q, v in exp.items():
                dd = tol_diff(v, float(row_f[q]))
                mx = 'v2_ddiff' if q == 'D_diff' else 'v5_gap'
                maxes[mx] = max(maxes[mx], abs(v - float(row_f[q])))
                if dd > 0:
                    fail('V2 diff vs frozen %s %d %d %d %s' % (pair, sid, di, mi, q))
            n2d += 1

    # V5:cost_diff = gap_B − gap_A(逐行,全区间)
    for (pair, sid, di, mi, band), r in dh.items():
        if not (r['ok'] == '1' and isfin(r['cost_diff'])):
            continue
        d5 = float(r['cost_diff']) - (float(r['gap_B']) - float(r['gap_A']))
        maxes['v5'] = max(maxes['v5'], abs(d5))
        if tol_diff(float(r['cost_diff']), float(r['gap_B']) - float(r['gap_A'])) > 0:
            fail('V5 identity %s %d %d %d %s' % (pair, sid, di, mi, band))
        n5 += 1
    doc['v2'] = {'ok': not any(f.startswith('V2') for f in FAILS),
                 'n_pair_cells': n2, 'n_diff_cells': n2d, 'n_frozen_absent': n2miss,
                 'max_abs': {k: v for k, v in maxes.items() if k.startswith('v2')}}
    doc['v3'] = {'ok': not any(f.startswith('V3') for f in FAILS), 'n': n3,
                 'max_abs_diff_full_vs_bandavg': maxes['v3']}
    doc['v4'] = {'ok': not any(f.startswith('V4') for f in FAILS),
                 'max_identity_resid': maxes['v4_ident'],
                 'full_level_direct_vs_bandavg_max_abs': full_vs_avg,
                 'note': 'full level/centered 由 96 步直接重算(非四段平均);'
                         'direct-vs-bandavg 差仅作信息记录'}
    doc['v5'] = {'ok': not any(f.startswith('V5') for f in FAILS), 'n': n5,
                 'max_abs_identity_resid': maxes['v5'],
                 'max_abs_gap_vs_recompute': maxes['v5_gap']}
    print('[verify] V2 ok=%s V3 ok=%s max3=%.2e V4 ident=%.2e V5 ok=%s'
          % (doc['v2']['ok'], doc['v3']['ok'], maxes['v3'], maxes['v4_ident'], doc['v5']['ok']))

    # ---------- V6:汇总表独立复算 + fig_data 同源 ----------
    def collect_diff():
        out = {}
        for r in read_csv_dicts(os.path.join(out_dir, DIFF_H)):
            k = (r['pair'], r['ds'], int(float(r['m_over_P'])), r['band'])
            rec = out.setdefault(k, {'sid': set(), 'q': {q: [] for q in QUANT_DIFF}})
            if r['ok'] == '1':
                rec['sid'].add(int(r['sid_t46']))
                for q in QUANT_DIFF:
                    v = float(r[q])
                    if v == v:
                        rec['q'][q].append(v)
        return out

    agg = collect_diff()
    del agg
    max6, n6 = 0.0, 0
    rows_raw = read_csv_dicts(os.path.join(out_dir, DIFF_H))
    by_pmb = {}
    for r in rows_raw:
        by_pmb.setdefault((r['pair'], r['ds'], int(float(r['m_over_P'])), r['band']),
                          []).append(r)

    def cellstats(vals):
        v = np.asarray(vals, dtype=np.float64)
        return float(v.mean()), float(np.median(v)), (v > 0).mean()

    def diff_filter(sel, keycols, tr):
        if keycols == 'var':
            return [r for r in sel if r['var'] == tr['var']]
        if keycols == 'direction':
            want = tr['direction']
            return [r for r in sel if ('down' if float(r['delta_mult']) < 0 else 'up') == want]
        if keycols == 'amplitude':
            return [r for r in sel if '%.1f' % abs(float(r['delta_mult'])) == tr['amplitude']]
        return sel

    for fn, keycols in (('table_main_t47h.csv', None), ('table_var_t47h.csv', 'var'),
                        ('table_direction_t47h.csv', 'direction'),
                        ('table_amplitude_t47h.csv', 'amplitude')):
        for tr in read_csv_dicts(os.path.join(out_dir, fn)):
            sel = diff_filter(by_pmb.get((tr['pair'], tr['ds'],
                                          int(float(tr['m_over_P'])), tr['band']), []),
                              keycols, tr)
            for q in QUANT_DIFF:
                vals = [float(r[q]) for r in sel if r['ok'] == '1' and isfin(r[q])]
                if vals:
                    m_, md_, pr_ = cellstats(vals)
                    max6 = max(max6, abs(m_ - float(tr[q + '_mean'])),
                               abs(md_ - float(tr[q + '_median'])))
                    if tol_diff(m_, float(tr[q + '_mean'])) > 0 \
                            or tol_diff(md_, float(tr[q + '_median'])) > 0:
                        fail('V6 %s %s %s mean/median' % (fn, q, str(tr)[:60]))
                    if '%.4f' % pr_ != tr[q + '_pos_rate']:
                        fail('V6 %s %s pos_rate' % (fn, q))
                elif tr[q + '_mean'] != 'nan':
                    fail('V6 %s %s expected nan' % (fn, q))
            sids = {int(r['sid_t46']) for r in sel if r['ok'] == '1' and isfin(r['D_diff'])}
            nfin = len([r for r in sel if r['ok'] == '1' and isfin(r['D_diff'])])
            if int(tr['n_pairs']) != nfin or int(tr['n_base']) != len(sids):
                fail('V6 %s n_pairs/n_base' % fn)
            if int(tr['n_cand']) != len(sel):
                fail('V6 %s n_cand' % fn)
            n6 += 1

    # 方法表
    rows_p = read_csv_dicts(os.path.join(out_dir, PAIRS_H))
    for tr in read_csv_dicts(os.path.join(out_dir, 'table_method_t47h.csv')):
        sel = [r for r in rows_p if r['method'] == tr['method'] and r['ds'] == tr['ds']
               and int(float(r['m_over_P'])) == int(float(tr['m_over_P']))
               and r['band'] == tr['band'] and r['ok_A'] == '1' and r['ok_B'] == '1']
        for q in ('D', 'cost'):
            vals = [float(r[q]) for r in sel if isfin(r[q])]
            if vals:
                m_, md_, pr_ = cellstats(vals)
                max6 = max(max6, abs(m_ - float(tr[q + '_mean'])),
                           abs(md_ - float(tr[q + '_median'])))
                if tol_diff(m_, float(tr[q + '_mean'])) > 0 or tol_diff(md_, float(tr[q + '_median'])) > 0:
                    fail('V6 method %s %s mean/median' % (tr['method'], q))
                if '%.4f' % pr_ != tr[q + '_pos_rate']:
                    fail('V6 method %s %s pos_rate' % (q,))
            for q2, col in (('mse_A', 'mseA_mean'), ('mse_B', 'mseB_mean'),
                            ('level_B', 'levelB_mean'), ('centered_B', 'centeredB_mean')):
                vals2 = [float(r[q2]) for r in sel if isfin(r[q2])]
                if vals2:
                    max6 = max(max6, abs(float(np.mean(vals2)) - float(tr[col])))
                    if tol_diff(float(np.mean(vals2)), float(tr[col])) > 0:
                        fail('V6 method %s %s' % (tr['method'], col))
        n6 += 1

    # fig_data ↔ CSV 重算
    fd = np.load(os.path.join(out_dir, 'fig_data_t47h.npz'))
    figq = {'Ddiff': 'D_diff', 'costdiff': 'cost_diff', 'gapB': 'gap_B'}
    max6f = 0.0
    for tag, q in figq.items():
        for ds in sorted({r['ds'] for r in rows_raw}):
            for mp in (1, 2, 4):
                for bi, bid in enumerate(BAND_IDS):
                    v = [float(r[q]) for r in rows_raw if r['pair'] == 'vts_vs_moirai'
                         and r['ds'] == ds and int(float(r['m_over_P'])) == mp
                         and r['band'] == bid and r['ok'] == '1' and isfin(r[q])]
                    med = float(np.median(v)) if v else np.nan
                    got = float(fd['%s_%s_m%d_bands' % (tag, ds, mp)][bi])
                    if v:
                        max6f = max(max6f, abs(med - got))
                        if tol_diff(med, got) > 0:
                            fail('V6 figdata %s %s m%d %s' % (tag, ds, mp, bid))
                    elif got == got:
                        fail('V6 figdata nan expected %s %s m%d %s' % (tag, ds, mp, bid))
                    if int(fd['%s_%s_m%d_nb' % (tag, ds, mp)][bi]) != len(v):
                        fail('V6 figdata n %s %s m%d %s' % (tag, ds, mp, bid))
                vf = [float(r[q]) for r in rows_raw if r['pair'] == 'vts_vs_moirai'
                      and r['ds'] == ds and int(float(r['m_over_P'])) == mp
                      and r['band'] == 'full' and r['ok'] == '1' and isfin(r[q])]
                gotf = float(fd['%s_%s_m%d_full' % (tag, ds, mp)])
                if vf:
                    max6f = max(max6f, abs(float(np.median(vf)) - gotf))
                    if tol_diff(float(np.median(vf)), gotf) > 0:
                        fail('V6 figdata full %s %s m%d' % (tag, ds, mp))
                elif gotf == gotf:
                    fail('V6 figdata full nan expected %s %s m%d' % (tag, ds, mp))
    doc['v6'] = {'ok': not any(f.startswith('V6') for f in FAILS),
                 'n_table_rows': n6, 'max_abs_stat': max6, 'max_abs_figdata': max6f}
    print('[verify] V6 ok=%s rows=%d max_stat=%.2e max_fig=%.2e'
          % (doc['v6']['ok'], n6, max6, max6f))

    wrjson(doc, out_dir)
    if FAILS:
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
