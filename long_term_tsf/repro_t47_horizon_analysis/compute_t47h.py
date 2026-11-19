"""EXP-047h compute:对齐核验 + 逐配对逐区间指标与误差分解(零推理)。

用法:$PY compute_t47h.py <T47输入根> <OUT输出目录>
步骤:
1. 记录 13 输入 + 本目录代码 SHA256(input_shas_t47h.json,先于一切输入读取);
2. 对齐核验:npz row_globals/item_ids/cond_ids ↔ manifest;true_p vs 重建真值
   (f64 y + manifest δ)max ≤1e-5;δ ↔ delta_mult·s ≤ 容差;冻结 pairs rg 一致;
3. 逐方法×配对×区间(b1–b4+full):MSE_A/B、D、cost、bias、level、centered、
   cost_level、cost_centered(f64;真值 = baseline f64 y + manifest δ,npz true_p 仅核对);
4. 逐对照×配对×区间:D_diff、gap_A、gap_B、cost_diff、cost_level_diff、
   cost_centered_diff(共同有效 ok,失效写 nan,行保留);
5. compute_meta_t47h.json(实测最大差/计数/耗时)。

有效性沿用冻结 pairs_t47.csv 的 ok_A/ok_B,不重判。
"""
import csv
import json
import os
import sys
import time

import numpy as np

from t47h_common import (BANDS, CONTRASTS, DIFF_H, H, INTERVALS, METHODS, PAIRS_H, SAVE_SUB,
                         TRUE_TOL, input_path, load_manifest, pair_index, read_csv_dicts,
                         record_shas, tol_diff)

PAIR_FIELDS = ['method', 'sid_t46', 'ds', 'var', 'channel', 'delta_idx', 'm_idx',
               'delta_mult', 'delta', 'm_over_P', 'rg_A', 'rg_B', 'ok_A', 'ok_B', 'band',
               'mse_A', 'mse_B', 'D', 'cost', 'bias_A', 'bias_B', 'level_A', 'level_B',
               'centered_A', 'centered_B', 'cost_level', 'cost_centered']
DIFF_FIELDS = ['pair', 'sid_t46', 'ds', 'var', 'channel', 'delta_idx', 'm_idx',
               'delta_mult', 'delta', 'm_over_P', 'rg_A', 'rg_B', 'ok', 'band',
               'D_diff', 'gap_A', 'gap_B', 'cost_diff', 'cost_level_diff',
               'cost_centered_diff']


def g12(x):
    return '%.12g' % x


def wr(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)


def main(repo_root, out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    shas = record_shas(repo_root, out_dir)
    print('[compute] shas recorded: %d inputs, %d code files' % (len(shas['inputs']), len(shas['code'])))

    man = load_manifest(repo_root)
    pidx = pair_index(man)
    meta_by_rg = {int(r['row_global']): r for r in man}

    # ---- npz 装载与对齐核验 ----
    base_cache, w_of = {}, {}
    for ds in dict.fromkeys(p['ds'] for p in pidx.values()):
        bz = np.load(input_path(repo_root, '%s/baseline_%s.npz' % (SAVE_SUB, ds)))
        base_cache[ds] = bz
        w_of[ds] = {int(s): i for i, s in enumerate(bz['sid_t46'])}
    preds, rg_pos = {}, {}
    max_true_diff, max_delta_diff, max_rg_mismatch = 0.0, 0.0, 0
    for m in METHODS:
        z = np.load(input_path(repo_root, '%s/pred_%s_t47.npz' % (SAVE_SUB, m)))
        rg_pos[m] = {int(rg): k for k, rg in enumerate(z['row_globals'])}
        preds[m] = z['pred'].astype(np.float64)
        true_p = z['true_p'].astype(np.float64)
        n_align = 0
        for k, rg in enumerate(z['row_globals']):
            rg = int(rg)
            r = meta_by_rg.get(rg)
            assert r is not None, ('%s rg not in manifest' % m, rg)
            assert int(z['item_ids'][k]) == int(r['sid_t46']), ('%s item_id' % m, rg)
            assert int(z['cond_ids'][k]) == int(r['cond_id']), ('%s cond_id' % m, rg)
            n_align += 1
            sid, di = int(r['sid_t46']), int(r['delta_idx'])
            ds = r['ds']
            w = w_of[ds][sid]
            yp = base_cache[ds]['y'][w] + float(r['delta'])
            d = float(np.abs(yp - true_p[k]).max())
            if d > max_true_diff:
                max_true_diff = d
            bz = base_cache[ds]
            delta_resid = abs(float(r['delta']) - float(r['delta_mult']) * float(bz['s'][w]))
            max_delta_diff = max(max_delta_diff, delta_resid)
            if tol_diff(float(r['delta']), float(r['delta_mult']) * float(bz['s'][w])) > 0:
                raise SystemExit('[compute] delta vs delta_mult*s exceeds tol at rg=%d' % rg)
        print('[compute] %s: %d rows aligned' % (m, n_align))

    # 冻结 pairs:ok 标志沿用 + rg 一致性
    frozen = {}
    for r in read_csv_dicts(input_path(repo_root, '%s/pairs_t47.csv' % SAVE_SUB)):
        key = (r['method'], int(r['sid_t46']), int(r['delta_idx']), int(r['m_idx']))
        frozen[key] = r
        p = pidx[(int(r['sid_t46']), int(r['delta_idx']), int(r['m_idx']))]
        if int(r['rg_A']) != p['rg_a'] or int(r['rg_B']) != p['rg_b']:
            max_rg_mismatch += 1
    assert max_rg_mismatch == 0, max_rg_mismatch

    # 真值缓存:rg → y'(每 (ds,sid) 逐 δ 一次)
    yp_cache = {}

    def yp_of(p):
        key = (p['ds'], p['sid'], p['di'])
        if key not in yp_cache:
            w = w_of[p['ds']][p['sid']]
            yp_cache[key] = base_cache[p['ds']]['y'][w] + p['delta']
        return yp_cache[key]

    def seg(arr, lo, hi):
        return arr[lo:hi]

    # ---- 逐方法×配对×区间 ----
    pair_rows, pm = [], {}   # pm[(m,sid,di,mi)] = {band: (mse_A, mse_B, D, cost, level…)}
    for (sid, di, mi), p in sorted(pidx.items()):
        yp = yp_of(p)
        for m in METHODS:
            fk = frozen.get((m, sid, di, mi))
            if fk is None:
                continue  # autoarima 范围外
            ok_a, ok_b = fk['ok_A'] == '1', fk['ok_B'] == '1'
            pa = preds[m][rg_pos[m][p['rg_a']]]
            pb = preds[m][rg_pos[m][p['rg_b']]]
            e_a, e_b = pa - yp, pb - yp
            vals = {}
            for bname, lo, hi in INTERVALS:
                if not (ok_a and ok_b):
                    v = (float('nan'),) * 12
                else:
                    ea, eb = seg(e_a, lo, hi), seg(e_b, lo, hi)
                    dvec = seg(pa, lo, hi) - seg(pb, lo, hi)
                    mse_a = float(np.mean(ea * ea))
                    mse_b = float(np.mean(eb * eb))
                    dstat = float(np.mean(dvec)) / p['delta']
                    bias_a = float(np.mean(ea))
                    bias_b = float(np.mean(eb))
                    lev_a = bias_a * bias_a
                    lev_b = bias_b * bias_b
                    cen_a = float(np.mean((ea - bias_a) ** 2))
                    cen_b = float(np.mean((eb - bias_b) ** 2))
                    v = (mse_a, mse_b, dstat, mse_b - mse_a, bias_a, bias_b,
                         lev_a, lev_b, cen_a, cen_b, lev_b - lev_a, cen_b - cen_a)
                vals[bname] = v
                pair_rows.append([m, sid, p['ds'], p['var'], p['channel'], di, mi,
                                  g12(p['delta_mult']), g12(p['delta']), p['m_over_P'],
                                  p['rg_a'], p['rg_b'], int(ok_a), int(ok_b), bname]
                                 + [g12(x) for x in v])
            pm[(m, sid, di, mi)] = vals
    wr(os.path.join(out_dir, PAIRS_H), PAIR_FIELDS, pair_rows)

    # ---- 逐对照×配对×区间(共同有效窗,失效行保留写 nan)----
    diff_rows = []
    for (mv, mc) in CONTRASTS:
        for (sid, di, mi), p in sorted(pidx.items()):
            fk_v = frozen.get((mv, sid, di, mi))
            fk_c = frozen.get((mc, sid, di, mi))
            if fk_v is None or fk_c is None:
                continue
            ok = (fk_v['ok_A'] == '1' and fk_v['ok_B'] == '1'
                  and fk_c['ok_A'] == '1' and fk_c['ok_B'] == '1')
            vv, cv = pm[(mv, sid, di, mi)], pm[(mc, sid, di, mi)]
            for bname, lo, hi in INTERVALS:
                if ok:
                    va, vb = vv[bname], cv[bname]
                    d_diff = va[2] - vb[2]                    # D_vts − D_c
                    gap_a = va[0] - vb[0]                     # MSE_A_vts − MSE_A_c
                    gap_b = va[1] - vb[1]                     # MSE_B_vts − MSE_B_c
                    cost_diff = va[3] - vb[3]                 # = gap_B − gap_A
                    # v 元组索引:0 mse_A,1 mse_B,2 D,3 cost,6 level_A,7 level_B,
                    # 8 centered_A,9 centered_B,10 cost_level,11 cost_centered
                    cl_diff = va[10] - vb[10]
                    cc_diff = va[11] - vb[11]
                    row_vals = [d_diff, gap_a, gap_b, cost_diff, cl_diff, cc_diff]
                else:
                    row_vals = [float('nan')] * 6
                diff_rows.append(['%s_vs_%s' % (mv, mc), sid, p['ds'], p['var'], p['channel'],
                                  di, mi, g12(p['delta_mult']), g12(p['delta']), p['m_over_P'],
                                  p['rg_a'], p['rg_b'], int(ok), bname] + [g12(x) for x in row_vals])
    wr(os.path.join(out_dir, DIFF_H), DIFF_FIELDS, diff_rows)

    meta = {
        'n_pairs_rows': len(pair_rows), 'n_diff_rows': len(diff_rows),
        'n_pairs_per_method': {m: sum(1 for k in frozen if k[0] == m) for m in METHODS},
        'max_true_p_diff': max_true_diff, 'true_tol': TRUE_TOL,
        'max_delta_minus_mult_s': max_delta_diff,
        'frozen_rg_mismatches': max_rg_mismatch,
        'elapsed_s': time.time() - t0,
    }
    assert max_true_diff <= TRUE_TOL, max_true_diff
    with open(os.path.join(out_dir, 'compute_meta_t47h.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('[compute] done: %d pair rows, %d diff rows; max_true_p_diff=%.3e; %.1fs'
          % (len(pair_rows), len(diff_rows), max_true_diff, meta['elapsed_s']))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
