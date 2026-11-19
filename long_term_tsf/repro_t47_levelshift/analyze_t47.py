"""EXP-047:配对分析与 §6 原始复核。

输入:per_sample_metrics_t47.csv + pred_*_t47.npz(f32→f64)+ baseline npz(f64)
     + t46 目录 features_t46.csv / per_sample_metrics_t46.csv(§6 复用)。

输出(save_t47_levelshift/):
- pairs_t47.csv           每 (sid,δ_idx,m_idx,method):D、cost、配对 ok
- method_diff_t47.csv     每 (sid,δ,m,对照方法):D_vts−D_c、cost_vts−cost_c(共同有效窗)
- table_D_cost_t47.csv    聚合(overall/ds/var/方向/幅度 × m × method,median+mean+n+D>0 占比)
- table_method_diff_t47.csv 同键聚合 ΔD/Δcost
- table_D_common_m.csv    "三 m 共同有效窗"子集的 D vs m(问题②口径)
- s6_recheck_t47.csv      1792 窗 a/b/c/s/类别 + t46 各方法误差复用列
- table_s6_t47.csv        类别 × (ds×var hist_std 三分位) × method 误差汇总
                          (scope=class_common = 四方法共同有效窗;各类 n 见该表)
- analysis_meta_t47.json  头条数字(报告由 gen_report_numbers 自动引用)

口径(variant_formulas.md §4/§5):
- D = mean_h(pred_A − pred_B)/δ(有符号,保留负值与 >1);D>0 = 冲突组向旧水平偏移;
- cost = MSE_B − MSE_A(同窗口配对);
- ΔD/Δcost 仅在两方法各自配对均 ok 的共同窗口上计算;
- 构造变体非独立样本,只作描述统计;§6 阈值(0.1/0.5)预先固定,不因结果调整。
"""
import csv
import json
import os
import sys

import numpy as np

from t47_common import M_MULTS, T46_DIR, manifest_path

METHODS = ['vts', 'moirai', 'autoets', 'autoarima']
CONTRASTS = ['moirai', 'autoets', 'autoarima']
PAIRS_VS = [('vts', c) for c in CONTRASTS]
STABLE_TH = 0.1
CHANGE_TH = 0.5
TERTILE_NAME = ['low', 'mid', 'high']


def read_csv_dicts(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def wr(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)


def agg_stats(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return dict(n=0, mean=None, median=None, pos_rate=None)
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)),
                pos_rate=float((v > 0).mean()))


def scope_keys(r):
    """行 → 各聚合口径的键(平行列表)。"""
    direction = 'down' if float(r['delta_mult']) < 0 else 'up'
    amplitude = '%.1f' % abs(float(r['delta_mult']))
    return {
        'overall': 'all',
        'ds': r['ds'],
        'var': r['var'],
        'direction': direction,
        'amplitude': amplitude,
    }


def main(repo_root, save_dir):
    with open(manifest_path()) as f:
        man = list(csv.DictReader(f))
    met = {int(r['row_global']): r for r in read_csv_dicts(
        os.path.join(save_dir, 'per_sample_metrics_t47.csv'))}

    # npz preds(f64)
    preds, rg_idx = {}, {}
    for m in METHODS:
        z = np.load(os.path.join(save_dir, 'pred_%s_t47.npz' % m))
        preds[m] = z['pred'].astype(np.float64)
        rg_idx[m] = {int(rg): j for j, rg in enumerate(z['row_globals'])}
    base_cache = {}

    def baseline(ds):
        if ds not in base_cache:
            base_cache[ds] = np.load(os.path.join(save_dir, 'baseline_%s.npz' % ds))
        return base_cache[ds]

    # 配对索引
    A_rg, B_rg, meta_pair = {}, {}, {}
    for r in man:
        sid, di, cid = int(r['sid_t46']), int(r['delta_idx']), int(r['cond_id'])
        if r['group'] == 'A':
            A_rg[(sid, di)] = int(r['row_global'])
        else:
            B_rg[(sid, di, int(r['m_idx']))] = int(r['row_global'])
        meta_pair[int(r['row_global'])] = r

    def ok_of(m, rg):
        return met[rg]['ok_%s' % m] == '1'

    pair_rows = []
    D_map = {}   # (method, sid, di, mi) -> D
    cost_map = {}
    valid_map = {}
    for (sid, di), rg_a in sorted(A_rg.items()):
        ra = met[rg_a]
        for mi in range(3):
            rg_b = B_rg[(sid, di, mi)]
            rb = met[rg_b]
            assert ra['ds'] == rb['ds'] and ra['var'] == rb['var']
            delta = float(ra['delta'])
            assert delta == float(rb['delta']) and delta != 0.0
            for m in METHODS:
                if rg_a not in rg_idx[m] or rg_b not in rg_idx[m]:
                    continue  # autoarima 范围外
                ok = ok_of(m, rg_a) and ok_of(m, rg_b)
                if ok:
                    pa = preds[m][rg_idx[m][rg_a]]
                    pb = preds[m][rg_idx[m][rg_b]]
                    D = float((pa - pb).mean() / delta)
                    cost = float(rb['mse_%s' % m]) - float(ra['mse_%s' % m])
                else:
                    D = cost = float('nan')
                D_map[(m, sid, di, mi)] = D
                cost_map[(m, sid, di, mi)] = cost
                valid_map[(m, sid, di, mi)] = ok
                pair_rows.append([m, sid, ra['ds'], ra['var'], ra['channel'], di, mi,
                                  ra['delta_mult'], '%.12g' % delta, M_MULTS[mi],
                                  rg_a, rg_b, int(ok_of(m, rg_a)), int(ok_of(m, rg_b)),
                                  '%.12g' % D, '%.12g' % cost])
    wr(os.path.join(save_dir, 'pairs_t47.csv'),
       ['method', 'sid_t46', 'ds', 'var', 'channel', 'delta_idx', 'm_idx', 'delta_mult',
        'delta', 'm_over_P', 'rg_A', 'rg_B', 'ok_A', 'ok_B', 'D', 'cost'], pair_rows)

    # 方法间差(共同有效窗)
    diff_rows = []
    DD_map, dcost_map = {}, {}
    for (sid, di), rg_a in sorted(A_rg.items()):
        ra = met[rg_a]
        for mi in range(3):
            rg_b = B_rg[(sid, di, mi)]
            for mv, mc in PAIRS_VS:
                if not (valid_map.get((mv, sid, di, mi)) and valid_map.get((mc, sid, di, mi))):
                    continue
                DD = D_map[(mv, sid, di, mi)] - D_map[(mc, sid, di, mi)]
                dc = cost_map[(mv, sid, di, mi)] - cost_map[(mc, sid, di, mi)]
                DD_map[(mv, mc, sid, di, mi)] = DD
                dcost_map[(mv, mc, sid, di, mi)] = dc
                diff_rows.append(['%s_vs_%s' % (mv, mc), sid, ra['ds'], ra['var'],
                                  ra['channel'], di, mi, ra['delta_mult'], M_MULTS[mi],
                                  '%.12g' % DD, '%.12g' % dc])
    wr(os.path.join(save_dir, 'method_diff_t47.csv'),
       ['pair', 'sid_t46', 'ds', 'var', 'channel', 'delta_idx', 'm_idx', 'delta_mult',
        'm_over_P', 'D_diff', 'cost_diff'], diff_rows)

    # 聚合表(scope key 值一律入聚合键:ds/var/方向/幅度必须真分组,不得塌缩)
    def agg_table(rows, get_D, get_cost):
        acc = {}
        for r in rows:
            keys = scope_keys(r)
            meth = r.get('method', r.get('pair'))
            for scope, key in keys.items():
                acc.setdefault((scope, key, meth, float(r['m_over_P'])), []).append(r)
        out = []
        for k, rs in sorted(acc.items()):
            ds_v = agg_stats([get_D(r) for r in rs])
            cs_v = agg_stats([get_cost(r) for r in rs])
            out.append(list(k) + [ds_v['n'], ds_v['mean'], ds_v['median'], ds_v['pos_rate'],
                                  cs_v['mean'], cs_v['median']])
        return out

    pr = [{**{k: v for k, v in zip(
        ['method', 'sid_t46', 'ds', 'var', 'channel', 'delta_idx', 'm_idx', 'delta_mult',
         'delta', 'm_over_P', 'rg_A', 'rg_B', 'ok_A', 'ok_B', 'D', 'cost'], map(str, row))},
        'D': float(row[14]), 'cost': float(row[15])} for row in pair_rows]
    dr = [{**{k: v for k, v in zip(
        ['pair', 'sid_t46', 'ds', 'var', 'channel', 'delta_idx', 'm_idx', 'delta_mult',
         'm_over_P', 'D_diff', 'cost_diff'], map(str, row))},
        'D_diff': float(row[9]), 'cost_diff': float(row[10])} for row in diff_rows]

    rows1 = agg_table(pr, lambda r: r['D'], lambda r: r['cost'])
    wr(os.path.join(save_dir, 'table_D_cost_t47.csv'),
       ['scope', 'key', 'method', 'm_over_P', 'n', 'D_mean', 'D_median', 'D_pos_rate',
        'cost_mean', 'cost_median'], rows1)
    rows2 = agg_table(dr, lambda r: r['D_diff'], lambda r: r['cost_diff'])
    wr(os.path.join(save_dir, 'table_method_diff_t47.csv'),
       ['scope', 'key', 'pair', 'm_over_P', 'n', 'Ddiff_mean', 'Ddiff_median', 'pos_rate',
        'costdiff_mean', 'costdiff_median'], rows2)

    # 问题②口径:每方法"三个 m 全部有效"的基准窗子集上 D vs m
    common_rows = []
    for m in METHODS:
        sids_di = sorted({(sid, di) for (mm, sid, di, mi) in valid_map
                          if mm == m and valid_map[(mm, sid, di, mi)] and mi == 0})
        keep = [(sid, di) for (sid, di) in sids_di
                if all(valid_map.get((m, sid, di, mi)) for mi in range(3))]
        for mi in range(3):
            vals = [D_map[(m, sid, di, mi)] for (sid, di) in keep]
            cs = [cost_map[(m, sid, di, mi)] for (sid, di) in keep]
            dv, cv = agg_stats(vals), agg_stats(cs)
            common_rows.append([m, M_MULTS[mi], len(keep), dv['mean'], dv['median'],
                                dv['pos_rate'], cv['mean'], cv['median']])
    wr(os.path.join(save_dir, 'table_D_common_m.csv'),
       ['method', 'm_over_P', 'n_common_windows', 'D_mean', 'D_median', 'D_pos_rate',
        'cost_mean', 'cost_median'], common_rows)

    # ---------------- §6 原始复核(零推理;复用 t46 全 1792 窗) ----------------
    from t46_common import DS_CFG, DS_ORDER, H, hist_features, load_csv, sample_list_path, \
        std_all, train_scaler

    samples = {int(r['sample_id']): r for r in read_csv_dicts(sample_list_path())}
    feats = {int(r['sample_id']): r for r in read_csv_dicts(os.path.join(T46_DIR, 'features_t46.csv'))}
    t46m = {int(r['sample_id']): r for r in read_csv_dicts(
        os.path.join(T46_DIR, 'save_t46_histdiag', 'per_sample_metrics.csv'))}
    assert len(samples) == len(feats) == len(t46m) == 1792

    s6_rows, cache = [], {}
    for sid in range(1792):
        s, ft = samples[sid], feats[sid]
        assert s['ds'] == ft['ds'] and s['var'] == ft['var']
        ds, cfg = s['ds'], DS_CFG[s['ds']]
        if ds not in cache:
            data = load_csv(os.path.join(repo_root, cfg['csv']))
            mean, scale = train_scaler(data, cfg['K'])
            cache[ds] = std_all(data, mean, scale)
        std = cache[ds]
        start, ch, L, P = int(s['start_row']), int(s['channel']), cfg['L'], cfg['P']
        x = std[start - L:start, ch]
        a = float(x[L - 4 * P:L - P].mean())   # 3P 个点整体标量均值(用户确认口径)
        b = float(x[L - P:L].mean())
        c = float(std[start:start + H, ch].mean())
        sv = float(ft['hist_std'])
        stable = abs(b - a) <= STABLE_TH * sv and abs(c - b) <= STABLE_TH * sv
        changed = abs(b - a) >= CHANGE_TH * sv and abs(c - b) <= CHANGE_TH * abs(b - a)
        cls = 'stable' if stable else ('changed_persist' if changed else 'other')
        row = [sid, ds, s['var'], ch, start, '%.12g' % a, '%.12g' % b, '%.12g' % c,
               '%.12g' % sv, int(stable), int(changed), cls]
        t46row = t46m[sid]
        for m in METHODS:
            row += [t46row.get('ok_%s' % m, ''), t46row.get('mse_%s' % m, ''),
                    t46row.get('mae_%s' % m, '')]
        s6_rows.append(row)

    # hist_std 三分位(t46 口径:每 (ds×var) 非近常数 64 窗内 1/3、2/3 分位)
    tert = {}
    for ds in DS_ORDER:
        for v in dict.fromkeys(r[2] for r in s6_rows if r[1] == ds):
            grp = [r for r in s6_rows if r[1] == ds and r[2] == v
                   and feats[r[0]]['near_constant'] == 'False']
            vals = [float(r[8]) for r in grp]
            q1, q2 = np.quantile(vals, 1 / 3), np.quantile(vals, 2 / 3)
            for r in grp:
                sv = float(r[8])
                tert[r[0]] = TERTILE_NAME[0] if sv <= q1 else (TERTILE_NAME[1] if sv <= q2
                                                               else TERTILE_NAME[2])
    wr(os.path.join(save_dir, 's6_recheck_t47.csv'),
       ['sample_id', 'ds', 'var', 'channel', 'start_row', 'a', 'b', 'c', 's',
        'stable', 'changed_persist', 'class']
       + [x for m in METHODS for x in ('ok_%s' % m, 'mse_%s' % m, 'mae_%s' % m)]
       + ['hist_tertile'],
       [r + [tert[r[0]]] for r in s6_rows])

    acc = {}
    for r in s6_rows:
        keys = (('overall', '', ''), ('ds', r[1], ''), ('class', r[11], ''),
                ('class_tert', r[11], tert[r[0]]))
        vals = {}
        for m in METHODS:
            try:
                ok = int(r[12 + 3 * METHODS.index(m)]) == 1
                mse = float(r[13 + 3 * METHODS.index(m)])
            except (ValueError, IndexError):
                continue
            if ok and np.isfinite(mse):
                vals[m] = mse
                for key in keys:
                    acc.setdefault((m,) + key, []).append(mse)
        if len(vals) == len(METHODS):  # 四方法共同有效窗(类级公平对比,ARIMA 覆盖少于全类)
            for m, mse in vals.items():
                acc.setdefault((m, 'class_common', r[11], ''), []).append(mse)
    s6_tab = []
    for key in sorted(acc):
        v = acc[key]
        s6_tab.append([key[0], key[1], key[2], key[3], len(v),
                       '%.12g' % float(np.mean(v)), '%.12g' % float(np.median(v))])
    wr(os.path.join(save_dir, 'table_s6_t47.csv'),
       ['method', 'scope', 'key1', 'key2', 'n_ok', 'mse_mean', 'mse_median'], s6_tab)

    meta = dict(n_pairs_per_method=len(A_rg) * 3, n_diff_pairs=len(diff_rows) // len(PAIRS_VS),
                stable_th=STABLE_TH, change_th=CHANGE_TH,
                s6_n=len(s6_rows),
                s6_class_counts={c: sum(1 for r in s6_rows if r[11] == c)
                                 for c in ('stable', 'changed_persist', 'other')})
    with open(os.path.join(save_dir, 'analysis_meta_t47.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('analyze done: pairs=%d diff=%d s6=%d classes=%s'
          % (len(pair_rows), len(diff_rows), len(s6_rows), meta['s6_class_counts']))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_t47_levelshift'))
