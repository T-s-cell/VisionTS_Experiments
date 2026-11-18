"""EXP-046:收尾门禁(全部 PASS 才算收官,任一 FAIL 非零退出)。

G1 样本清单:sample_list_sha.txt 与文件实际 sha 一致;按 t46_common.iter_samples
   重生成内容与文件逐字节一致(确定性重放);
G2 真值/对齐:四方法各 ds 数组内 true vs csv 标准化 f64 重算 maxdiff≤TRUTH_TOL,
   覆盖全部 1792 样本;item_ids/start_rows/arr_index 与清单逐行一致;历史窗不越界;
G3 覆盖率与台账:metrics csv 每方法 n_ok 与 collect_meta 一致;sfdx 两方法
   ledger (sid,status) 多重集 == npz status 数组;ledger 无重复样本行;
G4 表格重算:overall_t46.csv 由 per_sample_metrics.csv 独立重算,值一致(≤1e-9);
G5 有限性:各方法 ok 样本预测全有限;失败样本 pred 行不参与指标(NaN);
G6 item_id↔起点:moirai item_id↔start_rows 与清单逐行核对(G2 内执行,此处单列报告)。
G7 特征与分箱:features_t46.csv 由 raw csv 逐样本复算(≤1e-8);每 (ds×var×特征)
   三分位分箱计数由 features 值复算,与 coverage_by_group n_cell 一致;
G8 方向重算:direction_summary / direction_by_var 两表由 table_tertile 独立重算
   (计数精确相等,中位差 ≤1e-9);
G9 案例重算:cases_t46.csv 由 per_sample_metrics + 修复后规则(VTS=mse4[0])重算,
   id 序列精确一致、margin 一致;
G10 幅度分层重算:amplitude_strat_t46.csv 每行 n/rel_gap/delta 独立重算(≤1e-9)。
输出 verify_t46.json;打印各门禁结果。
"""
import csv
import io
import json
import os
import sys
from collections import Counter

import numpy as np

from t46_common import (CHANNELS, DS_CFG, DS_ORDER, H, TRUTH_TOL, iter_samples,
                        get_window, hist_features, load_csv, sample_list_path, sha256_file,
                        std_all, train_scaler)

FEATURES = ['period_corr', 'recent_old', 'trend', 'vol', 'hist_std']
TERTILE_NAME = ['low', 'mid', 'high']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

METHODS = ['vts', 'moirai', 'autoets', 'autoarima']


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main(repo_root, save_dir):
    gates = {}

    # ---------- G1 ----------
    list_path = sample_list_path()
    sha_path = os.path.join(os.path.dirname(list_path), 'sample_list_sha.txt')
    sha_actual = sha256_file(list_path)
    with open(sha_path) as f:
        sha_recorded = f.read().strip().split()[0]
    g1_sha = sha_actual == sha_recorded
    with open(list_path, newline='') as f:
        content = f.read()
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\r\n')  # 与 select_t46.py csv.writer 默认一致
    w.writerow(['sample_id', 'ds', 'seg', 'channel', 'var', 'start_row', 'arr_index', 'L', 'P'])
    for sid, ds, seg, ch, start, arr_index in iter_samples():
        cfg = DS_CFG[ds]
        w.writerow([sid, ds, seg, ch, CHANNELS[ch], start, arr_index, cfg['L'], cfg['P']])
    g1_replay = content == buf.getvalue()
    gates['G1_sample_list'] = dict(sha_match=g1_sha, replay_match=g1_replay,
                                   sha=sha_actual)
    assert g1_sha and g1_replay, 'G1 FAIL'

    samples = rows_of(list_path)
    assert len(samples) == 1792
    cache = {}

    def std_of(ds):
        if ds not in cache:
            cfg = DS_CFG[ds]
            data = load_csv(os.path.join(repo_root, cfg['csv']))
            assert data.shape[0] == cfg['csv_rows']
            mean, scale = train_scaler(data, cfg['K'])
            cache[ds] = std_all(data, mean, scale)
        return cache[ds]

    # ---------- G2/G6 ----------
    g2 = {}
    g6 = True
    for ds in DS_ORDER:
        cfg = DS_CFG[ds]
        ss = [s for s in samples if s['ds'] == ds]
        std = std_of(ds)
        true64 = np.stack([std[int(s['start_row']):int(s['start_row']) + H, int(s['channel'])]
                           for s in ss])
        arrs = {}
        zv = np.load(os.path.join(save_dir, 'arr_vts_t46.npz'))
        arrs['vts'] = (zv[f'{ds}_pred'], zv[f'{ds}_true'], None)
        zm = np.load(os.path.join(save_dir, f'arr_moirai_{ds}.npz'))
        g6 &= (list(zm['item_ids']) == [int(s['sample_id']) for s in ss]
               and list(zm['start_rows']) == [int(s['start_row']) for s in ss])
        arrs['moirai'] = (zm['pred'], zm['true'], None)
        for name in ('AutoETS', 'AutoARIMA'):
            zs = np.load(os.path.join(save_dir, f'{name}_{ds}.npz'))
            arrs[name.lower()] = (zs['pred'], zs['true'], zs['status'])
        dmax = 0.0
        for s in ss:
            start, L = int(s['start_row']), cfg['L']
            assert 12 * cfg['K'] <= start <= 16 * cfg['K'] - H and start - L >= 0, s
            assert int(s['arr_index']) == start - 12 * cfg['K'], s
        for m, (pred, tstored, _st) in arrs.items():
            assert pred.shape == (len(ss), H) and pred.dtype == np.float32, (ds, m, pred.shape, pred.dtype)
            dmax = max(dmax, float(np.abs(tstored.astype(np.float64) - true64).max()))
        g2[ds] = dict(truth_max_diff=dmax, ok=dmax <= TRUTH_TOL, n=len(ss))
        assert g2[ds]['ok'], ('G2 FAIL', ds, dmax)
    gates['G2_truth_alignment'] = g2
    gates['G6_item_id_start_rows'] = bool(g6)
    assert g6, 'G6 FAIL'

    # ---------- G3 ----------
    mets = rows_of(os.path.join(save_dir, 'per_sample_metrics.csv'))
    assert len(mets) == 1792
    with open(os.path.join(save_dir, 'collect_meta.json')) as f:
        cmeta = json.load(f)
    g3 = dict(per_method={})
    for m in METHODS:
        n_ok_csv = sum(1 for r in mets if r[f'ok_{m}'] == '1')
        n_ok_meta = sum(v['methods'][m]['n_ok'] for v in cmeta['per_ds'].values())
        g3['per_method'][m] = dict(n_ok_csv=n_ok_csv, n_ok_meta=n_ok_meta, ok=n_ok_csv == n_ok_meta)
        assert n_ok_csv == n_ok_meta, ('G3 FAIL', m)
    ledger_path = os.path.join(save_dir, 'failure_ledger.csv')
    led = rows_of(ledger_path)
    led_count = Counter((r['sample_id'], r['method'], r['status']) for r in led if r['method'] in ('AutoETS', 'AutoARIMA'))
    assert all(c == 1 for c in led_count.values()), 'G3 FAIL: duplicate ledger rows'
    for ds in DS_ORDER:
        ss = {int(s['sample_id']): s for s in samples if s['ds'] == ds}
        for name in ('AutoETS', 'AutoARIMA'):
            zs = np.load(os.path.join(save_dir, f'{name}_{ds}.npz'))
            npz_ms = Counter((int(i), str(st)) for i, st in zip(zs['item_ids'], zs['status']))
            csv_ms = Counter((int(sid), st) for (sid, meth, st), c in led_count.items()
                             if meth == name and int(sid) in ss)
            assert npz_ms == csv_ms, ('G3 FAIL ledger vs npz', ds, name)
    gates['G3_coverage_ledger'] = g3

    # ---------- G4 ----------
    ov = rows_of(os.path.join(save_dir, 'overall_t46.csv'))
    sids_all = [int(r['sample_id']) for r in mets]
    ds_of = {int(r['sample_id']): r['ds'] for r in mets}
    g4_max = 0.0
    for r in ov:
        ds, m = r['ds'], r['method']
        sel = [int(r2['sample_id']) for r2 in mets if ds == 'ALL' or ds_of[int(r2['sample_id'])] == ds]
        mse = [float(r2[f'mse_{m}']) for r2 in mets
               if (ds == 'ALL' or ds_of[int(r2['sample_id'])] == ds) and r2[f'ok_{m}'] == '1']
        recomputed = float(np.mean(mse)) if mse else None
        recorded = r['mse_own_ok']
        if recomputed is None or recorded == '':
            assert recomputed is None and recorded == '', ('G4 FAIL', ds, m)
        else:
            g4_max = max(g4_max, abs(recomputed - float(recorded)) / (abs(recomputed) + 1e-30))
    assert g4_max <= 1e-9, ('G4 FAIL', g4_max)
    gates['G4_table_recompute'] = dict(max_rel_diff=g4_max)

    # ---------- G5 ----------
    g5 = {}
    for ds in DS_ORDER:
        ss = [s for s in samples if s['ds'] == ds]
        mets_ds = {int(r['sample_id']): r for r in mets if r['ds'] == ds}
        zv = np.load(os.path.join(save_dir, 'arr_vts_t46.npz'))
        zm = np.load(os.path.join(save_dir, f'arr_moirai_{ds}.npz'))
        arrs = {'vts': zv[f'{ds}_pred'], 'moirai': zm['pred']}
        for name in ('AutoETS', 'AutoARIMA'):
            arrs[name.lower()] = np.load(os.path.join(save_dir, f'{name}_{ds}.npz'))['pred']
        for m, pred in arrs.items():
            ok_idx = [j for j, s in enumerate(ss) if mets_ds[int(s['sample_id'])][f'ok_{m}'] == '1']
            ok = pred[ok_idx].astype(np.float64)
            assert np.isfinite(ok).all(), ('G5 FAIL', ds, m)
            g5[f'{ds}/{m}'] = int(len(ok_idx))
    gates['G5_finiteness'] = dict(n_ok_checked=g5, ok=True)

    # ---------- G7 特征复算 + 三分位分箱复算 ----------
    feats_csv = rows_of(os.path.join(SCRIPT_DIR, 'features_t46.csv'))
    assert len(feats_csv) == 1792
    fc = {int(r['sample_id']): r for r in feats_csv}
    nearc = {sid: fc[sid]['near_constant'] == 'True' for sid in fc}
    g7_feat_max = 0.0
    for ds in DS_ORDER:
        cfg = DS_CFG[ds]
        std = std_of(ds)
        for s in samples:
            if s['ds'] != ds:
                continue
            sid = int(s['sample_id'])
            start, ch, P = int(s['start_row']), int(s['channel']), cfg['P']
            x = get_window(std, ds, start, ch)
            f = hist_features(x, P)
            r = fc[sid]
            assert r['ds'] == ds and r['var'] == CHANNELS[ch], ('G7 FAIL row', sid)
            assert (r['near_constant'] == 'True') == f['near_constant'], ('G7 FAIL nc', sid)
            for k in ('hist_std', 'period_corr', 'recent_old', 'trend', 'vol'):
                if f['near_constant'] and k != 'hist_std':
                    continue
                g7_feat_max = max(g7_feat_max, abs(float(r[k]) - f[k]))
    assert g7_feat_max <= 1e-8, ('G7 FAIL features', g7_feat_max)
    cov = rows_of(os.path.join(save_dir, 'coverage_by_group_t46.csv'))
    cov_n = {(r['ds'], r['var'], r['feature'], r['tertile']): int(r['n_cell'])
             for r in cov if r['method'] == 'vts'}
    fv2 = {}
    for r in feats_csv:
        if r['near_constant'] == 'True':
            continue
        fv2.setdefault((r['ds'], r['var']), []).append(r)
    for (ds, var), rs in fv2.items():
        assert len(rs) == 64, ('G7 FAIL cell size', ds, var, len(rs))
        for feat in FEATURES:
            v = np.array([float(r[feat]) for r in rs])
            q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
            counts = [int((v < q1).sum()), int(((v >= q1) & (v < q2)).sum()), int((v >= q2).sum())]
            for t, name in enumerate(TERTILE_NAME):
                assert cov_n[(ds, var, feat, name)] == counts[t], \
                    ('G7 FAIL tertile', ds, var, feat, name, cov_n[(ds, var, feat, name)], counts[t])
    gates['G7_features_bin_recompute'] = dict(feature_max_abs_diff=g7_feat_max, ok=True)

    # ---------- G8 方向统计重算(自 table_tertile 独立重算两表) ----------
    tert = rows_of(os.path.join(save_dir, 'table_tertile_t46.csv'))
    dmap = {}
    for r in tert:
        if r['rel_gap'] != '':
            dmap.setdefault((r['feature'], r['pair']), {})[
                (r['ds'], r['var'], TERTILE_NAME.index(r['tertile']))] = float(r['rel_gap'])
    g8_max = 0.0
    for r in rows_of(os.path.join(save_dir, 'direction_summary_t46.csv')):
        d = dmap[(r['feature'], r['pair'])]
        deltas, n_pos, n_neg, n_mono = [], 0, 0, 0
        for ds in DS_ORDER:
            for var in CHANNELS:
                lo, mi, hi = (d.get((ds, var, t)) for t in (0, 1, 2))
                if None in (lo, mi, hi):
                    continue
                n_pos += hi > lo
                n_neg += hi < lo
                n_mono += lo <= mi <= hi
                deltas.append(hi - lo)
        assert (len(deltas) == int(r['n_cells_valid']) and n_pos == int(r['n_high_gt_low'])
                and n_neg == int(r['n_low_gt_high'])
                and n_mono == int(r['n_monotone_nondecreasing'])), ('G8 FAIL', r['feature'], r['pair'])
        g8_max = max(g8_max, abs(float(np.median(deltas)) - float(r['median_rel_gap_high_minus_low'])))
    for r in rows_of(os.path.join(save_dir, 'direction_by_var_t46.csv')):
        d = dmap[(r['feature'], r['pair'])]
        deltas = [d[(ds, r['var'], 2)] - d[(ds, r['var'], 0)] for ds in DS_ORDER
                  if (ds, r['var'], 0) in d and (ds, r['var'], 2) in d]
        assert (len(deltas) == int(r['n_ds_valid'])
                and sum(x > 0 for x in deltas) == int(r['n_ds_high_gt_low'])
                and sum(x < 0 for x in deltas) == int(r['n_ds_low_gt_high'])), ('G8 FAIL var', r)
        g8_max = max(g8_max, abs(float(np.median(deltas)) - float(r['median_delta_across_ds'])))
    assert g8_max <= 1e-9, ('G8 FAIL', g8_max)
    gates['G8_direction_recompute'] = dict(max_abs_diff=g8_max, ok=True)

    # ---------- G9 案例重算(修复后规则:mse4[0] = VisionTS) ----------
    cases_csv = rows_of(os.path.join(save_dir, 'cases_t46.csv'))
    met_r = {int(r['sample_id']): r for r in mets}
    exp_rows = []
    exp_margin = {}
    for ds in DS_ORDER:
        cands = [sid for sid, r in met_r.items() if r['ds'] == ds
                 and all(r[f'ok_{m}'] == '1' for m in METHODS) and not nearc[sid]]
        better, worse = [], []
        for sid in cands:
            mse4 = [float(met_r[sid][f'mse_{m}']) for m in METHODS]
            order = np.argsort(mse4, kind='stable')
            rank_v = int(np.where(order == 0)[0][0])
            if rank_v == 0 and mse4[0] < 0.8 * mse4[order[1]]:
                better.append((mse4[order[1]] / mse4[0], sid))
            if rank_v == 3 and mse4[0] > 1.25 * mse4[order[2]]:
                worse.append((mse4[0] / mse4[order[2]], sid))
        for cls, lst in (('better', better), ('worse', worse)):
            lst.sort(key=lambda x: (-x[0], x[1]))
            for margin, sid in lst[:2]:
                exp_rows.append((cls, sid))
                exp_margin[(cls, sid)] = '%.4f' % margin
    got_rows = [(r['case'], int(r['sample_id'])) for r in cases_csv]
    assert exp_rows == got_rows, ('G9 FAIL ids', exp_rows, got_rows)
    for r in cases_csv:
        assert exp_margin[(r['case'], int(r['sample_id']))] == r['margin'], ('G9 FAIL margin', r)
    gates['G9_cases_recompute'] = dict(n_cases=len(got_rows), ok=True)

    # ---------- G10 幅度分层重算 ----------
    feats_f = {sid: {k: float(fc[sid][k]) for k in FEATURES} for sid in fc}
    var_of = {sid: fc[sid]['var'] for sid in fc}
    g10_max = 0.0
    for r in rows_of(os.path.join(save_dir, 'amplitude_strat_t46.csv')):
        ds, var, feat, ht = r['ds'], r['var'], r['feature'], r['hist_tertile']
        mv, mb = r['pair'].split('_vs_')
        ids = [sid for sid in fc if met_r[sid]['ds'] == ds and var_of[sid] == var and not nearc[sid]]
        hv = np.array([feats_f[sid]['hist_std'] for sid in ids])
        q1, q2 = np.quantile(hv, [1 / 3, 2 / 3])
        strat = [sid for sid in ids
                 if np.digitize(feats_f[sid]['hist_std'], [q1, q2]) == TERTILE_NAME.index(ht)]
        med = float(np.median([feats_f[sid][feat] for sid in ids]))
        per = {}
        for s2, sel in ((0, lambda sid: feats_f[sid][feat] < med),
                        (1, lambda sid: feats_f[sid][feat] >= med)):
            common = [sid for sid in strat if sel(sid)
                      and met_r[sid][f'ok_{mv}'] == '1' and met_r[sid][f'ok_{mb}'] == '1']
            if common:
                a = float(np.mean([float(met_r[sid][f'mse_{mv}']) for sid in common]))
                b = float(np.mean([float(met_r[sid][f'mse_{mb}']) for sid in common]))
                per[s2] = (len(common), (a - b) / (a + b + 1e-12))
            else:
                per[s2] = None
        for s2, nk, gk in ((0, 'n_feat_low', 'rel_gap_feat_low'),
                           (1, 'n_feat_high', 'rel_gap_feat_high')):
            if per[s2] is None:
                assert int(r[nk]) == 0 and r[gk] == '', ('G10 FAIL empty', r)
            else:
                assert per[s2][0] == int(r[nk]), ('G10 FAIL n', r)
                g10_max = max(g10_max, abs(per[s2][1] - float(r[gk])))
        if per[0] and per[1]:
            g10_max = max(g10_max, abs(per[1][1] - per[0][1] - float(r['delta'])))
    assert g10_max <= 1e-9, ('G10 FAIL', g10_max)
    gates['G10_amplitude_recompute'] = dict(max_abs_diff=g10_max, ok=True)

    with open(os.path.join(save_dir, 'verify_t46.json'), 'w') as f:
        json.dump(dict(all_pass=True, gates=gates), f, indent=1, default=str)
    print('verify: ALL GATES PASS')
    for k, v in gates.items():
        print(' ', k, 'ok' if not isinstance(v, dict) or 'ok' not in v else v['ok'])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'save_t46_histdiag')
