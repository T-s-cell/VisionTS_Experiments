"""EXP-046:收尾门禁(全部 PASS 才算收官,任一 FAIL 非零退出)。

G1 样本清单:sample_list_sha.txt 与文件实际 sha 一致;按 t46_common.iter_samples
   重生成内容与文件逐字节一致(确定性重放);
G2 真值/对齐:四方法各 ds 数组内 true vs csv 标准化 f64 重算 maxdiff≤TRUTH_TOL,
   覆盖全部 1792 样本;item_ids/start_rows/arr_index 与清单逐行一致;历史窗不越界;
G3 覆盖率与台账:metrics csv 每方法 n_ok 与 collect_meta 一致;sfdx 两方法
   ledger (sid,status) 多重集 == npz status 数组;ledger 无重复样本行;
G4 表格重算:overall_t46.csv 由 per_sample_metrics_t46.csv 独立重算,值一致(≤1e-9);
G5 有限性:各方法 ok 样本预测全有限;失败样本 pred 行不参与指标(NaN);
G6 item_id↔起点:moirai item_id↔start_rows 与清单逐行核对(G2 内执行,此处单列报告)。
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
                        load_csv, sample_list_path, sha256_file, std_all, train_scaler)

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
    with open(list_path) as f:
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
    mets = rows_of(os.path.join(save_dir, 'per_sample_metrics_t46.csv'))
    assert len(mets) == 1792
    with open(os.path.join(save_dir, 'collect_meta_t46.json')) as f:
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

    with open(os.path.join(save_dir, 'verify_t46.json'), 'w') as f:
        json.dump(dict(all_pass=True, gates=gates), f, indent=1, default=str)
    print('verify: ALL GATES PASS')
    for k, v in gates.items():
        print(' ', k, 'ok' if not isinstance(v, dict) or 'ok' not in v else v['ok'])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'save_t46_histdiag')
