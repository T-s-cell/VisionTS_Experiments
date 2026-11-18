"""EXP-047 STEP 1:生成并冻结基准窗口清单(先于一切推理)。

每 (ds×var) 从 EXP-046 features_t46.csv 选 recent_old 最低 16 个
(排除 near_constant 与非有限;同值按 sample_id 升序)→ 4×7×16 = 448。
只用输入历史特征;不经误差/未来信息。重跑必须字节一致(sha256 断言)。
"""
import csv
import os
import sys

import numpy as np

from t47_common import CHANNELS, DS_ORDER, SCRIPT_DIR, T46_DIR, baseline_list_path, sha256_file
from t46_common import DS_CFG, sample_list_path

COLUMNS = ['w_global', 'sid_t46', 'ds', 'var', 'channel', 'start_row', 'arr_index', 'L', 'P',
           'hist_std_csv', 'recent_old_csv', 'rank']
N_PER_CELL = 16


def read_csv_dicts(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    samples = {int(r['sample_id']): r for r in read_csv_dicts(sample_list_path())}
    feats = read_csv_dicts(os.path.join(T46_DIR, 'features_t46.csv'))
    assert len(samples) == 1792 and len(feats) == 1792

    cells = {}
    for r in feats:
        sid = int(r['sample_id'])
        if r['near_constant'] == 'True':
            continue
        vals = {k: float(r[k]) for k in ('hist_std', 'recent_old')}
        if not all(np.isfinite(v) for v in vals.values()):
            continue
        s = samples[sid]
        assert s['ds'] == r['ds'] and s['var'] == r['var']
        cells.setdefault((r['ds'], r['var']), []).append(
            (vals['recent_old'], sid, vals['hist_std'], s))

    rows = []
    w = 0
    for ds in DS_ORDER:
        for ch, var in enumerate(CHANNELS):
            cand = sorted(cells[(ds, var)], key=lambda t: (t[0], t[1]))[:N_PER_CELL]
            assert len(cand) == N_PER_CELL, (ds, var, len(cand))
            cfg = DS_CFG[ds]
            for rank, (ro, sid, hs, s) in enumerate(cand):
                assert int(s['arr_index']) == int(s['start_row']) - 12 * cfg['K']
                rows.append([w, sid, ds, var, s['channel'], s['start_row'], s['arr_index'],
                             cfg['L'], cfg['P'], '%.10g' % hs, '%.10g' % ro, rank])
                w += 1
    assert w == 448, w

    path = baseline_list_path()
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(COLUMNS)
        wr.writerows(rows)

    sha = sha256_file(path)
    with open(os.path.join(SCRIPT_DIR, 'baseline_list_sha_t47.txt'), 'w') as f:
        f.write(sha + '\n')
    prev = path + '.prev'
    if os.path.exists(prev):
        assert open(prev).read().strip() == sha, 'baseline list changed across runs'
    else:
        with open(prev, 'w') as f:
            f.write(sha + '\n')

    by_ds = {}
    for r in rows:
        by_ds[r[2]] = by_ds.get(r[2], 0) + 1
    print('baseline_list rows=448 by_ds=%s sha=%s' % (by_ds, sha))


if __name__ == '__main__':
    main()
