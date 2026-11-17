"""EXP-046:由输入历史计算逐样本特征(不读任何预测)。"""
import csv
import os
import sys

import numpy as np

from t46_common import (DS_CFG, DS_ORDER, load_csv, sample_list_path, std_all,
                        train_scaler, hist_features)

FIELDS = ['sample_id', 'ds', 'channel', 'var', 'start_row', 'L', 'P',
          'hist_std', 'period_corr', 'recent_old', 'trend', 'vol', 'near_constant']


def main(repo_root):
    samples = list(csv.DictReader(open(sample_list_path())))
    assert len(samples) == 1792
    cache = {}
    rows = []
    for s in samples:
        ds = s['ds']
        if ds not in cache:
            data = load_csv(os.path.join(repo_root, DS_CFG[ds]['csv']))
            assert data.shape[0] == DS_CFG[ds]['csv_rows']
            mean, scale = train_scaler(data, DS_CFG[ds]['K'])
            cache[ds] = std_all(data, mean, scale)
        std = cache[ds]
        start = int(s['start_row'])
        ch = int(s['channel'])
        x = std[start - DS_CFG[ds]['L']:start, ch]
        assert len(x) == DS_CFG[ds]['L']
        feat = hist_features(x, DS_CFG[ds]['P'])
        rows.append([s['sample_id'], ds, ch, s['var'], start, s['L'], s['P']] +
                    ['%.10g' % feat[k] for k in ('hist_std', 'period_corr', 'recent_old', 'trend', 'vol')] +
                    [str(feat['near_constant'])])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features_t46.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    nc = sum(1 for r in rows if r[-1] == 'True')
    print('features rows=%d near_constant=%d -> %s' % (len(rows), nc, path))


if __name__ == '__main__':
    main(sys.argv[1])
