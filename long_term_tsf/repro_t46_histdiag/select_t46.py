"""EXP-046 STEP 1:生成并冻结采样清单(先于一切预测运行)。

流程:合法预测起点([12K,16K-96])按时间 np.array_split 64 段,
np.random.default_rng(2021) 依 DS_ORDER 顺序每段取 1 起点;
每起点全 7 变量 → 448/数据集、1792 合计。
清单行序:ds → seg → channel(seg0 全 7 变量,再 seg1 …)。
不经误差/未来信息选样;重跑必须字节一致(sha256 断言)。
"""
import csv
import json
import os
import sys

from t46_common import (DS_CFG, DS_ORDER, CHANNELS, H, SAMPLE_LIST, iter_samples,
                        load_csv, sha256_file)

COLUMNS = ['sample_id', 'ds', 'seg', 'channel', 'var', 'start_row', 'arr_index', 'L', 'P']


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, SAMPLE_LIST)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for sid, ds, seg, ch, start, ai in iter_samples():
            w.writerow([sid, ds, seg, ch, CHANNELS[ch], start, ai, DS_CFG[ds]['L'], DS_CFG[ds]['P']])

    # 结构断言
    with open(path) as f:
        rows = list(csv.reader(f))[1:]
    assert len(rows) == 1792, len(rows)
    per = {}
    for r in rows:
        per.setdefault(r[1], []).append(r)
    assert sorted(per) == DS_ORDER and all(len(v) == 448 for v in per.values())
    for ds, rs in per.items():
        starts = sorted({int(r[5]) for r in rs})
        assert len(starts) == 64
        K = DS_CFG[ds]['K']
        assert starts[0] >= 12 * K and starts[-1] <= 16 * K - H
        for r in rs:
            assert int(r[6]) == int(r[5]) - 12 * K

    sha = sha256_file(path)
    with open(os.path.join(out_dir, 'sample_list_sha.txt'), 'w') as f:
        f.write(sha + '\n')
    if os.path.exists(path + '.prev'):
        prev = open(path + '.prev').read().strip()
        assert prev == sha, 'sample list changed across runs: %s vs %s' % (prev, sha)
    else:
        with open(path + '.prev', 'w') as f:
            f.write(sha + '\n')
    print('sample_list rows=1792 sha=%s' % sha)
    # 每数据集前 8 行(smoke 用)摘要
    smoke = {ds: [r for r in rows if r[1] == ds][:8] for ds in ('ETTh1', 'ETTm1')}
    print(json.dumps({ds: [(r[0], r[3], r[5], r[6]) for r in v] for ds, v in smoke.items()}))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
