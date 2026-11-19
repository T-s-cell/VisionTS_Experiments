"""S1-b 超限排查(非门禁):单变量 [32,L,1] vs 7 通道 [32,L,7] 前向组织差。

S1-b 实测 max=8.24e-4 > 5e-4。本脚本量化该偏差的分布与来源,供决策:
- 全 4 ds 各取 baseline_list 前 2 基准窗(冒烟只覆盖 ETTh1/ETTm1;ETTm2 L=4032 最长需测);
- 每窗:|out7[:, ch] − out1| 的 max/mean/p95 + |pred| 尺度;
- 偏差随 horizon 的轮廓(前段/末段);
- 两次 1-ch 前向逐位(确认确定性,排除数值噪声)。
"""
import csv
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from smoke_t47 import set_seeds
from t47_common import DS_CFG, DS_ORDER, T46_DIR
from t46_common import load_csv, std_all, train_scaler


def main(repo_root, save_dir):
    import torch
    from infer_vts_t47 import load_vm
    from t47_common import pos46_map

    with open(os.path.join(SCRIPT_DIR, 'baseline_list_t47.csv')) as f:
        bl = list(csv.DictReader(f))
    pos46 = pos46_map()
    arr46 = np.load(os.path.join(T46_DIR, 'save_t46_histdiag', 'arr_vts_t46.npz'))

    out, vm_cache, std_cache = {}, {}, {}
    for ds in DS_ORDER:
        cfg = DS_CFG[ds]
        L, P, K = cfg['L'], cfg['P'], cfg['K']
        rows = [r for r in bl if r['ds'] == ds][:2]
        if ds not in std_cache:
            data = load_csv(os.path.join(repo_root, cfg['csv']))
            mean, scale = train_scaler(data, K)
            std_cache[ds] = std_all(data, mean, scale)
        std = std_cache[ds]
        if ds not in vm_cache:
            vm_cache[ds] = load_vm(repo_root, L, P)
        vm = vm_cache[ds]

        batches = {}
        for r in rows:
            ai = int(r['arr_index'])
            batches.setdefault((ai // 32) * 32, []).append(r)
        for bstart, brows in sorted(batches.items()):
            ch = int(brows[0]['channel'])
            xb = np.empty((32, L, 7), dtype=np.float32)
            for j in range(32):
                st = K * 12 + bstart + j
                xb[j] = std[st - L:st, :].astype(np.float32)
            set_seeds(2021)
            with torch.no_grad():
                o7 = vm.forward(torch.tensor(xb).cuda()).cpu().numpy()          # [32,96,7]
            set_seeds(2021)
            with torch.no_grad():
                o1a = vm.forward(torch.tensor(xb[:, :, ch:ch + 1]).cuda()).cpu().numpy()
            set_seeds(2021)
            with torch.no_grad():
                o1b = vm.forward(torch.tensor(xb[:, :, ch:ch + 1]).cuda()).cpu().numpy()
            assert np.array_equal(o1a, o1b)   # 同输入同种子 → 逐位(确定性)

            for r in brows:
                ai = int(r['arr_index'])
                loc = ai - bstart
                j46 = pos46[ds][int(r['sid_t46'])]
                d47 = np.abs(o7[loc, :, ch].astype(np.float64)
                             - o1a[loc, :, 0].astype(np.float64))
                d_ref = float(np.max(np.abs(o7[loc, :, ch].astype(np.float64)
                                            - arr46['%s_pred' % ds][j46].astype(np.float64))))
                out['%s_w%d_ai%d' % (ds, int(r['rank']), ai)] = dict(
                    max=float(d47.max()), mean=float(d47.mean()),
                    p95=float(np.quantile(d47, 0.95)),
                    h_first32_mean=float(d47[:32].mean()), h_last32_mean=float(d47[-32:].mean()),
                    scale=float(np.abs(o1a[loc, :, 0]).max()),
                    s1a_vs_arr46=d_ref)
        del vm_cache[ds]
        torch.cuda.empty_cache()

    dmax = max(v['max'] for v in out.values())
    doc = dict(purpose='S1b exceedance investigation, not a gate',
               s1b_gate_max=8.242130279541016e-04, worst_here=dmax, per_window=out)
    with open(os.path.join(SCRIPT_DIR, 'diag_s1b_t47.json'), 'w') as f:
        json.dump(doc, f, indent=1)
    for k, v in out.items():
        print(k, {kk: round(vv, 6) for kk, vv in v.items()})
    print('worst max=%.6e (gate saw 8.2421e-04 on ETTh1/ETTm1 subset)' % dmax)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else os.path.join(SCRIPT_DIR, 'save_t47_levelshift'))
