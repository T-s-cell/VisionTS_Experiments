"""EXP-047 STEP 2:变换清单 + f64 基准数据(先于一切推理;不物化 7168 构造输入)。

- 每基准窗口重算 x/y(csv 标准化 f64)、s = x.std(ddof=0),与 features_t46.csv
  hist_std 交叉核对 ≤1e-8(超则中止);
- f64 基准按 ds 落盘 save_t47_levelshift/baseline_<ds>.npz:
  x (112,L) f64、y (112,96) f64、s (112,) f64、sid_t46/arr_index (112,) int64;
  行序 = baseline_list 该 ds 块序(var channel 序 → rank);
- manifest_t47.csv 7168 行 = 448 窗 × 16 条件(A4+B12),行序 window-major then cond;
  构造输入由推理脚本在线派生(A=x+δ;B=末 m 步+δ;y'=y+δ),verify 逐位重建核对。
"""
import csv
import json
import os
import sys

import numpy as np

from t47_common import (CHANNELS, CONDS, DELTA_MULTS, DS_ORDER, M_MULTS, N_CONDS, S_TOL,
                        SCRIPT_DIR, apply_variant, cond_params, load_csv, sha256_file,
                        std_all, train_scaler, baseline_list_path, manifest_path)
from t46_common import DS_CFG, get_window

MANIFEST_COLUMNS = ['row_global', 'sid_t46', 'ds', 'var', 'channel', 'start_row', 'arr_index',
                    'L', 'P', 'rank', 'cond_id', 'group', 'delta_idx', 'm_idx', 'delta_mult',
                    'delta', 'm_steps', 'm_over_P', 's']


def fmt(x):
    return '%.12g' % x


def main(repo_root, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(baseline_list_path()) as f:
        base = list(csv.DictReader(f))
    assert len(base) == 448

    std_cache = {}
    ds_rows = {}   # ds -> list of (x, y, s, sid, arr_index)
    s_by_sid = {}
    max_s_diff = 0.0
    for r in base:
        ds = r['ds']
        if ds not in std_cache:
            cfg = DS_CFG[ds]
            data = load_csv(os.path.join(repo_root, cfg['csv']))
            mean, scale = train_scaler(data, cfg['K'])
            std_cache[ds] = std_all(data, mean, scale)
        std = std_cache[ds]
        cfg = DS_CFG[ds]
        start, ch = int(r['start_row']), int(r['channel'])
        x = get_window(std, ds, start, ch)                      # (L,) f64
        y = std[start:start + 96, ch]                           # (96,) f64
        s = float(x.std())
        d = abs(s - float(r['hist_std_csv']))
        max_s_diff = max(max_s_diff, d)
        assert d <= S_TOL, (ds, r['sid_t46'], d)
        assert np.isfinite(x).all() and np.isfinite(y).all() and s > 0
        ds_rows.setdefault(ds, []).append((x, y, s, int(r['sid_t46']), int(r['arr_index'])))
        s_by_sid[int(r['sid_t46'])] = s

    # f64 基准数据(交付可独立复算)
    base_meta = {}
    for ds, rows in ds_rows.items():
        L = DS_CFG[ds]['L']
        x = np.stack([r[0] for r in rows]).astype(np.float64)     # (112, L)
        y = np.stack([r[1] for r in rows]).astype(np.float64)     # (112, 96)
        s = np.array([r[2] for r in rows], dtype=np.float64)
        sid = np.array([r[3] for r in rows], dtype=np.int64)
        ai = np.array([r[4] for r in rows], dtype=np.int64)
        p = os.path.join(save_dir, 'baseline_%s.npz' % ds)
        np.savez_compressed(p, x=x, y=y, s=s, sid_t46=sid, arr_index=ai)
        base_meta[ds] = dict(path=p, n=int(x.shape[0]), L=L, x_dtype=str(x.dtype),
                             y_dtype=str(y.dtype), sha256=sha256_file(p))
        # 构造自检(每 ds 首窗):A/B 末 m 步恒等 + y' 一致
        x0 = x[0]
        for (cid, g, di, mi) in CONDS:
            if g != 'B':
                continue
            delta = s[0] * DELTA_MULTS[di]
            m = M_MULTS[mi] * DS_CFG[ds]['P']
            xa = apply_variant(x0, delta, m, 'A')
            xb = apply_variant(x0, delta, m, 'B')
            assert np.array_equal(xa[-m:], xb[-m:])
            assert xb[-1] == x0[-1] + delta

    # manifest
    rows = []
    for r in base:
        ds = r['ds']
        s = s_by_sid[int(r['sid_t46'])]
        for cid in range(N_CONDS):
            g, di, mi, dmult, mop = cond_params(cid)
            delta = s * dmult
            m_steps = 0 if mi is None else M_MULTS[mi] * DS_CFG[ds]['P']
            rows.append([int(r['w_global']) * N_CONDS + cid, r['sid_t46'], ds, r['var'],
                         r['channel'], r['start_row'], r['arr_index'], r['L'], r['P'],
                         r['rank'], cid, g, di, '' if mi is None else mi,
                         fmt(dmult), fmt(delta), m_steps, '' if mop is None else mop, fmt(s)])
    assert len(rows) == 7168 and len({r[0] for r in rows}) == 7168
    assert len({(r[1], r[10]) for r in rows}) == 7168  # (sid_t46, cond_id) 唯一
    with open(manifest_path(), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(MANIFEST_COLUMNS)
        wr.writerows(rows)
    sha = sha256_file(manifest_path())

    meta = dict(manifest_sha256=sha, n_rows=len(rows), s_max_vs_features=max_s_diff,
                baselines=base_meta, delta_mults=DELTA_MULTS, m_mults=M_MULTS)
    with open(os.path.join(save_dir, 'build_meta_t47.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('manifest rows=7168 sha=%s s_max_vs_features=%.3e' % (sha, max_s_diff))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else os.path.join(SCRIPT_DIR, 'save_t47_levelshift'))
