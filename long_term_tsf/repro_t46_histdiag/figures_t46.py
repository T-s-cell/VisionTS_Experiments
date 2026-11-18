"""EXP-046:图输出。

- 案例图:cases_t46.csv 每案例一 png(压缩历史 + 真值 + 四方法预测);
  历史压缩 stride = ceil(L/720),末 4P 段以原密度叠加高亮;
  样本按 (ds, sample_id) 查清单(全局行),绘图真值与 arr_vts npz 真值
  逐案例核对(maxdiff ≤ 1e-5,与 verify G2 同容差),不一致即中止;
- 汇总图:fig_tertile_summary.png — 5 特征 × 3 对照的逐 (ds×var) rel_gap
  三分位均值条形(单元均值,非样本池化)。
确定性:案例顺序/配色/抽稀 stride 全由 csv 与常量决定。
"""
import csv
import math
import os
import sys

import numpy as np

from t46_common import DS_CFG, DS_ORDER, H, load_csv, sample_list_path, std_all, train_scaler

METHODS = ['vts', 'moirai', 'autoets', 'autoarima']
PAIRS = [('vts', 'moirai'), ('vts', 'autoets'), ('vts', 'autoarima')]
FEATURES = ['period_corr', 'recent_old', 'trend', 'vol', 'hist_std']
TERTILE_NAME = ['low', 'mid', 'high']
TRUE_TOL = 1e-5
COLORS = dict(vts='tab:red', moirai='tab:blue', autoets='tab:green', autoarima='tab:purple',
              true='black')


def pred_for(ds, sid, arrays, pos):
    j = pos[sid]
    return {m: arrays[m][j] for m in METHODS}


def main(repo_root, save_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    samples = list(csv.DictReader(open(sample_list_path())))
    cases = list(csv.DictReader(open(os.path.join(save_dir, 'cases_t46.csv'))))
    png_dir = os.path.join(save_dir, 'cases_png')
    os.makedirs(png_dir, exist_ok=True)

    cache = {}
    arr_cache = {}
    true_cache = {}
    pos_cache = {}
    row_cache = {}
    for ds in dict.fromkeys(c['ds'] for c in cases):
        cfg = DS_CFG[ds]
        if ds not in cache:
            data = load_csv(os.path.join(repo_root, cfg['csv']))
            mean, scale = train_scaler(data, cfg['K'])
            cache[ds] = std_all(data, mean, scale)
        full = [s for s in samples if s['ds'] == ds]
        pos_cache[ds] = {int(s['sample_id']): j for j, s in enumerate(full)}
        row_cache[ds] = {int(s['sample_id']): s for s in full}
        zv = np.load(os.path.join(save_dir, 'arr_vts_t46.npz'))
        zm = np.load(os.path.join(save_dir, f'arr_moirai_{ds}.npz'))
        ze = np.load(os.path.join(save_dir, f'AutoETS_{ds}.npz'))
        za = np.load(os.path.join(save_dir, f'AutoARIMA_{ds}.npz'))
        arr_cache[ds] = {'vts': zv[f'{ds}_pred'], 'moirai': zm['pred'],
                         'autoets': ze['pred'], 'autoarima': za['pred']}
        true_cache[ds] = zv[f'{ds}_true']

    max_true_diff = 0.0
    for c in cases:
        ds, sid = c['ds'], int(c['sample_id'])
        cfg = DS_CFG[ds]
        s = row_cache[ds][sid]
        start, ch, L, P = int(s['start_row']), int(s['channel']), cfg['L'], cfg['P']
        std = cache[ds]
        hist = std[start - L:start, ch]
        stride = max(1, math.ceil(L / 720))
        preds = pred_for(ds, sid, arr_cache[ds], pos_cache[ds])
        true = std[start:start + H, ch]
        j = pos_cache[ds][sid]
        d = float(np.max(np.abs(true.astype(np.float64) - true_cache[ds][j].astype(np.float64))))
        if d > TRUE_TOL:
            raise SystemExit(f'truth mismatch: {ds} sid={sid} start={start} '
                             f'j={j} maxdiff={d:.3e} > {TRUE_TOL:.0e}')
        max_true_diff = max(max_true_diff, d)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[1.6, 1])
        ax1.plot(np.arange(0, L, stride), hist[::stride], lw=0.7, color='gray', label='history')
        hp = np.arange(L - 4 * P, L)
        ax1.plot(hp, hist[-4 * P:], lw=1.0, color='gray')
        ax1.axvspan(L - 4 * P, L, color='gold', alpha=0.15, label='last 4P (full density)')
        ax1.set_xlim(0, L)
        ax1.set_title(f"{c['case']} {ds} {c['var']} sid={sid} start={start} "
                      f"L={L} P={P} margin={c['margin']} rank_vts={c['rank_vts']}", fontsize=10)
        ax1.legend(fontsize=8)
        ax2.plot(np.arange(H), true, color=COLORS['true'], lw=1.8, label='truth')
        for m in METHODS:
            mse = float(c[f'mse_{m}'])
            ax2.plot(np.arange(H), preds[m], lw=1.2,
                     color=COLORS[m], label=f'{m} (mse={mse:.4g})')
        ax2.set_xlim(0, H - 1)
        ax2.legend(fontsize=8)
        ax2.set_title('standardized scale', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(png_dir, f"case_{c['case']}_{ds}_sid{sid}.png"), dpi=120)
        plt.close(fig)

    # ---- tertile 汇总图(单元均值) ----
    tert = list(csv.DictReader(open(os.path.join(save_dir, 'table_tertile_t46.csv'))))
    fig, axes = plt.subplots(len(FEATURES), len(PAIRS), figsize=(4 * len(PAIRS), 3 * len(FEATURES)),
                             sharex=True)
    for i, feat in enumerate(FEATURES):
        for j, (mv, mb) in enumerate(PAIRS):
            ax = axes[i, j]
            pair = f'{mv}_vs_{mb}'
            by_t = {t: [] for t in range(3)}
            for r in tert:
                if r['feature'] == feat and r['pair'] == pair and r['rel_gap'] != '':
                    by_t[TERTILE_NAME.index(r['tertile'])].append(float(r['rel_gap']))
            means = [float(np.mean(by_t[t])) if by_t[t] else np.nan for t in range(3)]
            ax.bar(range(3), means, color=['#9ecae1', '#4292c6', '#08519c'])
            ax.axhline(0, color='k', lw=0.8)
            n_cells = [len(by_t[t]) for t in range(3)]
            ax.set_title(f'{feat} / {pair}\ncells={n_cells}', fontsize=8)
            ax.set_xticks(range(3))
            ax.set_xticklabels(TERTILE_NAME, fontsize=8)
    fig.suptitle('VisionTS relative gap by feature tertile (cell means, positive = VTS weaker)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig_tertile_summary.png'), dpi=130)
    plt.close(fig)
    print('figures done: %d case pngs + fig_tertile_summary.png; truth-vs-npz maxdiff=%.2e (tol %.0e)'
          % (len(cases), max_true_diff, TRUE_TOL))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'save_t46_histdiag')
