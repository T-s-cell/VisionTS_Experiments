"""EXP-047h figures:三主图(vts_vs_moirai)+ fig_data(verify V6 同源核验)。

用法:$PY figures_t47h.py <OUT输出目录>
- fig_Ddiff_band_t47h.png     D_diff = D_vts − D_moirai 随区间(>0 = VTS 牵引更强);
- fig_costdiff_band_t47h.png  cost_diff 随区间(>0 = VTS 冲突代价更大);
- fig_gapB_band_t47h.png      gap_B = MSE_B_vts − MSE_B_moirai 随区间(>0 = 误差更大);
每图:ds 2×2 分面、m/P∈{1,2,4} 三线、实点=median、空心=mean、零线;
b1–b4 为主图区间,full-96 单列(虚线分隔,不作第五时段)。
fig_data_t47h.npz:每图每 (ds,m) 的 bands[4]/full 中位数与均值 + n(供 V6)。
"""
import os
import sys

import numpy as np

from t47h_common import BAND_IDS, DIFF_H, DS_P, read_csv_dicts

PAIR = 'vts_vs_moirai'
MPS = (1, 2, 4)
MCOLOR = {1: 'tab:blue', 2: 'tab:orange', 4: 'tab:green'}
FIGS = [('Ddiff', 'D_diff', 'D_diff = D_vts - D_moirai  (>0: VTS pulled toward old level more)',
         'fig_Ddiff_band_t47h.png'),
        ('costdiff', 'cost_diff',
         'cost_diff = (MSE_B-MSE_A)_vts - _moirai  (>0: VTS conflict cost larger)',
         'fig_costdiff_band_t47h.png'),
        ('gapB', 'gap_B', 'gap_B = MSE_B_vts - MSE_B_moirai  (>0: VTS final error larger)',
         'fig_gapB_band_t47h.png')]
XBAND = np.arange(4, dtype=np.float64)
XFULL = 4.6


def panel_vals(rows, ds, mp, q):
    bands, full, nb, nf = [], [], [], []
    for bid in BAND_IDS:
        v = [r[q] for r in rows if r['pair'] == PAIR and r['ds'] == ds and r['m_over_P'] == mp
             and r['band'] == bid and r['_ok'] and r[q] == r[q]]
        bands.append(float(np.median(v)) if v else np.nan)
        nb.append(len(v))
    v = [r[q] for r in rows if r['pair'] == PAIR and r['ds'] == ds and r['m_over_P'] == mp
         and r['band'] == 'full' and r['_ok'] and r[q] == r[q]]
    full = float(np.median(v)) if v else np.nan
    nf = len(v)
    return np.array(bands), full, nb, nf


def draw(ax, rows, ds, q):
    note = ('each band = 1P' if DS_P[ds] == 24 else 'each band = 0.25P')
    for mp in MPS:
        col = MCOLOR[mp]
        med_b, med_f, _, nf = panel_vals(rows, ds, mp, q)
        mean_b, mean_f, _, _ = panel_vals_mean(rows, ds, mp, q)
        ax.plot(XBAND, med_b, '-o', color=col, label='m/P=%d median' % mp)
        ax.plot([XFULL], [med_f], 'o', color=col,
                label='m/P=%d full median (n=%d)' % (mp, nf))
        ax.plot(XBAND, mean_b, '--s', mfc='none', color=col, alpha=0.6,
                label='m/P=%d mean' % mp)
        ax.plot([XFULL], [mean_f], 's', mfc='none', color=col, alpha=0.6)
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(3.8, color='gray', lw=0.7, ls=':')
    ax.set_xticks(list(XBAND) + [XFULL])
    ax.set_xticklabels(['b1\n[0,24)', 'b2\n[24,48)', 'b3\n[48,72)', 'b4\n[72,96)',
                        'full\n[0,96)'], fontsize=7)
    ax.set_title('%s (%s)' % (ds, note), fontsize=9)
    ax.legend(fontsize=6)


def main(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw = read_csv_dicts(os.path.join(out_dir, DIFF_H))
    rows = []
    for r in raw:
        rec = dict(pair=r['pair'], ds=r['ds'], m_over_P=int(float(r['m_over_P'])),
                   band=r['band'], _ok=r['ok'] == '1', sid=int(r['sid_t46']))
        for _, q, _, _ in FIGS:
            rec[q] = float(r[q])
        rows.append(rec)
    dss = sorted({r['ds'] for r in rows})

    figdata = {'meta_pair': np.array(PAIR), 'ds_order': np.array(dss),
               'band_ids': np.array(BAND_IDS)}
    for figtag, q, supt, fname in FIGS:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
        for ax, ds in zip(axes.ravel(), dss):
            draw(ax, rows, ds, q)
            for mp in MPS:
                med_b, med_f, nb, nf = panel_vals(rows, ds, mp, q)
                figdata['%s_%s_m%d_bands' % (figtag, ds, mp)] = med_b
                figdata['%s_%s_m%d_full' % (figtag, ds, mp)] = np.float64(med_f)
                figdata['%s_%s_m%d_nb' % (figtag, ds, mp)] = np.array(nb, dtype=np.int64)
                figdata['%s_%s_m%d_nf' % (figtag, ds, mp)] = np.int64(nf)
                mean_b, mean_f, _, _ = panel_vals_mean(rows, ds, mp, q)
                figdata['%s_%s_m%d_bands_mean' % (figtag, ds, mp)] = mean_b
                figdata['%s_%s_m%d_full_mean' % (figtag, ds, mp)] = np.float64(mean_f)
        for ax in axes[1]:
            ax.set_xlabel('interval (same H=96 forecast sliced; full shown separately)')
        fig.suptitle(supt, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=130)
        plt.close(fig)
        print('[figures] %s done' % fname)
    np.savez_compressed(os.path.join(out_dir, 'fig_data_t47h.npz'), **figdata)
    print('[figures] fig_data_t47h.npz: %d arrays' % len(figdata))


def panel_vals_mean(rows, ds, mp, q):
    bands, nb, full, nf = [], [], np.nan, 0
    for bid in BAND_IDS:
        v = [r[q] for r in rows if r['pair'] == PAIR and r['ds'] == ds and r['m_over_P'] == mp
             and r['band'] == bid and r['_ok'] and r[q] == r[q]]
        bands.append(float(np.mean(v)) if v else np.nan)
        nb.append(len(v))
    v = [r[q] for r in rows if r['pair'] == PAIR and r['ds'] == ds and r['m_over_P'] == mp
         and r['band'] == 'full' and r['_ok'] and r[q] == r[q]]
    if v:
        full, nf = float(np.mean(v)), len(v)
    return np.array(bands), full, nb, nf


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
