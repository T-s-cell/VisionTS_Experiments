"""EXP-047:图输出。

- fig_D_vs_m.png / fig_cost_vs_m.png:D、冲突代价 vs m/P;列 = 幅度(|δ|/s ∈ {0.5,1}),
  线 = 方向(down=δ<0 / up=δ>0),面板 = 方法;实线点 = median,空心点 = mean;
- fig_method_diff_vs_m.png:ΔD(vts−对照,共同有效窗)vs m/P,直接回答"是否比对照更强";
- 配对案例图:cases_t47.csv 每 (ds, sid, δ_idx) 一 png — 历史 A/B(压缩 stride=ceil(L/720),
  末 4P 全密度高亮,B 偏移段橙色)+ 真值 y' + vts/moirai/autoets pred_A(实线)/pred_B(虚线);
  m 取该配对 |D_vts| 最大的 m_idx(确定性);绘图真值 vs f64 重算恒等、vs npz true_p ≤1e-5,
  违者中止;
- fig_data_t47.npz:案例图全部落盘数组(verify G10 复核用)。
"""
import csv
import math
import os
import sys

import numpy as np

from t47_common import DS_CFG, MAIN_METHODS, apply_variant, manifest_path

CONTRAST_PAIRS = [('vts', 'moirai'), ('vts', 'autoets'), ('vts', 'autoarima')]
COLORS = dict(vts='tab:red', moirai='tab:blue', autoets='tab:green', autoarima='tab:purple',
              true='black')
TRUE_TOL = 1e-5
N_CASES_PER_DS = 1


def read_csv_dicts(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def select_cases(save_dir):
    """每 ds 选 1 个 (sid,δ_idx):vts/moirai/autoets 全有效且 |D_vts| 最大(确定性)。"""
    pairs = read_csv_dicts(os.path.join(save_dir, 'pairs_t47.csv'))
    by = {}
    for r in pairs:
        by.setdefault((r['ds'], int(r['sid_t46']), int(r['delta_idx'])), {})[r['method']] = \
            (float(r['D']), int(r['ok_A']) and int(r['ok_B']))
    picked = []
    for ds in dict.fromkeys(k[0] for k in sorted(by)):
        cand = []
        for (d, sid, di), mm in sorted(by.items()):
            if d != ds:
                continue
            if all(m in mm and mm[m][1] for m in ('vts', 'moirai', 'autoets')):
                cand.append((abs(mm['vts'][0]), sid, di, mm['vts'][0]))
        if cand:
            _, sid, di, D = sorted(cand, key=lambda t: (-t[0], t[1], t[2]))[0]
            picked.append((ds, sid, di, D))
    with open(manifest_path()) as f:
        man_a = {(int(r['sid_t46']), int(r['delta_idx'])): r for r in csv.DictReader(f)
                 if r['group'] == 'A'}
    rows = []
    for ds, sid, di, D in picked:
        a = man_a[(sid, di)]
        rows.append([ds, sid, di, '%.12g' % D, a['cond_id'], a['var'], a['channel'], a['delta']])
    path = os.path.join(save_dir, 'cases_t47.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ds', 'sid_t46', 'delta_idx', 'D_vts', 'cond_id_A', 'var', 'channel', 'delta'])
        w.writerows(rows)
    return rows


def line_summary_fig(pairs, value, fname, ylab, save_dir):
    panels = MAIN_METHODS + ['autoarima']
    fig, axes = plt.subplots(len(panels), 2, figsize=(9, 2.6 * len(panels)), sharex=True)
    for i, m in enumerate(panels):
        for j, amp in enumerate(('0.5', '1.0')):
            ax = axes[i, j]
            for direc, style, marker in (('down', '-', 'o'), ('up', '--', 's')):
                pts = {}
                for r in pairs:
                    if (r['method'] == m and '%.1f' % abs(float(r['delta_mult'])) == amp
                            and ('down' if float(r['delta_mult']) < 0 else 'up') == direc
                            and float(r[value]) == float(r[value])):  # NaN 滤除
                        pts.setdefault(int(r['m_over_P']), []).append(float(r[value]))
                if pts:
                    xs = sorted(pts)
                    col = COLORS.get(m, 'k')
                    ax.plot(xs, [float(np.median(pts[x])) for x in xs], style + marker,
                            color=col, label='%s median (n=%s)' % (direc, [len(pts[x]) for x in xs]))
                    ax.plot(xs, [float(np.mean(pts[x])) for x in xs], marker, mfc='none',
                            color=col, alpha=0.6, label='%s mean' % direc)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xticks([1, 2, 4])
            ax.set_title('%s | amplitude %s' % (m, amp), fontsize=9)
            ax.legend(fontsize=6)
    fig.suptitle(ylab, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, fname), dpi=130)
    plt.close(fig)


def main(repo_root, save_dir):
    del repo_root
    global plt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pairs = read_csv_dicts(os.path.join(save_dir, 'pairs_t47.csv'))
    line_summary_fig(pairs, 'D', 'fig_D_vs_m.png',
                     'D (>0 = pulled toward old level) vs m/P', save_dir)
    line_summary_fig(pairs, 'cost', 'fig_cost_vs_m.png',
                     'conflict cost = MSE_B - MSE_A vs m/P', save_dir)

    # ---- ΔD vs m ----
    diffs = read_csv_dicts(os.path.join(save_dir, 'method_diff_t47.csv'))
    fig, axes = plt.subplots(len(CONTRAST_PAIRS), 1, figsize=(7, 2.6 * len(CONTRAST_PAIRS)),
                             sharex=True)
    for i, (mv, mc) in enumerate(CONTRAST_PAIRS):
        ax = axes[i]
        pts = {}
        for r in diffs:
            if r['pair'] == '%s_vs_%s' % (mv, mc) and float(r['D_diff']) == float(r['D_diff']):
                pts.setdefault(int(r['m_over_P']), []).append(float(r['D_diff']))
        if pts:
            xs = sorted(pts)
            col = COLORS.get(mc, 'k')
            ax.plot(xs, [float(np.median(pts[x])) for x in xs], '-o', color=col, label='median')
            ax.plot(xs, [float(np.mean(pts[x])) for x in xs], '--s', mfc='none', color=col,
                    alpha=0.7, label='mean')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks([1, 2, 4])
        ax.set_ylabel('D_diff')
        ax.set_title('D_vts - D_%s (common valid windows)' % mc, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle('method difference in old-level pull vs m/P', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig_method_diff_vs_m.png'), dpi=130)
    plt.close(fig)

    # ---- 配对案例图 ----
    cases = select_cases(save_dir)
    with open(manifest_path()) as f:
        man = list(csv.DictReader(f))
    arrs = {m: np.load(os.path.join(save_dir, 'pred_%s_t47.npz' % m)) for m in METHODS}
    rg_pos = {m: {int(rg): k for k, rg in enumerate(arrs[m]['row_globals'])} for m in METHODS}
    pr_by = {}
    for r in pairs:
        if r['method'] == 'vts' and int(r['ok_A']) and int(r['ok_B']):
            pr_by.setdefault((r['ds'], int(r['sid_t46']), int(r['delta_idx'])), []).append(r)
    base_cache = {}
    figdata = {}
    max_true_diff = 0.0
    H = 96
    png_dir = os.path.join(save_dir, 'cases_png')
    os.makedirs(png_dir, exist_ok=True)
    for ic, c in enumerate(cases):
        ds, sid, di = c[0], int(c[1]), int(c[2])
        cfg = DS_CFG[ds]
        L, P = cfg['L'], cfg['P']
        if ds not in base_cache:
            base_cache[ds] = np.load(os.path.join(save_dir, 'baseline_%s.npz' % ds))
        bz = base_cache[ds]
        w = {int(s): i for i, s in enumerate(bz['sid_t46'])}[sid]
        delta = float(c[7])
        mi_row = max(pr_by[(ds, sid, di)], key=lambda r: (abs(float(r['D'])), int(r['m_idx'])))
        m_idx, m_over_P = int(mi_row['m_idx']), int(mi_row['m_over_P'])
        m_steps = m_over_P * P
        cid_a, cid_b = di, 4 + 3 * di + m_idx
        rg_a = int(next(r['row_global'] for r in man if int(r['sid_t46']) == sid
                        and int(r['cond_id']) == cid_a))
        rg_b = int(next(r['row_global'] for r in man if int(r['sid_t46']) == sid
                        and int(r['cond_id']) == cid_b))
        x, y = bz['x'][w], bz['y'][w]
        xa, xb = apply_variant(x, delta, 0, 'A'), apply_variant(x, delta, m_steps, 'B')
        yp = y + delta
        d_npz = float(np.abs(yp - arrs['vts']['true_p'][rg_pos['vts'][rg_a]].astype(np.float64)).max())
        if d_npz > TRUE_TOL:
            raise SystemExit('case truth mismatch: %s sid=%d maxdiff=%.3e' % (ds, sid, d_npz))
        max_true_diff = max(max_true_diff, d_npz)

        stride = max(1, math.ceil(L / 720))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[1.6, 1])
        ax1.plot(np.arange(0, L, stride), xa[::stride], lw=0.7, color='gray',
                 label='history A (x+δ)')
        ax1.plot(np.arange(L - 4 * P, L), xa[-4 * P:], lw=1.0, color='gray')
        bs = np.arange(L)[-m_steps:]
        ax1.plot(bs, xb[-m_steps:], lw=1.0, color='tab:orange',
                 label='history B last %dP (+δ only)' % m_over_P)
        ax1.axvspan(L - 4 * P, L, color='gold', alpha=0.15)
        ax1.set_xlim(0, L)
        ax1.set_title('case %d: %s sid=%d var=%s δ=%.3g m=%dP  D_vts=%.3g'
                      % (ic, ds, sid, c[5], delta, m_over_P, float(c[3])), fontsize=10)
        ax1.legend(fontsize=8)
        ax2.plot(np.arange(H), yp, color=COLORS['true'], lw=1.8, label="truth y'=y+δ")
        for m in MAIN_METHODS:
            ax2.plot(np.arange(H), arrs[m]['pred'][rg_pos[m][rg_a]], lw=1.1, color=COLORS[m],
                     label='%s pred_A' % m)
            ax2.plot(np.arange(H), arrs[m]['pred'][rg_pos[m][rg_b]], lw=1.1, ls='--',
                     color=COLORS[m], label='%s pred_B' % m)
        ax2.set_xlim(0, H - 1)
        ax2.legend(fontsize=7)
        ax2.set_title('standardized scale', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(png_dir, 'case_%d_%s_sid%d_di%d.png' % (ic, ds, sid, di)), dpi=120)
        plt.close(fig)
        figdata['case%d_true' % ic] = yp
        figdata['case%d_truep_vts_f32' % ic] = arrs['vts']['true_p'][rg_pos['vts'][rg_a]]
        figdata['case%d_rgAB' % ic] = np.array([rg_a, rg_b], dtype=np.int64)
        for m in MAIN_METHODS:
            figdata['case%d_predA_%s' % (ic, m)] = arrs[m]['pred'][rg_pos[m][rg_a]]
            figdata['case%d_predB_%s' % (ic, m)] = arrs[m]['pred'][rg_pos[m][rg_b]]
    np.savez_compressed(os.path.join(save_dir, 'fig_data_t47.npz'), **figdata)
    print('figures done: %d cases; truth-vs-npz maxdiff=%.2e (tol %.0e)'
          % (len(cases), max_true_diff, TRUE_TOL))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_t47_levelshift'))
