"""EXP-046:分析(口径冻结于 feature_formulas.md,分析前落盘)。

输入:features_t46.csv / per_sample_metrics_t46.csv / sample_list_t46.csv / failure_ledger.csv
输出:
  overall_t46.csv             总体表(各方法 own-ok 与四方法共同样本双口径 + 覆盖率)
  table_tertile_t46.csv       (ds×var×特征×三分位) 各方法误差 + 两两绝对/相对 gap
  coverage_by_group_t46.csv   覆盖率细分到特征组
  direction_summary_t46.csv   方向一致性:high−low 符号计数 / 单调计数 / 中位差
  joint_hard_t46.csv          共同困难清单(90 分位阈值 per ds×var×method,允许空)
  joint_hard_summary_t46.csv  共同困难 vs 其余:各方法 MSE / gap
  cases_t46.csv               明显较好/较差案例(确定性规则,排除近常数)
  analyze_meta_t46.json       汇总(近常数/方向计数/ledger 状态统计)

口径:
- tertile:每 (ds×var) 内非近常数 64 样本,特征 np.quantile 1/3、2/3 分箱
  (np.digitize 语义:v<q1 low,q1≤v<q2 mid,v≥q2 high);
- 相对 gap = (MSE_VTS−MSE_B)/(MSE_VTS+MSE_B+1e-12),正 = VisionTS 相对弱势线索;
  两两比较用双方均成功样本;
- hist_std 作为第 5 分组维(描述性,检验幅度解释);
- 共同困难:四者共同成功且非近常数样本中,每方法 MSE ≥ 该 (ds×var×method) ok 样本
  90 分位阈值;允许某 (ds,var) 为空;
- 案例:排名(四者共同成功)中 VTS=1 且 MSE<0.8×次优 = 明显较好;
  VTS=4 且 MSE>1.25×次差(排名 3) = 明显较差;每类每 ds 取 margin 前 2,不足则空缺。
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

from t46_common import CHANNELS, DS_ORDER, sample_list_path

FEATURES = ['period_corr', 'recent_old', 'trend', 'vol', 'hist_std']
PAIRS = [('vts', 'moirai'), ('vts', 'autoets'), ('vts', 'autoarima')]
METHODS = ['vts', 'moirai', 'autoets', 'autoarima']
FEAT_LABEL = dict(period_corr='周期一致性', recent_old='近旧周期差', trend='趋势强度',
                  vol='波动程度', hist_std='历史幅度(描述)')
TERTILE_NAME = ['low', 'mid', 'high']


def read_csv_dict(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def wr(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)


def main(repo_root, save_dir):
    samples = read_csv_dict(sample_list_path())
    feats_raw = read_csv_dict(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'features_t46.csv'))
    mets_raw = read_csv_dict(os.path.join(save_dir, 'per_sample_metrics_t46.csv'))
    assert len(samples) == 1792 and len(feats_raw) == 1792 and len(mets_raw) == 1792

    feats = {}
    for r in feats_raw:
        feats[int(r['sample_id'])] = dict(
            ds=r['ds'], var=r['var'], channel=int(r['channel']),
            hist_std=float(r['hist_std']), near_constant=r['near_constant'] == 'True',
            **{k: float(r[k]) for k in FEATURES if k != 'hist_std'})
    met = {}
    for r in mets_raw:
        sid = int(r['sample_id'])
        met[sid] = dict(ds=r['ds'], var=r['var'],
                        ok={m: r[f'ok_{m}'] == '1' for m in METHODS},
                        mse={m: float(r[f'mse_{m}']) for m in METHODS},
                        mae={m: float(r[f'mae_{m}']) for m in METHODS})
    sids = sorted(feats)
    assert sids == sorted(met) and len(sids) == 1792
    by_ds_var = {}
    for sid in sids:
        f = feats[sid]
        assert f['ds'] == met[sid]['ds'] and f['var'] == met[sid]['var'], sid
        by_ds_var.setdefault((f['ds'], f['var']), []).append(sid)

    n_nc = sum(1 for sid in sids if feats[sid]['near_constant'])
    ledger_path = os.path.join(save_dir, 'failure_ledger.csv')
    ledger_stat = Counter()
    if os.path.exists(ledger_path):
        for r in read_csv_dict(ledger_path):
            ledger_stat[(r['method'], r['status'])] += 1

    # ---------- 总体表(own-ok 与四方法共同样本双口径) ----------
    ov_rows = []
    com4_all = [sid for sid in sids if all(met[sid]['ok'].values())]
    for ds in DS_ORDER + ['ALL']:
        sel = sids if ds == 'ALL' else [sid for sid in sids if met[sid]['ds'] == ds]
        com4 = [sid for sid in sel if all(met[sid]['ok'].values())]
        for m in METHODS:
            ok = [sid for sid in sel if met[sid]['ok'][m]]
            mse_ok = [met[sid]['mse'][m] for sid in ok]
            mse_c4 = [met[sid]['mse'][m] for sid in com4]
            mae_ok = [met[sid]['mae'][m] for sid in ok]
            mae_c4 = [met[sid]['mae'][m] for sid in com4]
            ov_rows.append([ds, m, len(sel), len(ok), '%.4f' % (len(ok) / len(sel)),
                            '%.12g' % np.mean(mse_ok) if ok else '',
                            '%.12g' % np.mean(mae_ok) if ok else '',
                            len(com4),
                            '%.12g' % np.mean(mse_c4) if com4 else '',
                            '%.12g' % np.mean(mae_c4) if com4 else ''])
    wr(os.path.join(save_dir, 'overall_t46.csv'),
       ['ds', 'method', 'n', 'n_ok', 'coverage', 'mse_own_ok', 'mae_own_ok',
        'n_common4', 'mse_common4', 'mae_common4'], ov_rows)

    # ---------- 三分位表 + 覆盖率 + 方向汇总 ----------
    tert_rows, cov_rows = [], []
    direction = {}  # (feat,pair) -> list of (rel_low, rel_mid, rel_high)
    for (ds, var) in sorted(by_ds_var):
        cell = by_ds_var[(ds, var)]
        valid = [sid for sid in cell if not feats[sid]['near_constant']]
        # 先按 hist_std(第 5 维)以外的特征各自分箱;hist_std 自身同款代码
        tertile_by_feat = {}
        for feat in FEATURES:
            v = np.array([feats[sid][feat] for sid in valid], dtype=np.float64)
            assert np.isfinite(v).all(), (ds, var, feat)
            q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
            tertile_by_feat[feat] = {sid: int(t) for sid, t in
                                     zip(valid, np.digitize(v, [q1, q2]))}
        for feat in FEATURES:
            for t in (0, 1, 2):
                ids_t = [sid for sid in valid if tertile_by_feat[feat][sid] == t]
                hs = float(np.mean([feats[sid]['hist_std'] for sid in ids_t])) if ids_t else float('nan')
                per_ok = {}
                for m in METHODS:
                    ok = [sid for sid in ids_t if met[sid]['ok'][m]]
                    per_ok[m] = (ok,
                                 float(np.mean([met[sid]['mse'][m] for sid in ok])) if ok else None,
                                 float(np.mean([met[sid]['mae'][m] for sid in ok])) if ok else None)
                    cov_rows.append([ds, var, feat, TERTILE_NAME[t], m, len(ids_t), len(ok),
                                     '%.4f' % (len(ok) / len(ids_t)) if ids_t else ''])
                for (mv, mb) in PAIRS:
                    common = [sid for sid in ids_t if met[sid]['ok'][mv] and met[sid]['ok'][mb]]
                    if common:
                        mvv = float(np.mean([met[sid]['mse'][mv] for sid in common]))
                        mbb = float(np.mean([met[sid]['mse'][mb] for sid in common]))
                        vvv = float(np.mean([met[sid]['mae'][mv] for sid in common]))
                        vbb = float(np.mean([met[sid]['mae'][mb] for sid in common]))
                        abs_gap = mvv - mbb
                        rel_gap = (mvv - mbb) / (mvv + mbb + 1e-12)
                    else:
                        mvv = mbb = vvv = vbb = abs_gap = rel_gap = None
                    tert_rows.append([ds, var, feat, FEAT_LABEL[feat], TERTILE_NAME[t],
                                      mv + '_vs_' + mb, len(ids_t), len(common),
                                      '%.12g' % mvv if mvv is not None else '',
                                      '%.12g' % mbb if mbb is not None else '',
                                      '%.12g' % abs_gap if abs_gap is not None else '',
                                      '%.12g' % rel_gap if rel_gap is not None else '',
                                      '%.12g' % vvv if vvv is not None else '',
                                      '%.12g' % vbb if vbb is not None else '',
                                      '%.12g' % hs])
                    if common and mvv is not None:
                        direction.setdefault((feat, (mv, mb)), {})[(ds, var, t)] = rel_gap
    wr(os.path.join(save_dir, 'table_tertile_t46.csv'),
       ['ds', 'var', 'feature', 'feature_cn', 'tertile', 'pair', 'n_cell', 'n_common',
        'mse_vts', 'mse_base', 'abs_gap', 'rel_gap', 'mae_vts', 'mae_base', 'hist_std_mean'],
       tert_rows)
    wr(os.path.join(save_dir, 'coverage_by_group_t46.csv'),
       ['ds', 'var', 'feature', 'tertile', 'method', 'n_cell', 'n_ok', 'coverage'], cov_rows)

    dir_rows = []
    for (feat, (mv, mb)), d in sorted(direction.items(), key=lambda kv: (kv[0][0], kv[0][1][0])):
        n_valid = 0
        n_pos = n_neg = n_mono = 0
        deltas = []
        for ds in DS_ORDER:
            for var in CHANNELS:
                triple = [d.get((ds, var, t)) for t in (0, 1, 2)]
                if any(x is None for x in triple):
                    continue
                lo, mi, hi = triple
                n_valid += 1
                n_pos += hi > lo
                n_neg += hi < lo
                n_mono += lo <= mi <= hi
                deltas.append(hi - lo)
        dir_rows.append([feat, f'{mv}_vs_{mb}', n_valid, n_pos, n_neg, n_mono,
                         '%.12g' % float(np.median(deltas)) if deltas else ''])
    wr(os.path.join(save_dir, 'direction_summary_t46.csv'),
       ['feature', 'pair', 'n_cells_valid', 'n_high_gt_low', 'n_low_gt_high',
        'n_monotone_nondecreasing', 'median_rel_gap_high_minus_low'], dir_rows)

    # ---------- 共同困难(90 分位 per ds×var×method;四者共同成功样本) ----------
    jh_rows = []
    jh_ids = []
    for (ds, var) in sorted(by_ds_var):
        cell = [sid for sid in by_ds_var[(ds, var)]
                if not feats[sid]['near_constant'] and all(met[sid]['ok'].values())]
        if not cell:
            continue
        thr = {}
        for m in METHODS:
            okall = [met[sid]['mse'][m] for sid in by_ds_var[(ds, var)]
                     if not feats[sid]['near_constant'] and met[sid]['ok'][m]]
            if okall:
                thr[m] = float(np.quantile(okall, 0.9))
        if len(thr) < 4:
            continue
        for sid in cell:
            if all(met[sid]['mse'][m] >= thr[m] for m in METHODS):
                jh_ids.append(sid)
                jh_rows.append([sid, ds, var]
                               + ['%.12g' % met[sid]['mse'][m] for m in METHODS]
                               + ['%.12g' % thr[m] for m in METHODS])
    wr(os.path.join(save_dir, 'joint_hard_t46.csv'),
       ['sample_id', 'ds', 'var'] + [f'mse_{m}' for m in METHODS] + [f'thr_{m}' for m in METHODS],
       jh_rows)

    jh_sum = []
    jh_set = set(jh_ids)
    rest = [sid for sid in com4_all if not feats[sid]['near_constant'] and sid not in jh_set]
    for grp_name, grp in (('joint_hard', jh_ids), ('rest', rest)):
        for (mv, mb) in PAIRS:
            mvv = float(np.mean([met[sid]['mse'][mv] for sid in grp])) if grp else None
            mbb = float(np.mean([met[sid]['mse'][mb] for sid in grp])) if grp else None
            if mvv is None:
                ab = rl = None
            else:
                ab = mvv - mbb
                rl = (mvv - mbb) / (mvv + mbb + 1e-12)
            jh_sum.append([grp_name, f'{mv}_vs_{mb}', len(grp),
                           '%.12g' % mvv if mvv is not None else '',
                           '%.12g' % mbb if mbb is not None else '',
                           '%.12g' % ab if ab is not None else '',
                           '%.12g' % rl if rl is not None else ''])
    wr(os.path.join(save_dir, 'joint_hard_summary_t46.csv'),
       ['group', 'pair', 'n', 'mse_vts', 'mse_base', 'abs_gap', 'rel_gap'], jh_sum)

    # ---------- 案例(确定性;四者共同成功、非近常数) ----------
    case_rows = []
    for ds in DS_ORDER:
        cands = [sid for sid in com4_all
                 if met[sid]['ds'] == ds and not feats[sid]['near_constant']]
        better, worse = [], []
        for sid in cands:
            mse4 = [met[sid]['mse'][m] for m in METHODS]
            order = np.argsort(mse4, kind='stable')  # 平手按方法序 vts<moirai<ets<arima
            rank_v = int(np.where(order == 0)[0][0])
            if rank_v == 0 and mse4[0] < 0.8 * mse4[order[1]]:
                better.append((mse4[order[1]] / mse4[0], sid, rank_v))
            if rank_v == 3 and mse4[3] > 1.25 * mse4[order[2]]:
                worse.append((mse4[3] / mse4[order[2]], sid, rank_v))
        for cls, lst in (('better', better), ('worse', worse)):
            lst.sort(key=lambda x: (-x[0], x[1]))
            for margin, sid, rank_v in lst[:2]:
                case_rows.append([cls, ds, sid, feats[sid]['var'], '%.4f' % margin, rank_v]
                                 + ['%.12g' % met[sid]['mse'][m] for m in METHODS])
    wr(os.path.join(save_dir, 'cases_t46.csv'),
       ['case', 'ds', 'sample_id', 'var', 'margin', 'rank_vts']
       + [f'mse_{m}' for m in METHODS], case_rows)

    meta = dict(n_total=1792, n_near_constant=n_nc,
                near_constant_ids=[sid for sid in sids if feats[sid]['near_constant']],
                n_common4=len(com4_all),
                direction_summary=[dict(zip(
                    ['feature', 'pair', 'n_cells_valid', 'n_high_gt_low',
                     'n_low_gt_high', 'n_monotone', 'median_delta'], r)) for r in dir_rows],
                n_joint_hard=len(jh_ids), n_cases=len(case_rows),
                ledger=dict([('{}_{}'.format(m, s), c) for (m, s), c in sorted(ledger_stat.items())]))
    with open(os.path.join(save_dir, 'analyze_meta_t46.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('analyze done: nc=%d com4=%d joint_hard=%d cases=%d'
          % (n_nc, len(com4_all), len(jh_ids), len(case_rows)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'save_t46_histdiag')
