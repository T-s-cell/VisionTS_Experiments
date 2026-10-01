"""Aggregate the 96 full-shot runs and compare against the paper.

Outputs (to ../repro_fullshot/outputs/):
  summary.csv          raw per-run rows (96): mse/mae + best_valid_loss/epoch
  aggregate.csv        per (DS,pred): 3-seed mean, std (ddof=1), per-dataset avg
  comparison.md        verdicts vs Table 21 (primary) / Table 20 std (reference)

Pre-declared criteria: per-cell |mean - paper| <= 0.01 -> "一致",
0.01 < |d| <= 0.02 -> "基本一致", > 0.02 -> "偏差". Overall pass: all cells
(incl. avg column) <= 0.02. Paper-internal Table20-vs-Table21 mismatches
(ECL-192/336/720) are recorded separately, never reconciled.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_BASE = os.path.join(os.getcwd(), 'save_fullshot')
OUT_DIR = os.path.join(HERE, 'outputs')

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Illness', 'Weather', 'Traffic', 'Electricity']
PREDS = {ds: [24, 36, 48, 60] if ds == 'Illness' else [96, 192, 336, 720] for ds in DATASETS}
SEEDS = [2021, 2022, 2023]

# Paper Table 21 (full results of Table 4), VisionTS column: (MSE, MAE)
PAPER_T21 = {
    ('ETTh1', 96): (0.347, 0.376), ('ETTh1', 192): (0.385, 0.400),
    ('ETTh1', 336): (0.407, 0.415), ('ETTh1', 720): (0.439, 0.443),
    ('ETTh1', 'avg'): (0.395, 0.409),
    ('ETTh2', 96): (0.269, 0.328), ('ETTh2', 192): (0.332, 0.374),
    ('ETTh2', 336): (0.351, 0.395), ('ETTh2', 720): (0.390, 0.430),
    ('ETTh2', 'avg'): (0.336, 0.382),
    ('ETTm1', 96): (0.281, 0.322), ('ETTm1', 192): (0.322, 0.353),
    ('ETTm1', 336): (0.356, 0.379), ('ETTm1', 720): (0.391, 0.413),
    ('ETTm1', 'avg'): (0.338, 0.367),
    ('ETTm2', 96): (0.169, 0.256), ('ETTm2', 192): (0.225, 0.294),
    ('ETTm2', 336): (0.278, 0.334), ('ETTm2', 720): (0.372, 0.392),
    ('ETTm2', 'avg'): (0.261, 0.319),
    ('Illness', 24): (2.034, 0.937), ('Illness', 36): (1.866, 0.888),
    ('Illness', 48): (1.784, 0.870), ('Illness', 60): (1.910, 0.912),
    ('Illness', 'avg'): (1.899, 0.902),
    ('Weather', 96): (0.142, 0.192), ('Weather', 192): (0.191, 0.238),
    ('Weather', 336): (0.246, 0.282), ('Weather', 720): (0.328, 0.337),
    ('Weather', 'avg'): (0.227, 0.262),
    ('Traffic', 96): (0.344, 0.236), ('Traffic', 192): (0.372, 0.249),
    ('Traffic', 336): (0.383, 0.257), ('Traffic', 720): (0.422, 0.280),
    ('Traffic', 'avg'): (0.380, 0.256),
    ('Electricity', 96): (0.126, 0.218), ('Electricity', 192): (0.144, 0.237),
    ('Electricity', 336): (0.162, 0.256), ('Electricity', 720): (0.192, 0.286),
    ('Electricity', 'avg'): (0.156, 0.249),
}

# Paper Table 20, VisionTS mean±std over three runs (reference only)
PAPER_T20_STD = {
    ('ETTh1', 96): (0.002, 0.000), ('ETTh1', 192): (0.001, 0.000),
    ('ETTh1', 336): (0.001, 0.001), ('ETTh1', 720): (0.001, 0.000),
    ('ETTh2', 96): (0.003, 0.002), ('ETTh2', 192): (0.001, 0.001),
    ('ETTh2', 336): (0.002, 0.002), ('ETTh2', 720): (0.003, 0.002),
    ('ETTm1', 96): (0.001, 0.001), ('ETTm1', 192): (0.006, 0.002),
    ('ETTm1', 336): (0.003, 0.002), ('ETTm1', 720): (0.001, 0.001),
    ('ETTm2', 96): (0.003, 0.002), ('ETTm2', 192): (0.003, 0.003),
    ('ETTm2', 336): (0.002, 0.001), ('ETTm2', 720): (0.002, 0.002),
    ('Weather', 96): (0.000, 0.001), ('Weather', 192): (0.000, 0.000),
    ('Weather', 336): (0.003, 0.001), ('Weather', 720): (0.004, 0.001),
    ('Traffic', 96): (0.001, 0.000), ('Traffic', 192): (0.001, 0.001),
    ('Traffic', 336): (0.001, 0.001), ('Traffic', 720): (0.001, 0.000),
    ('Electricity', 96): (0.000, 0.000), ('Electricity', 192): (0.001, 0.001),
    ('Electricity', 336): (0.001, 0.001), ('Electricity', 720): (0.000, 0.000),
}

PAPER_INTERNAL_MISMATCH = [
    'Electricity/192: Table20 mean 0.146/0.239 vs Table21 0.144/0.237',
    'Electricity/336: Table20 mean 0.161/0.255 vs Table21 0.162/0.256',
    'Electricity/720: Table20 mean 0.193/0.286 vs Table21 0.192/0.286',
]

JOB_NAMES = {'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2', 'ETTm1': 'ETTm1', 'ETTm2': 'ETTm2',
             'Illness': 'illness', 'Weather': 'Weather', 'Traffic': 'Traffic',
             'Electricity': 'Electricity'}


def verdict(delta):
    if delta <= 0.01:
        return '一致'
    if delta <= 0.02:
        return '基本一致'
    return '偏差'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = []
    for ds in DATASETS:
        for pred in PREDS[ds]:
            for seed in SEEDS:
                job = f'{JOB_NAMES[ds]}_{pred}_seed{seed}'
                mpath = os.path.join(SAVE_BASE, job, 'results', '_', 'metrics.npy')
                if not os.path.isfile(mpath):
                    print(f'[MISSING] {job}')
                    continue
                m = np.load(mpath)
                assert np.isfinite(m).all(), f'{job}: non-finite {m}'
                raw.append(dict(dataset=ds, pred=pred, seed=seed, job=job,
                                mae=float(m[0]), mse=float(m[1]),
                                rmse=float(m[2]), best_valid_loss=float(m[5]),
                                best_valid_epoch=float(m[6])))

    with open(os.path.join(OUT_DIR, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)
    print(f'summary.csv: {len(raw)} rows (expect 96)')

    def mean(vals):
        return sum(vals) / len(vals)

    def std(vals):  # sample std, ddof=1 (pre-declared)
        if len(vals) < 2:
            return 0.0
        m = mean(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    agg_rows = []
    lines = ['# VisionTS LN full-shot reproduction — comparison vs paper',
             '',
             'Criteria (pre-declared): per-cell |mean−paper| ≤0.01 一致; ≤0.02 基本一致; >0.02 偏差.',
             'std = sample std over 3 seeds (ddof=1). Table 20 used as reference only.',
             '',
             '## Per (dataset, pred_len): 3-seed mean±std vs Table 21',
             '',
             '| dataset | pred | MSE mean±std | paper MSE | Δ | MSE判定 | MAE mean±std | paper MAE | Δ | MAE判定 | paper std (MSE/MAE) |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    max_delta = 0.0
    for ds in DATASETS:
        for pred in PREDS[ds]:
            rows = [r for r in raw if r['dataset'] == ds and r['pred'] == pred]
            if len(rows) != 3:
                lines.append(f'| {ds} | {pred} | INCOMPLETE ({len(rows)}/3 runs) | — | — | — | — | — | — | — | — |')
                continue
            mses = [r['mse'] for r in rows]
            maes = [r['mae'] for r in rows]
            mse_m, mae_m = mean(mses), mean(maes)
            mse_s, mae_s = std(mses), std(maes)
            p_mse, p_mae = PAPER_T21[(ds, pred)]
            d_mse, d_mae = abs(mse_m - p_mse), abs(mae_m - p_mae)
            max_delta = max(max_delta, d_mse, d_mae)
            p_std = PAPER_T20_STD.get((ds, pred), ('—', '—'))
            agg_rows.append(dict(dataset=ds, pred=pred, mse_mean=mse_m, mse_std=mse_s,
                                 mae_mean=mae_m, mae_std=mae_s,
                                 paper_mse=p_mse, paper_mae=p_mae))
            lines.append(
                f'| {ds} | {pred} | {mse_m:.3f}±{mse_s:.3f} | {p_mse:.3f} | {mse_m - p_mse:+.3f} | {verdict(d_mse)} '
                f'| {mae_m:.3f}±{mae_s:.3f} | {p_mae:.3f} | {mae_m - p_mae:+.3f} | {verdict(d_mae)} '
                f'| ±{p_std[0]}/±{p_std[1]} |')

    lines += ['', '## Per-dataset avg over 4 pred_lens vs Table 21 avg', '',
              '| dataset | MSE avg | paper | Δ | 判定 | MAE avg | paper | Δ | 判定 |',
              '|---|---|---|---|---|---|---|---|---|']
    for ds in DATASETS:
        rows = [r for r in agg_rows if r['dataset'] == ds]
        if len(rows) != 4:
            lines.append(f'| {ds} | INCOMPLETE | — | — | — | — | — | — | — |')
            continue
        mse_avg = mean([r['mse_mean'] for r in rows])
        mae_avg = mean([r['mae_mean'] for r in rows])
        p_mse, p_mae = PAPER_T21[(ds, 'avg')]
        d_mse, d_mae = abs(mse_avg - p_mse), abs(mae_avg - p_mae)
        max_delta = max(max_delta, d_mse, d_mae)
        lines.append(f'| {ds} | {mse_avg:.3f} | {p_mse:.3f} | {mse_avg - p_mse:+.3f} | {verdict(d_mse)} '
                     f'| {mae_avg:.3f} | {p_mae:.3f} | {mae_avg - p_mae:+.3f} | {verdict(d_mae)} |')

    lines += ['', '## Paper-internal Table20 vs Table21 mismatches (recorded, not reconciled)', '']
    lines += [f'- {m}' for m in PAPER_INTERNAL_MISMATCH]
    lines += ['', f'Overall max |Δ| across all cells: {max_delta:.4f}',
              f'**Verdict: {"数值复现达标（阈值 0.02）" if max_delta <= 0.02 else "存在偏差格，见上表"}**',
              '运行完成（96 SUCCESS）与数值复现达标为两个独立结论；本节仅为数值判定。']

    with open(os.path.join(OUT_DIR, 'comparison.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    with open(os.path.join(OUT_DIR, 'aggregate.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    print('comparison.md + aggregate.csv written; max |Δ| =', round(max_delta, 4))


if __name__ == '__main__':
    main()
