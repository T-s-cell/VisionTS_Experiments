"""EXP-047h(预测距离×误差分解,零推理):公共常量/路径/SHA 记录。

只读复用 EXP-047 v2 冻结产物(分支 repro/ltsf-t47-levelshift HEAD 45b7e0c,
审计包 temp/VisionTS_t47_levelshift_20261119_v2.tar.gz sha 46ad7eb1…);
不重新推理/训练/筛窗。协议见 protocol_t47h.md。

固定区间:b1–b4 各 24 步(同一 H=96 预测的切片);full=[0,96) 仅原结果核对与
单列汇报。ETTh(P=24)每区间 1P;ETTm(P=96)每区间 0.25P、full 1P。
容差:atol=1e-9、rtol=1e-10(|a−b| ≤ atol + rtol·|b|);真值核对 1e-5。
"""
import csv
import json
import os
import sys

_T47 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'repro_t47_levelshift'))
sys.path.insert(0, _T47)
from t47_common import (DELTA_MULTS, DS_CFG, DS_ORDER, M_MULTS, METHODS,  # noqa: F401
                        baseline_list_path, cond_params, manifest_path, sha256_file)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H = 96
BANDS = [('b1', 0, 24), ('b2', 24, 48), ('b3', 48, 72), ('b4', 72, 96)]
INTERVALS = BANDS + [('full', 0, H)]  # full 仅核对/单列,不作第五时段解释
BAND_IDS = [b[0] for b in BANDS]
TOL_ATOL = 1e-9
TOL_RTOL = 1e-10
TRUE_TOL = 1e-5
SAVE_SUB = 'save_t47_levelshift'
CONTRASTS = [('vts', 'moirai'), ('vts', 'autoets'), ('vts', 'autoarima')]  # 主比较在前
PAIRS_H = 'pairs_horizon_t47h.csv'
DIFF_H = 'method_diff_horizon_t47h.csv'

INPUT_FILES = (['baseline_list_t47.csv', 'manifest_t47.csv']
               + ['%s/baseline_%s.npz' % (SAVE_SUB, ds) for ds in DS_ORDER]
               + ['%s/pred_%s_t47.npz' % (SAVE_SUB, m) for m in METHODS]
               + ['%s/per_sample_metrics_t47.csv' % SAVE_SUB,
                  '%s/pairs_t47.csv' % SAVE_SUB,
                  '%s/method_diff_t47.csv' % SAVE_SUB])
assert len(INPUT_FILES) == 13


def input_path(repo_root, rel):
    return os.path.join(repo_root, rel)


def check_inputs(repo_root):
    missing = [rel for rel in INPUT_FILES if not os.path.exists(input_path(repo_root, rel))]
    if missing:
        raise SystemExit('[t47h] missing inputs (fail-stop): %s' % missing)


def record_shas(repo_root, out_dir):
    """compute 前调用:记录 13 输入 + 本目录代码 SHA256(缺输入即停)。"""
    check_inputs(repo_root)
    code = {}
    for fn in sorted(os.listdir(SCRIPT_DIR)):
        if fn.endswith('.py') or fn == 'protocol_t47h.md':
            code[fn] = sha256_file(os.path.join(SCRIPT_DIR, fn))
    doc = {'input_root': os.path.abspath(repo_root),
           'inputs': {rel: sha256_file(input_path(repo_root, rel)) for rel in INPUT_FILES},
           'code': code}
    with open(os.path.join(out_dir, 'input_shas_t47h.json'), 'w') as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    return doc


def tol_diff(a, b):
    """门禁量:|a−b| − (atol + rtol·|b|);≤0 即过。"""
    return abs(a - b) - (TOL_ATOL + TOL_RTOL * abs(b))


def read_csv_dicts(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_manifest(repo_root):
    man = read_csv_dicts(manifest_path())
    assert len(man) == 7168, len(man)
    rgs = [int(r['row_global']) for r in man]
    assert len(set(rgs)) == 7168
    for r in man:
        di, cid, g = int(r['delta_idx']), int(r['cond_id']), r['group']
        if g == 'A':
            assert cid == di, (r['row_global'], cid, di)
        else:
            mi = int(r['m_idx'])
            assert cid == 4 + 3 * di + mi, (r['row_global'], cid, di, mi)
            assert int(r['m_steps']) == M_MULTS[mi] * int(r['P']), r['row_global']
            assert int(r['m_over_P']) == M_MULTS[mi], r['row_global']
        assert float(r['delta_mult']) == DELTA_MULTS[di], r['row_global']
    return man


def pair_index(man):
    """(sid_t46, delta_idx, m_idx) → 配对元数据;不依赖数组行序。

    rg_A/rg_B 由 manifest 条件表建立:A cond_id = delta_idx,B cond_id = 4+3·delta_idx+m_idx
    (load_manifest 已核 cond_id 一致)。"""
    A_rg, B_rg, meta = {}, {}, {}
    for r in man:
        sid, di = int(r['sid_t46']), int(r['delta_idx'])
        rg = int(r['row_global'])
        meta[rg] = r
        if r['group'] == 'A':
            assert (sid, di) not in A_rg
            A_rg[(sid, di)] = rg
        else:
            assert (sid, di, int(r['m_idx'])) not in B_rg
            B_rg[(sid, di, int(r['m_idx']))] = rg
    out = {}
    for (sid, di), rg_a in A_rg.items():
        ra = meta[rg_a]
        for mi in range(3):
            rg_b = B_rg[(sid, di, mi)]
            rb = meta[rg_b]
            assert ra['ds'] == rb['ds'] and ra['var'] == rb['var'] and ra['channel'] == rb['channel']
            delta = float(ra['delta'])
            assert delta == float(rb['delta']) and delta != 0.0
            out[(sid, di, mi)] = dict(sid=sid, di=di, mi=mi, rg_a=rg_a, rg_b=rg_b,
                                      ds=ra['ds'], var=ra['var'], channel=ra['channel'],
                                      delta_mult=float(ra['delta_mult']), delta=delta,
                                      m_over_P=M_MULTS[mi], P=int(ra['P']))
    assert len(out) == 5376, len(out)
    return out


def band_period_note(ds):
    """区间周期比例标注:ETTh(P=24)每 band=1P;ETTm(P=96)每 band=0.25P。"""
    P = DS_CFG[ds]['P']
    ratio = 24.0 / P
    return '%gP' % ratio if ratio != int(ratio) else '%dP' % int(ratio)
