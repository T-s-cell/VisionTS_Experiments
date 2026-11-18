"""EXP-047 冒烟门禁(S1 双层复现 / S2 δ=0 实推 / S3 构造恒等 / S4 id 对齐)。

env 不同分部运行:--part vts(visionts env)、moirai(t46diag env)、sfdx(t46diag env)、
finalize(任意含 numpy env;S3+S4+汇总)。每部写 smoke_t47_<part>.json;
finalize 汇总 smoke_report_t47.json,任一检查 FAIL 或缺部 → 退出码 1(不续全量)。

S1-a t43 原组织复现:batch32×7通道 [32,L,7] forward(t43 同 env/权重/配置:
mae_base/ln/ckpt sha 核验/bilinear/norm=align=0.4/periodicity=P/seed 2021),
冒烟窗通道输出 vs arr_vts_t46.npz 切片 ≤1e-5;
S1-b 新组织核对:同批输入走 t47 单变量 [32,L,1] 路径 vs S1-a ≤5e-4
(两项实测差值入 json;超限先排查,不自动放宽);
S2 δ=0:A/B 构造输入逐位相同,两次实际推理(每次前重置种子 2021)预测逐位一致;
S3 配对恒等式:A/B 末 m 步逐位相同、B 末点 = x[-1]+δ、真值 y'=y+δ 共享,
manifest δ 与 s·delta_mult 一致(rtol 1e-11,%.12g 落盘精度);
S4 对齐:npz item/cond/row_global ↔ manifest 冒烟子集逐行一致,
true_p vs f64 重算(y+δ)≤1e-5,sfdx ok 行有限/失败行 NaN。
"""
import csv
import json
import os
import random
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOL_TRUE = 1e-5

ALL_PARTS = ('vts', 'moirai', 'sfdx')


def set_seeds(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_smoke(save_dir):
    from t47_common import baseline_list_path, manifest_path, smoke_subset
    with open(manifest_path()) as f:
        man = list(csv.DictReader(f))
    smoke = smoke_subset(man)
    base = {}
    for ds in sorted({r['ds'] for r in smoke}):
        base[ds] = np.load(os.path.join(save_dir, 'baseline_%s.npz' % ds))
    return man, smoke, base


def write_part(part, checks, extra=None):
    doc = dict(part=part, checks=checks,
               all_pass=all(c['pass'] for c in checks.values()),
               ts=time.strftime('%F %T'))
    if extra:
        doc.update(extra)
    p = os.path.join(SCRIPT_DIR, 'smoke_t47_%s.json' % part)
    with open(p, 'w') as f:
        json.dump(doc, f, indent=1)
    print('[smoke %s] %s -> %s' % (part, 'PASS' if doc['all_pass'] else 'FAIL', p))
    return doc['all_pass']


def chk(name, ok, detail):
    d = dict(name=name, **detail)
    d['pass'] = bool(ok)
    return d


# ---------------- part vts (visionts env) ----------------

def part_vts(repo_root, save_dir):
    import torch
    from t47_common import (DS_CFG, DELTA_MULTS, M_MULTS, SMOKE_TOL_NEW, SMOKE_TOL_T43,
                            T46_DIR, apply_variant)
    from t46_common import load_csv, std_all, train_scaler

    man, smoke, base = load_smoke(save_dir)
    checks = {}
    std_cache, vm = {}, {}
    d_1a_all, d_1b_all, d2_all, n2 = [], [], [], 0

    def load_vm43(ds, L, P):
        if ds in vm:
            return vm[ds]
        from infer_vts_t47 import load_vm
        vm[ds] = load_vm(repo_root, L, P)  # t47 与 t43 配置逐参相同(sha 核验在 load_vm 内)
        return vm[ds]

    for ds in ('ETTh1', 'ETTm1'):
        rows = [r for r in smoke if r['ds'] == ds]
        cfg = DS_CFG[ds]
        L, P = cfg['L'], cfg['P']
        if ds not in std_cache:
            data = load_csv(os.path.join(repo_root, cfg['csv']))
            mean, scale = train_scaler(data, cfg['K'])
            std_cache[ds] = std_all(data, mean, scale)
        std = std_cache[ds]
        arr46 = np.load(os.path.join(T46_DIR, 'save_t46_histdiag', 'arr_vts_t46.npz'))
        ref_pred = arr46['%s_pred' % ds]

        # 该 ds 冒烟窗所在固定 batch(32 连续 val 窗;两窗同 batch 时只推理一次)
        batches = {}
        for r in rows:
            ai = int(r['arr_index'])
            batches.setdefault((ai // 32) * 32, []).append(r)
        out43, out1v = {}, {}
        d_1a, d_1b = [], []
        for bstart, brows in sorted(batches.items()):
            chs = {int(r['channel']) for r in brows}
            assert len(chs) == 1, chs  # 冒烟窗同 var → 同通道
            ch = chs.pop()
            xb = np.empty((32, L, 7), dtype=np.float32)
            for j in range(32):
                st = cfg['K'] * 12 + bstart + j
                xb[j] = std[st - L:st, :].astype(np.float32)
            set_seeds(2021)
            vm43 = load_vm43(ds, L, P)
            with torch.no_grad():
                o43 = vm43.forward(torch.tensor(xb).cuda()).cpu().numpy()   # [32,96,7]
            set_seeds(2021)
            with torch.no_grad():
                o1v = vm43.forward(torch.tensor(xb[:, :, ch:ch + 1]).cuda()).cpu().numpy()  # [32,96,1]
            out43[bstart], out1v[bstart] = o43, o1v
        for r in rows:
            ai, ch = int(r['arr_index']), int(r['channel'])
            bstart = (ai // 32) * 32
            loc = ai - bstart
            d_1a_all.append(float(np.max(np.abs(
                out43[bstart][loc, :, ch].astype(np.float64) - ref_pred[ai].astype(np.float64)))))
            d_1b_all.append(float(np.max(np.abs(
                out1v[bstart][loc, :, 0].astype(np.float64) - out43[bstart][loc, :, ch].astype(np.float64)))))

        # S2:δ=0 A/B 逐位相同,两次实际推理逐位一致(每 ds 2 个不同窗口)
        sids = []
        for r in rows:
            if r['sid_t46'] not in sids:
                sids.append(r['sid_t46'])
        assert len(sids) == 2
        sid_row = {int(s): i for i, s in enumerate(base[ds]['sid_t46'])}
        xs = []
        for sid in sids:
            x = base[ds]['x'][sid_row[int(sid)]]
            xa = apply_variant(x, 0.0, P, 'A')
            xb0 = apply_variant(x, 0.0, P, 'B')
            assert np.array_equal(xa, xb0) and np.array_equal(xa, x)
            xs.append(xa.astype(np.float32))
        xt = torch.tensor(np.stack(xs)).unsqueeze(-1).cuda()   # [2,L,1]
        set_seeds(2021)
        with torch.no_grad():
            p1 = vm43.forward(xt).cpu().numpy()
        set_seeds(2021)
        with torch.no_grad():
            p2 = vm43.forward(xt).cpu().numpy()
        d2_all.append(float(np.max(np.abs(p1.astype(np.float64) - p2.astype(np.float64)))))
        n2 += len(xs)

    checks['S1a_t43_repro'] = chk('S1a', max(d_1a_all) <= SMOKE_TOL_T43,
                                  dict(maxdiff=max(d_1a_all), tol=SMOKE_TOL_T43, n=len(d_1a_all)))
    checks['S1b_univar_path'] = chk('S1b', max(d_1b_all) <= SMOKE_TOL_NEW,
                                    dict(maxdiff=max(d_1b_all), tol=SMOKE_TOL_NEW, n=len(d_1b_all)))
    checks['S2_vts_delta0'] = chk('S2vts', max(d2_all) == 0.0,
                                  dict(maxdiff=max(d2_all), tol=0.0, n=n2))
    return write_part('vts', checks)


# ---------------- part moirai (t46diag env) ----------------

def part_moirai(save_dir):
    import pandas as pd
    from t47_common import DS_CFG, SEED, apply_variant
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    man, smoke, base = load_smoke(save_dir)
    checks = {}
    module = MoiraiModule.from_pretrained('Salesforce/moirai-1.0-R-base')
    PATCH_SIZE, NUM_SAMPLES, BATCH_SIZE, H = 32, 200, 64, 96
    d_all = []
    for ds in ('ETTh1', 'ETTm1'):
        rows = [r for r in smoke if r['ds'] == ds]
        sids = []
        for r in rows:
            if r['sid_t46'] not in sids:
                sids.append(r['sid_t46'])
        assert len(sids) == 2
        cfg = DS_CFG[ds]
        L, P = cfg['L'], cfg['P']
        sid_row = {int(s): i for i, s in enumerate(base[ds]['sid_t46'])}
        entries = []
        for sid in sids:  # 每 ds 2 个不同窗口,δ=0
            x = base[ds]['x'][sid_row[int(sid)]]
            xa = apply_variant(x, 0.0, P, 'A')
            xb0 = apply_variant(x, 0.0, P, 'B')
            assert np.array_equal(xa, xb0) and np.array_equal(xa, x)
            entries.append(dict(target=xa.astype(np.float32),
                                start=pd.Period('2020-01-01', freq=cfg['freq']),
                                feat_static_cat=[0], item_id=int(r['sid_t46'])))
        gds = ListDataset(entries, freq=cfg['freq'])
        preds = []
        for _ in range(2):  # 两次实际推理,各前重置种子
            set_seeds(SEED)
            model = MoiraiForecast(module=module, prediction_length=H, context_length=L,
                                   patch_size=PATCH_SIZE, num_samples=NUM_SAMPLES,
                                   target_dim=1, feat_dynamic_real_dim=0,
                                   past_feat_dynamic_real_dim=0)
            predictor = model.create_predictor(batch_size=BATCH_SIZE)
            pts = {}
            for fcst in predictor.predict(gds):
                pts[int(fcst.item_id)] = np.median(np.asarray(fcst.samples), axis=0)
            preds.append(pts)
        for sid in sids:
            sid = int(sid)
            assert np.array_equal(preds[0][sid], preds[1][sid])
            d_all.append(0.0)
    checks['S2_moirai_delta0'] = chk('S2moirai', len(d_all) == 4,
                                     dict(maxdiff=0.0, tol=0.0, n=len(d_all)))
    return write_part('moirai', checks)


# ---------------- part sfdx (t46diag env) ----------------

def part_sfdx(save_dir):
    from t47_common import DS_CFG, apply_variant

    man, smoke, base = load_smoke(save_dir)
    from statsforecast.models import AutoETS
    ds = 'ETTh1'
    cfg = DS_CFG[ds]
    L, P = cfg['L'], cfg['P']
    sid_row = {int(s): i for i, s in enumerate(base[ds]['sid_t46'])}
    sids = []
    for r in smoke:
        if r['ds'] == ds and r['sid_t46'] not in sids:
            sids.append(r['sid_t46'])
        if len(sids) == 2:
            break
    d_all = []
    for sid in sids:
        x = base[ds]['x'][sid_row[int(sid)]]
        xa = apply_variant(x, 0.0, P, 'A')
        xb0 = apply_variant(x, 0.0, P, 'B')
        assert np.array_equal(xa, xb0)
        ps = []
        for _ in range(2):
            m = AutoETS(season_length=P)
            m.fit(xa.astype(np.float64))
            ps.append(np.asarray(m.predict(h=96)['mean'], dtype=np.float64))
        assert np.array_equal(ps[0], ps[1])
        d_all.append(0.0)
    checks = {'S2_sfdx_delta0': chk('S2sfdx', len(d_all) == 2,
                                    dict(maxdiff=0.0, tol=0.0, n=len(d_all)))}
    return write_part('sfdx', checks)


# ---------------- finalize (numpy only) ----------------

def part_finalize(save_dir):
    from t47_common import (DELTA_MULTS, M_MULTS, apply_variant, smoke_subset)

    with open(os.path.join(SCRIPT_DIR, 'manifest_t47.csv')) as f:
        man = list(csv.DictReader(f))
    smoke = smoke_subset(man)
    checks = {}

    # S3:构造恒等式(全部 64 冒烟行;按 (sid, delta_idx) 配 A 与 3 个 B)
    by_sid_di = {}
    for r in smoke:
        by_sid_di.setdefault((r['ds'], int(r['sid_t46']), int(r['delta_idx'])), {})[
            int(r['cond_id'])] = r
    n_ident = 0
    for (ds, sid, di), grp in sorted(by_sid_di.items()):
        bz = base_npz(save_dir, ds)
        sid_row = {int(s): i for i, s in enumerate(bz['sid_t46'])}
        x, y = bz['x'][sid_row[sid]], bz['y'][sid_row[sid]]
        P = int(next(iter(grp.values()))['P'])
        delta = float(next(iter(grp.values()))['delta'])
        assert abs(delta - float(bz['s'][sid_row[sid]]) * DELTA_MULTS[di]) <= 1e-11 * max(1.0, abs(delta))
        for cid, r in grp.items():
            assert abs(float(r['delta']) - delta) == 0.0
            if r['group'] == 'A':
                m_steps, xa = 0, apply_variant(x, delta, 0, 'A')
                assert np.array_equal(xa, x + delta)
            else:
                m_steps = int(r['m_steps'])
                assert m_steps == M_MULTS[int(r['m_idx'])] * P
                xb = apply_variant(x, delta, m_steps, 'B')
                assert np.array_equal(xb[-m_steps:], (x + delta)[-m_steps:])
                assert xb[-1] == x[-1] + delta
                assert np.array_equal(xb[:-m_steps], x[:-m_steps])
            yp = y + delta
            assert np.isfinite(yp).all() and np.isfinite(x).all()
            n_ident += 1
    checks['S3_pair_identity'] = chk('S3', n_ident == 64, dict(n=n_ident, expect=64))

    # S4:npz ↔ manifest 对齐 + true_p vs f64 重算
    expected = dict(item=np.array([int(r['sid_t46']) for r in smoke], dtype=np.int64),
                    cond=np.array([int(r['cond_id']) for r in smoke], dtype=np.int64),
                    rg=np.array([int(r['row_global']) for r in smoke], dtype=np.int64))
    sfdx_status = {}
    for name, fn in (('vts', 'pred_vts_t47_smoke.npz'),
                     ('moirai', 'pred_moirai_t47_smoke.npz'),
                     ('autoets', 'pred_autoets_t47_smoke.npz'),
                     ('autoarima', 'pred_autoarima_t47_smoke.npz')):
        p = os.path.join(save_dir, fn)
        z = np.load(p)
        emask = np.isin(expected['rg'], z['row_globals'])
        ok = (np.array_equal(z['item_ids'], expected['item'][emask])
              and np.array_equal(z['cond_ids'], expected['cond'][emask])
              and np.array_equal(z['row_globals'], expected['rg'][emask]))
        # true_p vs f64 重算
        dmax = 0.0
        for j, rg in enumerate(z['row_globals']):
            r = next(q for q in smoke if int(q['row_global']) == int(rg))
            bz = base_npz(save_dir, r['ds'])
            sid_row = {int(s): i for i, s in enumerate(bz['sid_t46'])}
            yp = (bz['y'][sid_row[int(r['sid_t46'])]] + float(r['delta'])).astype(np.float64)
            dmax = max(dmax, float(np.max(np.abs(z['true_p'][j].astype(np.float64) - yp))))
        ok = ok and dmax <= TOL_TRUE
        if 'status' in z.files:
            st = z['status']
            okp = all(np.isfinite(z['pred'][j]).all() for j in range(len(st)) if st[j] == 'ok')
            okn = all(not np.isfinite(z['pred'][j]).any() for j in range(len(st)) if st[j] != 'ok')
            sfdx_status[name] = dict(ok=sum(1 for s in st if s == 'ok'),
                                     fail=sum(1 for s in st if s != 'ok'))
            ok = ok and okp and okn
        else:
            ok = ok and bool(np.isfinite(z['pred']).all())
        checks['S4_align_%s' % name] = chk('S4_%s' % name, ok,
                                           dict(true_p_maxdiff=dmax, n=int(len(z['row_globals'])),
                                                **sfdx_status.get(name, {})))

    # 汇总
    report = dict(parts={}, all_pass=all(c['pass'] for c in checks.values()))
    for part in ALL_PARTS:
        p = os.path.join(SCRIPT_DIR, 'smoke_t47_%s.json' % part)
        if os.path.exists(p):
            with open(p) as f:
                doc = json.load(f)
            report['parts'][part] = doc
            report['all_pass'] = report['all_pass'] and doc['all_pass']
        else:
            report['parts'][part] = dict(missing=True)
            report['all_pass'] = False
    with open(os.path.join(SCRIPT_DIR, 'smoke_report_t47.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print('[smoke finalize] all_pass=%s -> smoke_report_t47.json' % report['all_pass'])
    return report['all_pass']


def base_npz(save_dir, ds):
    global _BASE
    try:
        _BASE
    except NameError:
        _BASE = {}
    if ds not in _BASE:
        _BASE[ds] = np.load(os.path.join(save_dir, 'baseline_%s.npz' % ds))
    return _BASE[ds]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    part = None
    for a in sys.argv[1:]:
        if a.startswith('--part'):
            part = a.split('=', 1)[1] if '=' in a else sys.argv[sys.argv.index(a) + 1]
    assert part in ('vts', 'moirai', 'sfdx', 'finalize'), part
    repo_root = args[0]
    save_dir = args[1] if len(args) > 1 else os.path.join(SCRIPT_DIR, 'save_t47_levelshift')
    fn = dict(vts=part_vts, moirai=part_moirai, sfdx=part_sfdx,
              finalize=lambda r, s: part_finalize(s))[part]
    ok = fn(repo_root, save_dir)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
