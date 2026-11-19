"""S1-b 排查 v2(用户指定三项核查;非门禁,不改容差,不启动全量)。

A. 出处:实际模型类/源码路径/sha/commit/torch/GPU/ckpt sha;嵌入 diag_s1b_t47.json
   与 smoke_t47_vts.json 全文。
B. 通道独立性与"仅改其他 6 通道":同 [32,L,7] 输入,目标通道逐位不动,其余 6 通道
   内容改写(*-1+0.5),比较目标通道预测(预期逐位相同 → 无跨通道依赖);
   另核 [32,L,1] vs [32,L,7] 组织差(即 S1-b)。
C. 首个差异计算步骤:逐级 hook(resize 渲染/patch_embed/12 encoder 块/enc norm/
   decoder_embed/8 decoder 块/dec norm/decoder_pred/最终 y),定位组织差首次出现处。
D. 组织差对 D 与冲突代价的量化:每 ds 前 8 基准窗,δ=±1.0·s、m=P,单变量与
   7 通道两种组织各算 A/B 预测 → D、cost 逐窗对比(绝对差+相对典型量级)。
"""
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from smoke_t47 import set_seeds
from t47_common import DS_CFG, DS_ORDER, T46_DIR
from t46_common import load_csv, std_all, train_scaler


def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def provenance(repo_root):
    import torch
    from t47_common import CKPT_SHA
    vs_root = os.path.dirname(os.path.abspath(repo_root))
    prov = dict(
        model_class='VisionTS (visionts.model.VisionTS; infer_vts_t47.load_vm 实例化 '
                    "VisionTS(arch='mae_base', finetune_type='ln', load_ckpt=True))",
        channel_handling='Channel Independent: model.py L115 rearrange b s n -> b n s;'
                         ' L119 (b n) 1 f p → 通道并入 batch 维,逐通道独立重建,无跨通道 attention',
        source_dir=vs_root + '/visionts',
        sha_model_py=sha256(os.path.join(vs_root, 'visionts', 'model.py')),
        sha_init_py=sha256(os.path.join(vs_root, 'visionts', '__init__.py')),
        sha_ckpt=CKPT_SHA,
        torch=torch.__version__, cuda=torch.version.cuda or '',
        gpu=torch.cuda.get_device_name(0),
        local_git=dict(branch='repro/ltsf-t47-levelshift',
                       head='40475fde6010ad3ccf2c6a85c36ae974e076f37b'),
    )
    try:
        head = subprocess.run(['git', '-C', vs_root, 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        prov['eta_git_head'] = head or 'no-git-or-empty'
    except Exception as e:
        prov['eta_git_head'] = 'error:%s' % e
    for fn in ('diag_s1b_t47.json', 'smoke_t47_vts.json'):
        p = os.path.join(SCRIPT_DIR, fn)
        prov[fn] = json.load(open(p)) if os.path.exists(p) else dict(missing=True)
    return prov


def build_batch(std, K, L, bstart):
    xb = np.empty((32, L, 7), dtype=np.float32)
    for j in range(32):
        st = K * 12 + bstart + j
        xb[j] = std[st - L:st, :].astype(np.float32)
    return xb


def main(repo_root, save_dir):
    import torch
    from infer_vts_t47 import load_vm
    from t47_common import DELTA_MULTS, M_MULTS

    doc = dict(purpose='S1b exceedance investigation v2 (user-directed), NOT a gate',
               provenance=provenance(repo_root))

    with open(os.path.join(SCRIPT_DIR, 'baseline_list_t47.csv')) as f:
        bl = list(csv.DictReader(f))
    diag1 = json.load(open(os.path.join(SCRIPT_DIR, 'diag_s1b_t47.json')))
    worst_key = max(diag1['per_window'], key=lambda k: diag1['per_window'][k]['max'])
    import re
    m = re.match(r'(.+)_w\d+_ai(\d+)$', worst_key)
    ds_worst, ai_worst = m.group(1), int(m.group(2))
    doc['partC_window'] = worst_key

    vm_cache, std_cache = {}, {}
    d_all, d_s1b_all = [], []

    # ---------------- Part B: 通道独立性 ----------------
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
            x7 = build_batch(std, K, L, bstart)
            x7mod = x7.copy()
            for c in range(7):
                if c != ch:
                    x7mod[:, :, c] = x7[:, :, c] * (-1.0) + 0.5   # 确定改写,目标通道不动
            set_seeds(2021)
            with torch.no_grad():
                o1 = vm.forward(torch.tensor(x7).cuda()).cpu().numpy()      # [32,96,7]
            set_seeds(2021)
            with torch.no_grad():
                o2 = vm.forward(torch.tensor(x7mod).cuda()).cpu().numpy()   # [32,96,7]
            set_seeds(2021)
            with torch.no_grad():
                o1v = vm.forward(torch.tensor(x7[:, :, ch:ch + 1]).cuda()).cpu().numpy()
            for r in brows:
                ai = int(r['arr_index'])
                loc = ai - bstart
                d_target = np.abs(o1[loc, :, ch].astype(np.float64)
                                  - o2[loc, :, ch].astype(np.float64))
                d_all.append(float(d_target.max()))
                doc.setdefault('partB_windows', {})[
                    '%s_ai%d' % (ds, ai)] = dict(
                    bitwise=np.array_equal(o1[loc, :, ch], o2[loc, :, ch]),
                    max=float(d_target.max()),
                    s1b_max=float(np.abs(o1[loc, :, ch].astype(np.float64)
                                         - o1v[loc, :, 0].astype(np.float64)).max()))
                d_s1b_all.append(doc['partB_windows']['%s_ai%d' % (ds, ai)]['s1b_max'])

    doc['partB'] = dict(
        n=len(d_all),
        target_bitwise_all=bool(all(
            v['bitwise'] for v in doc['partB_windows'].values())),
        target_max=float(max(d_all)),
        s1b_max=float(max(d_s1b_all)),
        interpretation='若 target_bitwise_all=True:其余 6 通道内容改变不影响目标通道'
                       '预测 → 通道独立,S1-b 差异全部来自批次组织(224 vs 32)的数值效应')

    # ---------------- Part C: 首个差异步骤(最差窗所在 batch) ----------------
    ds = ds_worst
    cfg = DS_CFG[ds]
    L, P, K = cfg['L'], cfg['P'], cfg['K']
    vm = vm_cache[ds]
    std = std_cache[ds]
    row_w = next(r for r in bl if r['ds'] == ds and int(r['arr_index']) == ai_worst)
    ch = int(row_w['channel'])
    bstart = (ai_worst // 32) * 32
    loc = ai_worst - bstart
    x7 = build_batch(std, K, L, bstart)
    idx7 = loc * 7 + ch

    stages = {}
    hooks = []
    cur_idx = [idx7]   # 7-ch 行 idx7 = loc*7+ch;1-ch 行 = loc(每轮前切换)

    def grab(name, row_index_unused):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            stages[name] = t[cur_idx[0]].detach().to(torch.float64).cpu().numpy()
        return hook

    hooks.append(vm.input_resize.register_forward_hook(grab('S1_resize', idx7)))
    hooks.append(vm.vision_model.patch_embed.register_forward_hook(grab('S2_patch_embed', idx7)))
    for bi, blk in enumerate(vm.vision_model.blocks):
        hooks.append(blk.register_forward_hook(grab('S%d_enc_block%d' % (3 + bi, bi), idx7)))
    hooks.append(vm.vision_model.norm.register_forward_hook(
        grab('S15_enc_norm', idx7)))
    hooks.append(vm.vision_model.decoder_embed.register_forward_hook(
        grab('S16_dec_embed', idx7)))
    for bi, blk in enumerate(vm.vision_model.decoder_blocks):
        hooks.append(blk.register_forward_hook(
            grab('S%d_dec_block%d' % (17 + bi, bi), idx7)))
    hooks.append(vm.vision_model.decoder_norm.register_forward_hook(
        grab('S25_dec_norm', idx7)))
    hooks.append(vm.vision_model.decoder_pred.register_forward_hook(
        grab('S26_dec_pred', idx7)))

    # S0:实例归一化 means/stdev(目标行)
    def norm_row(x, b, nch):
        xt = torch.tensor(x).cuda()
        means = xt.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(xt - means, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev = stdev / 0.4
        return float(means[b, 0, nch]), float(stdev[b, 0, nch])

    m7, s7 = norm_row(x7, loc, ch)
    x1 = x7[:, :, ch:ch + 1]
    m1, s1 = norm_row(x1, loc, 0)

    set_seeds(2021)
    stages.clear()
    cur_idx[0] = idx7
    with torch.no_grad():
        y7 = vm.forward(torch.tensor(x7).cuda()).cpu().numpy()
    stage7 = {k: v.copy() for k, v in stages.items()}
    set_seeds(2021)
    stages.clear()
    cur_idx[0] = loc
    with torch.no_grad():
        y1 = vm.forward(torch.tensor(x1).cuda()).cpu().numpy()
    stage1 = stages
    for h in hooks:
        h.remove()

    pipeline_order = (['S2_patch_embed'] + ['S%d_enc_block%d' % (3 + i, i) for i in range(12)]
                      + ['S15_enc_norm', 'S16_dec_embed']
                      + ['S%d_dec_block%d' % (17 + i, i) for i in range(8)]
                      + ['S25_dec_norm', 'S26_dec_pred'])
    stages_cmp = {}
    first_div = None
    for k in pipeline_order:
        d = float(np.max(np.abs(stage7[k].astype(np.float64) - stage1[k].astype(np.float64))))
        stages_cmp[k] = d
        if d > 0 and first_div is None:
            first_div = k
    stages_cmp['S0_norm_mean'] = abs(m7 - m1)
    stages_cmp['S0_norm_stdev'] = abs(s7 - s1)
    stages_cmp['S27_final_y'] = float(np.max(np.abs(
        y7[loc, :, ch].astype(np.float64) - y1[loc, :, 0].astype(np.float64))))
    doc['partC'] = dict(window=worst_key, idx7=idx7, idx1=loc, ch=ch,
                        means7=m7, means1=m1, stdev7=s7, stdev1=s1,
                        first_divergence=first_div or '(none; all bitwise equal)',
                        stage_order=pipeline_order,
                        stage_maxabs=stages_cmp)

    # ---------------- Part D: 组织差对 D/cost 的量化 ----------------
    partD = {}
    for ds in DS_ORDER:
        cfg = DS_CFG[ds]
        L, P, K = cfg['L'], cfg['P'], cfg['K']
        vm = vm_cache[ds]
        std = std_cache[ds]
        rows = [r for r in bl if r['ds'] == ds][:8]
        assert len({r['channel'] for r in rows}) == 1
        ch = int(rows[0]['channel'])
        bz = np.load(os.path.join(save_dir, 'baseline_%s.npz' % ds))
        sid_row = {int(s): i for i, s in enumerate(bz['sid_t46'])}
        ws = [sid_row[int(r['sid_t46'])] for r in rows]
        xs = np.stack([bz['x'][w] for w in ws])          # (8,L) f64
        ys = np.stack([bz['y'][w] for w in ws])          # (8,96) f64
        ss = bz['s'][ws].astype(np.float64)              # (8,)

        # 1-ch 组织:8 窗 × {+δ,−δ} × {A,B} = 32 行,一个 batch
        rows1, trues1 = [], []
        for w in range(8):
            for dm in (1.0, -1.0):
                delta = ss[w] * dm
                xA = xs[w] + delta
                xB = xs[w].copy(); xB[-P:] += delta
                rows1 += [xA, xB]
                trues1 += [ys[w] + delta, ys[w] + delta]
        set_seeds(2021)
        with torch.no_grad():
            p1 = vm.forward(torch.tensor(np.stack(rows1).astype(np.float32))
                            .unsqueeze(-1).cuda()).squeeze(-1).cpu().numpy().astype(np.float64)

        # 7-ch 组织:同 8 窗全通道,[8,L,7],仅目标通道改
        x7base = np.empty((8, L, 7), dtype=np.float32)
        for w, r in enumerate(rows):
            st = int(r['start_row'])
            x7base[w] = std[st - L:st, :].astype(np.float32)
        p7 = {}
        for tag, part in (('A', 'all'), ('B', 'tail')):
            for dm in (1.0, -1.0):
                xg = x7base.copy()
                for w in range(8):
                    delta = ss[w] * dm
                    if part == 'all':
                        xg[w, :, ch] += delta
                    else:
                        xg[w, -P:, ch] += delta
                set_seeds(2021)
                with torch.no_grad():
                    o = vm.forward(torch.tensor(xg).cuda()).cpu().numpy()
                p7['%s_%.1f' % (tag, dm)] = o[:, :, ch].astype(np.float64)

        dD, dcost, absD, abscost = [], [], [], []
        for w in range(8):
            for di, dm in enumerate((1.0, -1.0)):
                delta = ss[w] * dm
                k = '%.1f' % dm
                pa1, pb1 = p1[(w * 4 + di * 2 + 0)], p1[(w * 4 + di * 2 + 1)]
                pa7, pb7 = p7['A_' + k][w], p7['B_' + k][w]
                yp = ys[w] + delta
                D1 = (pa1 - pb1).mean() / delta
                D7 = (pa7 - pb7).mean() / delta
                c1 = float(((pb1 - yp) ** 2).mean() - ((pa1 - yp) ** 2).mean())
                c7 = float(((pb7 - yp) ** 2).mean() - ((pa7 - yp) ** 2).mean())
                dD.append(abs(D1 - D7)); dcost.append(abs(c1 - c7))
                absD.append(abs(D1)); abscost.append(abs(c1))
        partD[ds] = dict(
            n=16,
            D1_median=float(np.median(absD)), cost1_median=float(np.median(abscost)),
            dD_max=float(np.max(dD)), dD_median=float(np.median(dD)), dD_mean=float(np.mean(dD)),
            dcost_max=float(np.max(dcost)), dcost_median=float(np.median(dcost)),
            dcost_mean=float(np.mean(dcost)),
            dD_rel_max=float(np.max(dD)) / max(float(np.median(absD)), 1e-12),
            dcost_rel_max=float(np.max(dcost)) / max(float(np.median(abscost)), 1e-12))
        print(ds, json.dumps(partD[ds]))

    doc['partD'] = partD
    with open(os.path.join(SCRIPT_DIR, 'diag_s1b2_t47.json'), 'w') as f:
        json.dump(doc, f, indent=1)
    print('---')
    print('partB: target_bitwise_all=%s target_max=%.3e s1b_max=%.3e'
          % (doc['partB']['target_bitwise_all'], doc['partB']['target_max'],
             doc['partB']['s1b_max']))
    print('partC: first_divergence=%s' % doc['partC']['first_divergence'])
    for k in pipeline_order:
        print('  %-18s %.3e' % (k, doc['partC']['stage_maxabs'][k]))
    print('written diag_s1b2_t47.json')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else os.path.join(SCRIPT_DIR, 'save_t47_levelshift'))
