"""EXP-047 预检:环境 + 资产(推理前运行,任何缺失 → 退出码 1)。

--env vts     检 torch/cuda/visionts 可导入
--env t46diag 检 pandas/gluonts/uni2ts/statsforecast 可导入
共通:ckpt sha、冻结链(baseline_list/manifest sha)、baseline npz ↔ build_meta、
     t46 资产(features/sample_list/arr_vts_t46.npz/per_sample_metrics_t46.csv)。
"""
import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main(repo_root, save_dir, env):
    from t47_common import CKPT_SHA, T46_DIR, baseline_list_path, manifest_path

    fails = []

    def need(cond, msg):
        print('[preflight] %s %s' % ('OK ' if cond else 'MISS', msg))
        if not cond:
            fails.append(msg)

    import numpy as np
    print('[preflight] python=%s numpy=%s' % (sys.version.split()[0], np.__version__))

    ckpt = os.path.join(os.path.dirname(os.path.abspath(repo_root)), 'ckpt',
                        'mae_visualize_vit_base.pth')
    need(os.path.exists(ckpt), 'ckpt exists: %s' % ckpt)
    if os.path.exists(ckpt):
        need(sha256(ckpt) == CKPT_SHA, 'ckpt sha256 == %s…' % CKPT_SHA[:12])

    need(os.path.exists(baseline_list_path()), 'baseline_list_t47.csv')
    sha_file = os.path.join(SCRIPT_DIR, 'baseline_list_sha_t47.txt')
    need(os.path.exists(sha_file)
         and open(sha_file).read().strip() == sha256(baseline_list_path()),
         'baseline_list sha == recorded')
    need(os.path.exists(manifest_path()), 'manifest_t47.csv')

    bm_path = os.path.join(save_dir, 'build_meta_t47.json')
    need(os.path.exists(bm_path), 'build_meta_t47.json')
    if os.path.exists(bm_path):
        with open(bm_path) as f:
            bm = json.load(f)
        need(bm['manifest_sha256'] == sha256(manifest_path()), 'manifest sha == build_meta')
        for ds, d in bm['baselines'].items():
            p = os.path.join(save_dir, 'baseline_%s.npz' % ds)
            need(os.path.exists(p) and sha256(p) == d['sha256'],
                 'baseline_%s.npz sha == build_meta' % ds)

    for rel in ('features_t46.csv', 'sample_list_t46.csv',
                os.path.join('save_t46_histdiag', 'arr_vts_t46.npz'),
                os.path.join('save_t46_histdiag', 'per_sample_metrics_t46.csv')):
        need(os.path.exists(os.path.join(T46_DIR, rel)), 't46 asset %s' % rel)

    if env == 'vts':
        import torch
        print('[preflight] torch=%s cuda=%s' % (torch.__version__, torch.cuda.is_available()))
        need(torch.cuda.is_available(), 'cuda available (vts env)')
        sys.path.insert(0, os.path.dirname(os.path.abspath(repo_root)))
        from visionts import VisionTS  # noqa: F401
        print('[preflight] visionts import OK')
    elif env == 't46diag':
        import gluonts  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401
        import statsforecast
        import uni2ts  # noqa: F401
        print('[preflight] t46diag imports OK (statsforecast %s)'
              % getattr(statsforecast, '__version__', '?'))

    print('[preflight] %s' % ('PASS' if not fails else 'FAIL (%d): %s'
                              % (len(fails), '; '.join(fails))))
    return not fails


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    env = 'vts' if '--env' not in sys.argv else sys.argv[sys.argv.index('--env') + 1]
    ok = main(args[0], args[1] if len(args) > 1
              else os.path.join(SCRIPT_DIR, 'save_t47_levelshift'), env)
    sys.exit(0 if ok else 1)
