"""EXP-046 预检(eta,推理前运行):版本快照 + 依赖/权重/数据可用性。

- 记录 python/torch/numpy/gluonts/uni2ts/statsforecast/pandas 版本 → config_t46.json;
- MoiraiForecast 构造签名必须含 num_samples kwarg(uni2ts 1.1.1 语义断言);
- AutoETS/AutoARIMA 构造签名与默认值快照;
- Salesforce/moirai-1.0-R-base 实际加载一次(HF_HOME 指向缓存;记录耗时);
- 4 个 ETT csv 行数;EXP-043 val 臂 4 组 arr_val.npz 在位且形状/dtype 正确;
- sample_list_t46.csv 在位且 sha 与 sample_list_sha.txt 一致。
任何一项失败 → 非零退出。
"""
import csv
import inspect
import json
import os
import sys
import time

from t46_common import DS_CFG, DS_ORDER, H, SAMPLE_LIST, sha256_file

EXPECTED_ARR_SHAPE_PER_DS = None  # 按 ds 记录 (4K-H+1, H, 7)


def main(repo_root, save_dir):
    import numpy as np
    import torch
    import gluonts
    import statsforecast
    import uni2ts
    import pandas
    from statsforecast.models import AutoARIMA, AutoETS
    from uni2ts.model.moirai import MoiraiForecast

    cfg = {}
    cfg['python'] = sys.version.split()[0]
    cfg['torch'] = torch.__version__
    cfg['numpy'] = np.__version__
    cfg['gluonts'] = gluonts.__version__
    cfg['statsforecast'] = statsforecast.__version__
    cfg['uni2ts'] = getattr(uni2ts, '__version__', '1.1.1')
    cfg['pandas'] = pandas.__version__
    cfg['cuda_available'] = torch.cuda.is_available()
    cfg['cuda_device'] = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
    cfg['HF_HOME'] = os.environ.get('HF_HOME', '(unset)')

    sig_m = str(inspect.signature(MoiraiForecast.__init__))
    assert 'num_samples' in sig_m, 'unexpected uni2ts API: ' + sig_m
    cfg['moirai_forecast_signature'] = sig_m
    cfg['autoets_signature'] = str(inspect.signature(AutoETS.__init__))
    cfg['autoarima_signature'] = str(inspect.signature(AutoARIMA.__init__))
    cfg['autoets_defaults'] = {k: repr(v) for k, v in vars(AutoETS(season_length=24)).items()
                               if not k.startswith('_')}

    t0 = time.time()
    from uni2ts.model.moirai import MoiraiModule
    module = MoiraiModule.from_pretrained('Salesforce/moirai-1.0-R-base')
    cfg['weights'] = 'Salesforce/moirai-1.0-R-base'
    cfg['weights_load_seconds'] = round(time.time() - t0, 1)
    n_params = sum(p.numel() for p in module.parameters())
    cfg['module_params'] = n_params
    del module

    for ds in DS_ORDER:
        c = DS_CFG[ds]
        with open(os.path.join(repo_root, c['csv'])) as f:
            n = sum(1 for _ in f) - 1
        assert n == c['csv_rows'], (ds, n, c['csv_rows'])
        p = os.path.join(repo_root, 'save_t43_period_val_val', c['group'], 'arr_val.npz')
        z = np.load(p)
        assert sorted(z.files) == ['pred', 'true'], (ds, z.files)
        n_expect = 4 * c['K'] - H + 1
        assert z['pred'].shape == z['true'].shape == (n_expect, H, 7), (ds, z['pred'].shape)
        assert z['pred'].dtype == z['true'].dtype == np.float32, (ds, z['pred'].dtype)
        cfg.setdefault('arr_val_groups', {})[c['group']] = dict(path=p, shape=list(z['pred'].shape))

    list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SAMPLE_LIST)
    assert os.path.exists(list_path), 'run select_t46.py first'
    sha = sha256_file(list_path)
    sha_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_list_sha.txt')
    with open(sha_file) as f:
        assert f.read().strip().split()[0] == sha, 'sample list sha mismatch'
    cfg['sample_list_sha'] = sha

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config_t46.json'), 'w') as f:
        json.dump(cfg, f, indent=1)
    print('preflight OK -> config_t46.json')
    print(json.dumps({k: v for k, v in cfg.items() if k != 'arr_val_groups'}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'save_t46_histdiag')
