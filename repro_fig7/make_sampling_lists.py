"""Step 1: generate and persist the sampling lists.

All later steps (sanity_check / extract_features) read these files, so the
sampled subset is fixed before any feature extraction.

Sampling method (supplementary setting, recorded in each file):
  rng = np.random.default_rng(2021), ONE rng used sequentially in the fixed
  group order ImageNet -> ETTm1 -> Weather -> Electricity,
  rng.choice(pool, k, replace=False).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

import numpy as np
import sklearn
import pandas as pd


def main():
    common.set_seeds()
    rng = np.random.default_rng(common.SEED)
    manifest = {'seed': common.SEED,
                'method': 'np.random.default_rng(2021).choice(pool, k, replace=False), '
                          'single rng consumed sequentially in group order ImageNet->ETTm1->Weather->Electricity',
                'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                'pred_len': common.PRED_LEN,
                'supplementary_note': common.SUPPLEMENTARY['note'],
                'groups': {}}

    # ---- ImageNet: filenames by 1-based index into the sorted val set ----
    idx = rng.choice(50000, common.N_IMAGENET, replace=False)
    names = [f'ILSVRC2012_val_{int(i) + 1:08d}.JPEG' for i in idx]
    common.save_json(os.path.join(common.LIST_DIR, 'ImageNet.json'), {
        'group': 'ImageNet', 'pool': 50000, 'n': common.N_IMAGENET,
        'index_rule': 'ILSVRC2012_val_{i+1:08d}.JPEG, 1-based position in the official 50000-file val tar',
        'filenames': names,
    })
    manifest['groups']['ImageNet'] = {'pool': 50000, 'n': common.N_IMAGENET}
    print(f'[ImageNet] sampled {len(names)}/{50000}')

    # ---- Time series: window x variable combos from the test split ----
    for name in ['ETTm1', 'Weather', 'Electricity']:
        cfg = common.TS_CONFIGS[name]
        ds = common.build_dataset(name)
        vars_ = common.variable_names(name)
        C = len(vars_)
        n_ds = len(ds)
        n_win = n_ds // C if cfg['data'] == 'custom' else n_ds
        pool = n_win * C
        ks = rng.choice(pool, common.N_TS, replace=False)
        b1 = common.border1(name)

        samples = []
        for k in sorted(int(k) for k in ks):
            w, ch = k // C, k % C
            t0, t1, s_abs, e_abs = common.timestamps_for(name, w)
            samples.append({
                'k': k, 'window_idx': w, 'channel_idx': ch, 'variable': vars_[ch],
                'abs_row_start': int(s_abs), 'abs_row_end': int(e_abs),
                't_start': t0, 't_end': t1,
            })
        common.save_json(os.path.join(common.LIST_DIR, f'{name}.json'), {
            'group': name, 'data': cfg['data'], 'seq_len': cfg['seq_len'],
            'periodicity': cfg['periodicity'], 'pred_len': common.PRED_LEN,
            'channels': C, 'test_windows': n_win, 'pool': pool, 'n': len(samples),
            'test_border1': b1,
            'scaler': 'StandardScaler fit on the train segment only (inside the Dataset class)',
            'samples': samples,
        })
        manifest['groups'][name] = {'pool': pool, 'n': len(samples), 'channels': C, 'test_windows': n_win}
        print(f'[{name}] pool={pool} ({n_win} windows x {C} vars), sampled {len(samples)}')

    common.save_json(os.path.join(common.LIST_DIR, 'manifest.json'), manifest)
    print('[done] sampling lists ->', common.LIST_DIR)


if __name__ == '__main__':
    main()
