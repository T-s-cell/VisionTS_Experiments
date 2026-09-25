"""Step 4: joint t-SNE on all 1900 x 54528 features at once.

Fixed parameters (supplementary setting — the paper does not disclose its t-SNE
configuration): perplexity=30, learning_rate=200, init='pca',
early_exaggeration=12, max_iter=1000, random_state=2021, euclidean metric.
No pre-PCA, no per-group standardization (init='pca' only seeds the 2-D init).

CPU-bound; intended to run under nohup. Writes:
  outputs/tsne_coords_1900x2.npy
  outputs/tsne_params.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

import numpy as np
from sklearn.manifold import TSNE
import sklearn

TSNE_PARAMS = dict(
    n_components=2, perplexity=30, learning_rate=200.0, init='pca',
    early_exaggeration=12.0, max_iter=1000, random_state=common.SEED,
    metric='euclidean',
)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main():
    feats = np.load(os.path.join(common.OUT_DIR, 'features_1900x54528.npy'))
    groups = np.load(os.path.join(common.OUT_DIR, 'groups.npy'), allow_pickle=True)
    assert feats.shape == (1900, common.FEATURE_DIM), feats.shape
    assert np.isfinite(feats).all()
    log(f'features {feats.shape}, dtype={feats.dtype}, groups={sorted(set(groups.tolist()))}')

    log(f'running TSNE with {TSNE_PARAMS}')
    t0 = time.time()
    emb = TSNE(**TSNE_PARAMS).fit_transform(feats)
    log(f'TSNE done in {(time.time() - t0) / 60:.1f} min, coords shape {emb.shape}')

    np.save(os.path.join(common.OUT_DIR, 'tsne_coords_1900x2.npy'), emb.astype(np.float64))
    common.save_json(os.path.join(common.OUT_DIR, 'tsne_params.json'), {
        'params': TSNE_PARAMS,
        'sklearn_version': sklearn.__version__,
        'input_shape': list(feats.shape),
        'input_dtype': str(feats.dtype),
        'note': 'one joint run over all 1900 samples; no pre-PCA, no per-group '
                'standardization (init=pca seeds the 2-D initialization only); '
                'parameters are supplementary settings, the paper does not disclose them.',
        'group_order': common.GROUPS,
        'elapsed_s': round(time.time() - t0, 1),
    })
    log('saved tsne_coords_1900x2.npy + tsne_params.json')


if __name__ == '__main__':
    main()
