"""Step 3 (after sanity_check passes): full feature extraction for all 1900 samples.

Fixed global order: ImageNet(1000) -> ETTm1(300) -> Weather(300) -> Electricity(300),
batch=32, FP32, eval mode, no_grad. Reads the sampling lists produced by
make_sampling_lists.py. Writes:
  outputs/features_1900x54528.npy  (float32)
  outputs/groups.npy               (1900 str)
  outputs/meta.csv                 (per-sample reference rows)
  outputs/env.json                 (versions, ckpt sha256, gpu, mask facts)
  outputs/mask_ref.json            (ids_keep / visible coordinates)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

import numpy as np
import torch
from PIL import Image


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def ts_batch(name, entries, device):
    ds = common.build_dataset(name)
    seqs = []
    for e in entries:
        if common.TS_CONFIGS[name]['data'] == 'custom':
            seq = ds[e['k']][0]
        else:
            seq = ds[e['window_idx']][0][:, e['channel_idx']:e['channel_idx'] + 1]
        seqs.append(torch.as_tensor(np.asarray(seq), dtype=torch.float32))
    return torch.stack(seqs).to(device)


def main():
    common.set_seeds()
    assert torch.cuda.is_available(), 'GPU required for the full extraction'
    device = 'cuda'
    tfm = common.imagenet_transform()
    model = common.load_model()
    model.eval().to(device)

    lists = {g: common.load_sampling_list(g) for g in common.GROUPS}
    # mask buffer is created by update_config; must configure before touching model.mask
    common.configure(model, 'ETTm1')
    ref_ids_keep = torch.argsort(common.get_noise(model, 1), dim=1)[:, :70]

    # record mask reference (visible patch coordinates, row-major grid)
    noise_row = common.get_noise(model, 1)[0]
    ids_keep = ref_ids_keep[0].tolist()
    coords = [{'id': i, 'row': i // 14, 'col': i % 14} for i in ids_keep]
    common.save_json(os.path.join(common.OUT_DIR, 'mask_ref.json'), {
        'grid': '14x14, patch 16, image 224; visible = left 5 columns (supplementary setting)',
        'num_visible': 70, 'mask_ratio': model.mask_ratio,
        'ids_keep_order': ids_keep,
        'visible_coords': coords,
        'all_ids_in_left_5_cols': all(c['col'] < 5 for c in coords),
    })

    feats = np.zeros((sum(len(lists[g]['filenames'] if g == 'ImageNet' else lists[g]['samples'])
                          for g in common.GROUPS), common.FEATURE_DIM), dtype=np.float32)
    groups = []
    meta_rows = ['group,sample_ref,window_idx,channel_idx,abs_row_start,abs_row_end,t_start,t_end']
    pos = 0

    for g in common.GROUPS:
        if g in common.TS_CONFIGS:
            common.configure(model, g)
        entries = lists[g]['filenames'] if g == 'ImageNet' else lists[g]['samples']
        log(f'{g}: {len(entries)} samples, num_patch_input={model.num_patch_input}, '
            f'mask_ratio={model.mask_ratio:.6f}')
        for s in range(0, len(entries), common.BATCH):
            batch = entries[s:s + common.BATCH]
            if g == 'ImageNet':
                imgs = []
                for fn in batch:
                    img = Image.open(os.path.join(common.IMAGENET_VAL_DIR, fn)).convert('RGB')
                    imgs.append(tfm(img))
                image = torch.stack(imgs).to(device)
                refs = batch
            else:
                x = ts_batch(g, batch, device)
                image = common.build_ts_image(model, x)
                refs = [e['variable'] for e in batch]
            with torch.no_grad():
                out, _, _ = model.vision_model.forward_encoder(
                    image, model.mask_ratio, common.get_noise(model, image.shape[0]))
            ik = torch.argsort(common.get_noise(model, image.shape[0]), dim=1)[:, :70]
            assert torch.equal(ik, ref_ids_keep.expand_as(ik)), f'{g} batch {s}: ids_keep drift'
            flat = out.flatten(1).to(torch.float32).cpu().numpy()
            assert np.isfinite(flat).all(), f'{g} batch {s}: non-finite features'
            feats[pos:pos + len(batch)] = flat
            groups.extend([g] * len(batch))
            for e, ref in zip(batch, refs):
                if g == 'ImageNet':
                    meta_rows.append(f'{g},{ref},,,,,'.rstrip(','))
                else:
                    meta_rows.append(
                        f'{g},{e["variable"]},{e["window_idx"]},{e["channel_idx"]},'
                        f'{e["abs_row_start"]},{e["abs_row_end"]},{e["t_start"]},{e["t_end"]}')
            pos += len(batch)
            if (s // common.BATCH) % 5 == 0:
                log(f'{g}: {pos} / done-batch {s}-{s + len(batch)}')
        log(f'{g}: DONE ({pos} total)')

    assert pos == len(feats)
    np.save(os.path.join(common.OUT_DIR, 'features_1900x54528.npy'), feats)
    np.save(os.path.join(common.OUT_DIR, 'groups.npy'), np.array(groups))
    with open(os.path.join(common.OUT_DIR, 'meta.csv'), 'w') as f:
        f.write('\n'.join(meta_rows) + '\n')

    import hashlib
    ckpt = os.path.join(common.CKPT_DIR, 'mae_visualize_vit_base.pth')
    h = hashlib.sha256()
    with open(ckpt, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    env = {
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'torchvision': __import__('torchvision').__version__,
        'numpy': np.__version__,
        'sklearn': __import__('sklearn').__version__,
        'gpu': torch.cuda.get_device_name(0),
        'ckpt_path': ckpt,
        'ckpt_sha256': h.hexdigest(),
        'mask_ratio': model.mask_ratio,
        'feature_dim': common.FEATURE_DIM,
        'seed': common.SEED,
        'supplementary_note': common.SUPPLEMENTARY['note'],
    }
    common.save_json(os.path.join(common.OUT_DIR, 'env.json'), env)
    log(f'ALL DONE: {pos} features -> outputs/features_1900x54528.npy')


if __name__ == '__main__':
    main()
