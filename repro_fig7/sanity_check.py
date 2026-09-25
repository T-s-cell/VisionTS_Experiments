"""Step 2 (after make_sampling_lists): small-sample sanity checks.

Checks (all must pass before the full run):
  1. shapes: image (B,3,224,224), encoder out (B,71,768), flat 54528, finite values
  2. mask buffer: left 5 of 14 columns visible (70 patches), mask_ratio = 9/14
  3. ids_keep identical across groups and across batches (token order consistency)
  4. equivalence: production forward() input tensor (captured via forward_pre_hook)
     == build_ts_image() tensor (proves the reused path is the audited one)
  5. example input images saved to outputs/example_inputs.png
  6. Weather/Electricity: raw csv has OT as last column -> column reorder is identity,
     so channel_index maps to the raw csv feature order
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

import numpy as np
import pandas as pd
import torch
from PIL import Image

N_CHECK = 8
report = []


def check(name, ok, detail=''):
    line = f'[{"PASS" if ok else "FAIL"}] {name}' + (f' | {detail}' if detail else '')
    report.append(line)
    print(line, flush=True)
    if not ok:
        raise AssertionError(line)


def ts_batch(name, entries, device):
    ds = common.build_dataset(name)
    seqs = []
    for e in entries:
        if common.TS_CONFIGS[name]['data'] == 'custom':
            seq = ds[e['k']][0]
        else:
            seq = ds[e['window_idx']][0][:, e['channel_idx']:e['channel_idx'] + 1]
        seq = torch.as_tensor(np.asarray(seq), dtype=torch.float32)
        seqs.append(seq)
    return torch.stack(seqs).to(device)


def imagenet_batch(entries, device, tfm):
    imgs = []
    for e in entries:
        p = os.path.join(common.IMAGENET_VAL_DIR, e if isinstance(e, str) else e['filenames'])
        img = Image.open(p).convert('RGB')
        imgs.append(tfm(img))
    return torch.stack(imgs).to(device)


def ids_keep_of(noise):
    return torch.argsort(noise, dim=1)[:, :70]


def encode_group(model, name, entries, device, tfm, ref_ids_keep):
    if name == 'ImageNet':
        x = imagenet_batch(entries, device, tfm)
        image = x
    else:
        x = ts_batch(name, entries, device)
        image = common.build_ts_image(model, x)
    with torch.no_grad():
        feats, _, _ = model.vision_model.forward_encoder(
            image, model.mask_ratio, common.get_noise(model, image.shape[0]))
    flat = feats.flatten(1)
    check(f'{name}: image shape', tuple(image.shape) == (len(entries), 3, 224, 224), str(tuple(image.shape)))
    check(f'{name}: encoder out shape', tuple(feats.shape) == (len(entries), 71, 768), str(tuple(feats.shape)))
    check(f'{name}: flat dim', flat.shape[1] == common.FEATURE_DIM, str(flat.shape[1]))
    check(f'{name}: finite', bool(torch.isfinite(flat).all()))
    ik = ids_keep_of(common.get_noise(model, image.shape[0]))
    check(f'{name}: ids_keep == ref', bool(torch.equal(ik, ref_ids_keep.expand_as(ik))))
    return image


def main():
    common.set_seeds()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}', flush=True)
    tfm = common.imagenet_transform()
    model = common.load_model()
    model.eval().to(device)

    # ---- mask buffer structure ----
    cfg0 = common.configure(model, 'ETTm1')
    mask2d = model.mask.reshape(14, 14)
    check('mask: left 5 cols visible', bool((mask2d[:, :5] == 0).all()))
    check('mask: right 9 cols masked', bool((mask2d[:, 5:] == 1).all()))
    check('mask: visible count 70', int((model.mask == 0).sum()) == 70)
    check('mask_ratio == 9/14 (float32 tol)', abs(model.mask_ratio - 9 / 14) < 1e-6, f'{model.mask_ratio}')

    # identical mask across the three TS configs
    masks = {}
    ratios = {}
    for name in ['ETTm1', 'Weather', 'Electricity']:
        common.configure(model, name)
        masks[name] = model.mask.detach().clone()
        ratios[name] = model.mask_ratio
        check(f'{name}: num_patch_input == 5', model.num_patch_input == 5)
    for name in ['Weather', 'Electricity']:
        check(f'mask identical ETTm1 vs {name}', bool(torch.equal(masks['ETTm1'], masks[name])))
        check(f'mask_ratio identical ETTm1 vs {name}', ratios['ETTm1'] == ratios[name])
    common.configure(model, 'ETTm1')

    ref_ids_keep = ids_keep_of(common.get_noise(model, 1))

    # ---- equivalence: production forward() input vs build_ts_image ----
    lists = {g: common.load_sampling_list(g) for g in common.GROUPS}
    for g in ['ETTm1', 'Weather', 'Electricity']:
        cfg = common.configure(model, g)
        check(f'{g}: equiv config seq_len/periodicity',
              model.context_len == cfg['seq_len'] and model.periodicity == cfg['periodicity'],
              f"seq_len={model.context_len}, periodicity={model.periodicity}")
        entries = lists[g]['samples'][:N_CHECK]
        x = ts_batch(g, entries, device)

        captured = {}
        hook = model.vision_model.register_forward_pre_hook(
            lambda mod, args: captured.update(img=args[0].detach().clone()))
        with torch.no_grad():
            _ = model.forward(x)
        hook.remove()
        check(f'{g}: production forward captured input', 'img' in captured)

        mine = common.build_ts_image(model, x)
        max_diff = (captured['img'] - mine).abs().max().item()
        check(f'{g}: build_ts_image == production image_input', max_diff <= 1e-6, f'max_abs_diff={max_diff:.3e}')

    # ---- per-group encode with ids_keep consistency + shapes ----
    imgs_for_plot = {}
    for g in common.GROUPS:
        if g in common.TS_CONFIGS:
            cfg = common.configure(model, g)
            check(f'{g}: encode config seq_len/periodicity',
                  model.context_len == cfg['seq_len'] and model.periodicity == cfg['periodicity'],
                  f"seq_len={model.context_len}, periodicity={model.periodicity}")
        entries = lists[g]['filenames'] if g == 'ImageNet' else lists[g]['samples']
        image = encode_group(model, g, entries[:N_CHECK], device, tfm, ref_ids_keep)
        imgs_for_plot[g] = image

    # second ETTm1 batch: ids_keep stable across batches
    entries2 = lists['ETTm1']['samples'][N_CHECK:2 * N_CHECK]
    ts_batch('ETTm1', entries2, device)
    ik2 = ids_keep_of(common.get_noise(model, len(entries2)))
    check('ids_keep stable across batches', bool(torch.equal(ik2, ref_ids_keep.expand_as(ik2))))

    # ---- variable-name ordering check (custom datasets: reorder is identity when OT is last) ----
    for g in ['Weather', 'Electricity']:
        csv = os.path.join(common.LTSF_ROOT, common.TS_CONFIGS[g]['root_path'], common.TS_CONFIGS[g]['data_path'])
        cols = list(pd.read_csv(csv, nrows=1).columns)
        check(f'{g}: raw csv last column is OT (reorder identity)', cols[-1] == 'OT', cols[-1])
        names = common.variable_names(g)
        check(f'{g}: variable names == raw csv feature order', names == [c for c in cols if c != 'date'])
    ettm1_names = common.variable_names('ETTm1')
    check('ETTm1: 7 variables', len(ettm1_names) == 7, str(ettm1_names))

    # ---- example inputs figure ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for j, g in enumerate(common.GROUPS):
        img = imgs_for_plot[g][0].detach().cpu()
        if g == 'ImageNet':
            disp = img * torch.tensor(common.IMAGENET_STD).view(3, 1, 1) + \
                torch.tensor(common.IMAGENET_MEAN).view(3, 1, 1)
            disp = disp.clamp(0, 1).permute(1, 2, 0).numpy()
            note = 'preprocessed (denorm. for display)'
        else:
            disp = (img - img.min()) / (img.max() - img.min() + 1e-12)
            disp = disp.permute(1, 2, 0).numpy()
        axes[0, j].imshow(disp)
        axes[0, j].set_title(f'{g} input')
        axes[0, j].axis('off')
        im_vis = imgs_for_plot[g][0].detach().cpu()[0]
        axes[1, j].imshow(im_vis.numpy(), cmap='gray')
        axes[1, j].set_title('channel 0 raw values')
        axes[1, j].axis('off')
    fig.suptitle('Example model inputs per group (top: RGB view, bottom: first channel as fed to MAE)')
    out_png = os.path.join(common.OUT_DIR, 'example_inputs.png')
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print('example inputs ->', out_png, flush=True)

    report_path = os.path.join(common.OUT_DIR, 'sanity_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report) + '\n')
    print('\nALL SANITY CHECKS PASSED ->', report_path, flush=True)


if __name__ == '__main__':
    main()
