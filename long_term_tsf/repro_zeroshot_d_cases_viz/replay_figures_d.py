"""EXP-032 step 2: replay EXP-001 test batches 46 and 83 on eta and build
the two Appendix D case figures.

Per user correction (2): the forecast is computed from the RAW decoded image
(unpatchify of MAE decoder tokens); the exported image_reconstructed only
swaps the visible region back for display. We therefore capture, in the SAME
forward pass, the true MAE input (masked region = 0), the fixed patch mask
and the raw decoded image, then re-derive the 96 steps from the raw decoded
image with the identical op sequence (grey mean -> output_resize ->
rearrange -> slice [2880:2976] -> *stdev + means).

Acceptance (correction 3), per window:
  (a) raw-decoded reconstruction vs the same forward's output: max|d| <= 1e-6
  (b) replay output vs EXP-001 frozen pred.npy row:      max|d| <= 1e-4
Any breach -> SystemExit (no fallback).
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d_common as D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import einops

SEED = 2021
NEEDED_BATCHES = (46, 83)
GATE_A = 1e-6
GATE_B = 1e-4

COL = {"visionts": "tab:red", "naive": "tab:blue", "avg": "tab:green"}


def make_args():
    return argparse.Namespace(
        task_name="long_term_forecast", is_training=0, model="VisionTS",
        vm_arch="mae_base", vm_ckpt="../ckpt/", vm_pretrained=1,
        seq_len=D.CTX, periodicity=D.PERIOD, pred_len=D.PRED_LEN,
        norm_const=0.4, align_const=0.4, ft_type="ln",
        interpolation="bilinear",
        data=D.DS, root_path="./dataset/ETT-small/", data_path="ETTh1.csv",
        features="M", target="OT", embed="timeF", freq="h",
        label_len=48, scale=True, seasonal_patterns=None,
        batch_size=D.BATCH_SIZE, num_workers=0)


def norm_params(x, vm):
    """L108-113 exactly (fp64=False path)."""
    means = x.mean(1, keepdim=True).detach()
    x_enc = x - means
    stdev = torch.sqrt(
        torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    stdev = stdev / vm.norm_const
    return means, stdev


def reconstruct_from_raw(raw, b, means, stdev, vm):
    """L135-149 exactly: raw decoded -> grey -> output_resize -> flatten ->
    slice [pad_left+ctx : +pred_len] -> *stdev -> +means."""
    y_grey = torch.mean(raw, 1, keepdim=True)
    y_seg = vm.output_resize(y_grey)
    y_flat = einops.rearrange(y_seg, "(b n) 1 f p -> b (p f) n",
                              b=b, f=vm.periodicity)
    y = y_flat[:, vm.pad_left + vm.context_len:
               vm.pad_left + vm.context_len + vm.pred_len, :]
    y = y * stdev.repeat(1, vm.pred_len, 1)
    y = y + means.repeat(1, vm.pred_len, 1)
    return y


def process_window(case, win, bi, batch_x, model, CAP, frozen, sel):
    vm = model.vm
    device = batch_x.device
    b = batch_x.shape[0]
    pos = win - bi * D.BATCH_SIZE
    ot_flat = pos * 7 + D.OT_IDX
    with torch.no_grad():
        y, img_in_disp, img_rec_disp = vm.forward(batch_x, export_image=True)
    assert CAP["y"].shape[0] == b * 7, CAP["y"].shape

    means, stdev = norm_params(batch_x, vm)
    raw = vm.vision_model.unpatchify(CAP["y"])              # (b*7,3,224,224)
    rec = reconstruct_from_raw(raw, b, means, stdev, vm)    # (b,96,7)

    ver_a = float((rec - y).abs().max().item())
    pred_replay = y[pos].cpu().numpy().astype(np.float64)   # (96,7)
    ver_b = float(np.abs(pred_replay
                         - frozen["visionts"][win].astype(np.float64)).max())
    ver_b_ot = float(np.abs(pred_replay[:, D.OT_IDX]
                            - frozen["visionts"][win][:, D.OT_IDX]
                            .astype(np.float64)).max())

    mask_full = vm.vision_model.unpatchify(
        CAP["mask"].unsqueeze(-1).repeat(
            1, 1, vm.vision_model.patch_embed.patch_size[0] ** 2 * 3))
    patch_size = vm.vision_model.patch_embed.patch_size[0]
    assert vm.num_patch_input == 5 and vm.pad_left == 0, \
        (vm.num_patch_input, vm.pad_left)

    img_true = CAP["image_input"][ot_flat].cpu().numpy().astype(np.float32)
    img_disp = img_rec_disp[pos, D.OT_IDX].cpu().numpy().astype(np.float32)
    img_raw = raw[ot_flat].cpu().numpy().astype(np.float32)
    mask_img = mask_full[ot_flat, 0].cpu().numpy().astype(np.float32)
    # figures show the pipeline's grey (mean over the 3 replicated channels),
    # matching visionts/model.py L138; npz keeps full 3-channel tensors
    grey_true, grey_disp, grey_raw = (img_true.mean(0), img_disp.mean(2),
                                      img_raw.mean(0))

    hist = batch_x[pos, -D.HIST_SHOW:, D.OT_IDX].cpu().numpy()
    truth = frozen["true"][win][:, D.OT_IDX]
    p_naive = frozen["naive"][win][:, D.OT_IDX]
    p_avg = frozen["avg"][win][:, D.OT_IDX]
    mae_replay = float(np.abs(pred_replay[:, D.OT_IDX] - truth
                              ).astype(np.float64).mean())
    mae_f64 = D.mae_f64_from_arrays(win)
    ts = D.timestamps(win)
    info = {"case": case, "window_index": win, "batch": bi,
            "pos_in_batch": pos, "ot_flat_index": ot_flat,
            "num_patch_input": int(vm.num_patch_input),
            "pad_left": int(vm.pad_left), "patch_size": int(patch_size),
            "mask_ratio": float(vm.mask_ratio),
            "ver_a_recon_vs_same_forward_maxdiff": ver_a,
            "ver_b_replay_vs_frozen_maxdiff_7var": ver_b,
            "ver_b_replay_vs_frozen_maxdiff_ot": ver_b_ot,
            "mae_ot_f64_from_frozen_arrays": mae_f64,
            "mae_ot_from_replay_pred": mae_replay,
            "mae_ot_npz": sel[case]["mae_npz"],
            "timestamps": ts,
            "recon_note": ("forecast re-derived from RAW decoded image "
                           "(unpatchify of decoder tokens) with the exact "
                           "L135-149 op sequence; the exported display image "
                           "swaps visible patches back and must NOT be used "
                           "(first output column bilinear taps input pixel "
                           "x=79 across the visible/masked boundary)"),
            "image_geometry": "visible patches 0-4 -> x in [0,80); 96 "
                              "forecast steps = columns [120,124) of the "
                              "24x336 resized image = x in [80, 82.67) at "
                              "224px scale",
            }
    gates_ok = ver_a <= GATE_A and ver_b <= GATE_B
    info["gates"] = {"a_threshold": GATE_A, "b_threshold": GATE_B,
                     "pass": bool(gates_ok)}

    save_dir = os.path.join(D.SAVE_D, "ETTh1_96")
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, "evidence_win%d.npz" % win)
    np.savez_compressed(
        npz_path,
        image_input_true=img_true,          # (3,224,224) masked region = 0
        image_recon_display=np.transpose(img_disp, (2, 0, 1)),
        image_raw_decoded=img_raw,          # (3,224,224) forecast source
        mask_image_ot=mask_img,             # (224,224) 1=masked patch
        mask_tokens=CAP["mask"][ot_flat].cpu().numpy().astype(np.float32),
        means=means[pos, 0].cpu().numpy().astype(np.float32),
        stdev=stdev[pos, 0].cpu().numpy().astype(np.float32),
        pred_vts_replay=pred_replay.astype(np.float32),
        true_frozen=frozen["true"][win],
        pred_vts_frozen_ot=frozen["visionts"][win][:, D.OT_IDX],
        pred_naive_ot=p_naive, pred_avg_ot=p_avg,
        ver_a=np.float64(ver_a), ver_b=np.float64(ver_b),
        window_index=np.int64(win), win_start_row=np.int64(ts["start_row"]),
        ot_flat_index=np.int64(ot_flat))
    info["evidence_npz"] = os.path.relpath(npz_path, D.ROOT)

    png_path = os.path.join(save_dir, "%s_win%d.png" % (case, win))
    make_figure(case, win, pos, bi, hist, truth, pred_replay[:, D.OT_IDX],
                p_naive, p_avg, grey_true, grey_disp, grey_raw, ts, mae_f64,
                mae_replay, png_path)
    info["figure_png"] = os.path.relpath(png_path, D.ROOT)
    return info, gates_ok


def make_figure(case, win, pos, bi, hist, truth, p_vts, p_naive, p_avg,
                img_true, img_disp, img_raw, ts, mae_f64, mae_replay,
                out_png):
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2))

    for ax, img, title in (
            (axes[0], img_true,
             "MAE 真实输入 (掩区=0)\n红虚线=可见区边界 x=80"),
            (axes[1], img_disp,
             "MAE 补全图 (展示版: 可见区换回输入)"),
            (axes[2], img_raw,
             "原始解码图 (预测来源, 3通道灰度均值)\n红带=24×336 图列 [120,124) = 未来96步")):
        m = max(2.0, float(np.abs(img).max()))
        ax.imshow(img, cmap="RdBu_r", vmin=-m, vmax=m)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    axes[0].axvline(79.5, color="red", ls="--", lw=1.2)
    axes[1].axvline(79.5, color="red", ls="--", lw=1.2)
    axes[2].axvspan(80, 124 / 336 * 224, color="red", alpha=0.4, lw=0)
    axes[2].axvline(80, color="red", lw=0.8)

    ax = axes[3]
    xh = np.arange(-D.HIST_SHOW, 0)
    xf = np.arange(0, D.PRED_LEN)
    ax.plot(xh, hist, color="0.55", lw=0.9, label="近期历史 (标准化)")
    ax.plot(xf, truth, color="black", lw=2.0, label="真值")
    ax.plot(xf, p_vts, color=COL["visionts"], lw=1.4,
            label="VisionTS (回放) MAE=%.4f" % mae_replay)
    ax.plot(xf, p_naive, color=COL["naive"], lw=1.2, ls="--",
            label="S-Naïve MAE=%.4f" % mae_f64["naive"])
    ax.plot(xf, p_avg, color=COL["avg"], lw=1.2, ls="--",
            label="S-Avg MAE=%.4f" % mae_f64["avg"])
    ax.axvline(-0.5, color="red", lw=1.0, ls=":")
    ax.set_xlabel("相对预测起点的步数", fontsize=9)
    ax.set_title("标准化空间: 近336步历史 + 未来96步", fontsize=10)
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "%s 案例: win=%d  起始行 %d (%s) → 末行 %d (%s)  "
        "batch %d / 批内位 %d  |  OT MAE: vts %.4f, naive %.4f, avg %.4f"
        % (case, win, ts["start_row"], ts["start_time"], ts["end_row"],
           ts["end_time"], bi, pos, mae_f64["visionts"], mae_f64["naive"],
           mae_f64["avg"]), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    sel = D.select_windows()
    wins = {c: sel[c]["window_index"] for c in ("success", "failure")}
    os.chdir(D.ROOT)
    sys.path.insert(0, D.ROOT)

    ckpt = os.path.join(D.ROOT, "..", "ckpt", "mae_visualize_vit_base.pth")
    assert os.path.isfile(ckpt), ckpt

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from data_provider.data_factory import data_provider
    from models import VisionTS

    args = make_args()
    model = VisionTS.Model(args).to(device).eval()

    CAP = {}
    orig_fwd = model.vm.vision_model.forward

    def cap_fwd(imgs, mask_ratio=0.75, noise=None):
        out = orig_fwd(imgs, mask_ratio=mask_ratio, noise=noise)
        CAP["image_input"] = imgs.detach()
        CAP["y"] = out[1].detach()
        CAP["mask"] = out[2].detach()
        return out

    model.vm.vision_model.forward = cap_fwd

    frozen = {}
    for m, rd in D.frozen_arrays_paths().items():
        frozen[m] = np.load(os.path.join(rd, "pred.npy")).astype(np.float64)
    frozen["true"] = np.load(
        os.path.join(D.frozen_arrays_paths()["visionts"], "true.npy"))

    _, loader = data_provider(args, "test")
    assert len(loader.dataset) == 2785, len(loader.dataset)

    infos, ok_all = {}, True
    for bi, batch in enumerate(loader):
        if bi == max(NEEDED_BATCHES) + 1:
            break
        if bi not in NEEDED_BATCHES:
            continue
        batch_x = batch[0].float().to(device)
        for case, win in wins.items():
            if win // D.BATCH_SIZE != bi:
                continue
            info, ok = process_window(case, win, bi, batch_x, model, CAP,
                                      frozen, sel)
            infos[case] = info
            ok_all = ok_all and ok
            print("[replay] %s win=%d ver_a=%.3e ver_b=%.3e "
                  "mae_replay=%.6f (frozen %.4f)"
                  % (case, win, info["ver_a_recon_vs_same_forward_maxdiff"],
                     info["ver_b_replay_vs_frozen_maxdiff_7var"],
                     info["mae_ot_from_replay_pred"],
                     D.FROZEN[case]["mae"]["visionts"]))

    env = {"device": str(device),
           "torch": torch.__version__,
           "cuda_device": torch.cuda.get_device_name(0)
           if torch.cuda.is_available() else "cpu",
           "seed": SEED, "batch_size": D.BATCH_SIZE,
           "ckpt_sha256": D.sha256_of(ckpt), "code_md5": D.code_md5(),
           "gates": {"a_recon_vs_same_forward<=1e-6": GATE_A,
                     "b_replay_vs_frozen<=1e-4": GATE_B},
           "criterion_note": "win/loss judged against EXP-031 seasonal "
                             "baselines; paper Fig.14 used MOIRAI-Large"}
    if not ok_all:
        D.dump_json(os.path.join(D.PKG, "logs", "replay_evidence.json"),
                    {"windows": infos, "env": env, "GATES_FAILED": True})
        raise SystemExit("[replay_figures_d] verification gate breached "
                         "(a<=1e-6, b<=1e-4); see logs/replay_evidence.json")
    D.dump_json(os.path.join(D.PKG, "logs", "replay_evidence.json"),
                {"windows": infos, "env": env, "ALL_GATES_PASS": True})
    print("[replay_figures_d] ALL GATES PASS")


if __name__ == "__main__":
    main()
