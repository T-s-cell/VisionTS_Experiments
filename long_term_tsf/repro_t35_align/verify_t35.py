"""EXP-035 verification. Four modes (each merges into logs/verify_t35.json):

  --baseline H         EXP-001 ETTm1 group full gate (baseline integrity,
                       NO rerun path) + EXP-031 column alignment + probe:
                       first test batch forward bitwise vs EXP-001
                       pred.npy[:32], geometry asserted at c=0.4
  --group H C          new-group gate (config incl. align_const token,
                       true double bitwise, finite values, pred bitwise !=
                       c=0.4, metrics == stored) + probe at this c
  --first_window H C   smoke extra: window 0 history rows [43776, 46080)
                       (L=2304) bitwise vs std_array + variable order +
                       window-0 future truth (seq_y last H steps) bitwise
  --smoke_hash H C     smoke pair hard assertions: BOTH models (c=0.4 and
                       c) replay the SAME first batch — inputs bitwise
                       identical across loaders, each replay bitwise vs its
                       stored pred.npy[:32], pinned geometry pairs for
                       (H=96, c=0.4->0.6): npi 5->8, input 224x80->224x128,
                       output 96x67->96x42 — and SHA-256 of the shared
                       input plus both predictions (pred hashes must differ)

Gates per group (any failure -> SystemExit, no fallback):
  * SUCCESS.txt present; config tokens (L=2304/P=96, r=0.4, align token,
    bs=32, train_epochs=0, seed=2021); ckpt sha256 == pinned EXP-001
    weights AND == the recorded line; every recorded md5 re-hashed
  * pred/true shape (N, H, 7), N == 11521-H; all values finite;
    no checkpoint FILES in the run dir
  * true bitwise == truth_recomputed(std_array) AND == the fixed-reshape
    restore of EXP-003 seed2021 truth (baseline) / EXP-001 truth (variants)
  * metric(pred, true) == metrics.npy[:5] exactly
  * new groups: pred bitwise != EXP-001 pred (c change took effect)
  * probe: actual num_patch_input / resize sizes == exact update_config
    formulas at the requested c; first batch bitwise vs stored pred

Missing baseline artifacts -> stop with a re-sync hint; md5/sha drift ->
stop and locate the cause (a rerun cannot legitimize a drifted tree).
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t35_common as C

VERIFY_JSON = os.path.join(C.PKG, "logs", "verify_t35.json")


def fail(msg):
    raise SystemExit("[verify_t35] FAIL: %s" % msg)


def check_run_dir_clean(run_dir):
    """No weight/checkpoint FILES anywhere; the empty checkpoints/<model_id>/
    scaffolding dir is framework behavior (EXP-001 dirs have it too)."""
    arts = []
    for pat in ("**/*.pth", "**/*.ckpt", "**/*.bin", "checkpoints/**"):
        for p in glob.glob(os.path.join(run_dir, pat), recursive=True):
            if os.path.isfile(p):
                arts.append(os.path.relpath(p, run_dir))
    return arts


def require_arrays(dir_, h, label, resync_hint):
    rd = os.path.join(dir_, "results", "_")
    for f in ("pred.npy", "true.npy", "metrics.npy"):
        if not os.path.isfile(os.path.join(rd, f)):
            fail("%s ETTm1_%d missing %s under %s\n  -> %s"
                 % (label, h, f, rd, resync_hint))
    return rd


def gate_fingerprints(run_dir, h, want_extra, label):
    rec = C.parse_success(os.path.join(run_dir, "SUCCESS.txt"))
    cfg = rec["config"]
    want_cfg = {
        "dataset": "ETTm1", "seq_len": "2304", "periodicity": "96",
        "pred_len": str(h), "norm_const": "0.4", "align_const": C.c_str(0.4),
        "batch_size": "32", "train_epochs": "0", "seed": "2021",
        "arch": "mae_base", "features": "M",
    }
    if want_extra:
        want_cfg.update(want_extra)
    for k, want in want_cfg.items():
        got = cfg.get(k)
        if got != want:
            fail("%s ETTm1_%d SUCCESS.txt %s=%s != %s — config/protocol "
                 "drift, locate the cause (no rerun)"
                 % (label, h, k, got, want))
    ckpt = os.path.join(C.ROOT, C.CKPT_REL)
    got_sha = C.sha256_of(ckpt)
    if got_sha != C.CKPT_SHA256:
        fail("%s ETTm1_%d current ckpt sha256 %s != pinned EXP-001 weights "
             "%s — weights changed, locate the cause first"
             % (label, h, got_sha, C.CKPT_SHA256))
    if cfg.get("ckpt_sha256") != got_sha:
        fail("%s ETTm1_%d recorded ckpt_sha256 %s != current %s — locate the "
             "cause (no rerun)" % (label, h, cfg.get("ckpt_sha256"), got_sha))
    bad = []
    for f in C.FP_FILES:
        want = rec["md5"].get(f)
        cur = C.md5_of(os.path.join(C.ROOT, f))
        if want != cur:
            bad.append("%s: recorded=%s current=%s" % (f, want, cur))
    if bad:
        fail("%s ETTm1_%d fingerprint md5 drift — locate the cause (no "
             "rerun): %s" % (label, h, "; ".join(bad)))
    arts = check_run_dir_clean(run_dir)
    if arts:
        fail("%s ETTm1_%d unexpected checkpoint artifacts: %s"
             % (label, h, arts))
    return {"config": cfg, "ckpt_sha256": got_sha,
            "md5_all_match": True, "checkpoint_artifacts": []}


def gate_arrays(rd, h, metric, label, truth_second):
    """Shared array-level gates; truth_second: 'exp003' (fixed-reshape
    restore) or 'exp001' (the reused c=0.4 baseline's true.npy)."""
    pred = np.load(os.path.join(rd, "pred.npy"))
    true = np.load(os.path.join(rd, "true.npy"))
    mets = np.load(os.path.join(rd, "metrics.npy"))
    n_exp = C.expected_n_windows(h)

    problems = []
    if true.shape != (n_exp, h, 7):
        problems.append("true shape %s != (%d, %d, 7)" % (true.shape, n_exp, h))
    if pred.shape != true.shape:
        problems.append("pred shape %s != true %s" % (pred.shape, true.shape))
    if problems:
        fail("%s ETTm1_%d: %s" % (label, h, "; ".join(problems)))
    if not (np.isfinite(pred).all() and np.isfinite(true).all()):
        fail("%s ETTm1_%d: non-finite values in pred/true" % (label, h))

    tr = C.truth_recomputed(h)
    eq_std = bool(np.array_equal(true, tr))
    if not eq_std:
        fail("%s ETTm1_%d: true.npy NOT bitwise == std_array recomputation"
             % (label, h))
    if truth_second == "exp003":
        t2 = C.exp003_truth_restored(h)
        src = "EXP-003 restored truth"
    else:
        t2 = np.load(os.path.join(C.exp001_dir(h), "results", "_", "true.npy"))
        src = "EXP-001 truth"
    eq_2 = bool(np.array_equal(true, t2))
    if not eq_2:
        d = int(np.argmax(np.abs(true.astype(np.float64)
                                 - t2.astype(np.float64)).sum(axis=(1, 2))))
        fail("%s ETTm1_%d: true.npy NOT bitwise == %s (worst window %d)"
             % (label, h, src, d))

    rec = C.metrics_from_arrays(metric, pred, true)
    eq_m = bool(np.array_equal(rec, mets[:5].astype(np.float64), equal_nan=True))
    if not eq_m:
        fail("%s ETTm1_%d: metric(pred,true) != metrics.npy "
             "(recomputed=%s stored=%s)" % (label, h, rec.tolist(),
                                            mets[:5].tolist()))
    return {"dir": os.path.relpath(os.path.dirname(rd), C.ROOT),
            "n": int(true.shape[0]),
            "true_bitwise_vs_std_array": eq_std,
            "true_bitwise_vs_%s" % truth_second: eq_2,
            "finite": True,
            "metrics_recompute_equals_stored": eq_m,
            "mse": float(rec[1]), "mae": float(rec[0]),
            "first_pred_abs_row0": C.first_pred_abs_row0()}


def build_model(args, c, label, h):
    import torch

    os.chdir(C.ROOT)
    if C.ROOT not in sys.path:
        sys.path.insert(0, C.ROOT)
    from models.VisionTS import Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(args).float().to(device)
    model.eval()
    vm = model.vm
    if not (vm.patch_size == C.MAE_PATCH_SIZE and vm.image_size == C.MAE_IMAGE_SIZE
            and vm.num_patch == C.MAE_NUM_PATCH):
        fail("%s ETTm1_%d probe ckpt geometry patch=%s image=%s numpatch=%s "
             "!= pinned ViT-B/16 (16/224/14)"
             % (label, h, vm.patch_size, vm.image_size, vm.num_patch))
    return model, device


def probe_geometry(model, h, c, label):
    geom = C.expected_geometry(h, c)
    vm = model.vm
    actual = {}
    for name in ("input_resize", "output_resize"):
        t = getattr(vm, name)
        mode = str(getattr(t, "interpolation", t))
        size = [int(x) for x in t.size] if isinstance(t.size, (tuple, list)) \
            else [int(t.size)]
        aa = getattr(t, "antialias", None)
        actual[name] = {"mode": mode, "size": size, "antialias": aa}
        if "BILINEAR" not in mode:
            fail("%s ETTm1_%d probe %s interpolation %s != bilinear"
                 % (label, h, name, mode))
        if aa is not None and aa is not False:
            fail("%s ETTm1_%d probe %s antialias=%s != False"
                 % (label, h, name, aa))
    if actual["input_resize"]["size"] != geom["input_size"]:
        fail("%s ETTm1_%d probe input_resize size %s != formula %s"
             % (label, h, actual["input_resize"]["size"], geom["input_size"]))
    if actual["output_resize"]["size"] != geom["output_size"]:
        fail("%s ETTm1_%d probe output_resize size %s != formula %s"
             % (label, h, actual["output_resize"]["size"], geom["output_size"]))
    if int(vm.num_patch_input) != geom["num_patch_input"]:
        fail("%s ETTm1_%d probe num_patch_input %d != formula %d"
             % (label, h, int(vm.num_patch_input), geom["num_patch_input"]))
    return geom, actual


def forward_first_batch(model, h, device):
    import torch

    from data_provider.data_factory import data_provider
    _, loader = data_provider(model.args_for_loader, "test")
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    dec_inp = torch.zeros_like(batch_y[:, -h:, :]).float()
    dec_inp = torch.cat([batch_y[:, :48, :], dec_inp], dim=1).float().to(device)
    with torch.no_grad():
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    return batch_x, outputs.detach().cpu().numpy()


def run_probe(h, c, run_dir, label):
    """Build the model exactly as run.py/exp_basic do for this (H, c);
    assert probe geometry against the exact update_config formulas, then
    forward the real first test batch and require bitwise equality with
    the stored pred.npy[:32]."""
    import torch

    args = argparse.Namespace(
        task_name="long_term_forecast", model="VisionTS",
        data="ETTm1", root_path="./dataset/ETT-small/", data_path="ETTm1.csv",
        features="M", target="OT", scale=True, embed="timeF", freq="h",
        seq_len=2304, label_len=48, pred_len=h, seasonal_patterns=None,
        batch_size=32, num_workers=0,
        vm_arch="mae_base", vm_ckpt=C.CKPT_DIR_REL, vm_pretrained=1,
        ft_type="ln", interpolation="bilinear",
        norm_const=0.4, align_const=c, periodicity=96)
    model, device = build_model(args, c, label, h)
    model.args_for_loader = args
    geom, actual = probe_geometry(model, h, c, label)
    batch_x, probe_pred = forward_first_batch(model, h, device)
    bs = int(probe_pred.shape[0])
    if bs != 32:
        fail("%s ETTm1_%d probe first batch size %d != 32" % (label, h, bs))
    pred = np.load(os.path.join(run_dir, "results", "_", "pred.npy"))
    eq = bool(np.array_equal(probe_pred, pred[:bs]))
    maxd = 0.0 if eq else float(np.max(np.abs(
        probe_pred.astype(np.float64) - pred[:bs].astype(np.float64))))
    if not eq:
        fail("%s ETTm1_%d probe first batch NOT bitwise == stored "
             "pred.npy[:32] (max|diff|=%.3e) — probe protocol mismatch, "
             "locate the cause" % (label, h, maxd))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"batch_size": bs, "pred_bitwise_vs_stored": True,
            "max_abs_diff": maxd, "device": str(device),
            "geometry": {"expected": geom, "actual": actual}}


def mode_baseline(h):
    metric = C.load_metric()
    d = C.exp001_dir(h)
    hint = ("re-sync EXP-001 artifacts from eta (save_repro_zeroshot/ETTm1_%d)"
            % h)
    if not os.path.isfile(os.path.join(d, "SUCCESS.txt")):
        fail("EXP-001 ETTm1_%d SUCCESS.txt missing under %s\n  -> %s"
             % (h, d, hint))
    rd = require_arrays(d, h, "exp001-baseline", hint)
    fp = gate_fingerprints(d, h, None, "exp001-baseline")
    ev = gate_arrays(rd, h, metric, "exp001-baseline", "exp003")
    col = C.exp031_ettm1_column()
    v = col[h]
    aligned = (abs(ev["mse"] - v["mse"]) <= 1e-12
               and abs(ev["mae"] - v["mae"]) <= 1e-12 and ev["n"] == v["n"])
    if not aligned:
        fail("EXP-031 column alignment FAIL ETTm1_%d: recomputed "
             "%.17g/%.17g n=%d vs EXP-031 %.17g/%.17g n=%s — locate the cause"
             % (h, ev["mse"], ev["mae"], ev["n"], v["mse"], v["mae"], v["n"]))
    pev = run_probe(h, C.BASELINE_C, d, "exp001-baseline")
    ev.update({"kind": "exp001_baseline_group", "dataset": "ETTm1",
               "horizon": h, "align_const": C.BASELINE_C, "fingerprints": fp,
               "exp031_visionts_column": {"mse": v["mse"], "mae": v["mae"],
                                          "n": v["n"], "aligned": True},
               "probe": pev})
    C.merge_json(VERIFY_JSON, "baseline_%d" % h, ev)
    print("[verify_t35] baseline ETTm1_%d OK: n=%d true bitwise (std & "
          "EXP-003), metrics==stored, EXP-031 column aligned, probe bitwise; "
          "mse=%.6f mae=%.6f" % (h, ev["n"], ev["mse"], ev["mae"]))


def mode_group(h, c):
    metric = C.load_metric()
    d = C.group_dir(h, c)
    if not os.path.isfile(os.path.join(d, "SUCCESS.txt")):
        fail("ETTm1_%d c=%s SUCCESS.txt missing under %s" % (h, C.c_str(c), d))
    rd = require_arrays(d, h, "t35-c%s" % C.c_str(c), "rerun this group")
    fp = gate_fingerprints(d, h, {"align_const": C.c_str(c)},
                           "t35-c%s" % C.c_str(c))
    ev = gate_arrays(rd, h, metric, "t35-c%s" % C.c_str(c), "exp001")
    pred = np.load(os.path.join(rd, "pred.npy"))
    pred_bl = np.load(os.path.join(C.exp001_dir(h), "results", "_", "pred.npy"))
    eq_bl = bool(np.array_equal(pred, pred_bl))
    maxd_bl = 0.0 if eq_bl else float(np.max(np.abs(
        pred.astype(np.float64) - pred_bl.astype(np.float64))))
    if eq_bl:
        fail("ETTm1_%d c=%s pred.npy bitwise == c=0.4 baseline — align_const "
             "had no effect, locate the cause" % (h, C.c_str(c)))
    pev = run_probe(h, c, d, "t35-c%s" % C.c_str(c))
    ev.update({"kind": "t35_align_group", "dataset": "ETTm1", "horizon": h,
               "align_const": c, "fingerprints": fp,
               "pred_bitwise_ne_baseline": True,
               "max_abs_diff_vs_baseline": maxd_bl, "probe": pev})
    C.merge_json(VERIFY_JSON, "group_%d_%s" % (h, C.c_str(c)), ev)
    print("[verify_t35] group ETTm1_%d c=%s OK: n=%d true bitwise (std & "
          "EXP-001), finite, pred != c0.4 (max|d|=%.4g), metrics==stored, "
          "probe bitwise; mse=%.6f mae=%.6f"
          % (h, C.c_str(c), ev["n"], maxd_bl, ev["mse"], ev["mae"]))


def mode_first_window(h, c):
    """Smoke extra: window 0 of the real test loader — L=2304 history rows
    [43776, 46080) and variable order vs std_array, plus the window-0
    future truth (seq_y's last H steps) rows [46080, 46080+H), bitwise."""
    import random
    import torch

    args = argparse.Namespace(
        task_name="long_term_forecast", is_training=0, model="VisionTS",
        data="ETTm1", root_path="./dataset/ETT-small/", data_path="ETTm1.csv",
        features="M", target="OT", scale=True, embed="timeF", freq="h",
        seq_len=2304, label_len=48, pred_len=h, seasonal_patterns=None,
        batch_size=32, num_workers=0)
    os.chdir(C.ROOT)
    if C.ROOT not in sys.path:
        sys.path.insert(0, C.ROOT)
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    from data_provider.data_factory import data_provider
    _, loader = data_provider(args, "test")
    n = len(loader.dataset)
    n_exp = C.expected_n_windows(h)
    if n != n_exp:
        fail("ETTm1_%d loader N=%d != expected %d" % (h, n, n_exp))
    sample = loader.dataset[0]
    seq_x = np.asarray(sample[0]).astype(np.float32)  # (L, 7)
    if seq_x.shape != (2304, 7):
        fail("ETTm1_%d window0 history shape %s != (2304, 7)"
             % (h, seq_x.shape))
    t0 = C.first_pred_abs_row0()  # 46080
    want = C.std_array()[t0 - 2304:t0]
    if not np.array_equal(seq_x, want):
        d = np.abs(seq_x.astype(np.float64) - want.astype(np.float64))
        wv = int(np.unravel_index(int(np.argmax(d)), d.shape)[1])
        fail("ETTm1_%d window0 history NOT bitwise == std[%d:%d] (worst var "
             "col %d — check border1/L or variable order)"
             % (h, t0 - 2304, t0, wv))
    seq_y = np.asarray(sample[1]).astype(np.float32)
    fut = seq_y[-h:]
    want_fut = C.std_array()[t0:t0 + h]
    if not np.array_equal(fut, want_fut):
        fail("ETTm1_%d window0 future (seq_y last %d steps) NOT bitwise == "
             "std[%d:%d]" % (h, h, t0, t0 + h))
    ev = {"kind": "first_window_check", "dataset": "ETTm1", "horizon": h,
          "align_const": c, "loader_n": n,
          "history_len": int(seq_x.shape[0]),
          "history_bitwise_vs_std_array": True,
          "history_rows": [t0 - 2304, t0],
          "future_truth_bitwise_vs_std_array": True,
          "future_truth_rows": [t0, t0 + h],
          "variable_order_check": "all 7 columns bitwise match std_array "
                                  "column order (HUFL, HULL, MUFL, MULL, "
                                  "LUFL, LULL, OT)"}
    C.merge_json(VERIFY_JSON, "first_window_%d_%s" % (h, C.c_str(c)), ev)
    print("[verify_t35] first_window ETTm1_%d c=%s OK: loader N=%d, win0 "
          "history rows [%d,%d) and future truth [%d,%d) bitwise == "
          "std_array (L=2304, 7 vars, order ok)"
          % (h, C.c_str(c), n, t0 - 2304, t0, t0, t0 + h))


def mode_smoke_hash(h, c):
    """Smoke-pair HARD assertions + SHA-256 evidence. Both models (c=0.4
    baseline protocol and the requested c) are built and replay the SAME
    first test batch:
      * loaders built under both protocols produce bitwise-identical inputs
      * each replay is bitwise vs its own stored pred.npy[:32]
      * pinned geometry pairs for (H=96, c=0.4->0.6): npi 5->8, input
        224x80->224x128, output 96x67->96x42 (always also formula-checked)
      * SHA-256 of the shared input bytes and of both stored predictions;
        the two prediction hashes must differ
    """
    import torch

    if (h, c) != (96, 0.6):
        fail("smoke_hash is the ETTm1_96 c=0.4->0.6 evidence; got (%d, %s)"
             % (h, C.c_str(c)))

    def make_args(cc):
        return argparse.Namespace(
            task_name="long_term_forecast", model="VisionTS",
            data="ETTm1", root_path="./dataset/ETT-small/",
            data_path="ETTm1.csv", features="M", target="OT", scale=True,
            embed="timeF", freq="h", seq_len=2304, label_len=48, pred_len=h,
            seasonal_patterns=None, batch_size=32, num_workers=0,
            vm_arch="mae_base", vm_ckpt=C.CKPT_DIR_REL, vm_pretrained=1,
            ft_type="ln", interpolation="bilinear",
            norm_const=0.4, align_const=cc, periodicity=96)

    os.chdir(C.ROOT)
    if C.ROOT not in sys.path:
        sys.path.insert(0, C.ROOT)
    import torch
    from data_provider.data_factory import data_provider

    _, loader_a = data_provider(make_args(C.BASELINE_C), "test")
    bx_a, by_a, bxm_a, bym_a = next(iter(loader_a))
    _, loader_b = data_provider(make_args(c), "test")
    bx_b, by_b, bxm_b, bym_b = next(iter(loader_b))
    inputs_equal = bool(np.array_equal(bx_a.numpy(), bx_b.numpy()))
    if not inputs_equal:
        fail("smoke pair: first-batch inputs differ between the c=0.4 and "
             "c=%s protocols — data pipeline drift" % C.c_str(c))
    in_sha = C.sha256_of_bytes(np.ascontiguousarray(bx_a.numpy()).tobytes())

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stored = {}
    hashes = {}
    geoms = {}
    for cc in (C.BASELINE_C, c):
        args = make_args(cc)
        model, _ = build_model(args, cc, "smoke-c%s" % C.c_str(cc), h)
        model.args_for_loader = args
        geom, actual = probe_geometry(model, h, cc, "smoke-c%s" % C.c_str(cc))
        geoms[cc] = {"expected": geom, "actual": actual}
        bx, out = forward_first_batch(model, h, dev)
        # forward path casts to float32; the raw loader batch is float64 —
        # compare after the same cast (f64->f32 is deterministic)
        if not np.array_equal(bx.cpu().numpy(),
                              bx_a.numpy().astype(np.float32)):
            fail("smoke pair c=%s: forward input != shared first batch"
                 % C.c_str(cc))
        rd = C.exp001_dir(h) if cc == C.BASELINE_C else C.group_dir(h, cc)
        pred = np.load(os.path.join(rd, "results", "_", "pred.npy"))
        if not np.array_equal(out, pred[:32]):
            maxd = float(np.max(np.abs(out.astype(np.float64)
                                       - pred[:32].astype(np.float64))))
            fail("smoke pair c=%s: replay NOT bitwise == stored pred.npy[:32] "
                 "(max|diff|=%.3e)" % (C.c_str(cc), maxd))
        stored[cc] = True
        hashes[cc] = C.sha256_of_bytes(np.ascontiguousarray(pred[:32]).tobytes())
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if hashes[C.BASELINE_C] == hashes[c]:
        fail("smoke pair: identical prediction SHA-256 for c=0.4 and c=%s — "
             "align_const had no effect" % C.c_str(c))
    g_base = geoms[C.BASELINE_C]["expected"]
    g_new = geoms[c]["expected"]
    pinned = ((g_base["num_patch_input"], g_new["num_patch_input"]) == (5, 8)
              and g_base["input_size"] == [224, 80]
              and g_new["input_size"] == [224, 128]
              and g_base["output_size"] == [96, 67]
              and g_new["output_size"] == [96, 42])
    if not pinned:
        fail("smoke pair pinned geometry FAIL: c=0.4 %s vs c=0.6 %s "
             "(want npi 5->8, in 224x80->224x128, out 96x67->96x42)"
             % (g_base, g_new))
    ev = {"kind": "smoke_pair_hash", "dataset": "ETTm1", "horizon": h,
          "align_pair": [C.BASELINE_C, c],
          "inputs_bitwise_identical": inputs_equal,
          "input_sha256": in_sha,
          "pred_sha256": {"0.4": hashes[C.BASELINE_C],
                          C.c_str(c): hashes[c]},
          "pred_hashes_differ": True,
          "replay_bitwise_vs_stored": stored,
          "geometry": {"0.4": geoms[C.BASELINE_C],
                       C.c_str(c): geoms[c]},
          "pinned_geometry_pairs_ok": True,
          "device": str(dev)}
    C.merge_json(VERIFY_JSON, "smoke_hash_%d_%s" % (h, C.c_str(c)), ev)
    print("[verify_t35] smoke_hash ETTm1_%d c=0.4->%s OK: inputs bitwise "
          "identical, both replays bitwise vs stored, npi 5->8 / in "
          "224x80->224x128 / out 96x67->96x42; sha256 in=%s.. pred0.4=%s.. "
          "pred%s=%s.." % (h, C.c_str(c), in_sha[:12],
                           hashes[C.BASELINE_C][:12], C.c_str(c),
                           hashes[c][:12]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs=1, metavar=("H",), type=int)
    ap.add_argument("--group", nargs=2, metavar=("H", "C"))
    ap.add_argument("--first_window", nargs=2, metavar=("H", "C"))
    ap.add_argument("--smoke_hash", nargs=2, metavar=("H", "C"))
    a = ap.parse_args()
    if a.baseline:
        mode_baseline(a.baseline[0])
    elif a.group:
        mode_group(int(a.group[0]), float(a.group[1]))
    elif a.first_window:
        mode_first_window(int(a.first_window[0]), float(a.first_window[1]))
    elif a.smoke_hash:
        mode_smoke_hash(int(a.smoke_hash[0]), float(a.smoke_hash[1]))
    else:
        ap.error("one of --baseline/--group/--first_window/--smoke_hash "
                 "required")


if __name__ == "__main__":
    main()
