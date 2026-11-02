"""EXP-034 verification. Three modes (each merges into logs/verify_t18.json):

  --bilinear DS H            EXP-001 group full gate (baseline integrity,
                             NO rerun path) + bilinear probe: first test
                             batch forward bitwise vs EXP-001 pred.npy[:32]
  --group DS H INTERP        new-group gate (config incl. interpolation,
                             true double bitwise, pred != bilinear,
                             metrics == stored) + probe vs this group's
                             pred.npy[:32]
  --first_window DS H INTERP smoke extra: window 0 history (Table-8 L)
                             and variable order vs std_array, bitwise

Gates per group (any failure -> SystemExit, no fallback):
  * SUCCESS.txt present; config tokens (Table-8 L/P, r=c=0.4, bs=32,
    train_epochs=0, seed=2021; +interpolation for new groups); ckpt
    sha256 == pinned EXP-001 weights AND == the recorded line; every
    recorded md5 re-hashed against the current tree
  * pred/true shape (N, H, 7), N == corrected window count (2881-H hourly,
    11521-H minute); no checkpoint FILES in the run dir
  * true bitwise == truth_recomputed(std_array) AND == the fixed-reshape
    restore of EXP-003 seed2021 truth (bilinear) / EXP-001 truth (variants)
  * metric(pred, true) == metrics.npy[:5] exactly
  * new groups: pred bitwise != EXP-001 pred (interpolation took effect)
  * probe: the model that would generate pred.npy (same constructor path,
    recorded input_resize/output_resize mode/size/antialias, geometry ==
    exact update_config formulas) reproduces the stored first batch bitwise

Missing baseline artifacts -> stop with a re-sync hint; md5/sha drift ->
stop and locate the cause (a rerun cannot legitimize a drifted tree).
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t18_common as C

VERIFY_JSON = os.path.join(C.PKG, "logs", "verify_t18.json")
MODE_NAME = {"bilinear": "BILINEAR", "bicubic": "BICUBIC", "nearest": "NEAREST"}


def fail(msg):
    raise SystemExit("[verify_t18] FAIL: %s" % msg)


def check_run_dir_clean(run_dir):
    """No weight/checkpoint FILES anywhere; the empty checkpoints/<model_id>/
    scaffolding dir is framework behavior (EXP-001 dirs have it too)."""
    arts = []
    for pat in ("**/*.pth", "**/*.ckpt", "**/*.bin", "checkpoints/**"):
        for p in glob.glob(os.path.join(run_dir, pat), recursive=True):
            if os.path.isfile(p):
                arts.append(os.path.relpath(p, run_dir))
    return arts


def require_arrays(dir_, ds, h, label, resync_hint):
    rd = os.path.join(dir_, "results", "_")
    for f in ("pred.npy", "true.npy", "metrics.npy"):
        if not os.path.isfile(os.path.join(rd, f)):
            fail("%s %s_%d missing %s under %s\n  -> %s"
                 % (label, ds, h, f, rd, resync_hint))
    return rd


def gate_fingerprints(run_dir, ds, h, want_extra, label):
    rec = C.parse_success(os.path.join(run_dir, "SUCCESS.txt"))
    cfg = rec["config"]
    c = C.DATASETS_T8[ds]
    want_cfg = {
        "dataset": ds, "seq_len": str(c["ctx"]), "periodicity": str(c["period"]),
        "pred_len": str(h), "norm_const": "0.4", "align_const": "0.4",
        "batch_size": "32", "train_epochs": "0", "seed": "2021",
        "arch": "mae_base", "features": "M",
    }
    if want_extra:
        want_cfg.update(want_extra)
    for k, want in want_cfg.items():
        got = cfg.get(k)
        if got != want:
            fail("%s %s_%d SUCCESS.txt %s=%s != %s — config/protocol drift, "
                 "locate the cause (no rerun)" % (label, ds, h, k, got, want))
    ckpt = os.path.join(C.ROOT, C.CKPT_REL)
    got_sha = C.sha256_of(ckpt)
    if got_sha != C.CKPT_SHA256:
        fail("%s %s_%d current ckpt sha256 %s != pinned EXP-001 weights %s — "
             "weights changed, locate the cause first"
             % (label, ds, h, got_sha, C.CKPT_SHA256))
    if cfg.get("ckpt_sha256") != got_sha:
        fail("%s %s_%d recorded ckpt_sha256 %s != current %s — locate the "
             "cause (no rerun)" % (label, ds, h, cfg.get("ckpt_sha256"), got_sha))
    bad = []
    for f in C.FP_FILES:
        want = rec["md5"].get(f)
        cur = C.md5_of(os.path.join(C.ROOT, f))
        if want != cur:
            bad.append("%s: recorded=%s current=%s" % (f, want, cur))
    if bad:
        fail("%s %s_%d fingerprint md5 drift — locate the cause (no rerun): %s"
             % (label, ds, h, "; ".join(bad)))
    arts = check_run_dir_clean(run_dir)
    if arts:
        fail("%s %s_%d unexpected checkpoint artifacts: %s" % (label, ds, h, arts))
    return {"config": cfg, "ckpt_sha256": got_sha,
            "md5_all_match": True, "checkpoint_artifacts": []}


def gate_arrays(rd, ds, h, metric, label, truth_second):
    """Shared array-level gates; truth_second: 'exp003' (fixed-reshape
    restore) or 'exp001' (the reused bilinear baseline's true.npy)."""
    pred = np.load(os.path.join(rd, "pred.npy"))
    true = np.load(os.path.join(rd, "true.npy"))
    mets = np.load(os.path.join(rd, "metrics.npy"))
    n_exp = C.expected_n_windows(ds, h)

    problems = []
    if true.shape != (n_exp, h, 7):
        problems.append("true shape %s != (%d, %d, 7)" % (true.shape, n_exp, h))
    if pred.shape != true.shape:
        problems.append("pred shape %s != true %s" % (pred.shape, true.shape))
    if problems:
        fail("%s %s_%d: %s" % (label, ds, h, "; ".join(problems)))

    tr = C.truth_recomputed(C.std_array(ds), ds, h)
    eq_std = bool(np.array_equal(true, tr))
    if not eq_std:
        fail("%s %s_%d: true.npy NOT bitwise == std_array recomputation"
             % (label, ds, h))
    if truth_second == "exp003":
        t2 = C.exp003_truth_restored(ds, h)
        src = "EXP-003 restored truth"
    else:
        t2 = np.load(os.path.join(C.exp001_dir(ds, h), "results", "_", "true.npy"))
        src = "EXP-001 truth"
    eq_2 = bool(np.array_equal(true, t2))
    if not eq_2:
        d = int(np.argmax(np.abs(true.astype(np.float64)
                                 - t2.astype(np.float64)).sum(axis=(1, 2))))
        fail("%s %s_%d: true.npy NOT bitwise == %s (worst window %d)"
             % (label, ds, h, src, d))

    rec = C.metrics_from_arrays(metric, pred, true)
    eq_m = bool(np.array_equal(rec, mets[:5].astype(np.float64), equal_nan=True))
    if not eq_m:
        fail("%s %s_%d: metric(pred,true) != metrics.npy "
             "(recomputed=%s stored=%s)" % (label, ds, h, rec.tolist(),
                                            mets[:5].tolist()))
    return {"dir": os.path.relpath(os.path.dirname(rd), C.ROOT),
            "n": int(true.shape[0]),
            "true_bitwise_vs_std_array": eq_std,
            "true_bitwise_vs_%s" % truth_second: eq_2,
            "metrics_recompute_equals_stored": eq_m,
            "mse": float(rec[1]), "mae": float(rec[0]),
            "first_pred_abs_row0": C.first_pred_abs_row0(ds)}


def run_probe(ds, h, interp, run_dir, label):
    """Build the model exactly as run.py/exp_basic do for this (ds, H,
    interp); record input_resize/output_resize mode+size+antialias, assert
    geometry against the exact update_config formulas, then forward the
    real first test batch and require bitwise equality with the stored
    pred.npy[:32] — evidence that the probe's configuration IS the one
    that generated the arrays, with BOTH resizes on the requested mode."""
    import torch

    c = C.DATASETS_T8[ds]
    args = argparse.Namespace(
        task_name="long_term_forecast", model="VisionTS",
        data=c["data"], root_path=c["root_path"], data_path=c["data_path"],
        features="M", target="OT", scale=True, embed="timeF", freq="h",
        seq_len=c["ctx"], label_len=48, pred_len=h, seasonal_patterns=None,
        batch_size=32, num_workers=0,
        vm_arch="mae_base", vm_ckpt=C.CKPT_REL, vm_pretrained=1,
        ft_type="ln", interpolation=interp,
        norm_const=0.4, align_const=0.4, periodicity=c["period"])
    os.chdir(C.ROOT)
    if C.ROOT not in sys.path:
        sys.path.insert(0, C.ROOT)
    from models.VisionTS import Model
    from data_provider.data_factory import data_provider

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(args).float().to(device)
    model.eval()
    vm = model.vm
    if not (vm.patch_size == C.MAE_PATCH_SIZE and vm.image_size == C.MAE_IMAGE_SIZE
            and vm.num_patch == C.MAE_NUM_PATCH):
        fail("%s %s_%d probe ckpt geometry patch=%s image=%s numpatch=%s != "
             "pinned ViT-B/16 (16/224/14)" % (label, ds, h, vm.patch_size,
                                              vm.image_size, vm.num_patch))
    geom = C.expected_geometry(ds, h)
    actual = {}
    for name in ("input_resize", "output_resize"):
        t = getattr(vm, name)
        mode = str(getattr(t, "interpolation", t))
        size = [int(x) for x in t.size] if isinstance(t.size, (tuple, list)) \
            else [int(t.size)]
        aa = getattr(t, "antialias", None)
        actual[name] = {"mode": mode, "size": size, "antialias": aa}
        if MODE_NAME[interp] not in mode:
            fail("%s %s_%d probe %s interpolation %s != requested %s"
                 % (label, ds, h, name, mode, MODE_NAME[interp]))
        if aa is not None and aa is not False:
            fail("%s %s_%d probe %s antialias=%s != False" % (label, ds, h, name, aa))
    if actual["input_resize"]["size"] != geom["input_size"]:
        fail("%s %s_%d probe input_resize size %s != formula %s"
             % (label, ds, h, actual["input_resize"]["size"], geom["input_size"]))
    if actual["output_resize"]["size"] != geom["output_size"]:
        fail("%s %s_%d probe output_resize size %s != formula %s"
             % (label, ds, h, actual["output_resize"]["size"], geom["output_size"]))
    if int(vm.num_patch_input) != geom["num_patch_input"]:
        fail("%s %s_%d probe num_patch_input %d != formula %d"
             % (label, ds, h, int(vm.num_patch_input), geom["num_patch_input"]))

    _, loader = data_provider(args, "test")
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    dec_inp = torch.zeros_like(batch_y[:, -h:, :]).float()
    dec_inp = torch.cat([batch_y[:, :48, :], dec_inp], dim=1).float().to(device)
    with torch.no_grad():
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    probe_pred = outputs.detach().cpu().numpy()  # features='M': no f_dim slice
    pred = np.load(os.path.join(run_dir, "results", "_", "pred.npy"))
    bs = int(probe_pred.shape[0])
    if bs != 32:
        fail("%s %s_%d probe first batch size %d != 32" % (label, ds, h, bs))
    eq = bool(np.array_equal(probe_pred, pred[:bs]))
    maxd = 0.0 if eq else float(np.max(np.abs(
        probe_pred.astype(np.float64) - pred[:bs].astype(np.float64))))
    if not eq:
        fail("%s %s_%d probe first batch NOT bitwise == stored pred.npy[:32] "
             "(max|diff|=%.3e) — probe protocol mismatch, locate the cause"
             % (label, ds, h, maxd))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"batch_size": bs, "pred_bitwise_vs_stored": True,
            "max_abs_diff": maxd, "device": str(device),
            "geometry": {"expected": geom, "actual": actual}}


def mode_bilinear(ds, h):
    metric = C.load_metric()
    d = C.exp001_dir(ds, h)
    hint = ("re-sync EXP-001 artifacts from eta (save_repro_zeroshot/%s_%d)"
            % (ds, h))
    if not os.path.isfile(os.path.join(d, "SUCCESS.txt")):
        fail("EXP-001 %s_%d SUCCESS.txt missing under %s\n  -> %s" % (ds, h, d, hint))
    rd = require_arrays(d, ds, h, "exp001-bilinear", hint)
    fp = gate_fingerprints(d, ds, h, None, "exp001-bilinear")
    ev = gate_arrays(rd, ds, h, metric, "exp001-bilinear", "exp003")
    col = C.exp031_visionts_column()
    v = col["per_group"][(ds, h)]
    aligned = (abs(ev["mse"] - v["mse"]) <= 1e-12
               and abs(ev["mae"] - v["mae"]) <= 1e-12 and ev["n"] == v["n"])
    if not aligned:
        fail("EXP-031 column alignment FAIL %s_%d: recomputed %.17g/%.17g n=%d "
             "vs EXP-031 %.17g/%.17g n=%s — locate the cause"
             % (ds, h, ev["mse"], ev["mae"], ev["n"], v["mse"], v["mae"], v["n"]))
    pev = run_probe(ds, h, "bilinear", d, "exp001-bilinear")
    ev.update({"kind": "exp001_bilinear_group", "dataset": ds, "horizon": h,
               "fingerprints": fp,
               "exp031_visionts_column": {"mse": v["mse"], "mae": v["mae"],
                                          "n": v["n"], "aligned": True},
               "probe": pev})
    C.merge_json(VERIFY_JSON, "bilinear_%s_%d" % (ds, h), ev)
    print("[verify_t18] bilinear %s_%d OK: n=%d true bitwise (std & EXP-003), "
          "metrics==stored, EXP-031 column aligned, probe bitwise; "
          "mse=%.6f mae=%.6f" % (ds, h, ev["n"], ev["mse"], ev["mae"]))


def mode_group(ds, h, interp):
    metric = C.load_metric()
    d = C.interp_dir(ds, h, interp)
    if not os.path.isfile(os.path.join(d, "SUCCESS.txt")):
        fail("%s_%d %s SUCCESS.txt missing under %s" % (ds, h, interp, d))
    rd = require_arrays(d, ds, h, "t18-" + interp, "rerun this group")
    fp = gate_fingerprints(d, ds, h, {"interpolation": interp}, "t18-" + interp)
    ev = gate_arrays(rd, ds, h, metric, "t18-" + interp, "exp001")
    pred = np.load(os.path.join(rd, "pred.npy"))
    pred_bl = np.load(os.path.join(C.exp001_dir(ds, h), "results", "_", "pred.npy"))
    eq_bl = bool(np.array_equal(pred, pred_bl))
    maxd_bl = 0.0 if eq_bl else float(np.max(np.abs(
        pred.astype(np.float64) - pred_bl.astype(np.float64))))
    if eq_bl:
        fail("%s_%d %s pred.npy bitwise == bilinear baseline — interpolation "
             "had no effect, locate the cause" % (ds, h, interp))
    pev = run_probe(ds, h, interp, d, "t18-" + interp)
    ev.update({"kind": "t18_interp_group", "dataset": ds, "horizon": h,
               "interpolation": interp, "fingerprints": fp,
               "pred_bitwise_ne_bilinear": True,
               "max_abs_diff_vs_bilinear": maxd_bl, "probe": pev})
    C.merge_json(VERIFY_JSON, "group_%s_%d_%s" % (ds, h, interp), ev)
    print("[verify_t18] group %s_%d %s OK: n=%d true bitwise (std & EXP-001), "
          "pred != bilinear (max|d|=%.4g), metrics==stored, probe bitwise; "
          "mse=%.6f mae=%.6f"
          % (ds, h, interp, ev["n"], maxd_bl, ev["mse"], ev["mae"]))


def mode_first_window(ds, h, interp):
    """Smoke extra: window 0 of the real test loader — Table-8 L-step
    history and variable order vs std_array, bitwise."""
    import random
    import torch

    c = C.DATASETS_T8[ds]
    args = argparse.Namespace(
        task_name="long_term_forecast", is_training=0, model="VisionTS",
        data=c["data"], root_path=c["root_path"], data_path=c["data_path"],
        features="M", target="OT", scale=True, embed="timeF", freq="h",
        seq_len=c["ctx"], label_len=48, pred_len=h, seasonal_patterns=None,
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
    n_exp = C.expected_n_windows(ds, h)
    if n != n_exp:
        fail("%s_%d loader N=%d != expected %d" % (ds, h, n, n_exp))
    sample = loader.dataset[0]
    seq_x = np.asarray(sample[0]).astype(np.float32)  # (L, 7)
    if seq_x.shape != (c["ctx"], 7):
        fail("%s_%d window0 history shape %s != (%d, 7)"
             % (ds, h, seq_x.shape, c["ctx"]))
    t0 = C.first_pred_abs_row0(ds)
    want = C.std_array(ds)[t0 - c["ctx"]:t0]
    if not np.array_equal(seq_x, want):
        d = np.abs(seq_x.astype(np.float64) - want.astype(np.float64))
        wv = int(np.unravel_index(int(np.argmax(d)), d.shape)[1])
        fail("%s_%d window0 history NOT bitwise == std[%d:%d] (worst var "
             "col %d — check border1/L or variable order)"
             % (ds, h, t0 - c["ctx"], t0, wv))
    ev = {"kind": "first_window_check", "dataset": ds, "horizon": h,
          "interpolation": interp, "loader_n": n,
          "history_len": int(seq_x.shape[0]),
          "history_bitwise_vs_std_array": True,
          "history_rows": [t0 - c["ctx"], t0],
          "variable_order_check": "all 7 columns bitwise match std_array "
                                  "column order (HUFL, HULL, MUFL, MULL, "
                                  "LUFL, LULL, OT)"}
    C.merge_json(VERIFY_JSON, "first_window_%s_%d_%s" % (ds, h, interp), ev)
    print("[verify_t18] first_window %s_%d %s OK: loader N=%d, win0 history "
          "rows [%d,%d) bitwise == std_array (L=%d, 7 vars, order ok)"
          % (ds, h, interp, n, t0 - c["ctx"], t0, c["ctx"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bilinear", nargs=2, metavar=("DS", "H"))
    ap.add_argument("--group", nargs=3, metavar=("DS", "H", "INTERP"))
    ap.add_argument("--first_window", nargs=3, metavar=("DS", "H", "INTERP"))
    a = ap.parse_args()
    if a.bilinear:
        mode_bilinear(a.bilinear[0], int(a.bilinear[1]))
    elif a.group:
        mode_group(a.group[0], int(a.group[1]), a.group[2])
    elif a.first_window:
        mode_first_window(a.first_window[0], int(a.first_window[1]), a.first_window[2])
    else:
        ap.error("one of --bilinear/--group/--first_window required")


if __name__ == "__main__":
    main()
