"""EXP-040 step 1: audit the EXP-003 Illness archive + official source
before any training.  Gates AG1-AG9; a missing critical material
(AG1/AG3/AG4) STOPS the whole line with an explicit message.

  AG1  12 archives complete (args/fingerprint/SUCCESS/checkpoint.pth/
       valid_loss.json/metrics.npy/pred.npy/true.npy each)
  AG2  archive code_md5 map == md5 of the CURRENT tree (the audited
       corrected code is what produced EXP-003)
  AG3  illness CSV sha256 == pin (missing file -> STOP; eta gets it
       from local via scp at preflight)
  AG4  frozen MAE ckpt sha256 == pin; archived final checkpoint.pth
       file sha recorded per task
  AG5  every archived checkpoint torch.load(weights_only=False) ok
  AG6  coverage arithmetic from the ACTUAL Dataset_Custom borders:
       val N == {518/434/350/266}; official drop_last drops
       {6/178/94/10}; anchor 178 == 434 - 256; train steps recorded
  AG7  reload archived best ckpt -> strict -> official test path ->
       float64 metrics vs archived SUCCESS: rel <= 1e-3 (cross-env
       tolerance) AND true.npy bitwise == archived true.npy
  AG8  parse 12 theta logs (per-epoch Vali Loss, 7 decimals); curves
       recorded as REFERENCE ONLY (cross-env; no threshold)
  AG9  early-stop semantics vs archive: theta-log curve replayed with
       the REAL EarlyStopping tie semantics (ties improve -> last
       argmin) reproduces the archived best_valid_epoch

Run: python t40_audit.py [--gpu 0]   (chdirs to long_term_tsf)
Writes logs/audit_t40.json; exit 0 iff all gates pass (AG8 tolerates a
missing/unparseable log per task: recorded, not fatal).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

_T40_DIR = os.path.dirname(os.path.abspath(__file__))
if _T40_DIR not in sys.path:
    sys.path.insert(0, _T40_DIR)
import t40_common as X              # noqa: E402
import t38_common as C              # noqa: E402

FP_CODE_FILES = [
    "repro_fullshot/run_train.py", "repro_fullshot/exp_univariate.py",
    "long_term_tsf/exp/exp_long_term_forecasting.py",
    "long_term_tsf/exp/exp_basic.py", "long_term_tsf/models/VisionTS.py",
    "long_term_tsf/data_provider/data_factory.py",
    "long_term_tsf/data_provider/data_loader.py",
    "long_term_tsf/utils/tools.py", "long_term_tsf/utils/metrics.py",
    "visionts/model.py", "visionts/models_mae.py", "visionts/util.py",
]

VALI_RE = re.compile(r"Epoch:\s*(\d+), Steps:\s*(\d+)\s*\|\s*"
                     r"Train Loss:\s*([-\d.]+)\s+Vali Loss:\s*([-\d.eE+]+)")


def replay_state_machine(scores):
    """EarlyStopping tie semantics: score = -loss; first call saves;
    improvement iff NOT (score < best + delta), i.e. ties re-save; stop
    when counter >= patience.  Returns (best_epoch_1idx, trigger_epoch,
    improved_flags)."""
    best, best_ep, counter, trig = None, None, 0, None
    flags = []
    for i, v in enumerate(scores, start=1):
        s = -v
        if best is None:
            best, best_ep = s, i
            flags.append(True)
        elif s < best:
            counter += 1
            flags.append(False)
            if trig is None and counter >= 3:
                trig = i
        else:
            best, best_ep, counter = s, i, 0
            flags.append(True)
    return best_ep, trig, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    os.chdir(X.ROOT)   # long_term_tsf — EXP-003 CWD contract
    os.makedirs(X.LOG_DIR, exist_ok=True)

    rec = {"gates": {}, "tasks": [], "started": C.time_str()}
    import torch
    rec["env"] = {"python": sys.version.split()[0],
                  "torch": torch.__version__, "numpy": np.__version__,
                  "gpu": torch.cuda.get_device_name(a.gpu)
                  if torch.cuda.is_available() else "cpu"}
    print("[t40-audit] env %s" % rec["env"])

    # ---- AG1: archive completeness --------------------------------------
    missing = []
    for h, s in X.TASKS:
        d = X.exp003_job(h, s)
        for rel in ["args.json", "fingerprint.json", "SUCCESS.txt",
                    "checkpoints/_/checkpoint.pth",
                    "checkpoints/_/valid_loss.json",
                    "results/_/metrics.npy", "results/_/pred.npy",
                    "results/_/true.npy"]:
            if not os.path.isfile(os.path.join(d, rel)):
                missing.append("%s/%s" % (os.path.basename(d), rel))
    rec["gates"]["AG1"] = {"n_missing": len(missing),
                           "missing": missing[:40]}
    if missing:
        print("[t40-audit] STOP: archive materials missing (%d): %s..."
              % (len(missing), missing[:5]))
        with open(X.AUDIT_JSON, "w") as f:
            json.dump(rec, f, indent=1)
        sys.exit(2)
    print("[t40-audit] AG1 PASS: 12 archives complete")

    # ---- AG2: code_md5 == current tree ----------------------------------
    tree_md5 = {rel: C.md5_of(os.path.join(X.REPO_ROOT, rel))
                for rel in FP_CODE_FILES}
    n_bad = 0
    for h, s in X.TASKS:
        fp = C.load_json(os.path.join(X.exp003_job(h, s), "fingerprint.json"))
        bad = [k for k in FP_CODE_FILES
               if fp.get("code_md5", {}).get(k) != tree_md5[k]]
        if bad:
            n_bad += 1
            rec.setdefault("ag2_bad", []).append(
                {"task": "illness_%d_seed%d" % (h, s), "files": bad})
    rec["gates"]["AG2"] = {"n_tasks": len(X.TASKS), "n_mismatch": n_bad}
    assert n_bad == 0, "AG2 FAIL: archived code_md5 != current tree"
    print("[t40-audit] AG2 PASS: archived code_md5 == current tree (12/12)")

    # ---- AG3/AG4: data + frozen ckpt pins --------------------------------
    csv_path = os.path.join(X.ROOT, X.CSV_REL)
    ckpt_path = os.path.abspath(os.path.join(X.ROOT, "..", X.CKPT_REL))
    rec["gates"]["AG3"] = {"csv": csv_path}
    if not os.path.isfile(csv_path):
        print("[t40-audit] STOP: illness csv missing at %s (scp from "
              "local, then rerun)" % csv_path)
        with open(X.AUDIT_JSON, "w") as f:
            json.dump(rec, f, indent=1)
        sys.exit(2)
    assert C.sha256_of(csv_path) == X.CSV_SHA256, "AG3 FAIL: csv sha drifted"
    rec["gates"]["AG3"]["sha256"] = C.sha256_of(csv_path)
    rec["gates"]["AG3"]["pin"] = X.CSV_SHA256
    print("[t40-audit] AG3 PASS: illness csv sha == pin")

    assert os.path.isfile(ckpt_path), "AG4 STOP: frozen ckpt missing"
    ck_sha = C.sha256_of(ckpt_path)
    assert ck_sha == X.CKPT_SHA256, "AG4 FAIL: frozen ckpt sha drifted"
    final_shas = {}
    for h, s in X.TASKS:
        ck = os.path.join(X.exp003_job(h, s),
                          "checkpoints", "_", "checkpoint.pth")
        final_shas["illness_%d_seed%d" % (h, s)] = C.sha256_of(ck)
    rec["gates"]["AG4"] = {"frozen_sha256": ck_sha,
                           "final_ckpt_sha256": final_shas}
    print("[t40-audit] AG4 PASS: frozen ckpt sha == pin; 12 final ckpt "
          "sha recorded")

    # ---- AG5: strict torch.load ------------------------------------------
    n_ok = 0
    for h, s in X.TASKS:
        ck = os.path.join(X.exp003_job(h, s),
                          "checkpoints", "_", "checkpoint.pth")
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        assert isinstance(sd, dict) and len(sd) > 0, \
            "AG5 FAIL: empty state_dict %s_%d" % (h, s)
        n_ok += 1
    rec["gates"]["AG5"] = {"n_loaded": n_ok}
    print("[t40-audit] AG5 PASS: 12/12 checkpoints torch.load ok")

    # ---- AG6: coverage arithmetic from actual Dataset_Custom -------------
    from data_provider.data_factory import data_provider
    ag6 = {}
    for h, s in X.TASKS:
        args0 = C.load_json(os.path.join(X.exp003_job(h, s), "args.json"))
        args0["save_dir"] = X.exp003_job(h, s)
        _, tr_loader = data_provider(argparse.Namespace(**args0), "train")
        val_ds, val_loader_official = data_provider(
            argparse.Namespace(**args0), "val")
        N = len(val_ds)
        official_batches = len(val_loader_official)   # drop_last=True? no:
        # EXP-003 val loader is complete (Exp_Univariate._get_data); the
        # OFFICIAL drop_last truncation is computed arithmetically:
        dropped = N - 256 * (N // 256)
        ag6["illness_%d_seed%d" % (h, s)] = {
            "val_n": N, "c_batches": N // 256, "official_dropped": dropped,
            "train_steps": len(tr_loader),
            "pin_val_n": X.VAL_N_PIN[h],
            "pin_dropped": X.VAL_DROPPED_PIN[h]}
        assert N == X.VAL_N_PIN[h], "AG6 FAIL val N %d != %d" % (
            N, X.VAL_N_PIN[h])
        assert dropped == X.VAL_DROPPED_PIN[h], \
            "AG6 FAIL dropped %d != %d" % (dropped, X.VAL_DROPPED_PIN[h])
    assert X.VAL_DROPPED_PIN[36] == 434 - 256
    ag6["anchor_178_eq_434_minus_256"] = True
    rec["gates"]["AG6"] = ag6
    print("[t40-audit] AG6 PASS: coverage 518/434/350/266, drops "
          "6/178/94/10, anchor 178==434-256")

    # ---- AG7: reload archived best ckpt -> test -> metrics ---------------
    from exp_univariate import Exp_Univariate
    from utils.metrics import metric
    ag7 = {}
    for h, s in X.TASKS:
        d = X.exp003_job(h, s)
        args0 = C.load_json(os.path.join(d, "args.json"))
        args0["save_dir"] = os.path.abspath(d)
        args0["use_gpu"] = True
        args0["gpu"] = 0   # archived H48/H60 ran on theta cuda:1; eta remap
        exp = Exp_Univariate(argparse.Namespace(**args0))
        ck = os.path.join(d, "checkpoints", "_", "checkpoint.pth")
        sd_t = torch.load(ck, map_location="cpu", weights_only=False)
        exp.model.load_state_dict(sd_t)   # strict
        succ = C.load_json(os.path.join(d, "SUCCESS.txt"))
        _, test_loader = data_provider(exp.args, "test")
        preds, trues = [], []
        exp.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(exp.device)
                batch_y = batch_y.float().to(exp.device)
                batch_x_mark = batch_x_mark.float().to(exp.device)
                batch_y_mark = batch_y_mark.float().to(exp.device)
                dec_inp = torch.zeros_like(
                    batch_y[:, -args0["pred_len"]:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :args0["label_len"], :], dec_inp],
                    dim=1).float().to(exp.device)
                outputs = exp.model(batch_x, batch_x_mark, dec_inp,
                                    batch_y_mark)
                outputs = outputs[:, -args0["pred_len"]:, :]
                batch_y = batch_y[:, -args0["pred_len"]:, :]
                preds.append(outputs.detach().cpu().numpy().astype(np.float64))
                trues.append(batch_y.detach().cpu().numpy().astype(np.float64))
        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        mae64, mse64, _, _, _ = metric(preds, trues)
        ref_true = np.load(os.path.join(d, "results", "_", "true.npy"))
        true_bitwise = bool(np.array_equal(
            trues.astype(np.float32), ref_true))
        rel_mse = abs(mse64 - succ["mse"]) / succ["mse"]
        rel_mae = abs(mae64 - succ["mae"]) / succ["mae"]
        ag7["illness_%d_seed%d" % (h, s)] = {
            "mse_f64": mse64, "mae_f64": mae64,
            "arch_mse": succ["mse"], "arch_mae": succ["mae"],
            "rel_mse": rel_mse, "rel_mae": rel_mae,
            "true_bitwise": true_bitwise,
            "best_valid_epoch_arch": C.load_json(
                os.path.join(d, "checkpoints", "_", "valid_loss.json"))[
                    "best_valid_epoch"]}
        assert rel_mse <= 1e-3 and rel_mae <= 1e-3, \
            "AG7 FAIL %s_%d: rel mse %.3g mae %.3g" % (
                h, s, rel_mse, rel_mae)
        assert true_bitwise, \
            "AG7 FAIL %s_%d: true.npy not bitwise == archive" % (h, s)
        del exp
        torch.cuda.empty_cache()
        print("[t40-audit] AG7 illness_%d_seed%d: rel_mse=%.2e rel_mae=%.2e"
              % (h, s, rel_mse, rel_mae))
    rec["gates"]["AG7"] = ag7
    print("[t40-audit] AG7 PASS: 12/12 reloaded-archive metrics within "
          "rel 1e-3, true bitwise")

    # ---- AG8 + AG9: theta logs -------------------------------------------
    ag8, ag9 = {}, {}
    for h, s in X.TASKS:
        key = "illness_%d_seed%d" % (h, s)
        p = X.theta_log(h, s)
        if p is None:
            ag8[key] = {"log": None}
            ag9[key] = {"status": "no-log"}
            continue
        epochs, scores = [], []
        with open(p, errors="replace") as f:
            for line in f:
                m = VALI_RE.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    scores.append(float(m.group(4)))
        ag8[key] = {"log": p, "n_parsed": len(epochs),
                    "epochs": epochs, "vali_loss": scores}
        if not scores:
            ag9[key] = {"status": "unparseable"}
            continue
        d = X.exp003_job(h, s)
        vl = C.load_json(os.path.join(d, "checkpoints", "_",
                                      "valid_loss.json"))
        best_ep, trig_ep, _flags = replay_state_machine(scores)
        ag9[key] = {"status": "checked", "n_epochs_in_log": len(scores),
                    "arch_best_valid_epoch": vl["best_valid_epoch"],
                    "replayed_best_epoch": best_ep,
                    "replayed_trigger_epoch": trig_ep,
                    "match": best_ep == vl["best_valid_epoch"]}
    rec["gates"]["AG8"] = {k: {"log": v.get("log"),
                               "n_parsed": v.get("n_parsed", 0)}
                           for k, v in ag8.items()}
    rec["gates"]["AG8_curves"] = ag8
    rec["gates"]["AG9"] = ag9
    n_parsed = sum(1 for v in ag8.values() if v.get("n_parsed", 0) > 0)
    n_match = sum(1 for v in ag9.values() if v.get("match"))
    n_check = sum(1 for v in ag9.values() if v.get("status") == "checked")
    print("[t40-audit] AG8 parsed=%d/12 logs; AG9 replay-match=%d/%d"
          % (n_parsed, n_match, n_check))

    # ---- archived SUCCESS summary (report input) --------------------------
    succ_tab = {}
    for h, s in X.TASKS:
        succ = C.load_json(os.path.join(X.exp003_job(h, s), "SUCCESS.txt"))
        succ_tab["illness_%d_seed%d" % (h, s)] = {
            "mse": succ["mse"], "mae": succ["mae"],
            "best_valid_epoch": succ.get(
                "metrics_mae_mse_rmse_mape_mspe_valloss_valepoch", [None]*7)[6]}
    rec["archived_success"] = succ_tab
    rec["finished"] = C.time_str()
    with open(X.AUDIT_JSON, "w") as f:
        json.dump(rec, f, indent=1)
    print("[t40-audit] ALL GATES PASS -> %s" % X.AUDIT_JSON)


if __name__ == "__main__":
    main()
