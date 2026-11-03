"""EXP-036 step 1: verify the three EXP-003 LN checkpoints under their own
training configs, then verify the encoder-LN splice evidence.

Per dataset (ETTm1 / Electricity / Weather, H=96 seed=2021):
  1. fingerprint re-check: the 12 EXP-003 code files md5 + dataset csv
     sha256 must match the job's fingerprint.json (deploy tree == training tree);
  2. SUCCESS.txt unrounded mse/mae == pinned anchors;
  3. strict checkpoint reload (keys 'vm.*'), first test batch forward in the
     job's own data order/batch -> max|pred - saved pred.npy[:B]| <= 1e-4;
  4. splice evidence: 50 encoder-LN keys present in the checkpoint; every
     non-LN vision_model tensor (params+buffers, encoder and decoder)
     bitwise == frozen MAE; at least one encoder-LN key differs, per-key
     max|delta| recorded; splicing into a fresh frozen model lands each key
     bitwise-equal to the source checkpoint.

All four weight files (frozen + 3 LN) are sha256-recorded. Merges into
logs/verify_t36.json section 'weights'.
"""
import argparse
import json
import os
import sys

import t36_common as C


def log(msg):
    print(msg, flush=True)


def fail(msg):
    raise SystemExit("[verify_t36_weights] FAIL: %s" % msg)


def check_fingerprints(ds):
    job = C.exp003_job(ds)
    fp = C.load_json(os.path.join(job, "fingerprint.json"))
    bad = {}
    for rel, want in fp["code_md5"].items():
        p = os.path.join(C.REPO_ROOT, rel)
        got = C.md5_of(p) if os.path.isfile(p) else "MISSING:" + p
        if got != want:
            bad[rel] = (want, got)
    if bad:
        fail("%s code fingerprint mismatch: %s" % (ds, list(bad.items())[:3]))
    for rel, want in fp["data_sha256"].items():
        p = os.path.join(C.ROOT, rel)  # keys like './dataset/ETT-small/ETTm1.csv', CWD=long_term_tsf
        got = C.sha256_of(p)
        if got != want:
            fail("%s data sha mismatch %s: %s != %s" % (ds, rel, got, want))
    ckpt_items = list(fp["ckpt_sha256"].items())
    got = C.sha256_of(C.CKPT_PATH)
    if ckpt_items and ckpt_items[0][1] != got:
        fail("frozen ckpt sha drift vs fingerprint.json: %s != %s"
             % (got, ckpt_items[0][1]))
    return {"code_files": len(fp["code_md5"]), "data_sha_ok": True, "ckpt_sha256": got}


def check_anchors(ds):
    with open(C.exp003_success(ds)) as f:
        s = json.load(f)
    mse, mae = float(s["mse"]), float(s["mae"])
    a = C.ANCHORS[ds]
    if mse != a["mse"] or mae != a["mae"]:
        fail("%s SUCCESS metrics %r/%r != pinned %r/%r" % (ds, mse, mae, a["mse"], a["mae"]))
    return {"success_mse": mse, "success_mae": mae, "anchor_match": True}


def probe_first_batch(ds):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from exp_univariate import Exp_Univariate

    job = C.exp003_job(ds)
    args = SimpleNamespace(**C.load_json(os.path.join(job, "args.json")))
    args.use_gpu = bool(torch.cuda.is_available())

    exp = Exp_Univariate(args)
    ckpt = os.path.join(job, "checkpoints", "_", "checkpoint.pth")
    state = torch.load(ckpt, map_location="cpu")
    exp.model.load_state_dict(state)  # strict
    exp.model.eval()

    _, loader = exp._get_data("test")
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))
    with torch.no_grad():
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
        outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        pred = outputs[:, -args.pred_len:, :].detach().cpu().numpy()

    pred_path = C.exp003_pred(ds)
    if not os.path.isfile(pred_path):
        fail("%s saved pred.npy missing (%s) -> decide: full-test replay fallback"
             % (ds, pred_path))
    ref = np.ascontiguousarray(np.load(pred_path, mmap_mode="r")[:pred.shape[0]])
    d = float(np.abs(pred.astype(np.float32) - ref.astype(np.float32)).max())
    if d > C.PROBE_TOL:
        fail("%s first-batch replay max|diff|=%.3e > %g (freeze site, do not retune)"
             % (ds, d, C.PROBE_TOL))
    n = pred.shape[0]

    del exp, state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("[verify_t36_weights] %s first-batch replay OK: B=%d max|diff|=%.3e" % (ds, n, d))
    return {"batch": int(n), "max_abs_diff": d, "tol": C.PROBE_TOL, "pred_path": pred_path}


def splice_evidence(ds):
    import torch
    src = C.load_exp003_visionts_dict(ds)
    missing = [k for k in C.ENC_LN_KEYS if k not in src]
    if missing:
        fail("%s checkpoint missing encoder-LN keys: %s..." % (ds, missing[:5]))

    frozen = torch.load(C.CKPT_PATH, map_location="cpu")
    mae = frozen["model"]

    enc_delta, nonln_equal, nonln_diff = {}, 0, []
    for k in src:
        if not k.startswith("vision_model."):
            continue
        mae_k = k[len("vision_model."):]
        if mae_k not in mae:
            continue
        if k in C.ENC_LN_KEYS:
            enc_delta[k] = float((src[k].float() - mae[mae_k].float()).abs().max())
        else:
            if bool(torch.equal(src[k], mae[mae_k])):
                nonln_equal += 1
            else:
                nonln_diff.append(k)
    if nonln_diff:
        fail("%s non-LN vision_model tensors differ from frozen: %s..."
             % (ds, nonln_diff[:5]))
    changed = {k: v for k, v in enc_delta.items() if v > 0.0}
    if not changed:
        fail("%s no encoder-LN key changed vs frozen" % ds)
    top = sorted(changed.items(), key=lambda kv: -kv[1])[:5]

    fig7, model = C.load_vision_model()
    evidence = C.splice_encoder_ln(model, src)
    del model
    log("[verify_t36_weights] %s splice OK: %d/%d keys changed (top %s), "
        "non-LN bitwise-equal=%d" % (ds, len(changed), len(enc_delta),
                                     ["%s:%.2e" % kv for kv in top[:2]], nonln_equal))
    return {
        "enc_ln_keys": len(C.ENC_LN_KEYS),
        "n_changed": len(changed),
        "max_delta": max(changed.values()),
        "top_deltas": {k: v for k, v in top},
        "per_key_delta": enc_delta,
        "nonln_bitwise_equal": nonln_equal,
        "splice_keys_equal_src": len(evidence),
        "ckpt_sha256": C.sha256_of(os.path.join(C.exp003_job(ds),
                                                "checkpoints", "_", "checkpoint.pth")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=C.EXP003_DS, action="append")
    args = ap.parse_args()
    dss = args.ds or C.EXP003_DS

    if C.sha256_of(C.CKPT_PATH) != C.CKPT_SHA256 or \
            os.path.getsize(C.CKPT_PATH) != C.CKPT_BYTES:
        fail("frozen ckpt sha/bytes mismatch: %s" % C.CKPT_PATH)

    out = {}
    for ds in dss:
        log("[verify_t36_weights] === %s ===" % ds)
        sec = {"job": C.exp003_job(ds)}
        sec.update(check_fingerprints(ds))
        sec.update(check_anchors(ds))
        sec.update(probe_first_batch(ds))
        sec.update(splice_evidence(ds))
        sec["frozen_ckpt_sha256"] = C.CKPT_SHA256
        out[ds] = sec
        data = C.load_json(C.VERIFY_JSON) if os.path.isfile(C.VERIFY_JSON) else {}
        data.setdefault("weights", {})[ds] = sec
        C.dump_json(C.VERIFY_JSON, data)
    log("[verify_t36_weights] ALL PASS for %s" % dss)


if __name__ == "__main__":
    main()
