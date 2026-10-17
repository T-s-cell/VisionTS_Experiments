"""Per-group REPLAY as an independent process: strict ckpt restore ->
recompute the whole test split -> compare against the cached test npz
(pred atol/rtol + identity exact + truth bitwise). Also proves immutability
of the ckpt npz and the cached test npz by hashing before/after.
"""
import argparse
from pathlib import Path

import numpy as np

from repro_timemmd_common_ln_v1 import protocol as P
from repro_timemmd_common_ln_v1.adapter import build_visionts, configure, \
    freeze_to_ln, predict_split
from repro_timemmd_common_ln_v1.common import bitwise_equal, dump_json, \
    identity_equal, pred_close, setup_deterministic_env, sha256_of
from repro_timemmd_common_ln_v1.data_timemmd import DomainData
from repro_timemmd_common_ln_v1.ln_train import restore_ln_ckpt_strict


def run_group(dom, h, atol=P.ATOL, rtol=P.RTOL, ckpt_override=None,
              meta_override=None):
    setup_deterministic_env()
    pinfo = P.verify_frozen()
    out_dir = P.out_root()
    gid = P.group_id(dom, h)
    gdir = out_dir / "runs" / gid
    test_path = gdir / P.test_npz_name(dom, h)
    replay_path = gdir / P.replay_npz_name(dom, h)

    sha_test_pre = sha256_of(test_path)
    ckpt_path = Path(ckpt_override) if ckpt_override else \
        out_dir / P.ckpt_npz_name(dom, h)
    meta_path = Path(meta_override) if meta_override else \
        out_dir / P.ckpt_meta_name(dom, h)
    sha_ckpt_pre = sha256_of(ckpt_path)

    model = build_visionts().to("cuda:0")
    configure(model, int(P.TABLE[dom]["L"]), h, int(P.FIXED_P[dom]))
    freeze_to_ln(model)
    meta = restore_ln_ckpt_strict(model, ckpt_path, meta_path,
                                  pinfo, dom, h)

    base = dict(np.load(str(test_path)))
    dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], dom)
    sp = dd.load_split(h, "test")
    pred = predict_split(model, sp["X"], "cuda:0", batch=P.BATCH)
    np.savez(str(replay_path), pred=pred.astype(np.float32),
             truth=sp["Y"].astype(np.float32),
             s=sp["s"], orig_row=sp["orig_row"],
             label_dates=sp["label_dates"],
             window_index=sp["window_index"])

    close, max_abs, max_viol = pred_close(pred, base["pred"], atol, rtol)
    ident = (identity_equal(sp["s"], base["s"])
             and identity_equal(sp["orig_row"], base["orig_row"])
             and identity_equal(sp["label_dates"], base["label_dates"])
             and identity_equal(sp["window_index"], base["window_index"]))
    truth_ok = bitwise_equal(sp["Y"], base["truth"])
    ok = close and ident and truth_ok
    report = {
        "gid": P.group_id(dom, h), "domain": dom, "h": int(h),
        "windows": int(sp["X"].shape[0]), "close": close,
        "max_abs_diff": max_abs, "max_tol_violation": max_viol,
        "atol": atol, "rtol": rtol,
        "identity_ok": bool(ident), "truth_bitwise_ok": bool(truth_ok),
        "ckpt_npz_sha256": meta["npz_sha256"],
        "sha_test_before": sha_test_pre, "sha_ckpt_before": sha_ckpt_pre,
        "sha_test_after": sha256_of(test_path),
        "sha_ckpt_after": sha256_of(ckpt_path),
        "immutability_ok": (sha_test_pre == sha256_of(test_path)
                            and sha_ckpt_pre == sha256_of(ckpt_path)),
        "all_pass": bool(ok and (sha_test_pre == sha256_of(test_path))
                         and (sha_ckpt_pre == sha256_of(ckpt_path))),
    }
    dump_json(gdir / ("replay_report__%s.json" % gid), report)
    print("REPLAY %s %s close=%s max_abs=%.3e viol=%.3e"
          % (report["gid"], "PASS" if report["all_pass"] else "FAIL",
             close, max_abs, max_viol))
    if not report["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--meta", default=None)
    args = ap.parse_args()
    run_group(args.dom, args.h, ckpt_override=args.ckpt,
              meta_override=args.meta)
