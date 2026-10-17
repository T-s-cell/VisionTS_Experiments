"""Subprocess probe for preflight B4: strict ckpt restore in a FRESH
process, then predict the first 8 val windows. Writes pred npz + digests
for the parent to compare against its in-process model.
"""
import argparse
from pathlib import Path

import numpy as np

from repro_timemmd_common_ln_v1 import protocol as P
from repro_timemmd_common_ln_v1.adapter import build_visionts, configure, \
    freeze_to_ln, predict_split
from repro_timemmd_common_ln_v1.common import dump_json, model_state_digest, \
    setup_deterministic_env
from repro_timemmd_common_ln_v1.data_timemmd import DomainData
from repro_timemmd_common_ln_v1.ln_train import restore_ln_ckpt_strict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--out", required=True,
                    help="output .npz path for probe predictions")
    ap.add_argument("--ckpt", default=None,
                    help="ckpt npz (default: formal out_root layout)")
    ap.add_argument("--meta", default=None,
                    help="ckpt meta json (default: formal out_root layout)")
    ap.add_argument("--n_probe", type=int, default=8)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen()
    model = build_visionts().to("cuda:0")
    configure(model, int(P.TABLE[args.dom]["L"]), args.h,
              int(P.FIXED_P[args.dom]))
    freeze_to_ln(model)
    out_dir = P.out_root()
    ckpt = Path(args.ckpt) if args.ckpt else \
        out_dir / P.ckpt_npz_name(args.dom, args.h)
    meta_path = Path(args.meta) if args.meta else \
        out_dir / P.ckpt_meta_name(args.dom, args.h)
    meta = restore_ln_ckpt_strict(model, ckpt, meta_path, pinfo,
                                  args.dom, args.h)
    dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], args.dom)
    spv = dd.load_split(args.h, "val")
    n = min(args.n_probe, spv["X"].shape[0])
    pred = predict_split(model, spv["X"][:n], "cuda:0", batch=P.BATCH)
    np.savez(str(args.out), pred=pred.astype(np.float32),
             window_index=spv["window_index"][:n])
    dump_json(str(args.out) + ".json", {
        "dom": args.dom, "h": args.h, "n": int(n),
        "digest_after": model_state_digest(model),
        "packed_sha256": meta["packed_sha256"],
        "ckpt_npz_sha256": meta["npz_sha256"]})
    print("RESTORE_PROBE OK %s_H%d n=%d"
          % (args.dom, args.h, n))


if __name__ == "__main__":
    main()
