#!/usr/bin/env python3
"""Forward-pass caching for stage-1: per (variable, candidate P), run VisionTS
zero-shot over the frozen val+test rows and store predictions to cache/.

Isolation: imports visionts.* read-only; writes only under repro_zeroshot_ext/.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import torch

from timesx_data import find_zip, iter_variables
from record_fingerprint import RUN_CONFIG, ckpt_path, make_fingerprint, verify_fingerprint

EXPECTED_GEOM = {  # P -> (pad_left, pad_right, num_patch_input) with ctx=96, pred=12, c=0.4
    1: (0, 0, 4),
    5: (4, 3, 4),
    52: (8, 40, 3),
}


def run_variable(v, P, rows_by_split, by_id, model, device, batch_size):
    """Forward all frozen val+test rows of one variable under a single P."""
    model.update_config(context_len=RUN_CONFIG["context_len"],
                        pred_len=RUN_CONFIG["pred_len"],
                        periodicity=P,
                        norm_const=RUN_CONFIG["norm_const_r"],
                        align_const=RUN_CONFIG["align_const_c"],
                        interpolation=RUN_CONFIG["interpolation"])
    geom_tuple = (model.pad_left, model.pad_right, model.num_patch_input)
    assert geom_tuple == EXPECTED_GEOM[P], \
        f"P={P} geometry {geom_tuple} != expected {EXPECTED_GEOM[P]}"
    geom = {"pad_left": model.pad_left, "pad_right": model.pad_right,
            "num_patch_input": model.num_patch_input,
            "num_patch": model.num_patch, "mask_ratio": model.mask_ratio}

    ids, labels = [], []
    for split in ("val", "test"):
        for r in rows_by_split[split]:
            ids.append(r["sample_id"])
            labels.append(split)
    preds = np.zeros((len(ids), RUN_CONFIG["pred_len"]), dtype=np.float64)
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        xs = np.stack([by_id[b] for b in batch_ids])
        x = torch.from_numpy(xs).float().unsqueeze(-1).to(device)  # [b,96,1]
        assert x.shape[1] == RUN_CONFIG["context_len"], \
            "input length must stay 96 (padding is internal to the model only)"
        with torch.no_grad():
            y = model(x)  # [b,pred,1]
        preds[i:i + len(batch_ids)] = y.squeeze(-1).double().cpu().numpy()
    return {"sample_ids": ids, "splits": labels, "pred": preds, "geom": geom}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--splits", default=os.path.join(HERE, "splits.json"))
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--only-var", default=None, help="substring filter on var_key (smoke)")
    ap.add_argument("--only-p", type=int, nargs="*", default=None, help="restrict P values")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "cache"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    zip_path = find_zip(args.zip)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from visionts.model import VisionTS

    fp = make_fingerprint(zip_path, args.splits, args.ckpt_dir)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = {"fingerprint": fp, "entries": {}}
    if os.path.isfile(manifest_path):
        old = json.load(open(manifest_path))
        if old.get("fingerprint") == fp:
            manifest["entries"] = old.get("entries", {})
        else:
            verify_fingerprint(old.get("fingerprint", {}), zip_path, args.splits,
                               args.ckpt_dir, what="cache/manifest.json")
            manifest["entries"] = old.get("entries", {})

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda unavailable, falling back to cpu", file=sys.stderr)
        device = "cpu"
    model = VisionTS(arch=RUN_CONFIG["arch"], finetune_type=RUN_CONFIG["finetune_type"],
                     ckpt_dir=os.path.dirname(ckpt_path(args.ckpt_dir)), load_ckpt=True)
    model.eval().to(device)

    splits = json.load(open(args.splits))
    t0 = time.time()
    n_fwd = 0
    for v in iter_variables(zip_path):
        if args.only_var and args.only_var not in v.var_key:
            continue
        entry = splits["variables"].get(v.var_key)
        if entry is None:
            raise SystemExit(f"variable {v.var_key} missing from frozen splits.json")
        rows_by_split = {"val": entry["val"], "test": entry["test"]}
        wanted = [r["sample_id"] for r in entry["val"] + entry["test"]]
        want_set = set(wanted)
        by_id, true_by_id, stats = {}, {}, {}
        for s in v.samples:
            if s.sample_id in want_set:
                arr = np.asarray(s.past_val, dtype=np.float64)
                by_id[s.sample_id] = arr
                true_by_id[s.sample_id] = np.asarray(s.future_val, dtype=np.float64)
                stats[s.sample_id] = (float(arr.mean()), float(arr.std(ddof=0)))
        assert len(by_id) == len(wanted), f"{v.var_key}: missing past values for frozen rows"

        for P in v.p_candidates:
            if args.only_p and P not in args.only_p:
                continue
            key = f"{v.var_key}__P{P}"
            npz_path = os.path.join(args.out_dir, f"{key}.npz")
            ent = manifest["entries"].get(key)
            if ent and os.path.isfile(npz_path) and ent.get("n") == len(wanted):
                continue
            out = run_variable(v, P, rows_by_split, by_id, model, device, args.batch_size)
            ids = out["sample_ids"]
            mus = np.array([stats[i][0] for i in ids])
            stds = np.array([stats[i][1] for i in ids])
            cstds = np.full(len(ids), entry["calib_std"], dtype=np.float64)
            trues = np.stack([true_by_id[i] for i in ids])
            np.savez_compressed(npz_path,
                                sample_ids=np.array(ids), splits=np.array(out["splits"]),
                                pred=out["pred"], true=trues, mu=mus, win_std=stds,
                                calib_std=cstds, geom=np.array(json.dumps(out["geom"])))
            manifest["entries"][key] = {"n": len(ids), "geom": out["geom"],
                                        "file": os.path.basename(npz_path)}
            n_fwd += len(ids)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
    print(f"done: {len(manifest['entries'])} (var,P) entries, forwards_executed={n_fwd} "
          f"(== forwards_saved; one forward per (var,P,sample)), "
          f"{time.time() - t0:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
