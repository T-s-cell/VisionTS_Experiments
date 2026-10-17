"""Validation-set period search: 136 combos (domain x P x H), then freeze
selected_p.json BEFORE any test inference. Combo-level ledger -> resumable.

Selection rule (frozen): per domain, per P -> equal-weight mean of the 4
per-H standardized val MSE (float32 full-element); argmin; tie -> smaller P.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_visionts, configure, \
    predict_split
from repro_timemmd_common_v1.common import all_finite, die, dump_json, \
    load_json, seed_everything, setup_deterministic_env, sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData


def metrics(pred, true):
    d = pred - true
    mse = float(np.mean(d.astype(np.float32) ** 2))
    mae = float(np.mean(np.abs(d.astype(np.float32))))
    p64 = pred.astype(np.float64)
    t64 = true.astype(np.float64)
    d64 = p64 - t64
    return mse, mae, float(np.mean(d64 ** 2)), float(np.mean(np.abs(d64)))


def load_status(out_root):
    sp = Path(out_root) / "psearch" / "status.json"
    if sp.exists():
        return load_json(sp)
    return {"combos": {}}


def save_status(out_root, status):
    dump_json(Path(out_root) / "psearch" / "status.json", status)


def run_combo(model, dev, domdata, dom, p, h, out_root):
    from repro_timemmd_common_v1 import protocol as PP
    (Path(out_root) / "psearch").mkdir(parents=True, exist_ok=True)
    configure(model, PP.TABLE[dom]["L"], h, p)
    sp = domdata.load_split(h, "val")
    seed_everything(PP.SEED)
    t0 = time.time()
    pred = predict_split(model, sp["X"], dev, batch=PP.BATCH)
    elapsed = time.time() - t0
    pred = pred[:, :, None] if pred.ndim == 2 else pred
    truth = sp["Y"]
    if pred.shape != truth.shape:
        raise ValueError("shape %s vs truth %s" % (pred.shape, truth.shape))
    if not all_finite(pred):
        raise ValueError("non-finite predictions")
    mse, mae, mse64, mae64 = metrics(pred, truth)
    name = PP.val_npz_name(dom, p, h)
    np.savez_compressed(
        Path(out_root) / "psearch" / name,
        pred=pred, truth=truth, s=sp["s"], orig_row=sp["orig_row"],
        label_dates=sp["label_dates"], window_index=sp["window_index"])
    return {"state": "done", "mse": mse, "mae": mae, "mse_f64": mse64,
            "mae_f64": mae64, "windows": int(pred.shape[0]),
            "npz": name, "elapsed_s": round(elapsed, 3)}


def select_periods(status):
    """argmin over P of 4H equal-weight mean MSE; tie -> smaller P."""
    selection = {}
    for dom in P.ORDER:
        table = {}
        for p in P.TABLE[dom]["P"]:
            ms = [status["combos"][P.combo_id(dom, p, h)]["mse"]
                  for h in P.TABLE[dom]["H"]]
            table[str(p)] = float(np.mean(ms))
        best_p = min(P.TABLE[dom]["P"], key=lambda q: (table[str(q)], q))
        selection[dom] = {"selected": int(best_p), "mean_val_mse_by_P":
                          table}
    return selection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated combo ids (preflight subset)")
    ap.add_argument("--select_only", action="store_true",
                    help="skip combos; only (re)run selection+freeze")
    ap.add_argument("--force_select", action="store_true")
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())
    (out_root / "psearch").mkdir(parents=True, exist_ok=True)

    status = load_status(out_root)
    combos = P.all_psearch_combos()
    if args.only:
        want = set(args.only.split(","))
        combos = [c for c in combos if P.combo_id(*c) in want]

    if not args.select_only:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_visionts()
        model.to(dev)
        domcache = {}
        for dom, p, h in combos:
            cid = P.combo_id(dom, p, h)
            ent = status["combos"].get(cid, {})
            npz_ok = (out_root / "psearch" /
                      P.val_npz_name(dom, p, h)).exists()
            if ent.get("state") == "done" and npz_ok and "mse" in ent:
                print("SKIP %s (done)" % cid, flush=True)
                continue
            if dom not in domcache:
                domcache[dom] = DomainData(pinfo["protocol_dir"],
                                           pinfo["splits"], dom)
            status["combos"][cid] = {"state": "running"}
            save_status(out_root, status)
            try:
                res = run_combo(model, dev, domcache[dom], dom, p, h,
                                out_root)
                status["combos"][cid] = res
            except Exception as e:  # noqa: BLE001
                status["combos"][cid] = {"state": "failed",
                                         "error": repr(e)}
                save_status(out_root, status)
                raise
            save_status(out_root, status)
            print("DONE %s mse=%.6f (%.1fs)"
                  % (cid, res["mse"], res["elapsed_s"]), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing = [P.combo_id(d, p, h) for d, p, h in P.all_psearch_combos()
               if status["combos"].get(P.combo_id(d, p, h), {})
               .get("state") != "done"]
    if missing:
        die("[FATAL] %d combos not done, refusing to select: %s..."
            % (len(missing), missing[:5]))

    sel_path = out_root / "selected_p.json"
    if sel_path.exists() and not args.force_select:
        die("[FATAL] %s exists; use --force_select to refreeze"
            % sel_path)
    selection = select_periods(status)
    payload = {"selection": selection, "seed": P.SEED,
               "protocol_json_sha256": P.PROTOCOL_JSON_SHA256,
               "splits_json_sha256": P.SPLITS_JSON_SHA256,
               "rule": "argmin of equal-weight mean of 4 per-H val MSE "
                       "(f32); tie -> smaller P"}
    dump_json(sel_path, payload)
    print("SELECTED P:", {d: v["selected"] for d, v in
                          selection.items()})
    print("selected_p.json sha256 = %s" % sha256_of(sel_path))


if __name__ == "__main__":
    main()
