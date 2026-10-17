"""Validation-set period search: 136 combos (domain x P x H), then freeze
selected_p.json BEFORE any test inference. Combo-level ledger -> resumable;
a cached combo is only skipped when its stored fingerprint (protocol pins,
weights, code md5, config) matches the current one AND the cached array
passes integrity checks vs the protocol split.

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
from repro_timemmd_common_v1.common import all_finite, bitwise_equal, die, \
    dump_json, load_json, seed_everything, setup_deterministic_env, \
    sha256_of
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


def combo_fingerprint(pinfo, dom, p, h):
    """Identity of everything that determines a combo's result. Stored in
    the ledger at done-time; a skip is only allowed when the stored
    fingerprint equals the current one AND the cached array passes
    integrity checks."""
    return {
        "protocol_json_sha256": P.PROTOCOL_JSON_SHA256,
        "splits_json_sha256": P.SPLITS_JSON_SHA256,
        "weights_sha256": pinfo["weights_sha256"],
        "code_md5": P.package_code_md5s(),
        "domain": dom, "p": int(p), "h": int(h),
        "seed": P.SEED, "batch": P.BATCH,
        "norm_const": P.NORM_CONST, "align_const": P.ALIGN_CONST,
        "interpolation": P.INTERPOLATION, "fp64": P.FP64,
    }


def verify_combo_arrays(npz_path, sp):
    """Integrity of a cached val-combo array vs the protocol split:
    keys, count, window_index permutation, identity/truth bitwise,
    finiteness. Independent of status.json metrics."""
    try:
        arr = dict(np.load(npz_path))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "load failed: %r" % e}
    for k in ("pred", "truth", "s", "orig_row", "label_dates",
              "window_index"):
        if k not in arr:
            return {"ok": False, "reason": "missing key %s" % k}
    n = arr["pred"].shape[0]
    if arr["pred"].shape[1:] != sp["Y"].shape[1:]:
        return {"ok": False, "reason": "pred shape %s != protocol %s"
                % (arr["pred"].shape[1:], sp["Y"].shape[1:])}
    if not np.array_equal(np.sort(arr["window_index"]), np.arange(n)):
        return {"ok": False, "reason": "window_index not permutation"}
    if n != len(sp["X"]):
        return {"ok": False, "reason": "count %d != protocol %d"
                % (n, len(sp["X"]))}
    for k in ("s", "orig_row", "label_dates"):
        if not bitwise_equal(arr[k], sp[k]):
            return {"ok": False, "reason": "%s mismatch vs protocol" % k}
    if not bitwise_equal(arr["truth"], sp["Y"]):
        return {"ok": False, "reason": "truth mismatch vs protocol"}
    if not np.isfinite(arr["pred"].astype(np.float64)).all():
        return {"ok": False, "reason": "non-finite pred"}
    return {"ok": True, "reason": "", "windows": int(n)}


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
            if dom not in domcache:
                domcache[dom] = DomainData(pinfo["protocol_dir"],
                                           pinfo["splits"], dom)
            npz_path = out_root / "psearch" / P.val_npz_name(dom, p, h)
            fp = combo_fingerprint(pinfo, dom, p, h)
            if (ent.get("state") == "done" and "mse" in ent
                    and ent.get("fingerprint") == fp
                    and npz_path.exists()):
                chk = verify_combo_arrays(
                    npz_path, domcache[dom].load_split(h, "val"))
                if chk["ok"]:
                    print("SKIP %s (fingerprint+integrity ok, %d windows)"
                          % (cid, chk["windows"]), flush=True)
                    continue
                print("REDO %s: cached array rejected (%s)"
                      % (cid, chk["reason"]), flush=True)
            else:
                print("REDO %s: no cached fingerprint match" % cid,
                      flush=True)
            status["combos"][cid] = {"state": "running"}
            save_status(out_root, status)
            try:
                res = run_combo(model, dev, domcache[dom], dom, p, h,
                                out_root)
                res["fingerprint"] = fp
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
    selection = select_periods(status)
    if sel_path.exists() and not args.force_select:
        old = load_json(sel_path)["selection"]
        old_sel = {d: old[d]["selected"] for d in old}
        new_sel = {d: selection[d]["selected"] for d in selection}
        if old_sel != new_sel:
            die("[TIER-1] selected_p.json exists with DIFFERENT selection "
                "%s vs %s; refusing to refreeze" % (old_sel, new_sel))
        print("SELECTED P UNCHANGED:", new_sel)
        print("selected_p.json sha256 = %s" % sha256_of(sel_path))
        return
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
