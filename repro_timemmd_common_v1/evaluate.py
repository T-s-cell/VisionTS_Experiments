"""Independent metric computation from cached arrays + TaTS comparison.

Recomputes (never trusts run_infer jsons):
  - per group: f32 full-element MSE/MAE (main, TaTS-consistent), f64 audit
    MSE/MAE, per-step f64
  - per domain: equal-weight mean of 4 horizons
  - overall: 9-domain equal-weight AND 36-group equal-weight
  - TaTS comparison: identity guard (s/orig_row/label_dates exact, truth
    bitwise), then per-group/per-domain/overall deltas and relative change
  - asserts TaTS overall anchor (frozen check values)
Writes evaluation.json.
"""
import argparse
from pathlib import Path

import numpy as np

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import bitwise_equal, dump_json, \
    setup_deterministic_env


def metrics_from(pred, true):
    d32 = (pred - true).astype(np.float32)
    mse = float(np.mean(d32 ** 2))
    mae = float(np.mean(np.abs(d32)))
    p64, t64 = pred.astype(np.float64), true.astype(np.float64)
    d64 = p64 - t64
    per_step = [{"step": k + 1, "mse": float(np.mean(d64[:, k] ** 2)),
                 "mae": float(np.mean(np.abs(d64[:, k])))}
                for k in range(pred.shape[1])]
    return {"mse": mse, "mae": mae,
            "mse_f64": float(np.mean(d64 ** 2)),
            "mae_f64": float(np.mean(np.abs(d64))),
            "per_step_f64": per_step}


def load_pred_true(gdir, dom, h):
    arr = dict(np.load(gdir / P.test_npz_name(dom, h)))
    pred = arr["pred"]
    true = arr["truth"]
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    if true.ndim == 3:
        true = true[:, :, 0]
    return arr, pred, true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--tats_ref", default=None,
                    help="'' disables TaTS comparison")
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())
    runs = out_root / "runs"

    tats_ref = args.tats_ref
    if tats_ref is None:
        tats_ref = str(P.tats_ref_dir())
    use_tats = bool(tats_ref) and Path(tats_ref).exists()

    groups, by_domain = {}, {}
    for dom, h in P.all_groups():
        gid = P.group_id(dom, h)
        _, pred, true = load_pred_true(runs / gid, dom, h)
        m = metrics_from(pred, true)
        m["group"] = gid
        m["windows"] = int(pred.shape[0])
        if use_tats:
            tnpz = (Path(tats_ref) / gid / "test_arrays.npz")
            if not tnpz.exists():
                tnpz = (Path(tats_ref) / ("%s_seed2021" % gid) /
                        "test_arrays.npz")
            t = dict(np.load(tnpz))
            tt = t["truth"][:, :, 0] if t["truth"].ndim == 3 else t["truth"]
            if not (bitwise_equal(t["s"], dict(np.load(
                    runs / gid / P.test_npz_name(dom, h)))["s"])
                    and bitwise_equal(tt, true)):
                raise SystemExit("[FATAL] identity/truth guard failed %s"
                                 % gid)
            tm = metrics_from(pred, tt)
            tatm = metrics_from(t["pred_mixed"][:, :, 0]
                                if t["pred_mixed"].ndim == 3
                                else t["pred_mixed"], tt)
            m["tats"] = {
                "mse": tatm["mse"], "mae": tatm["mae"],
                "delta_mse": m["mse"] - tatm["mse"],
                "delta_mae": m["mae"] - tatm["mae"],
                "rel_mse": (m["mse"] - tatm["mse"]) / tatm["mse"],
                "rel_mae": (m["mae"] - tatm["mae"]) / tatm["mae"],
            }
        groups[gid] = m

    for dom in P.ORDER:
        rows = [groups[P.group_id(dom, h)] for h in P.TABLE[dom]["H"]]
        ent = {"mse": float(np.mean([r["mse"] for r in rows])),
               "mae": float(np.mean([r["mae"] for r in rows]))}
        if use_tats:
            ent["tats_mse"] = float(np.mean([r["tats"]["mse"]
                                             for r in rows]))
            ent["tats_mae"] = float(np.mean([r["tats"]["mae"]
                                             for r in rows]))
            ent["delta_mse"] = ent["mse"] - ent["tats_mse"]
            ent["delta_mae"] = ent["mae"] - ent["tats_mae"]
            ent["rel_mse"] = ent["delta_mse"] / ent["tats_mse"]
            ent["rel_mae"] = ent["delta_mae"] / ent["tats_mae"]
        by_domain[dom] = ent

    overall9 = {
        "mse": float(np.mean([by_domain[d]["mse"] for d in P.ORDER])),
        "mae": float(np.mean([by_domain[d]["mae"] for d in P.ORDER])),
    }
    allm = list(groups.values())
    overall36 = {"mse": float(np.mean([r["mse"] for r in allm])),
                 "mae": float(np.mean([r["mae"] for r in allm]))}
    result = {"groups": groups, "by_domain": by_domain,
              "overall_9domain_equal": overall9,
              "overall_36group_equal": overall36,
              "test_windows": int(sum(r["windows"] for r in allm)),
              "tats_comparison_used": use_tats}

    if use_tats:
        t_mse = float(np.mean([r["tats"]["mse"] for r in allm]))
        t_mae = float(np.mean([r["tats"]["mae"] for r in allm]))
        result["tats_overall_36group"] = {"mse": t_mse, "mae": t_mae}
        if abs(t_mse - P.TATS_OVERALL_MSE) > 1e-9 or \
                abs(t_mae - P.TATS_OVERALL_MAE) > 1e-9:
            raise SystemExit(
                "[FATAL] TaTS anchor mismatch: recomputed (%.16f, %.16f) "
                "vs frozen (%.16f, %.16f)"
                % (t_mse, t_mae, P.TATS_OVERALL_MSE, P.TATS_OVERALL_MAE))
        result["tats_overall_36group"]["anchor_ok"] = True
        result["overall_36group_equal"]["delta_mse"] = \
            overall36["mse"] - t_mse
        result["overall_36group_equal"]["rel_mse"] = \
            (overall36["mse"] - t_mse) / t_mse
        result["overall_36group_equal"]["delta_mae"] = \
            overall36["mae"] - t_mae
        result["overall_36group_equal"]["rel_mae"] = \
            (overall36["mae"] - t_mae) / t_mae

    dump_json(out_root / "evaluation.json", result)
    print("EVAL OK: windows=%d" % result["test_windows"])
    print("Overall 9-domain equal: MSE %.6f MAE %.6f"
          % (overall9["mse"], overall9["mae"]))
    print("Overall 36-group equal: MSE %.6f MAE %.6f"
          % (overall36["mse"], overall36["mae"]))
    if use_tats:
        print("vs TaTS 36-group: dMSE %+.6f (%+.2f%%)"
              % (result["overall_36group_equal"]["delta_mse"],
                 100 * result["overall_36group_equal"]["rel_mse"]))


if __name__ == "__main__":
    main()
