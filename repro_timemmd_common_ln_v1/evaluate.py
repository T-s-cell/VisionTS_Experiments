"""Independent metric computation from cached arrays (EXP-028).

Recomputes from npz (never trusts inference jsons):
  - per group: f32 full-element MSE/MAE (main, TaTS-consistent), f64 audit,
    per-step f64
  - per domain: equal-weight mean of 4 horizons
  - overall: 9-domain equal-weight AND 36-group equal-weight
  - PRIMARY comparison EXP-027: recompute ITS 36-group overall from its
    runs npz and assert the frozen anchor (|diff| <= 1e-9); identity guard
    (window_index exact + truth bitwise) then per-group/domain/overall
    deltas (positive delta = LN worse than zero-shot)
  - SECONDARY TaTS: per-group comparison + frozen anchor assert
    (36-group equal-weight), unchanged from EXP-027 convention
  - epoch0 vs epoch1 val diagnostics (report-only): epoch0 val MSE
    recomputed from EXP-027 psearch val arrays at the FIXED P; epoch1 val
    MSE from training logs
Writes evaluation.json.
"""
import argparse
from pathlib import Path

import numpy as np

from repro_timemmd_common_ln_v1 import protocol as P
from repro_timemmd_common_ln_v1.common import bitwise_equal, dump_json, \
    load_json, setup_deterministic_env


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


def load_pred_true(gdir, dom, h, name=None):
    arr = dict(np.load(str(gdir / (name or P.test_npz_name(dom, h)))))
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
    runs27 = P.exp027_out() / "runs"
    ps27 = P.exp027_out() / "psearch"

    tats_ref = args.tats_ref
    if tats_ref is None:
        tats_ref = str(P.tats_ref_dir())
    use_tats = bool(tats_ref) and Path(tats_ref).exists()

    groups, by_domain = {}, {}
    for dom, h in P.all_groups():
        gid = P.group_id(dom, h)
        arr, pred, true = load_pred_true(runs / gid, dom, h)
        m = metrics_from(pred, true)
        m["group"] = gid
        m["windows"] = int(pred.shape[0])

        # PRIMARY: EXP-027 zero-shot, identity-guarded
        arr27, pred27, true27 = load_pred_true(runs27 / gid, dom, h)
        if not (np.array_equal(arr["window_index"], arr27["window_index"])
                and bitwise_equal(arr["truth"], arr27["truth"])
                and bitwise_equal(arr["s"], arr27["s"])
                and bitwise_equal(arr["orig_row"], arr27["orig_row"])
                and bitwise_equal(arr["label_dates"],
                                  arr27["label_dates"])):
            raise SystemExit("[FATAL] identity guard vs EXP-027 failed %s"
                             % gid)
        m27 = metrics_from(pred27, true27)
        m["exp027"] = {
            "mse": m27["mse"], "mae": m27["mae"],
            "delta_mse": m["mse"] - m27["mse"],
            "delta_mae": m["mae"] - m27["mae"],
            "rel_mse": (m["mse"] - m27["mse"]) / m27["mse"],
            "rel_mae": (m["mae"] - m27["mae"]) / m27["mae"],
        }

        # SECONDARY: TaTS supervised reference
        if use_tats:
            tnpz = Path(tats_ref) / gid / "test_arrays.npz"
            if not tnpz.exists():
                tnpz = (Path(tats_ref) / ("%s_seed2021" % gid) /
                        "test_arrays.npz")
            t = dict(np.load(str(tnpz)))
            tt = t["truth"][:, :, 0] if t["truth"].ndim == 3 else t["truth"]
            if not (bitwise_equal(t["s"], arr["s"])
                    and bitwise_equal(tt, true)):
                raise SystemExit("[FATAL] identity/truth guard failed %s"
                                 % gid)
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
        for key, src in (("exp027", "exp027"), ("tats", "tats")):
            if key == "tats" and not use_tats:
                continue
            ent["%s_mse" % key] = float(np.mean([r[src]["mse"]
                                                 for r in rows]))
            ent["%s_mae" % key] = float(np.mean([r[src]["mae"]
                                                 for r in rows]))
            ent["delta_mse_%s" % key] = ent["mse"] - ent["%s_mse" % key]
            ent["delta_mae_%s" % key] = ent["mae"] - ent["%s_mae" % key]
            ent["rel_mse_%s" % key] = (ent["delta_mse_%s" % key]
                                       / ent["%s_mse" % key])
            ent["rel_mae_%s" % key] = (ent["delta_mae_%s" % key]
                                       / ent["%s_mae" % key])
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

    # ---- EXP-027 anchor: recompute its 36-group overall, assert frozen pin
    m27_all = [metrics_from(*load_pred_true(runs27 / gid, dom, h)[1:])
               for (dom, h) in P.all_groups()
               for gid in [P.group_id(dom, h)]]
    e27_mse = float(np.mean([r["mse"] for r in m27_all]))
    e27_mae = float(np.mean([r["mae"] for r in m27_all]))
    if abs(e27_mse - P.EXP027_OVERALL_MSE) > 1e-9 or \
            abs(e27_mae - P.EXP027_OVERALL_MAE) > 1e-9:
        raise SystemExit(
            "[FATAL] EXP-027 anchor mismatch: recomputed (%.16f, %.16f) "
            "vs frozen (%.16f, %.16f)"
            % (e27_mse, e27_mae, P.EXP027_OVERALL_MSE,
               P.EXP027_OVERALL_MAE))
    result["exp027_overall_36group"] = {
        "mse": e27_mse, "mae": e27_mae, "anchor_ok": True}
    result["overall_36group_equal"]["delta_mse_exp027"] = \
        overall36["mse"] - e27_mse
    result["overall_36group_equal"]["rel_mse_exp027"] = \
        (overall36["mse"] - e27_mse) / e27_mse
    result["overall_36group_equal"]["delta_mae_exp027"] = \
        overall36["mae"] - e27_mae
    result["overall_36group_equal"]["rel_mae_exp027"] = \
        (overall36["mae"] - e27_mae) / e27_mae

    # ---- TaTS anchor (secondary, same convention as EXP-027)
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
        result["overall_36group_equal"]["delta_mse_tats"] = \
            overall36["mse"] - t_mse
        result["overall_36group_equal"]["rel_mse_tats"] = \
            (overall36["mse"] - t_mse) / t_mse
        result["overall_36group_equal"]["delta_mae_tats"] = \
            overall36["mae"] - t_mae
        result["overall_36group_equal"]["rel_mae_tats"] = \
            (overall36["mae"] - t_mae) / t_mae

    # ---- epoch0 (EXP-027 zero-shot val) vs epoch1 (LN) val diagnostics
    val_diag = {}
    tstat = load_json(out_root / "status.json")["groups"]
    for dom in P.ORDER:
        p = int(P.FIXED_P[dom])
        rows = []
        for h in P.TABLE[dom]["H"]:
            e0 = dict(np.load(str(ps27 / P.psearch_val_npz_name(dom, p,
                                                                h))))
            d0 = e0["pred"].astype(np.float64) - \
                e0["truth"].astype(np.float64)
            if d0.ndim == 3:
                d0 = d0[:, :, 0]
            mse0 = float(np.mean(d0 * d0))
            ent = tstat.get(P.group_id(dom, h), {})
            rows.append({"h": h,
                         "epoch0_val_mse_exp027": mse0,
                         "epoch1_val_mse_ln": ent.get("val_mse"),
                         "delta_val_mse": (ent.get("val_mse") - mse0
                                           if ent.get("val_mse")
                                           is not None else None)})
        val_diag[dom] = rows
    result["val_epoch0_vs_epoch1"] = val_diag

    # improved/degraded vs EXP-027 (f32 main, per group)
    imp = sorted([(gid, r["exp027"]["rel_mse"]) for gid, r in
                  groups.items() if r["exp027"]["delta_mse"] < 0],
                 key=lambda kv: kv[1])
    deg = sorted([(gid, r["exp027"]["rel_mse"]) for gid, r in
                  groups.items() if r["exp027"]["delta_mse"] >= 0],
                 key=lambda kv: kv[1])
    result["vs_exp027"] = {"n_improved": len(imp), "n_degraded": len(deg),
                           "improved": imp, "degraded": deg}

    dump_json(out_root / "evaluation.json", result)
    print("EVAL OK: windows=%d" % result["test_windows"])
    print("Overall 9-domain equal: MSE %.6f MAE %.6f"
          % (overall9["mse"], overall9["mae"]))
    print("Overall 36-group equal: MSE %.6f MAE %.6f"
          % (overall36["mse"], overall36["mae"]))
    print("vs EXP-027 zero-shot: dMSE %+.6f (%+.2f%%), improved %d / "
          "degraded %d groups"
          % (result["overall_36group_equal"]["delta_mse_exp027"],
             100 * result["overall_36group_equal"]["rel_mse_exp027"],
             len(imp), len(deg)))
    if use_tats:
        print("vs TaTS: dMSE %+.6f (%+.2f%%)"
              % (result["overall_36group_equal"]["delta_mse_tats"],
                 100 * result["overall_36group_equal"]["rel_mse_tats"]))


if __name__ == "__main__":
    main()
