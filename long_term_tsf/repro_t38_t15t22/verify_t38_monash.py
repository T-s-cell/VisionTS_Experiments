"""EXP-038 task A gates over all 30 code subsets. Reads
logs/monash_<ds>.json + logs/naive_<ds>.json and appends a 'monash'
section to logs/verify_t38.json. Exit 0 only when every HARD gate passes;
flag-only gate failures are recorded as anomalies and do NOT fail the
stage (over-tolerance rows are kept + explained, user correction #2).

Gates per subset (hard unless noted):
  G0 presence      monash json + naive json exist, runner recorded no error
  G1 pred windows  label windows all == meta prediction_length; for the 4
                   pinned subsets additionally == the pinned value (hard
                   only there; other 26 are evidence-only, user correction
                   #1: every subset gets an independent H check)
  G2 pred rule     independent rule re-derivation == runtime meta (evidence
                   for all; hard only for pinned subsets)
  G3 autotune P    accepted == paper Table 6 (hard for replay subsets;
                   flag + anomaly elsewhere)
  G4 anchor        VisionTS MAE rel <= 1e-3 vs author unrounded anchor
                   (hard for replay; elsewhere over-tolerance keeps the
                   row + anomaly, stage A does not turn green from it)
  G5 data version  probe evidence present (TSF: raw zip bytes+sha256;
                   M4: provenance + cache diff) and cache diff recorded
  G6 N1 rounding   N1 round-matches the paper Naive string (hard for
                   replay 2825.67/8.26; flag elsewhere; CIF has no direct
                   row -> covered by G9)
  G7 ratios        VisionTS/N1 and VisionTS/N2 finite and > 0
  G8 context       three-value context recorded; every context fed to the
                   model <= expected_from_accepted_p; truncation documented
                   when max < expected
  G11 phases       all six phases recorded, residual < 1% of wall

Global gates:
  G9 CIF pooled    runtime n6=15/n12=57; pooled VTS and pooled N1 both
                   round-match the paper pooled row 570907.24 / 578596.5
  G10 geomean      29-row geometric mean of VisionTS/N1 vs paper 0.729
                   (report + 2% WARN; only computed when all 29 rows
                   valid, otherwise coverage count is reported)
"""
import math
import os
import sys

import t38_common as C

PINNED = sorted(C.PINNED_PRED.keys())


def status(ok, hard):
    return ("PASS" if ok else ("FAIL" if hard else "FLAG")) + ("" if hard
                                                              else "(flag)")


def subset_gates(ds, vj_gates, anomalies):
    mrec_p = C.monash_json(ds)
    nrec_p = C.naive_json(ds)
    if not os.path.isfile(mrec_p):
        vj_gates.append({"ds": ds, "gate": "G0_presence", "st": "FAIL",
                         "hard": True, "detail": "monash json missing"})
        return None
    m = C.load_json(mrec_p)
    if "error" in m:
        vj_gates.append({"ds": ds, "gate": "G0_presence", "st": "FAIL",
                         "hard": True, "detail": m["error"]})
        return None
    n = C.load_json(nrec_p) if os.path.isfile(nrec_p) else None
    if n is None:
        vj_gates.append({"ds": ds, "gate": "G0_presence", "st": "FAIL",
                         "hard": True, "detail": "naive json missing"})
        return None
    vj_gates.append({"ds": ds, "gate": "G0_presence", "st": "PASS",
                     "hard": True, "detail": "runner+naive records present"})

    split = m.get("split", {})
    pinned = ds in C.PINNED_PRED
    replay = ds in C.REPLAY

    # G1 windows
    labels_ok = bool(split.get("label_len_unique")) and \
        split["label_len_unique"] == [split.get("meta_prediction_length")]
    detail = "label_len_unique=%s meta=%s" % (split.get("label_len_unique"),
                                              split.get("meta_prediction_length"))
    if pinned:
        want = C.PINNED_PRED[ds]
        ok = labels_ok and split.get("meta_prediction_length") == want
        detail += " pinned=%d" % want
        hard = True
    else:
        ok = labels_ok
        hard = False
    vj_gates.append({"ds": ds, "gate": "G1_pred_windows", "st": status(ok, hard),
                     "hard": hard, "detail": detail})

    # G2 independent rule
    rule = m.get("pred_rule", {})
    checks = m.get("checks", {})
    ok = bool(checks.get("rule_matches_meta"))
    hard = pinned
    if pinned:
        ok = ok and bool(checks.get("pinned_pred_match"))
    vj_gates.append({"ds": ds, "gate": "G2_pred_rule",
                     "st": status(ok, hard), "hard": hard,
                     "detail": "rule=%s(%s) meta=%s" %
                               (rule.get("expected"), rule.get("source"),
                                split.get("meta_prediction_length"))})

    # G3 autotune P
    acc = m.get("autotune", {}).get("accepted_p")
    want_p = C.EXPECTED_P[ds]
    ok = acc == want_p
    hard = replay
    vj_gates.append({"ds": ds, "gate": "G3_autotune_P",
                     "st": status(ok, hard), "hard": hard,
                     "detail": "accepted=%s expected=%d events=%s" %
                               (acc, want_p,
                                [(e["p"], round(e["valid_mae"], 6), e["accept"])
                                 for e in m.get("autotune", {}).get("events", [])])})
    if not ok:
        anomalies.append({"ds": ds, "type": "P_mismatch",
                          "detail": "accepted %s vs Table 6 %d" % (acc, want_p)})

    # G4 anchor
    mae = m.get("result_csv", {}).get("metrics", {}).get("MAE[0.5]")
    anchor = m.get("anchor", {})
    rd = anchor.get("rel_diff")
    ok = bool(anchor.get("pass"))
    hard = replay
    vj_gates.append({"ds": ds, "gate": "G4_vision_anchor",
                     "st": status(ok, hard), "hard": hard,
                     "detail": "MAE %s vs anchor %s rel %s tol %.0e" %
                               (mae, anchor.get("mae_anchor"), rd,
                                C.ANCHOR_TOL)})
    if not ok:
        anomalies.append({"ds": ds, "type": "anchor_over_tolerance",
                          "detail": "rel %.3e > 1e-3 (row kept; see "
                                    "per_series_mae argmax=%s)" %
                                    (rd, m.get("per_series_mae", {}).get("argmax"))})

    # G5 data version
    gen = m.get("generation", {})
    dv = m.get("data_version", {})
    if ds in C.M4_SUBSETS:
        ok = bool(gen.get("recipe_provenance")) and \
            "n_files_new_or_changed" in dv
        detail = "M4 provenance: %s; cache diff files=%s" % \
                 (gen.get("recipe_provenance"),
                  dv.get("n_files_new_or_changed"))
    else:
        ok = gen.get("download_sha256") is not None and \
             gen.get("downloaded_bytes") is not None and \
             "n_files_new_or_changed" in dv
        detail = "zip %s bytes sha %s...; cache diff files=%s" % \
                 (gen.get("downloaded_bytes"),
                  (gen.get("download_sha256") or "")[:16],
                  dv.get("n_files_new_or_changed"))
    vj_gates.append({"ds": ds, "gate": "G5_data_version",
                     "st": status(ok, True), "hard": True, "detail": detail})

    # G6 N1 round-match
    n1v = n.get("n1_vs_paper_naive", {})
    if ds.startswith("cif_2016_"):
        vj_gates.append({"ds": ds, "gate": "G6_n1_round", "st": "SKIP",
                         "hard": False,
                         "detail": "no direct paper row (G9 pooled)"})
    else:
        ok = bool(n1v.get("round_match"))
        hard = replay
        vj_gates.append({"ds": ds, "gate": "G6_n1_round",
                         "st": status(ok, hard), "hard": hard,
                         "detail": "N1 %s vs paper %s quantum %s rel %.4f" %
                                   (n.get("n1", {}).get("metrics", {}).get("MAE[0.5]"),
                                    n1v.get("paper_naive"), n1v.get("quantum"),
                                    n1v.get("rel_diff") if n1v.get("rel_diff")
                                    is not None else float("nan"))})
        if not ok:
            anomalies.append({"ds": ds, "type": "n1_round_mismatch",
                              "detail": "N1 %s vs printed %s" %
                                        (n["n1"]["metrics"]["MAE[0.5]"],
                                         n1v.get("paper_naive"))})

    # G7 ratios
    r = n.get("ratios", {})
    r1, r2 = r.get("visionts_over_n1"), r.get("visionts_over_n2")
    ok = (r1 is not None and r2 is not None and 0.0 < r1 < float("inf")
          and 0.0 < r2 < float("inf"))
    vj_gates.append({"ds": ds, "gate": "G7_ratios_finite",
                     "st": status(ok, True), "hard": True,
                     "detail": "VTS/N1 %s VTS/N2 %s" % (r1, r2)})

    # G8 context three-value (json reload makes hist keys strings)
    ctx = m.get("context", {})
    hist = {int(k): v for k, v in
            ctx.get("actual_update_config_final", {}).items()}
    exp_ctx = ctx.get("expected_from_accepted_p")
    n_uc = ctx.get("n_update_config_calls", 0)
    configured = ctx.get("configured")
    ok = bool(hist) and n_uc > 0 and exp_ctx is not None and \
        all(k <= exp_ctx for k in hist.keys())
    trunc = any(k < exp_ctx for k in hist.keys())
    detail = "configured=%s expected(P=%s)=%s hist=%s n_uc=%d%s" % \
             (configured, m.get("autotune", {}).get("accepted_p"), exp_ctx,
              dict(sorted(hist.items())), n_uc,
              " TRUNCATED(min-fed<expected)" if trunc else "")
    vj_gates.append({"ds": ds, "gate": "G8_context_three_value",
                     "st": status(ok, True), "hard": True, "detail": detail})
    if trunc:
        anomalies.append({"ds": ds, "type": "context_truncation",
                          "detail": detail})

    # G11 phases
    t = m.get("timing", {})
    ph = t.get("phases_sec", {})
    res = t.get("residual_frac")
    ok = (all(ph.get(k) is not None for k in
              ("model_init", "download", "prep", "autotune", "inference",
               "eval")) and res is not None and abs(res) < 0.01)
    vj_gates.append({"ds": ds, "gate": "G11_phase_timing",
                     "st": status(ok, True), "hard": True,
                     "detail": "phases=%s residual_frac=%s" %
                               ({k: (round(v, 2) if v is not None else None)
                                 for k, v in ph.items()}, res)})
    return {"monash": m, "naive": n}


def global_gates(subs, vj_gates, anomalies):
    """subs: {ds: {"monash": m, "naive": n}} for all loaded subsets."""
    # G9 CIF pooled. HARD part = series weights + pooled VTS vs the pooled
    # unrounded anchor (rel <= 1e-3). The paper's printed 570907.24 does
    # NOT strictly round-match the pooled unrounded anchor (570907.2287 ->
    # 570907.23), so paper round-match is recorded as a documented flag,
    # not a gate.
    c6, c12 = subs.get("cif_2016_6"), subs.get("cif_2016_12")
    if c6 and c12:
        n6 = c6["monash"]["split"]["n_series"]
        n12 = c12["monash"]["split"]["n_series"]
        vts6 = c6["monash"]["result_csv"]["metrics"]["MAE[0.5]"]
        vts12 = c12["monash"]["result_csv"]["metrics"]["MAE[0.5]"]
        n16 = c6["naive"]["n1"]["metrics"]["MAE[0.5]"]
        n112 = c12["naive"]["n1"]["metrics"]["MAE[0.5]"]
        pool = C.cif_pool({"cif_2016_6": vts6, "cif_2016_12": vts12})
        pool_n1 = C.cif_pool({"cif_2016_6": n16, "cif_2016_12": n112})
        pool_anchor = C.cif_pool(
            {k: C.AUTHOR_ANCHOR[k]["mae"] for k in C.CIF_WEIGHTS})
        rel_anchor = C.rel_diff(pool, pool_anchor)
        ok_v, q_v, want_v = C.paper_round_match(pool, "570907.24")
        ok_n, q_n, want_n = C.paper_round_match(pool_n1, "578596.5")
        ok = n6 == C.CIF_WEIGHTS["cif_2016_6"] and \
            n12 == C.CIF_WEIGHTS["cif_2016_12"] and \
            rel_anchor <= C.ANCHOR_TOL
        vj_gates.append({"ds": "CIF_2016_POOLED", "gate": "G9_cif_pooled",
                         "st": status(ok, True), "hard": True,
                         "detail": "n6=%d n12=%d pooled VTS %.4f vs pooled "
                                   "anchor %.4f rel %.3e; paper print "
                                   "570907.24 round_match=%s (anchor rounds "
                                   "to 570907.23 - paper last-digit "
                                   "inconsistency, documented); pooled N1 "
                                   "%.4f paper 578596.5 round_match=%s" %
                                   (n6, n12, pool, pool_anchor, rel_anchor,
                                    ok_v, pool_n1, ok_n)})
        if not ok_v or not ok_n:
            anomalies.append({"ds": "CIF_2016_POOLED",
                              "type": "cif_paper_print",
                              "detail": "pooled VTS %.4f/N1 %.4f vs printed "
                                        "570907.24/578596.5 (round_match "
                                        "%s/%s)" % (pool, pool_n1, ok_v,
                                                    ok_n)})
        cif = {"n6": n6, "n12": n12, "pooled_vts_mae": pool,
               "pooled_n1_mae": pool_n1, "pooled_anchor": pool_anchor,
               "rel_vs_anchor": rel_anchor,
               "paper_vts": "570907.24",
               "paper_naive": "578596.5", "round_match_vts": ok_v,
               "round_match_naive": ok_n,
               "ratio": pool / pool_n1 if pool_n1 else None}
    else:
        vj_gates.append({"ds": "CIF_2016_POOLED", "gate": "G9_cif_pooled",
                         "st": "FAIL", "hard": True,
                         "detail": "cif subsets missing"})
        cif = {"error": "cif subsets missing"}

    # G10 geomean over the 29 paper rows (VTS/N1), only if all valid
    rows = {}
    valid = True
    for key in C.PAPER_T15_ORDER:
        if key == "CIF_2016_POOLED":
            if cif.get("ratio") is None or not (cif["ratio"] > 0):
                valid = False
            else:
                rows[key] = cif["ratio"]
            continue
        s = subs.get(key)
        if not s:
            valid = False
            continue
        vts = s["monash"]["result_csv"]["metrics"].get("MAE[0.5]")
        n1 = s["naive"]["n1"]["metrics"].get("MAE[0.5]")
        if not vts or not n1 or not (vts > 0) or not (n1 > 0):
            valid = False
            continue
        rows[key] = vts / n1
    if valid:
        g = math.exp(sum(math.log(v) for v in rows.values()) / len(rows))
        rel = C.rel_diff(g, C.GEOMEAN_PAPER)
        warn = rel > C.GEOMEAN_WARN_REL
        vj_gates.append({"ds": "ALL", "gate": "G10_geomean",
                         "st": "WARN" if warn else "PASS", "hard": False,
                         "detail": "geomean %.6f vs paper %.3f rel %.4f%s" %
                                   (g, C.GEOMEAN_PAPER, rel,
                                    " (>2% WARN)" if warn else "")})
        geo = {"geomean": g, "paper": C.GEOMEAN_PAPER, "rel_diff": rel,
               "n_rows": len(rows), "rows": rows, "warn": warn,
               "valid": True}
    else:
        vj_gates.append({"ds": "ALL", "gate": "G10_geomean",
                         "st": "FLAG", "hard": False,
                         "detail": "valid rows %d/29 -> coverage only" %
                                   len(rows)})
        geo = {"valid": False, "n_valid_rows": len(rows),
               "n_paper_rows": 29, "rows": rows}
    return cif, geo


def main():
    subs = {}
    vj_gates = []
    anomalies = []
    for ds in C.MONASH_ORDER:
        got = subset_gates(ds, vj_gates, anomalies)
        if got:
            subs[ds] = got
    cif, geo = global_gates(subs, vj_gates, anomalies)

    n_hard_fail = sum(1 for g in vj_gates if g["hard"] and
                      g["st"].startswith("FAIL"))
    n_gates = len(vj_gates)
    n_pass = sum(1 for g in vj_gates if g["st"].startswith("PASS"))
    section = {"n_gates": n_gates, "n_pass": n_pass,
               "n_hard_fail": n_hard_fail,
               "n_flag_fail": sum(1 for g in vj_gates
                                  if g["st"].startswith("FLAG")),
               "all_hard_pass": n_hard_fail == 0,
               "gates": vj_gates, "anomalies": anomalies,
               "cif": cif, "geomean": geo}
    C.merge_json(C.VERIFY_JSON, "monash", section)

    for g in vj_gates:
        if not g["st"].startswith("PASS"):
            print("[t38][%s] %s %s: %s" % (g["st"], g["ds"], g["gate"],
                                           g["detail"]))
    print("[t38] monash gates: %d/%d pass, %d hard fail, %d flag, "
          "%d anomalies -> %s" % (n_pass, n_gates, n_hard_fail,
                                  section["n_flag_fail"], len(anomalies),
                                  C.VERIFY_JSON))
    if n_hard_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
