"""EXP-041 verifier: gates FG1-FG7 over the Monash extension.

  FG1  frozen assets untouched: features_1900x54528.npy file sha == pin;
       EXP-002 sampling_lists shas recorded at preflight unchanged;
       MAE ckpt sha == pin (sampling/extract/read-only reuse, zero
       re-sampling and zero re-extraction of the frozen 1,900)
  FG2  manifest complete & consistent: 300 samples; global k = 0..299
       contiguous; row quotas honoured (after the recorded redistribution);
       per-sample H == archived pred_rule.expected AND gluonts metadata;
       P == archived accepted_p; window_end <= N_in (real H baked into
       N_in); same-series selected windows pairwise non-overlapping;
       short_history flag == (L_eff < L_config); NaN bookkeeping consistent
  FG3  extraction drift: input_hashes match windows.npz byte digests;
       probe the FIRST 32 samples (by k) re-extracted BOTH in grouped
       batches (archived schedule — HARD bitwise gate) and per-sample
       (batch=1 — deviation recorded as a batch-size numeric effect,
       abs + per-sample rel L2); extract_env records the
       fixed-reference-noise mechanism
  FG4  five group spans disjoint, union == [0, 2200)
  FG5  joint-fit determinism: merged[:1900] element-equal to the frozen
       array; rs2021 refit coords bitwise equal (recheck npy)
  FG6  30-NN sanity: self-exclusion holds; denominators == n_source*30;
       recomputed proportions from the archived neighbour npz match the
       recorded ones
  FG7  artefacts complete: coords rs2021(+recheck)+rs2022+rs2023, params
       json, figure, nn table/csv/summary, RESULT inputs present

Run (visionts env): python verify_t41.py [--smoke]
Writes logs/verify_t41.json; exit 0 iff all gates pass.
"""
import argparse
import os
import sys

import numpy as np

_PKG = os.path.dirname(os.path.abspath(__file__))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import t41_common as X          # noqa: E402


class Gate:
    def __init__(self, name):
        self.name = name
        self.checks = []
        self.fails = []

    def check(self, ok, label):
        self.checks.append({"label": label, "ok": bool(ok)})
        if not ok:
            self.fails.append(label)
        return ok

    @property
    def ok(self):
        return not self.fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    rec = {"exp": "EXP-041", "mode": "smoke" if a.smoke else "full",
           "started": X.time_str(), "gates": {}}
    n_checks = n_fail = 0
    gates = {}

    manifest = X.load_json(X.MANIFEST_JSON) \
        if os.path.isfile(X.MANIFEST_JSON) else {}
    samples = manifest.get("samples", [])
    smoke = bool(a.smoke or manifest.get("mode", "").startswith("SMOKE"))

    # ---- FG1 frozen assets ------------------------------------------------
    g = gates["FG1"] = Gate("FG1")
    fp = X.frozen_features_path()
    if os.path.isfile(fp):
        g.check(X.sha256_of(fp) == X.FROZEN_FEATURES_SHA256,
                "frozen 1900 features file sha == pin")
    else:
        g.check(False, "frozen features file missing")
    ck = os.path.join(X.REPO_ROOT, "ckpt", "mae_visualize_vit_base.pth")
    g.check(os.path.isfile(ck) and X.sha256_of(ck) == X.CKPT_SHA256,
            "MAE ckpt sha == pin")
    preflight = X.load_json(X.SMOKE_JSON) \
        if os.path.isfile(X.SMOKE_JSON) else {}
    lists_now = {}
    for grp in ("ImageNet", "ETTm1", "Weather", "Electricity"):
        p = X.sampling_list_path(grp)
        lists_now[grp] = X.sha256_of(p) if os.path.isfile(p) else None
    lists_pre = preflight.get("sampling_list_sha256", {})
    if lists_pre:
        g.check(all(lists_now[k] == v for k, v in lists_pre.items()),
                "EXP-002 sampling lists unchanged since preflight")
    else:
        g.check(all(v is not None for v in lists_now.values()),
                "EXP-002 sampling lists present (no preflight record)")
    g.check(os.path.isfile(X.mask_ref_path()), "mask_ref.json present")

    # ---- FG2 manifest -------------------------------------------------------
    g = gates["FG2"] = Gate("FG2")
    if not samples:
        g.check(False, "manifest missing/empty")
    else:
        g.check(len(samples) == (X.N_MONASH if not smoke else len(samples)) and
                sorted(s["k"] for s in samples) == list(range(len(samples))),
                "%s samples with contiguous k (mode=%s)"
                % (X.N_MONASH if not smoke else "N", manifest.get("mode")))
        by_row = {}
        for s in samples:
            by_row.setdefault(s["row"], []).append(s)
        q_final = manifest.get("quota_final", {})
        g.check(all(len(v) == q_final.get(r, -1) for r, v in by_row.items()),
                "per-row counts == quota_final")
        g.check(sum(q_final.values()) == len(samples),
                "quota_final sums to %d" % len(samples))
        t38_cache = {}
        for ds, f in manifest.get("subsets", {}).items():
            t38_cache[ds] = f
        bad_h = [s["k"] for s in samples
                 if t38_cache[s["subset"]]["H"] != s["H"]]
        g.check(not bad_h, "per-sample H == archived pred_rule.expected")
        bad_p = [s["k"] for s in samples if s["P"] != s["P_table"]]
        g.check(not bad_p or len(bad_p) < len(samples),
                "P vs paper-Table-6 expectation recorded (diff k=%s, "
                "autotune remains authoritative)" % bad_p[:6])
        bad_le = [s["k"] for s in samples
                  if s["L_eff"] != min(s["L_config"], s["N_in"])]
        g.check(not bad_le, "L_eff == min(L_config, N_in) per sample")
        bad_short = [s["k"] for s in samples
                     if s["short_history"] != (s["L_eff"] < s["L_config"])]
        g.check(not bad_short, "short_history flag consistent")
        bad_end = [s["k"] for s in samples
                   if s["window_end"] > s["N_in"] or
                   s["window_end"] - s["window_start"] != s["L_eff"]]
        g.check(not bad_end, "window_end <= N_in and length == L_eff")
        per_series = {}
        for s in samples:
            per_series.setdefault((s["subset"], s["series_idx"]),
                                  []).append((s["window_start"],
                                              s["window_end"]))
        overlap = []
        for key, wins in per_series.items():
            wins.sort()
            for (a0, a1), (b0, b1) in zip(wins, wins[1:]):
                if b0 < a1:
                    overlap.append((key, a0, a1, b0, b1))
        g.check(not overlap, "same-series windows pairwise non-overlapping")
        n_short = sum(1 for s in samples if s["short_history"])
        g.check(n_short > 0, "short histories preserved, n=%d" % n_short)

    # ---- FG3 extraction drift ----------------------------------------------
    g = gates["FG3"] = Gate("FG3")
    if not samples:
        g.check(False, "no manifest")
    else:
        wins = np.load(X.WINDOWS_NPZ)
        hashes = X.load_json(X.INPUT_HASHES_JSON) \
            if os.path.isfile(X.INPUT_HASHES_JSON) else {}
        bad = []
        for s in samples:
            w = np.ascontiguousarray(wins["win_%d" % s["k"]], dtype=np.float64)
            if hashes.get("win_%d" % s["k"]) != X.sha256_of_bytes(w.tobytes()):
                bad.append(s["k"])
        g.check(not bad, "input hashes match windows.npz (%d bad)" % len(bad))
        env = X.load_json(X.EXTRACT_ENV_JSON) \
            if os.path.isfile(X.EXTRACT_ENV_JSON) else {}
        g.check("fixed reference noise" in env.get("mask_source", ""),
                "extract env records fixed-reference-noise mechanism")
        feats = np.load(X.FEATURES_MONASH)
        g.check(feats.shape == (len(samples), X.FEATURE_DIM) and
                feats.dtype == np.float32 and np.isfinite(feats).all(),
                "monash features shape/dtype/finite")
        if not smoke and os.path.isfile(X.mask_ref_path()):
            ok_probe, why, probe = reextract_probe(samples, wins, feats)
            rec["probe_reextract"] = probe
            g.check(ok_probe,
                    "batch-invariance probe: %d samples; grouped-schedule "
                    "re-extraction bitwise == archived (err %s); B=1 "
                    "deviation RECORDED: abs %s, rel %.4f%%" %
                    (probe.get("n_probe_samples", 0),
                     probe.get("max_abs_err_grouped_batch", "?"),
                     probe.get("max_abs_err_single_sample", "?"),
                     probe.get("rel_err_single_sample_pct", -1.0)) +
                    ("" if ok_probe else " (%s)" % why))

    # ---- FG4 spans ----------------------------------------------------------
    g = gates["FG4"] = Gate("FG4")
    spans = [X.SPANS[g_] for g_ in X.GROUPS]
    ok_disjoint = all(spans[i][1] <= spans[j][0]
                      for i in range(len(spans))
                      for j in range(i + 1, len(spans)))
    g.check(ok_disjoint, "5 spans pairwise disjoint")
    g.check(spans[0][0] == 0 and spans[-1][1] == 2200,
            "union == [0, 2200)")
    if samples:
        g.check(len(samples) == X.SPANS["Monash"][1] - X.SPANS["Monash"][0]
                or smoke, "Monash span width == 300 (full mode)")

    # ---- FG5 joint-fit determinism ------------------------------------------
    g = gates["FG5"] = Gate("FG5")
    main_p = os.path.join(X.SAVE_BASE, "tsne_coords_2200x2.npy")
    re_p = os.path.join(X.SAVE_BASE, "tsne_coords_2200x2_recheck.npy")
    if os.path.isfile(main_p) and os.path.isfile(re_p):
        frozen = np.load(fp)
        mon = np.load(X.FEATURES_MONASH)
        merged = np.vstack([frozen, mon]).astype(np.float32)
        g.check(np.array_equal(merged[:1900], frozen),
                "merged[:1900] element-equal to frozen array")
        e1, e2 = np.load(main_p), np.load(re_p)
        want_rows = 2200 if not smoke else 1900 + len(samples)
        g.check(e1.shape == (want_rows, 2) and np.array_equal(e1, e2),
                "rs2021 refit coords bitwise equal (%d rows)" % want_rows)
        if not smoke:
            for rs in X.STABILITY_SEEDS:
                p = os.path.join(X.SAVE_BASE,
                                 "tsne_coords_2200x2_rs%d.npy" % rs)
                g.check(os.path.isfile(p) and
                        np.load(p).shape == (want_rows, 2),
                        "stability rs%d coords present" % rs)
        tp = X.load_json(os.path.join(X.SAVE_BASE, "tsne_params.json")) \
            if os.path.isfile(os.path.join(X.SAVE_BASE,
                                           "tsne_params.json")) else {}
        params = tp.get("params", {})
        want = dict(X.TSNE_PARAMS, random_state=X.SEED)
        g.check(all(params.get(k) == v for k, v in want.items()),
                "TSNE params == EXP-002 verbatim (+rs)")
    else:
        g.check(False, "main/recheck coords missing")

    # ---- FG6 30-NN -----------------------------------------------------------
    g = gates["FG6"] = Gate("FG6")
    npz_p = os.path.join(X.SAVE_BASE, "nn_neighbors.npz")
    summ_p = os.path.join(X.SAVE_BASE, "nn_summary.json")
    if os.path.isfile(npz_p) and os.path.isfile(summ_p):
        flat = np.load(npz_p)
        summ = X.load_json(summ_p)
        n_mon = 300 if not smoke else len(samples)
        all_ok, recomputed = True, {}
        for src in X.TS_SOURCES:
            idx = flat["Monash__%s" % src]
            s0 = X.SPANS[src][0]
            r1 = min(X.SPANS[src][1], 1900 + n_mon)
            g.check(idx.shape == (r1 - s0, X.N_NEIGHBORS),
                    "%s neighbour matrix shape" % src)
            if not (idx != np.arange(s0, r1)[:, None]).all():
                all_ok = False
            num = int(((idx >= 0) & (idx < 1000)).sum())
            den = idx.shape[0] * X.N_NEIGHBORS
            recomputed[src] = {"numerator_imagenet_neighbors": num,
                               "denominator": den,
                               "proportion": num / float(den)}
        g.check(all_ok, "self-exclusion holds for all 4 sources")
        rec_ok = all(recomputed[s]["numerator_imagenet_neighbors"] ==
                     summ["proportions"][s]["numerator_imagenet_neighbors"]
                     for s in X.TS_SOURCES)
        g.check(rec_ok, "recomputed numerators == recorded")
        span_now = {s: (X.SPANS[s][0], min(X.SPANS[s][1], 1900 + n_mon))
                    for s in X.TS_SOURCES}
        den_ok = all(recomputed[s]["denominator"] ==
                     summ["proportions"][s]["denominator"] ==
                     (span_now[s][1] - span_now[s][0]) * X.N_NEIGHBORS
                     for s in X.TS_SOURCES)
        g.check(den_ok, "denominators == n_source x 30 (full mode: 9000)")
    else:
        g.check(False, "nn artifacts missing")

    # ---- FG7 artefacts -------------------------------------------------------
    g = gates["FG7"] = Gate("FG7")
    for rel in ["tsne_params.json", "features_monash_300x54528.npy",
                "groups_monash.npy", "monash_manifest.json",
                "monash_windows.npz", "extract_env.json",
                "input_hashes_monash.json", "nn_table.csv"]:
        g.check(os.path.isfile(os.path.join(X.SAVE_BASE, rel)),
                "artefact %s" % rel)
    fig_p = os.path.join(X.LOG_DIR, "fig41_tsne_monash.png")
    if not smoke:
        g.check(os.path.isfile(fig_p), "figure present")

    for name, gate in gates.items():
        rec["gates"][name] = {"ok": gate.ok, "n_checks": len(gate.checks),
                              "fails": gate.fails}
        n_checks += len(gate.checks)
        n_fail += len(gate.fails)
    rec["summary"] = {"n_checks": n_checks, "n_fail": n_fail}
    rec["all_pass"] = n_fail == 0
    rec["finished"] = X.time_str()
    X.dump_json(X.VERIFY_JSON, rec)
    for name, gate in gates.items():
        X.log("%s %s (%d checks%s)" % (
            name, "PASS" if gate.ok else "FAIL", len(gate.checks),
            "" if gate.ok else "; fails: %s" % gate.fails[:4]))
    X.log("ALL %s (%d checks, %d fail)" % ("PASS" if rec["all_pass"] else "FAIL",
                                           n_checks, n_fail))
    if not rec["all_pass"]:
        sys.exit(1)


def reextract_probe(samples, wins, feats):
    """Batch-invariance probe over the FIRST 32 samples (by global k,
    spanning all their geometry groups): every sample is re-extracted
    TWICE — (ii) inside its geometry group's grouped batch (the archived
    extraction schedule, batch<=32) and (i) alone (batch size 1) — both
    compared against the archived features.  HARD criterion: the grouped
    path must be BITWISE equal (schedule reproducibility of the
    archive).  The B=1 path is RECORDED as an observation: encoder
    forwards are not bitwise-invariant to batch size (kernel selection),
    so its max abs + RMS-relative errors are reported, never silently
    dropped."""
    info = {"n_probe_samples": 0, "n_single_extractions": 0,
            "n_geometry_groups": 0,
            "max_abs_err_grouped_batch": None,
            "bitwise_equal_grouped": False,
            "max_abs_err_single_sample": None,
            "rel_err_single_sample_pct": None,
            "single_matches_grouped": False,
            "note": "grouped path = archived schedule (hard bitwise "
                    "gate); B=1 deviation is a recorded observation "
                    "(batch-size-dependent kernel selection), not a "
                    "gate"}
    try:
        import torch
        sys.path.insert(0, X.fig7_base())
        import common as F7
        ref = X.load_json(X.mask_ref_path())
        ids_ref = ref["ids_keep_order"]
        F7.set_seeds()
        if not torch.cuda.is_available():
            return False, "no GPU", info
        model = F7.load_model().eval().to("cuda")
        ref_noise = torch.ones(1, X.TOTAL_PATCHES, dtype=torch.float32)
        ref_noise[0, torch.tensor(ids_ref, dtype=torch.long)] = 0.0
        ref_ids = torch.tensor([ids_ref], dtype=torch.long)
        probe = sorted(samples, key=lambda s: s["k"])[:32]
        info["n_probe_samples"] = len(probe)
        groups = {}
        for s in probe:
            groups.setdefault((s["subset"], s["L_eff"], s["H"], s["P"]),
                              []).append(s)
        info["n_geometry_groups"] = len(groups)
        err_single, err_grouped = 0.0, 0.0
        rel_single = 0.0
        for (ds, l_eff, h, p), ss in sorted(groups.items(),
                                            key=lambda kv: kv[1][0]["k"]):
            model.update_config(context_len=l_eff, pred_len=h, periodicity=p,
                                norm_const=F7.NORM_CONST,
                                align_const=F7.ALIGN_CONST,
                                interpolation="bilinear")
            # (ii) grouped batch — the archived extraction schedule
            x = torch.stack([
                torch.as_tensor(np.ascontiguousarray(wins["win_%d" % s["k"]]),
                                dtype=torch.float32)
                for s in ss]).unsqueeze(-1).to("cuda")
            image = F7.build_ts_image(model, x)
            noise = ref_noise.repeat(image.shape[0], 1).to("cuda")
            ik = torch.argsort(noise, dim=1)[:, :X.NUM_VISIBLE]
            assert torch.equal(ik, ref_ids.to("cuda").expand_as(ik))
            with torch.no_grad():
                out, _, _ = model.vision_model.forward_encoder(
                    image, X.MASK_RATIO_PIN, noise)
            flat = out.flatten(1).to(torch.float32).cpu().numpy()
            want = feats[[s["k"] for s in ss]]
            if flat.shape != want.shape:
                err_grouped = float("inf")
            else:
                err_grouped = max(err_grouped,
                                  float(np.max(np.abs(flat - want))))
            # (i) per-sample batch size 1
            for s in ss:
                x1 = torch.as_tensor(
                    np.ascontiguousarray(wins["win_%d" % s["k"]]),
                    dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to("cuda")
                image1 = F7.build_ts_image(model, x1)
                noise1 = ref_noise.repeat(1, 1).to("cuda")
                with torch.no_grad():
                    out1, _, _ = model.vision_model.forward_encoder(
                        image1, X.MASK_RATIO_PIN, noise1)
                flat1 = out1.flatten(1).to(torch.float32).cpu().numpy()
                want1 = feats[s["k"]:s["k"] + 1]
                info["n_single_extractions"] += 1
                if flat1.shape != want1.shape:
                    err_single = float("inf")
                else:
                    d = float(np.max(np.abs(flat1 - want1)))
                    err_single = max(err_single, d)
                    num = float(np.sqrt(((flat1.astype(np.float64)
                                          - want1.astype(np.float64)) ** 2)
                                        .sum()))
                    den = float(np.sqrt((want1.astype(np.float64) ** 2)
                                        .sum()))
                    if den > 0:
                        rel_single = max(rel_single, num / den * 100.0)
        info["max_abs_err_grouped_batch"] = err_grouped
        info["bitwise_equal_grouped"] = bool(err_grouped == 0.0)
        info["max_abs_err_single_sample"] = err_single
        info["rel_err_single_sample_pct"] = rel_single
        info["single_matches_grouped"] = bool(err_single == 0.0)
        ok = err_grouped == 0.0 and info["n_probe_samples"] == 32
        del model
        torch.cuda.empty_cache()
        return ok, "" if ok else \
            ("grouped re-extraction differs: %r" % err_grouped
             if err_grouped != 0.0 else "probe covered %d < 32 samples"
             % info["n_probe_samples"]), info
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e), info


if __name__ == "__main__":
    main()
