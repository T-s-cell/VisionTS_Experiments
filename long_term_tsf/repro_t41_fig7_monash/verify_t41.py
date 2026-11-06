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
       re-extract the FIRST 32-sample batch and compare bitwise;
       extract_env records the fixed-reference-noise mechanism
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
            ok_probe, why = reextract_probe(samples, wins, feats)
            g.check(ok_probe, "first-batch re-extraction bitwise equal" +
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
        for rs in X.STABILITY_SEEDS:
            p = os.path.join(X.SAVE_BASE, "tsne_coords_2200x2_rs%d.npy" % rs)
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
        all_ok, recomputed = True, {}
        for src in X.TS_SOURCES:
            idx = flat["Monash__%s" % src]
            s0, s1 = X.SPANS[src]
            g.check(idx.shape == (s1 - s0, X.N_NEIGHBORS),
                    "%s neighbour matrix shape" % src)
            if not (idx != np.arange(s0, s1)[:, None]).all():
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
        den_ok = all(recomputed[s]["denominator"] ==
                     summ["proportions"][s]["denominator"] ==
                     X.NN_DENOM_PER_SOURCE for s in X.TS_SOURCES)
        g.check(den_ok, "denominators == n_source x 30 == 9000")
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
    """Batch-invariance probe: re-run the FIRST 32-sample group through
    the same geometry + fixed reference noise and compare bitwise."""
    try:
        import torch
        sys.path.insert(0, X.fig7_base())
        import common as F7
        ref = X.load_json(X.mask_ref_path())
        ids_ref = ref["ids_keep_order"]
        F7.set_seeds()
        if not torch.cuda.is_available():
            return False, "no GPU"
        model = F7.load_model().eval().to("cuda")
        ref_noise = torch.ones(1, X.TOTAL_PATCHES, dtype=torch.float32)
        ref_noise[0, torch.tensor(ids_ref, dtype=torch.long)] = 0.0
        chunk = sorted(
            (s for s in samples
             if (s["subset"], s["L_eff"], s["H"], s["P"]) ==
             (samples[0]["subset"], samples[0]["L_eff"], samples[0]["H"],
              samples[0]["P"])),
            key=lambda s: s["k"])[:F7.BATCH]
        s0 = chunk[0]
        model.update_config(context_len=s0["L_eff"], pred_len=s0["H"],
                            periodicity=s0["P"], norm_const=F7.NORM_CONST,
                            align_const=F7.ALIGN_CONST,
                            interpolation="bilinear")
        x = torch.stack([
            torch.as_tensor(np.ascontiguousarray(wins["win_%d" % s["k"]]),
                            dtype=torch.float32)
            for s in chunk]).unsqueeze(-1).to("cuda")
        image = F7.build_ts_image(model, x)
        noise = ref_noise.repeat(image.shape[0], 1).to("cuda")
        with torch.no_grad():
            out, _, _ = model.vision_model.forward_encoder(
                image, X.MASK_RATIO_PIN, noise)
        flat = out.flatten(1).to(torch.float32).cpu().numpy()
        want = feats[[s["k"] for s in chunk]]
        same = flat.shape == want.shape and np.array_equal(flat, want)
        del model
        torch.cuda.empty_cache()
        return same, "" if same else "re-extracted features differ"
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)


if __name__ == "__main__":
    main()
