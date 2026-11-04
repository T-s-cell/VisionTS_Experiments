"""EXP-038 task A runner: one Monash subset through the OFFICIAL
eval_gluonts/run.py evaluate(), in-process, with observation-only patches
that archive what the official code actually did. Author-pinned env
(python 3.8 compatible; no walrus, no 3.9+ syntax).

Run: python t38_monash_runner.py --dataset <name> [--gpu 0]
(cwd-independent; chdirs to eval_gluonts, exactly like run_monash.sh)

Evidence per subset (logs/monash_<ds>.json):
  - probe: TSF subsets -> raw zip bytes+sha256, meta; M4 subsets ->
    provenance (+ optional local-mirror read_csv redirect, bytes+sha256
    recorded per file; same URLs, synced 2026-11-04 after GitHub direct
    access stalled from eta),
    forecast_horizon presence/value, converted offset name, raw series
    count). M4 subsets (GitHub M4-methods, no zenodo zip) -> recipe
    provenance + cache diff instead.
  - pred_len rule re-derivation: cif meta-forced / hard override /
    raw forecast_horizon / official frequency fallback (with the pandas
    >=2.2 alias shim recorded when it fires). Pinned subsets (fred_md,
    nn5_daily, cif_2016_6, cif_2016_12) hard-exit on mismatch; every other
    subset records the check as evidence (user correction #1: all 30 code
    subsets get an independent H check vs the actual test windows).
  - phase timing: model_init (t0 -> download entry; covers VisionTS() +
    its .to(device) H2D copy) / download / prep / autotune (validation) /
    inference (final forecast) / eval; residual must stay <1% of wall
    (user: download, validation and inference timed separately).
  - context three-value check: configured (1000/300), rounded
    (convert_context_len), actually fed to the model (every
    VisionTS.update_config call tagged with its forecast-call index).
  - final forecast matrix -> save_t38_monash/forecast_monash_<ds>.npz
    (NOT part of the audit pack) + per-series MAE array in the json for
    over-tolerance troubleshooting.
Replay subsets (fred_md, nn5_daily) additionally hard-gate: autotuned P,
author anchor rel <= 1e-3, exit 2 otherwise.
"""
import argparse
import io
import os
import sys
import time

import numpy as np

import t38_common as C


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams
        self._buf = []

    def write(self, s):
        for st in self._streams:
            st.write(s)
        self._buf.append(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()

    def text(self):
        return "".join(self._buf)


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def snapshot_dir(root):
    out = {}
    if not os.path.isdir(root):
        return out
    for dirpath, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            try:
                out[os.path.relpath(p, root)] = [os.path.getsize(p), C.sha256_of(p)]
            except OSError:
                out[os.path.relpath(p, root)] = [None, None]
    return out


def parse_autotune(text):
    events, accepted = [], None
    for line in text.splitlines():
        if "autotune: P =" in line:
            body = line.split("autotune: P =", 1)[1]
            p_str, rest = body.split("|", 1)
            p = int(p_str.strip())
            mae = float(rest.split("valid mae =")[1].split(",")[0])
            ok = "accept!" in line
            if ok:
                accepted = p
            events.append({"p": p, "valid_mae": mae, "accept": ok})
    return accepted, events


def rule_pred_len(ds, gen):
    """Independent re-derivation of pred_len from raw metadata / official
    rules (user correction #1)."""
    if ds.startswith("cif_2016_"):
        return int(ds[len("cif_2016_"):]), "cif_meta_forced"
    if ds in C.HARD_OVERRIDE:
        return C.HARD_OVERRIDE[ds], "hard_override"
    if gen.get("has_forecast_horizon"):
        return int(gen["forecast_horizon"]), "raw_tsf"
    if ds in C.M4_SUBSETS and gen.get("m4_recipe_prediction_length"):
        return int(gen["m4_recipe_prediction_length"]), "m4_recipe_constant"
    name = gen.get("freq_offset_name")
    if name in C.FALLBACK_MAP:
        return C.FALLBACK_MAP[name], "freq_fallback"
    if name in C.FALLBACK_SHIM:
        return C.FALLBACK_SHIM[name], "freq_fallback_pandas_shim"
    return None, "unknown(official rules exhausted)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    ds = args.dataset
    assert ds in C.MONASH_ORDER, "unknown subset %s" % ds
    is_m4 = ds in C.M4_SUBSETS

    os.makedirs(C.LOG_DIR, exist_ok=True)
    os.makedirs(C.SAVE_A_BASE, exist_ok=True)
    os.chdir(C.EVAL_GLUONTS)

    rec = {"dataset": ds, "started": C.time_str(), "gpu": args.gpu,
           "is_m4": is_m4}

    import torch
    import gluonts
    import pandas
    rec["env"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                  "gluonts": gluonts.__version__, "pandas": pandas.__version__,
                  "numpy": np.__version__,
                  "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "")}

    if C.EVAL_GLUONTS not in sys.path:
        sys.path.insert(0, C.EVAL_GLUONTS)
    import run
    import dataset as eval_dataset_mod
    assert os.path.samefile(run.__file__,
                            os.path.join(C.EVAL_GLUONTS, "run.py")), run.__file__
    assert os.path.samefile(eval_dataset_mod.__file__,
                            os.path.join(C.EVAL_GLUONTS, "dataset.py")), \
        eval_dataset_mod.__file__
    import gluonts.dataset.repository._tsf_datasets as tsf_mod
    import gluonts.dataset.repository.datasets as gds
    _gen = {"fired": False, "fallback_compat": None}

    # pandas>=2.2 alias shim (inert on the author env); applies ONLY the
    # renamed keys with the OFFICIAL values (ME->12, YE->4).
    _orig_fallback = eval_dataset_mod.default_prediction_length_from_frequency

    def fallback_compat(freq):
        try:
            return _orig_fallback(freq)
        except ValueError:
            from pandas.tseries.frequencies import to_offset
            name = str(to_offset(freq).name)
            if name not in C.FALLBACK_SHIM:
                raise
            _gen["fallback_compat"] = {"requested_freq": str(freq),
                                       "offset_name": name,
                                       "value": C.FALLBACK_SHIM[name]}
            return C.FALLBACK_SHIM[name]

    eval_dataset_mod.default_prediction_length_from_frequency = fallback_compat
    tsf_mod.default_prediction_length_from_frequency = fallback_compat
    orig_recipe = gds.dataset_recipes[ds]
    local_gen = eval_dataset_mod.generate_forecasting_dataset

    def probe_recipe(*args_, **kwargs_):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from zipfile import ZipFile
        from gluonts.dataset.repository._tsf_reader import frequency_converter
        from pandas.tseries.frequencies import to_offset
        if is_m4:
            # GitHub M4-methods provenance; the gluonts _m4 generator owns
            # the download (no zenodo zip to fingerprint)
            import gluonts.dataset.repository._m4 as m4mod
            _gen.update({
                "source_module": m4mod.__file__,
                "source_gluonts_version": gluonts.__version__,
                "recipe_provenance": "gluonts _m4 generator (GitHub "
                                     "M4-methods download)",
                "raw_frequency": "M4 native (%s)" % ds,
                "has_forecast_horizon": None,
                "forecast_horizon": None,
                "m4_recipe_prediction_length":
                    orig_recipe.keywords.get("prediction_length"),
            })
            try:
                attr_hits = {}
                for name in dir(m4mod):
                    if "pred" in name.lower() or "length" in name.lower():
                        v = getattr(m4mod, name)
                        if isinstance(v, dict):
                            attr_hits[name] = {str(k): int(x) for k, x in
                                               v.items()
                                               if isinstance(x, (int, np.integer))}
                if attr_hits:
                    _gen["m4_length_maps"] = attr_hits
            except Exception as e:  # best effort only
                _gen["m4_length_maps_error"] = str(e)
            _gen["fired"] = True
            # local mirror: same-URL files synced from a network mirror;
            # redirect the generator's pd.read_csv and record bytes+sha256
            mir = C.M4_MIRROR_DIR
            if os.path.isdir(mir):
                import pandas as _pd
                orig_read_csv = _pd.read_csv
                urlpfx = ("https://github.com/M4Competition/M4-methods/"
                          "raw/master/Dataset/")
                hits = []

                def read_csv_mirror(path, *a, **k):
                    if isinstance(path, str) and path.startswith(urlpfx):
                        lp = os.path.join(mir,
                                          path[len(urlpfx):])
                        if os.path.isfile(lp):
                            hits.append({"url": path, "local": lp,
                                         "bytes": os.path.getsize(lp),
                                         "sha256": C.sha256_of(lp)})
                            return orig_read_csv(lp, *a, **k)
                    return orig_read_csv(path, *a, **k)

                _pd.read_csv = read_csv_mirror
                try:
                    return orig_recipe(*args_, **kwargs_)
                finally:
                    _pd.read_csv = orig_read_csv
                    if hits:
                        _gen["m4_local_mirror"] = {"dir": mir,
                                                   "n_files": len(hits),
                                                   "files": hits}
            return orig_recipe(*args_, **kwargs_)
        from gluonts.dataset.repository._tsf_datasets import TSFReader
        entry = tsf_mod.datasets[ds]
        with TemporaryDirectory() as td:
            zpath = entry.download(Path(td))
            with ZipFile(zpath) as archive:
                archive.extractall(path=td)
                meta, raw = TSFReader(
                    os.path.join(td, archive.namelist()[0])).read()
            try:
                offset_name = str(to_offset(
                    frequency_converter(str(meta.frequency))).name)
            except Exception as e:
                offset_name = "unresolvable(%s)" % e
            _gen.update({
                "source_file": getattr(entry, "file_name", None),
                "source_root": str(getattr(entry, "ROOT", None)),
                "source_record": str(getattr(entry, "record", None)),
                "downloaded_bytes": os.path.getsize(zpath),
                "download_sha256": C.sha256_of(zpath),
                "raw_frequency": str(getattr(meta, "frequency", None)),
                "freq_offset_name": offset_name,
                "has_forecast_horizon": hasattr(meta, "forecast_horizon"),
                "forecast_horizon": int(getattr(meta, "forecast_horizon", -1)),
                "n_series_raw": len(raw),
            })
        _gen["fired"] = True
        return orig_recipe(*args_, **kwargs_)

    gds.dataset_recipes[ds] = probe_recipe
    _gen["recipe_orig_module"] = orig_recipe.func.__module__
    _gen["recipe_orig_qualname"] = orig_recipe.func.__qualname__
    _gen["recipe_orig_is_eval_local_gen"] = bool(orig_recipe.func is local_gen)

    # ---- wrap test-split getter --------------------------------------
    _split = {}
    _labels = {"targets": None}   # kept in memory only, never dumped
    orig_get = run.get_gluonts_test_dataset
    phase = {"model_init": None, "download": None, "prep": None,
             "autotune": None, "inference": None, "eval": None}
    ph_t = {}

    want = C.PINNED_PRED.get(ds)

    def get_wrapper(name, prediction_length=None, regenerate=False):
        ph_t["download0"] = time.time()
        test_data, metadata = orig_get(name, prediction_length=prediction_length,
                                       regenerate=True)
        ph_t["download1"] = time.time()
        inputs = list(test_data.input)
        labels = list(test_data.label)
        _split.update({
            "meta_prediction_length": int(metadata.prediction_length),
            "meta_freq": str(metadata.freq),
            "n_series": len(inputs),
            "input_len_min": int(min(len(x["target"]) for x in inputs)),
            "input_len_median": float(np.median([len(x["target"]) for x in inputs])),
            "input_len_max": int(max(len(x["target"]) for x in inputs)),
            "label_len_unique": sorted(set(int(len(x["target"])) for x in labels)),
            "input_nan_total": int(sum(int(np.isnan(np.asarray(x["target"],
                                                            dtype=float)).sum())
                                       for x in inputs)),
            "starts_first3": [str(x["start"]) for x in inputs[:3]],
            "item_ids_first3": [str(x.get("item_id", "")) for x in inputs[:3]],
        })
        _labels["targets"] = [np.asarray(x["target"], dtype=np.float64)
                              for x in labels]
        return test_data, metadata

    run.get_gluonts_test_dataset = get_wrapper

    # ---- phase-timing + context + final-matrix capture ----------------
    uc_log = []              # every VisionTS.update_config call
    fc_log = []              # every run.forecast call
    fc_counter = {"n": 0}
    final_capture = {"matrix": None, "test_lens": None}

    _orig_vts = run.VisionTS

    class VTSRecorder(_orig_vts):
        def __init__(self, *a, **k):
            t0 = time.time()
            _orig_vts.__init__(self, *a, **k)
            phase["model_init"] = time.time() - t0
            orig_uc = self.update_config

            def uc_rec(context_len, pred_len, periodicity):
                orig_uc(context_len, pred_len, periodicity)
                uc_log.append({"call": fc_counter["n"],
                               "context_len": int(context_len),
                               "pred_len": int(pred_len),
                               "periodicity": int(periodicity)})
            self.update_config = uc_rec

    run.VisionTS = VTSRecorder

    _orig_forecast = run.forecast

    def forecast_rec(model, train_list, test_list, batch_size, device,
                     periodicity):
        fc_counter["n"] += 1
        idx = fc_counter["n"]
        ph_t["fc%d" % idx] = time.time()
        out = _orig_forecast(model, train_list, test_list, batch_size,
                             device, periodicity)
        entry = {"call": idx, "wall_sec": time.time() - ph_t["fc%d" % idx],
                 "n_series": len(train_list),
                 "test_len_unique": sorted(set(int(len(t)) for t in test_list))[:8]}
        fc_log.append(entry)
        final_capture["matrix"] = np.asarray(out, dtype=np.float32)
        final_capture["test_lens"] = [int(len(t)) for t in test_list]
        return out

    run.forecast = forecast_rec

    _orig_evalf = run.evaluate_forecasts
    eval_state = {}

    def evalf_rec(*a, **k):
        ph_t["eval0"] = time.time()
        out = _orig_evalf(*a, **k)
        eval_state["wall_sec"] = time.time() - ph_t["eval0"]
        eval_state["n_forecasts"] = int(len(a[0]) if a else len(k.get("forecast", [])))
        return out

    run.evaluate_forecasts = evalf_rec

    # ---- run the official evaluate() -----------------------------------
    if os.path.isfile(C.vision_csv(ds)):   # run.evaluate appends to CSVs
        os.remove(C.vision_csv(ds))
    cache = C.gluonts_cache()
    before = snapshot_dir(cache)
    t0 = time.time()

    tee = Tee(sys.__stdout__)
    sys.stdout = tee
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda:%d" % args.gpu)
            torch.cuda.reset_peak_memory_stats(args.gpu)
        ph_t["t0"] = t0
        # official entry wraps evaluate() in torch.no_grad() (run.py:214);
        # in-process replay must do the same or activation graphs of every
        # forecast batch are retained (m4_monthly 94 batches -> 42GB OOM)
        with torch.no_grad():
            run.evaluate(
                ds,
                C.vision_csv(ds),
                context_len=C.MONASH_CONTEXT,
                no_periodicity_context_len=C.MONASH_NO_P_CONTEXT,
                device="cuda:%d" % args.gpu,
                checkpoint_dir="../ckpt/",
                batch_size=C.MONASH_BATCH,
                periodicity="autotune",
            )
    except Exception as exc:
        sys.stdout = tee._streams[0]
        rec["generation"] = dict(_gen)
        if _split:
            rec["split"] = dict(_split)
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        rec["finished"] = C.time_str()
        C.dump_json(C.monash_json(ds), rec)
        raise
    finally:
        sys.stdout = tee._streams[0]

    wall = time.time() - t0
    after = snapshot_dir(cache)
    rec["generation"] = dict(_gen)
    rec["split"] = dict(_split)

    # ---- assemble phase timing ------------------------------------------
    n_calls = fc_counter["n"]
    # model_init = everything before the dataset fetch (official line
    # `VisionTS(...).to(device)` — the H2D copy sits outside the patched
    # constructor, so bucket from t0 to download entry instead)
    if "download0" in ph_t:
        phase["model_init"] = ph_t["download0"] - ph_t["t0"]
    ph_t["prep0"] = ph_t.get("download1", ph_t.get("t0", wall))
    phase["download"] = (ph_t.get("download1", t0) - ph_t.get("download0", t0)) \
        if "download0" in ph_t else None
    first_fc = ph_t.get("fc1")
    phase["prep"] = (first_fc - ph_t["prep0"]) if first_fc else None
    if n_calls >= 2:
        # autotune window = fc1 start -> final-call start (each candidate's
        # forecast wall is INSIDE this interval, so it needs no subtraction)
        phase["autotune"] = ph_t["fc%d" % n_calls] - ph_t["fc1"]
        phase["inference"] = fc_log[-1]["wall_sec"]
    elif n_calls == 1:
        phase["autotune"] = None
        phase["inference"] = fc_log[0]["wall_sec"]
    phase["eval"] = eval_state.get("wall_sec")
    booked = sum(v for v in (phase["model_init"], phase["download"],
                             phase["prep"], phase["autotune"],
                             phase["inference"], phase["eval"]) if v)
    residual = wall - booked
    rec["timing"] = {"wall_sec": wall, "phases_sec": phase,
                     "residual_sec": residual,
                     "residual_frac": residual / wall if wall else None,
                     "n_forecast_calls": n_calls,
                     "forecast_calls": fc_log,
                     "eval_n_forecasts": eval_state.get("n_forecasts")}
    if torch.cuda.is_available():
        rec["gpu_peak"] = {"allocated_mb": torch.cuda.max_memory_allocated(args.gpu) / 1024 ** 2,
                           "reserved_mb": torch.cuda.max_memory_reserved(args.gpu) / 1024 ** 2}

    # ---- autotune parse (accepted P feeds the expected context) ----------
    accepted, events = parse_autotune(tee.text())
    rec["autotune"] = {"accepted_p": accepted, "events": events}
    rec["p_expected"] = C.EXPECTED_P[ds]
    rec["p_flag"] = (accepted == C.EXPECTED_P[ds])

    # ---- context three-value + final-matrix evidence ---------------------
    last_call = n_calls
    ctx_final = [e["context_len"] for e in uc_log if e["call"] == last_call]
    hist = {}
    for v in ctx_final:
        hist[v] = hist.get(v, 0) + 1
    rec["context"] = {
        "configured": C.MONASH_CONTEXT,
        "no_periodicity_configured": C.MONASH_NO_P_CONTEXT,
        "expected_from_accepted_p": expected_context(accepted),
        "actual_update_config_final": hist,
        "n_update_config_calls": len(uc_log),
        "final_call_idx": last_call,
    }
    if final_capture["matrix"] is not None:
        np.savez_compressed(C.forecast_npz(ds),
                            forecast=final_capture["matrix"])
        rec["final_matrix"] = {"npz": C.forecast_npz(ds),
                               "shape": list(final_capture["matrix"].shape)}

    # ---- checks ----------------------------------------------------------
    pred_len_got = _split.get("meta_prediction_length")
    rule_val, rule_src = rule_pred_len(ds, dict(_gen))
    rec["pred_rule"] = {"expected": rule_val, "source": rule_src}
    checks = {}
    checks["label_windows_match_pred"] = \
        _split.get("label_len_unique") == [pred_len_got]
    checks["rule_matches_meta"] = (rule_val == pred_len_got)
    if want is not None:
        checks["pinned_pred_match"] = (pred_len_got == want and
                                       rule_val == want)
    rec["checks"] = checks
    if want is not None and not checks.get("pinned_pred_match"):
        rec["aborted"] = "pinned_pred_mismatch"
        rec["finished"] = C.time_str()
        C.dump_json(C.monash_json(ds), rec)
        raise SystemExit("[t38] %s pinned pred_len mismatch: rule=%s meta=%s "
                         "want=%d" % (ds, rule_val, pred_len_got, want))

    # ---- anchor evidence ---------------------------------------------------
    row, ds_in_csv = C.parse_csv_metrics(C.vision_csv(ds))
    rec["result_csv"] = {"path": C.vision_csv(ds), "dataset_field": ds_in_csv,
                         "metrics": row}
    anchor = C.AUTHOR_ANCHOR[ds]["mae"]
    mae = row["MAE[0.5]"]
    rd = C.rel_diff(mae, anchor)
    rec["anchor"] = {"mae_got": mae, "mae_anchor": anchor,
                     "mse_got": row.get("MSE[mean]"),
                     "mse_anchor": C.AUTHOR_ANCHOR[ds]["mse"],
                     "rel_diff": rd, "tol": C.ANCHOR_TOL,
                     "pass": bool(rd <= C.ANCHOR_TOL)}
    if ds in C.REPLAY:
        rec["replay_hard"] = {
            "p_pass": accepted == C.EXPECTED_P[ds],
            "anchor_pass": bool(rd <= C.ANCHOR_TOL),
            "pass": bool(accepted == C.EXPECTED_P[ds] and rd <= C.ANCHOR_TOL)}
    # per-series MAE from the captured final matrix vs the label windows
    # (same pairing order the gluonts evaluator uses); the mean must agree
    # with the evaluator's MAE[0.5] up to float association noise.
    if final_capture["matrix"] is not None and _labels["targets"]:
        mat = final_capture["matrix"]
        truths = _labels["targets"]
        if mat.shape[0] == len(truths):
            per_mae = np.array([float(np.mean(np.abs(mat[i] - t)))
                                for i, t in enumerate(truths)])
            rec["per_series_mae"] = {
                "n": int(len(per_mae)),
                "mean_of_series_mae": float(per_mae.mean()),
                "evaluator_mae": mae,
                "abs_diff": abs(float(per_mae.mean()) - mae),
                "max": float(per_mae.max()), "argmax": int(np.argmax(per_mae)),
                "array": [float(v) for v in per_mae]}
        else:
            rec["per_series_mae"] = {"error": "shape mismatch mat=%s n_truth=%d"
                                     % (mat.shape, len(truths))}

    new_files = {k: v for k, v in after.items() if before.get(k) != v}
    rec["data_version"] = {"cache_dir": cache,
                           "n_files_new_or_changed": len(new_files),
                           "files": dict(list(new_files.items())[:200])}
    rec["finished"] = C.time_str()
    C.dump_json(C.monash_json(ds), rec)

    log("%s done: pred %s(%s) P %s(exp %s) MAE %.6f anchor rel %.3e (%.1f s)"
        % (ds, rule_val, rule_src, accepted, C.EXPECTED_P[ds], mae, rd, wall))
    if ds in C.REPLAY and not rec["replay_hard"]["pass"]:
        sys.exit(2)
    log("DONE")


def expected_context(p):
    """run.convert_context_len expectation for the FINAL forecast call
    (identical formula to run.py:88-93): P=1 -> no_periodicity_context_len,
    else int(round(context_len / P)) * P."""
    if p is None:
        return None
    if p == 1:
        return C.MONASH_NO_P_CONTEXT
    return int(round(float(C.MONASH_CONTEXT) / p)) * p


if __name__ == "__main__":
    main()
