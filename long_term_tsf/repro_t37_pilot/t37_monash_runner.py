"""EXP-037 task A runner: one Monash subset through the OFFICIAL
eval_gluonts/run.py evaluate(), in-process, with observation-only patches
that archive what the official code actually did.

Run with: python t37_monash_runner.py --dataset <name> [--gpu 0]
(cwd-independent; chdirs to eval_gluonts itself, exactly like run_monash.sh)

Patches (all wrapper-only, no upstream edits):
  1. gluonts dataset_recipes[<ds>] replaced with a probe that downloads the
     raw .tsf zip itself (bytes + sha256 archived) and reads the meta via
     TSFReader BEFORE delegating to the original recipe partial: records
     forecast_horizon presence/value, frequency, raw series count. This is
     the runtime proof of WHERE pred_len came from (NN5 Daily must show
     forecast_horizon == 56; 30 would be the frequency fallback and the
     runner aborts the subset BEFORE any model work). Recipe provenance is
     recorded: the built-in fred_md/nn5 recipes bind the ORIGINAL gluonts
     generate_forecasting_dataset at gluonts import time — the rebind of
     _tsf_datasets.generate_forecasting_dataset in eval_gluonts/dataset.py
     only feeds the additional_datasets recipes, so it is inert for these
     two subsets (fact documented via recipe_orig_module in the evidence).
  2. run.get_gluonts_test_dataset wrapped: forces regenerate=True so the
     probe fires and the data-version evidence always corresponds to this
     run, and captures the test split: metadata.prediction_length, freq,
     per-series input/label lengths, NaN counts, start dates. Label lengths
     must all equal pred_len.
  3. stdout Tee'd so the tqdm.write "autotune: P = X | valid mae = Y,
     accept!" lines are parsed in-process as well as logged.

Everything else is the official code path: run.evaluate() with
run_monash.sh flags (context 1000 / no-P context 300 / bs 512 /
periodicity=autotune), frozen MAE-base from ../ckpt, results CSV written
to save_t37_monash/ (the tracked result.csv is never touched).
get_dataset(regenerate=True) forces a fresh zenodo download so the data
version evidence (cache-file sha256 before/after) always corresponds to
this run.
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

import t37_common as C


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
    """-> (accepted_P, [{p, mae, accept}...]) from tqdm.write lines."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    ds = args.dataset
    assert ds in C.SUBSETS, "unknown subset %s" % ds

    os.makedirs(C.LOG_DIR, exist_ok=True)
    os.makedirs(C.SAVE_A_BASE, exist_ok=True)
    os.chdir(C.EVAL_GLUONTS)

    rec = {"dataset": ds, "started": C.time_str(), "gpu": args.gpu}

    import torch
    import gluonts
    rec["env"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                  "gluonts": gluonts.__version__,
                  "numpy": np.__version__,
                  "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "")}

    # ---- wrap generation (raw .tsf meta evidence) --------------------
    # The effective entry point for get_dataset() is dataset_recipes[<ds>]:
    # a partial of the ORIGINAL gluonts generate bound at gluonts import
    # time (eval_gluonts/dataset.py rebinds _tsf_datasets.generate_forecasting_dataset
    # but only the additional_datasets recipes use that local function).
    # We replace the recipe key AFTER `import run` and record provenance.
    # Script-mode sys.path[0] is THIS package dir, not the chdir'd cwd —
    # insert eval_gluonts explicitly and assert identity (a stray site-packages
    # module named run/dataset would silently poison every later gate).
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
    _gen = {"fired": False}
    orig_recipe = gds.dataset_recipes[ds]
    local_gen = eval_dataset_mod.generate_forecasting_dataset

    def probe_recipe(*args, **kwargs):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from zipfile import ZipFile
        from gluonts.dataset.repository._tsf_datasets import TSFReader
        entry = tsf_mod.datasets[ds]
        with TemporaryDirectory() as td:
            zpath = entry.download(Path(td))
            with ZipFile(zpath) as archive:
                archive.extractall(path=td)
                meta, raw = TSFReader(os.path.join(td, archive.namelist()[0])).read()
            _gen.update({
                "source_file": getattr(entry, "file_name", None),
                "source_root": str(getattr(entry, "ROOT", None)),
                "source_record": str(getattr(entry, "record", None)),
                "downloaded_bytes": os.path.getsize(zpath),
                "download_sha256": C.sha256_of(zpath),
                "raw_frequency": str(meta.frequency),
                "has_forecast_horizon": hasattr(meta, "forecast_horizon"),
                "forecast_horizon": int(getattr(meta, "forecast_horizon", -1)),
                "n_series_raw": len(raw),
            })
        _gen["fired"] = True
        return orig_recipe(*args, **kwargs)

    gds.dataset_recipes[ds] = probe_recipe
    _gen["recipe_orig_module"] = orig_recipe.func.__module__
    _gen["recipe_orig_qualname"] = orig_recipe.func.__qualname__
    _gen["recipe_orig_is_eval_local_gen"] = bool(orig_recipe.func is local_gen)

    # ---- wrap test-split getter --------------------------------------
    _split = {}
    orig_get = run.get_gluonts_test_dataset
    want = C.SUBSETS[ds]["pred_len"]

    def pred_len_check():
        got = _split.get("meta_prediction_length")
        if got != want:
            return "FAIL(got %s want %d)" % (got, want)
        if _split.get("label_len_unique") != [want]:
            return "FAIL(label lens %s)" % _split.get("label_len_unique")
        return "OK"

    def get_wrapper(name, prediction_length=None, regenerate=False):
        # regenerate=True unconditionally: run.py passes no regenerate, so a
        # pre-existing cache would skip the recipe and leave the probe and
        # data-version evidence empty.
        test_data, metadata = orig_get(name, prediction_length=prediction_length,
                                       regenerate=True)
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
            "input_nan_total": int(sum(int(np.isnan(np.asarray(x["target"], dtype=float)).sum())
                                       for x in inputs)),
            "starts_first3": [str(x["start"]) for x in inputs[:3]],
            "item_ids_first3": [str(x.get("item_id", "")) for x in inputs[:3]],
        })
        # user correction #1: pause the subset BEFORE any forecast work if
        # pred_len came from the frequency fallback instead of the raw meta
        if pred_len_check() != "OK":
            rec.update({"split": dict(_split), "pred_len_check": pred_len_check(),
                        "generation": dict(_gen), "aborted": "pred_len_check",
                        "finished": C.time_str()})
            C.dump_json(C.monash_json(name), rec)
            raise SystemExit("[t37] %s pred_len gate %s (want %d) — subset paused, "
                             "see %s" % (name, pred_len_check(), want,
                                         C.monash_json(name)))
        return test_data, metadata

    run.get_gluonts_test_dataset = get_wrapper

    # ---- data-version snapshots ---------------------------------------
    if os.path.isfile(C.vision_csv(ds)):   # run.evaluate appends to existing CSVs
        os.remove(C.vision_csv(ds))
    cache = C.gluonts_cache()
    before = snapshot_dir(cache)
    t0 = time.time()

    tee = Tee(sys.__stdout__)
    sys.stdout = tee
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda:%d" % args.gpu)  # lazy-init: torch>=2.6 rejects reset before first alloc
            torch.cuda.reset_peak_memory_stats(args.gpu)
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
    finally:
        sys.stdout = tee._streams[0]

    wall = time.time() - t0
    after = snapshot_dir(cache)

    # ---- collect evidence ---------------------------------------------
    rec["generation"] = dict(_gen)
    rec["split"] = dict(_split)
    accepted, events = parse_autotune(tee.text())
    rec["autotune"] = {"accepted_p": accepted, "events": events}
    rec["timing"] = {"wall_sec": wall}
    if torch.cuda.is_available():
        rec["gpu_peak"] = {"allocated_mb": torch.cuda.max_memory_allocated(args.gpu) / 1024 ** 2,
                           "reserved_mb": torch.cuda.max_memory_reserved(args.gpu) / 1024 ** 2}
    new_files = {k: v for k, v in after.items() if before.get(k) != v}
    rec["data_version"] = {"cache_dir": cache,
                           "n_files_new_or_changed": len(new_files),
                           "files": new_files}

    row, ds_in_csv = C.parse_csv_metrics(C.vision_csv(ds))
    rec["result_csv"] = {"path": C.vision_csv(ds), "dataset_field": ds_in_csv,
                         "metrics": row}
    rec["pred_len_check"] = pred_len_check()
    rec["p_check"] = ("OK" if accepted == C.SUBSETS[ds]["p"]
                      else "FAIL(accepted %s want %s)" % (accepted, C.SUBSETS[ds]["p"]))
    rec["finished"] = C.time_str()
    C.dump_json(C.monash_json(ds), rec)

    log("%s done: pred_len %d, P %s, MAE %.6f (%.1f s)%s"
        % (ds, rec["split"]["meta_prediction_length"], accepted,
           row["MAE[0.5]"], wall,
           "" if rec["pred_len_check"] == "OK" and rec["p_check"] == "OK"
           else "  [GATE %s|%s]" % (rec["pred_len_check"], rec["p_check"])))
    if rec["pred_len_check"] != "OK" or rec["p_check"] != "OK":
        sys.exit(2)
    log("DONE")


if __name__ == "__main__":
    main()
