"""36-group test inference (9 domains x 4H) with tier-1 guards and a
status machine. selected_p.json MUST exist and be frozen first.

Guards (tier-1, hard stop on mismatch): protocol pins, weights sha, code
md5s, repo commit, selected_p sha, preflight report sha, batch, env.
Skip rule: state==done AND guard set identical AND artifacts pass
verify_cache group checks; otherwise corrupt/failed groups are quarantined
and recomputed. No precision changes to evade OOM.
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_visionts, configure, \
    predict_split
from repro_timemmd_common_v1.common import all_finite, core_guards, die, \
    dump_json, guards_equal, load_json, model_state_digest, \
    seed_everything, setup_deterministic_env, sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData
from repro_timemmd_common_v1.p_search import metrics


def group_artifacts_ok(out_root, dom, h, want_windows):
    gdir = Path(out_root) / "runs" / P.group_id(dom, h)
    npz = gdir / P.test_npz_name(dom, h)
    if not npz.exists():
        return False
    try:
        arr = np.load(npz)
        for k in ("pred", "truth", "s", "orig_row", "label_dates",
                  "window_index"):
            if k not in arr.files:
                return False
        if arr["pred"].shape[0] != want_windows:
            return False
        idx = arr["window_index"]
        if not np.array_equal(np.sort(idx), np.arange(want_windows)):
            return False
    except Exception:
        return False
    mj = gdir / ("metrics__%s.json" % P.group_id(dom, h))
    if not mj.exists():
        return False
    m = load_json(mj)
    return (m.get("evaluated_windows") == want_windows
            and m.get("npz_sha256") == sha256_of(npz)
            and all_finite(arr["pred"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--groups", default=None,
                    help="comma-separated Domain_H ids (debug reruns)")
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())
    runs = out_root / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    sel_path = out_root / "selected_p.json"
    if not sel_path.exists():
        die("[TIER-1] selected_p.json missing; run p_search first")
    sel_sha = sha256_of(sel_path)
    selected = {d: v["selected"] for d, v in
                load_json(sel_path)["selection"].items()}

    pf_path = out_root / "preflight_report.json"
    if not pf_path.exists():
        die("[TIER-1] preflight_report.json missing")
    pf_sha = sha256_of(pf_path)

    guards = core_guards(pinfo, P.BATCH, sel_sha, pf_sha)
    guards["selected_p"] = selected
    gpath = out_root / "guards.json"
    if gpath.exists():
        old = load_json(gpath)
        ok, diff = guards_equal(old, guards)
        if not ok:
            die("[TIER-1] guards mismatch (%s); refusing to run. "
                "Delete guards.json only after human review."
                % ",".join(diff))
    else:
        dump_json(gpath, guards)

    splits = pinfo["splits"]
    groups = P.all_groups()
    if args.groups:
        want = set(args.groups.split(","))
        groups = [g for g in groups if P.group_id(*g) in want]

    status_path = out_root / "status.json"
    status = load_json(status_path) if status_path.exists() else \
        {"groups": {}}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_visionts()
    model.to(dev)
    digest_before = model_state_digest(model)
    domcache = {}
    n_fail = 0

    for dom, h in groups:
        gid = P.group_id(dom, h)
        want = int(splits[dom]["window_counts"][str(h)]["test"])
        ent = status["groups"].get(gid, {})
        if ent.get("state") == "done" and \
                group_artifacts_ok(out_root, dom, h, want):
            print("SKIP %s (done, artifacts ok)" % gid, flush=True)
            continue
        gdir = runs / gid
        if gdir.exists():
            qdir = out_root / "quarantine" / gid
            qdir.parent.mkdir(parents=True, exist_ok=True)
            if qdir.exists():
                shutil.rmtree(qdir)
            shutil.move(str(gdir), str(qdir))
            print("QUARANTINED old %s" % gid, flush=True)
        gdir.mkdir(parents=True)
        status["groups"][gid] = {"state": "running"}
        dump_json(status_path, status)
        try:
            if dom not in domcache:
                domcache[dom] = DomainData(pinfo["protocol_dir"],
                                           pinfo["splits"], dom)
            dd = domcache[dom]
            configure(model, dd.L, h, selected[dom])
            sp = dd.load_split(h, "test")
            seed_everything(P.SEED)
            t0 = time.time()
            pred = predict_split(model, sp["X"], dev, batch=P.BATCH)
            elapsed = time.time() - t0
            pred = pred[:, :, None] if pred.ndim == 2 else pred
            truth = sp["Y"]
            if pred.shape != truth.shape:
                raise ValueError("pred %s vs truth %s"
                                 % (pred.shape, truth.shape))
            if pred.shape[0] != want:
                raise ValueError("coverage %d != %d"
                                 % (pred.shape[0], want))
            if not all_finite(pred):
                raise ValueError("non-finite predictions")
            mse, mae, mse64, mae64 = metrics(pred, truth)
            npz = gdir / P.test_npz_name(dom, h)
            np.savez_compressed(
                npz, pred=pred, truth=truth, s=sp["s"],
                orig_row=sp["orig_row"], label_dates=sp["label_dates"],
                window_index=sp["window_index"])
            m = {"group": gid, "domain": dom, "horizon": h, "P":
                 int(selected[dom]), "batch": P.BATCH,
                 "mse": mse, "mae": mae, "mse_f64": mse64, "mae_f64": mae64,
                 "evaluated_windows": int(pred.shape[0]),
                 "expected_windows": want,
                 "inference_seconds": round(elapsed, 3),
                 "model_digest_before": digest_before,
                 "model_digest_after": model_state_digest(model),
                 "npz_sha256": sha256_of(npz)}
            dump_json(gdir / ("metrics__%s.json" % gid), m)
            status["groups"][gid] = {"state": "done", "mse": mse,
                                     "mae": mae}
            print("DONE %s P=%d mse=%.6f mae=%.6f (%d win, %.1fs)"
                  % (gid, m["P"], mse, mae, want, elapsed), flush=True)
        except Exception as e:  # noqa: BLE001
            status["groups"][gid] = {"state": "failed", "error": repr(e)}
            n_fail += 1
            dump_json(status_path, status)
            raise
        dump_json(status_path, status)

    done = sum(1 for g in status["groups"].values()
               if g.get("state") == "done")
    print("RUN_INFER EXIT: done=%d failed=%d" % (done, n_fail))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
