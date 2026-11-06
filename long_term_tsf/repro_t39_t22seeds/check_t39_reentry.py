#!/usr/bin/env python
"""EXP-039 re-entry completeness check (post-delivery hardening).

One implementation, three modes:

  --one DS_H arm seed     decide mode for run_t39_all.sh job_decide:
                          prints SKIP\\t<detail> / MOVE:<dir>\\t<reason> /
                          RUN.  (MOVE semantics preserved for a future
                          queue re-entry; the archival closing round
                          itself never runs the queue.)

  --all                   READ-ONLY archival verification over all 64
                          tasks: never moves anything, reports
                          PASS:<detail> or FAIL:<reason>, writes
                          logs/reentry_check_t39.json.  Exit 0 iff
                          64/64 PASS.

  --generate-accepted     one-time READ-ONLY generation of
                          logs/accepted_ckpt_digests.json over the
                          frozen save_t39_full state.  Hard-gated: a
                          dir is only digested AFTER its checkpoint
                          file sha256 matches the DELIVERED train
                          json's final_ckpt_sha256 (BG8 pairwise
                          distinctness does not prove the file was not
                          substituted; the delivered evidence does).

Checks per task (superset of the queue's job_decide):
  presence + SUCCESS(seed/mse/mae) + train json(seed pins, success match)
  + job fingerprint == train json fingerprint
  + INDEPENDENT anchors vs EXP-003: code_md5 / data_sha256 / ckpt_sha256
    maps equal to the config's EXP-003 fingerprint
  + FULL config pin: job args.json AND fingerprint.args == EXP-003
    args.json rebuilt with the driver's exact overrides
    {model_id=job, save_dir=abspath(jd), ft_type=full,
     vm_pretrained=1|0 by arm, gpu=0, seed=<s>} -- no free variation of
    whitelisted keys (seed/ft_type/vm_pretrained are pinned to this
    task's expected values)
  + arrays ACTUALLY LOADED: pred/true shapes == EXP-003 archived shapes,
    np.isfinite over all elements (not just the \\x93NUMPY header)
  + checkpoint: file sha256 == delivered train json final_ckpt_sha256
    FIRST, then torch.load(weights_only=False, map_location=cpu) must
    succeed and its state_dict tensor digest must equal the accepted
    digest manifest.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


def build_pkg(pkg):
    sys.path.insert(0, pkg)
    import t39_common as X   # first import: fixes sys.path for t38 pkg
    import t38_common as C
    return X, C


def tensor_digest(sd):
    import torch
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(b"\x00")
        t = sd[k]
        if torch.is_tensor(t):
            t = t.detach().cpu().contiguous()
            h.update(str(tuple(t.shape)).encode())
            h.update(b"\x00")
            h.update(t.numpy().tobytes())
        else:
            h.update(repr(t).encode())
        h.update(b"\x01")
    return h.hexdigest()


def expected_args(X, C, ds, h, arm, seed, jd):
    job = X.config_job(ds, h, arm, seed)
    exp003_args = C.load_json(os.path.join(C.exp003_job(ds, h), "args.json"))
    expected = dict(exp003_args)
    expected.update({
        "model_id": job,
        "save_dir": os.path.abspath(jd),
        "ft_type": "full",
        "vm_pretrained": 1 if arm == "all" else 0,
        "gpu": 0,
        "seed": seed,
    })
    return expected, exp003_args


def check_one(X, C, ds, h, arm, seed, mode, accepted):
    """mode: 'decide' (SKIP/MOVE/RUN) or 'report' (PASS/FAIL/RUN)."""
    jd = X.b_job_dir(ds, h, arm, seed)
    job = X.config_job(ds, h, arm, seed)

    def _why(msg):
        if mode == "decide":
            return "MOVE:%s\t%s" % (jd, msg)
        return "FAIL[%s]: %s" % (job, msg)

    if not os.path.isdir(jd):
        return "RUN", "no job dir"
    succ = os.path.join(jd, "SUCCESS.txt")
    if not os.path.isfile(succ):
        return _why("no SUCCESS.txt"), ""
    try:
        s = C.load_json(succ)
        assert isinstance(s.get("mse"), float) and isinstance(s.get("mae"), float)
        assert s.get("seed") == seed
    except Exception as e:
        return _why("SUCCESS unreadable: %s" % e), ""
    p = X.train_json(ds, h, arm, seed)
    try:
        rec = C.load_json(p)
        assert rec["success"]["mse"] == s["mse"]
        assert rec["fingerprint"]["seed"] == seed
        assert rec["fingerprint"]["args"]["seed"] == seed
        assert rec.get("requested_seed") == seed
    except Exception as e:
        return _why("train json incomplete: %s" % e), ""
    try:
        fpj = C.load_json(os.path.join(jd, "fingerprint.json"))
        assert fpj == rec["fingerprint"], "job-dir fingerprint != train json"
        assert fpj.get("seed") == seed and fpj.get("args", {}).get("seed") == seed
        # all three are dict-valued maps (file md5 / dataset sha / ckpt sha)
        assert all(fpj.get(k) for k in ("code_md5", "data_sha256", "ckpt_sha256"))
    except Exception as e:
        return _why("fingerprint incomplete: %s" % e), ""
    # ---- independent anchors vs EXP-003 provenance (not self-referential)
    try:
        fp3 = C.load_json(os.path.join(C.exp003_job(ds, h), "fingerprint.json"))
        assert fpj["code_md5"] == fp3.get("code_md5"), "code_md5 != EXP-003"
        assert fpj["data_sha256"] == fp3.get("data_sha256"), "data sha != EXP-003"
        assert fpj["ckpt_sha256"] == fp3.get("ckpt_sha256"), "ckpt sha != EXP-003"
    except Exception as e:
        return _why("independent anchors: %s" % e), ""
    # ---- full config pin (whitelisted keys pinned to expected values)
    try:
        expected, _ = expected_args(X, C, ds, h, arm, seed, jd)
        got_args = C.load_json(os.path.join(jd, "args.json"))
        assert got_args == expected, "job args.json != expected config"
        assert rec["fingerprint"]["args"] == expected, \
            "train json fingerprint.args != expected config"
    except Exception as e:
        return _why("config pin: %s" % e), ""
    # ---- arrays actually loaded: shape == EXP-003, finite everywhere
    try:
        rd = os.path.join(jd, "results", "_")
        for name in ("pred.npy", "true.npy"):
            q = os.path.join(rd, name)
            if not os.path.isfile(q) or os.path.getsize(q) <= 0:
                raise AssertionError("missing/empty %s" % name)
            with open(q, "rb") as fh:
                if fh.read(6) != b"\x93NUMPY":
                    raise AssertionError("invalid npy header: %s" % name)
            ref = np.load(os.path.join(C.exp003_job(ds, h), "results", "_",
                                      name), mmap_mode="r")
            arr = np.load(q, allow_pickle=False)
            assert arr.shape == ref.shape, \
                "%s shape %s != EXP-003 %s" % (name, arr.shape, ref.shape)
            assert np.isfinite(arr).all(), "non-finite values in %s" % name
    except Exception as e:
        return _why("array validity: %s" % e), ""
    # ---- checkpoint: delivered file sha FIRST, then load + digest
    try:
        ck = os.path.join(jd, "checkpoints", "_", "checkpoint.pth")
        if not os.path.isfile(ck):
            raise AssertionError("missing final checkpoint")
        fsha = C.sha256_of(ck)
        assert fsha == rec.get("final_ckpt_sha256"), \
            "ckpt file sha != delivered train json final_ckpt_sha256"
        import torch
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        assert isinstance(sd, dict) and len(sd) > 0, "empty state_dict"
        if accepted is not None:
            ent = accepted.get(job)
            assert ent, "no accepted digest for %s" % job
            assert ent["file_sha256"] == fsha, "accepted file sha stale"
            assert tensor_digest(sd) == ent["tensor_digest"], \
                "state_dict digest != accepted manifest"
    except Exception as e:
        return _why("checkpoint validity: %s" % e), ""
    detail = "mse=%.6f mae=%.6f" % (s["mse"], s["mae"])
    if mode == "decide":
        return "SKIP\t%s (json=%s)" % (detail, os.path.basename(p)), ""
    return "PASS", detail


def load_accepted(X):
    path = os.path.join(X.LOG_DIR, "accepted_ckpt_digests.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def generate_accepted(X, C):
    acc = {}
    for ds, h, arm, seed in X.TASKS:
        jd = X.b_job_dir(ds, h, arm, seed)
        job = X.config_job(ds, h, arm, seed)
        ck = os.path.join(jd, "checkpoints", "_", "checkpoint.pth")
        p = X.train_json(ds, h, arm, seed)
        assert os.path.isfile(ck), "missing ckpt: %s" % ck
        rec = C.load_json(p)
        fsha = C.sha256_of(ck)
        assert fsha == rec.get("final_ckpt_sha256"), \
            "REFUSING %s: file sha != delivered final_ckpt_sha256" % job
        import torch
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        assert isinstance(sd, dict) and len(sd) > 0
        acc[job] = {"file_sha256": fsha,
                    "tensor_digest": tensor_digest(sd),
                    "n_keys": len(sd)}
        print("[t39-check] accepted %s (%d keys)" % (job, len(sd)))
    out = os.path.join(X.LOG_DIR, "accepted_ckpt_digests.json")
    with open(out, "w") as f:
        json.dump(acc, f, indent=1, sort_keys=True)
    print("[t39-check] wrote %s (%d entries)" % (out, len(acc)))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--one", nargs=3, metavar=("DS_H", "ARM", "SEED"))
    g.add_argument("--all", action="store_true")
    g.add_argument("--generate-accepted", action="store_true")
    ap.add_argument("pkg", nargs="?", default=None,
                    help="repro_t39_t22seeds package dir")
    a = ap.parse_args()
    pkg = a.pkg or os.path.dirname(os.path.abspath(__file__))
    X, C = build_pkg(pkg)

    if a.generate_accepted:
        generate_accepted(X, C)
        return
    if a.one:
        ds, h = a.one[0].rsplit("_", 1)
        verdict, _ = check_one(X, C, ds, int(h), a.one[1], int(a.one[2]),
                               "decide", load_accepted(X))
        print(verdict)
        sys.exit(0 if verdict == "RUN" or verdict.startswith("SKIP") else 0)
    # --all: READ-ONLY report
    accepted = load_accepted(X)
    assert accepted is not None, \
        "accepted_ckpt_digests.json missing: run --generate-accepted first"
    records, n_pass = [], 0
    for ds, h, arm, seed in X.TASKS:
        verdict, detail = check_one(X, C, ds, h, arm, seed, "report", accepted)
        ok = verdict == "PASS"
        n_pass += ok
        records.append({"ds": ds, "h": h, "arm": arm, "seed": seed,
                        "verdict": verdict, "detail": detail})
        print("[%s] %s_%d %s seed%d %s" % ("PASS" if ok else verdict.split(":")[0],
              ds, h, arm, seed, detail or verdict))
    summary = {"n_tasks": len(records), "n_pass": n_pass,
               "n_fail": len(records) - n_pass,
               "accepted_digests": len(accepted)}
    out = os.path.join(X.LOG_DIR, "reentry_check_t39.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1)
    print("[t39-check] %d/%d PASS (%d FAIL) -> %s"
          % (n_pass, summary["n_tasks"], summary["n_fail"], out))
    sys.exit(0 if n_pass == len(records) else 1)


if __name__ == "__main__":
    main()
