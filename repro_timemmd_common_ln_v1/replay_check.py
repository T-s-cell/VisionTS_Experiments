"""36-group independent replay driver: re-executes replay_one.py (fresh
process, strict ckpt restore) per group and aggregates the per-group
reports. Also verifies cache immutability (report carries before/after
hashes) and provides --perturb_verify: a tampered checkpoint must be
detected — variant A (bytes only) by the sha gate, variant B (hashes
recomputed consistently) by the numeric replay comparison. Exit 3 if a
negative control is NOT detected.
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from repro_timemmd_common_ln_v1 import protocol as P
from repro_timemmd_common_ln_v1.common import dump_json, load_json, \
    setup_deterministic_env, sha256_of
from repro_timemmd_common_ln_v1.ln_train import npz_key, packed_sha


def run_group(dom, h):
    cmd = [sys.executable, "-m",
           "repro_timemmd_common_ln_v1.replay_one",
           "--dom", dom, "--h", str(h)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    gid = P.group_id(dom, h)
    rep_path = P.out_root() / "runs" / gid / ("replay_report__%s.json"
                                              % gid)
    if rep_path.exists():
        return load_json(rep_path)
    return {"gid": gid, "all_pass": False,
            "error": "rc=%d %s" % (r.returncode, r.stderr[-2000:])}


def numeric_npz_key(z):
    """First sorted float array key in an npz mapping (skips string arrays
    like ln_names/label_dates). Raises if the npz has no float array."""
    for k in sorted(z.keys()):
        if z[k].dtype.kind == "f":
            return k
    raise SystemExit("[TIER-1] perturb target: no float array in npz")


def perturb_verify(pinfo):
    """Negative control on Security/H6 (smallest group). Formal artifacts
    (replay npz/report, test npz, ckpt npz/meta) are hash-watched before
    and after both variants: the NC must stay fully inside
    negative_control/."""
    import shutil
    dom, h = "Security", 6
    out_dir = P.out_root()
    nc = out_dir / "negative_control"
    nc.mkdir(parents=True, exist_ok=True)
    gid = P.group_id(dom, h)
    gdir = out_dir / "runs" / gid
    watch = [out_dir / P.replay_npz_name(dom, h),
             gdir / ("replay_report__%s.json" % gid),
             gdir / P.test_npz_name(dom, h),
             out_dir / P.ckpt_npz_name(dom, h),
             out_dir / P.ckpt_meta_name(dom, h)]

    def snap():
        return {str(p): sha256_of(p) for p in watch}

    results = {"formal_hashes_pre": snap()}
    src = out_dir / P.ckpt_npz_name(dom, h)
    meta_src = out_dir / P.ckpt_meta_name(dom, h)

    # variant A: tamper bytes, leave meta untouched -> sha gate must trip
    pa = nc / "naive.npz"
    shutil.copyfile(str(src), str(pa))
    z = dict(np.load(str(pa)))
    k0 = numeric_npz_key(z)
    z[k0] = (z[k0] + np.float32(1e-3)).astype(np.float32)
    np.savez(str(pa), **z)
    r = subprocess.run(
        [sys.executable, "-m", "repro_timemmd_common_ln_v1.replay_one",
         "--dom", dom, "--h", str(h), "--ckpt", str(pa),
         "--meta", str(meta_src), "--nc_tag", "variant_a_naive"],
        capture_output=True, text=True)
    results["variant_a_naive_detected"] = (r.returncode != 0)
    results["variant_a_rc"] = r.returncode
    results["variant_a_stderr_tail"] = r.stderr[-800:]

    # variant B: tamper bytes AND recompute meta hashes consistently ->
    # restore succeeds; the numeric replay comparison must catch it
    pb = nc / "consistent.npz"
    shutil.copyfile(str(pa), str(pb))
    meta = load_json(meta_src)
    names = meta["ln_names"]
    raw = np.load(str(pb))
    meta_b = dict(meta)
    meta_b["npz_sha256"] = sha256_of(pb)
    meta_b["packed_sha256"] = packed_sha(names,
                                         {n: raw[npz_key(n)]
                                          for n in names})
    mb_path = nc / "consistent.meta.json"
    dump_json(mb_path, meta_b)
    r = subprocess.run(
        [sys.executable, "-m", "repro_timemmd_common_ln_v1.replay_one",
         "--dom", dom, "--h", str(h), "--ckpt", str(pb),
         "--meta", str(mb_path), "--nc_tag", "variant_b_consistent"],
        capture_output=True, text=True)
    # rc==2 means replay comparison failed -> tamper DETECTED
    results["variant_b_consistent_detected"] = (r.returncode == 2)
    results["variant_b_rc"] = r.returncode
    results["variant_b_stderr_tail"] = r.stderr[-800:]
    b_rep = nc / ("replay_report__variant_b_consistent__%s.json" % gid)
    if b_rep.exists():
        results["variant_b_max_abs_diff"] = load_json(b_rep) \
            .get("max_abs_diff")
    results["formal_hashes_post"] = snap()
    results["formal_artifacts_unchanged"] = \
        (results["formal_hashes_pre"] == results["formal_hashes_post"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--perturb_verify", action="store_true")
    ap.add_argument("--groups", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)

    groups = P.all_groups()
    if args.groups:
        want = set(args.groups.split(","))
        groups = [g for g in groups if P.group_id(*g) in want]

    reports = {}
    n_fail = 0
    for dom, h in groups:
        rep = run_group(dom, h)
        reports[rep["gid"]] = rep
        if not rep.get("all_pass"):
            n_fail += 1
            print("REPLAY FAIL %s: %s" % (rep["gid"],
                                          rep.get("error", "mismatch")),
                  flush=True)
        else:
            print("REPLAY OK %s max_abs=%.3e"
                  % (rep["gid"], rep["max_abs_diff"]), flush=True)

    summary = {
        "n_groups": len(groups), "n_pass": len(groups) - n_fail,
        "reports": reports,
        "immutability_ok": all(
            r.get("immutability_ok") for r in reports.values()),
    }
    if args.perturb_verify:
        pv = perturb_verify(pinfo)
        summary["perturb_verify"] = pv
        if not pv["formal_artifacts_unchanged"]:
            dump_json(P.out_root() / "replay_summary.json", summary)
            print("[FATAL] perturb_verify MODIFIED formal artifacts")
            raise SystemExit(3)
        if not (pv["variant_a_naive_detected"]
                and pv["variant_b_consistent_detected"]):
            dump_json(P.out_root() / "replay_summary.json", summary)
            print("[FATAL] perturb_verify NOT detected: %s" % pv)
            raise SystemExit(3)
        print("PERTURB_VERIFY: both variants detected, formal artifacts "
              "unchanged", flush=True)

    # Summary must equal the on-disk reports at exit time (gate added
    # after the summary/report divergence incident).
    diverged = []
    for gid, rep in reports.items():
        try:
            d = load_json(P.out_root() / "runs" / gid
                          / ("replay_report__%s.json" % gid))
        except Exception:
            diverged.append(gid)
            continue
        if (d.get("all_pass") != rep.get("all_pass")
                or d.get("max_abs_diff") != rep.get("max_abs_diff")):
            diverged.append(gid)
    summary["on_disk_consistent"] = not diverged
    if diverged:
        dump_json(P.out_root() / "replay_summary.json", summary)
        print("[FATAL] on-disk replay reports diverge from summary: %s"
              % diverged)
        raise SystemExit(3)

    dump_json(P.out_root() / "replay_summary.json", summary)
    print("REPLAY: %d/%d pass, immutability_ok=%s"
          % (summary["n_pass"], len(groups), summary["immutability_ok"]))
    if n_fail or not summary["immutability_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
