#!/usr/bin/env python3
"""Pre-GPU verification of BOTH references (user ruling 2026-10-11: must
run and pass BEFORE any GPU inference).

visionts_ref (EXP-016 VisionTS, old zeta environment):
  - MANIFEST.json sha256 for every file; archived protocol.json token ==
    a4c01683793a; archive-commit pin 0932a5ac...;
  - per subset: shard identity sequence == CSV row order (no missing/
    duplicate/reorder), saved target bitwise == CSV-parsed target, every
    pred finite;
  - ladder recompute (same code path as this round's scoring) vs the frozen
    EXP-016 {ds}_metrics.json and sequence_level_{ds}.csv at the DOUBLE
    tolerance rtol=atol=1e-12.

aurora_ref (EXP-017 Aurora, 3 seeds, frozen — not re-run):
  - MANIFEST.json sha256; COMMIT.txt == 3d20862a...; metrics sha pins;
  - per subset per seed: identity order == CSV, target bitwise == CSV,
    finite preds; ladder per seed; mean +- std(ddof=1) vs the frozen
    user-approved {ds}_metrics.json at rtol=atol=1e-12.

Nonzero exit on ANY failure. Writes results/verify_refs.json."""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (HERE, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

import common as C
import protocol as P
from evaluate import ladder, load_aurora_seed, load_vts_old
from load_cgtsf import iter_rows

RTOL = P.RECOMPUTE_RTOL
ATOL = P.RECOMPUTE_ATOL


def close(got, ref):
    got, ref = float(got), float(ref)
    ad = abs(got - ref)
    rd = ad / abs(ref) if ref != 0 else ad
    return (ad <= ATOL + RTOL * abs(ref)), ad, rd


def csv_table(ds):
    ids, tgts, order = {}, {}, []
    for ix, date, _h, ps in iter_rows(ds):
        key = f"{ix}|{date}"
        if key in ids:
            sys.exit(f"{ds}: duplicate CSV identity {key}")
        ids[key] = np.asarray(json.loads(ps), dtype=np.float64)
        order.append(key)
    return ids, order


def check_rows(ds, rows, ids, order, tag):
    got_order = [f"{ix}|{dt}" for ix, dt, _p, _t in rows]
    if got_order != order:
        sys.exit(f"HARD STOP {tag} {ds}: shard identity order != CSV row "
                 f"order (missing/duplicate/reorder)")
    tgt_bad = nonfinite = 0
    for _ix, _dt, pred, tgt in rows:
        if not np.isfinite(pred).all():
            nonfinite += 1
    for (_ix, _dt, _p, tgt), key in zip(rows, order):
        if not np.array_equal(tgt.astype(np.float64), ids[key]):
            tgt_bad += 1
    if tgt_bad or nonfinite:
        sys.exit(f"HARD STOP {tag} {ds}: target mismatches {tgt_bad}, "
                 f"nonfinite preds {nonfinite}")
    return len(rows)


def verify_vts_old(rep):
    ok = True
    man = C.load_json(os.path.join(P.VTS_REF_DIR, "MANIFEST.json"))
    bad = [rel for rel, sha in man.get("files", {}).items()
           if not os.path.isfile(os.path.join(P.VTS_REF_DIR, rel))
           or C.sha256_file(os.path.join(P.VTS_REF_DIR, rel)) != sha]
    if bad:
        sys.exit(f"HARD STOP: visionts_ref manifest sha mismatches: "
                 f"{bad[:10]} ({len(bad)})")
    if man.get("exp016_archive_commit") != P.EXP016_ARCHIVE_COMMIT:
        sys.exit("HARD STOP: visionts_ref archive-commit pin mismatch")
    proto_sha = C.sha256_file(os.path.join(P.VTS_REF_DIR, "protocol.json"))
    if proto_sha[:12] != P.EXP016_TOKEN:
        sys.exit(f"HARD STOP: EXP-016 protocol token {proto_sha[:12]} != "
                 f"{P.EXP016_TOKEN}")
    inf_sha = C.sha256_file(os.path.join(P.VTS_REF_DIR,
                                         "logs/infer_summary.json"))
    if inf_sha != P.EXP016_INFER_SUMMARY_SHA256:
        sys.exit("HARD STOP: EXP-016 infer_summary sha mismatch")
    rep["visionts_old"] = {"manifest_files": len(man["files"]),
                           "exp016_token_ok": True,
                           "exp016_commit_ok": True, "subsets": {}}
    for ds in P.ORDER:
        m = C.load_json(os.path.join(P.VTS_REF_DIR, "results",
                                     f"{ds}_metrics.json"))
        if C.sha256_file(os.path.join(P.VTS_REF_DIR, "results",
                                      f"{ds}_metrics.json")) \
                != P.EXP016_METRICS_SHA256[ds]:
            sys.exit(f"HARD STOP: EXP-016 {ds} metrics sha != pin")
        ids, order = csv_table(ds)
        rows = load_vts_old(ds)
        n = check_rows(ds, rows, ids, order, "vts_old")
        mse, mae, _pt, seqs = ladder(rows)
        with open(os.path.join(P.VTS_REF_DIR, "results",
                               f"sequence_level_{ds}.csv")) as f:
            r = list(csv.reader(f))[1:]
        ref_seq = {int(row[0]): (int(row[1]), float(row[2]), float(row[3]))
                   for row in r}
        seq_bad, seq_max_ad = 0, 0.0
        for ix, nw, mm, aa in seqs:
            if ref_seq[ix][0] != nw:
                seq_bad += 1
                continue
            for got, expect in ((mm, ref_seq[ix][1]), (aa, ref_seq[ix][2])):
                c, ad, _rd = close(got, expect)
                seq_max_ad = max(seq_max_ad, ad)
                if not c:
                    seq_bad += 1
        m_ok, m_ad, m_rd = close(mse, m["mse_seq_equal_weight"])
        a_ok, a_ad, a_rd = close(mae, m["mae_seq_equal_weight"])
        ds_ok = (m_ok and a_ok and seq_bad == 0
                 and len(seqs) == m["n_sequences"] and n == m["n_windows"])
        ok &= ds_ok
        rep["visionts_old"]["subsets"][ds] = {
            "n_windows": n,
            "mse": {"got": mse, "ref": m["mse_seq_equal_weight"],
                    "abs_diff": m_ad, "rel_diff": m_rd, "ok": bool(m_ok)},
            "mae": {"got": mae, "ref": m["mae_seq_equal_weight"],
                    "abs_diff": a_ad, "rel_diff": a_rd, "ok": bool(a_ok)},
            "sequence_entries_mismatched": seq_bad,
            "max_seq_abs_diff": seq_max_ad, "ok": bool(ds_ok)}
        print(f"[verify-refs] vts_old {ds}: dMSE={m_ad:.3e} dMAE={a_ad:.3e} "
              f"seq_bad={seq_bad} -> {'OK' if ds_ok else 'FAIL'}")
    return bool(ok)


def verify_aurora(rep):
    ok = True
    man = C.load_json(os.path.join(P.AURORA_REF_DIR, "MANIFEST.json"))
    bad = [rel for rel, sha in man.get("files", {}).items()
           if not os.path.isfile(os.path.join(P.AURORA_REF_DIR, rel))
           or C.sha256_file(os.path.join(P.AURORA_REF_DIR, rel)) != sha]
    if bad:
        sys.exit(f"HARD STOP: aurora_ref manifest sha mismatches: "
                 f"{bad[:10]} ({len(bad)})")
    commit = open(os.path.join(P.AURORA_REF_DIR, "COMMIT.txt")).read().strip()
    if commit != P.AURORA_CODE_COMMIT:
        sys.exit(f"HARD STOP: aurora_ref code commit {commit} != "
                 f"{P.AURORA_CODE_COMMIT}")
    if man.get("aurora_archive_commit") != P.AURORA_ARCHIVE_COMMIT:
        sys.exit("HARD STOP: aurora_ref archive-commit pin mismatch")
    rep["aurora"] = {"manifest_files": len(man["files"]),
                     "code_commit_ok": True, "archive_commit_ok": True,
                     "subsets": {}}
    for ds in P.ORDER:
        ref = C.load_json(os.path.join(P.AURORA_REF_DIR, "results",
                                       f"{ds}_metrics.json"))
        if C.sha256_file(os.path.join(P.AURORA_REF_DIR, "results",
                                      f"{ds}_metrics.json")) \
                != P.AURORA_METRICS_SHA256[ds]:
            sys.exit(f"HARD STOP: Aurora {ds} metrics sha != pin")
        ids, order = csv_table(ds)
        per_seed_mse, per_seed_mae = {}, {}
        seed_detail = {}
        for seed in P.AURORA_SEEDS:
            rows = load_aurora_seed(ds, seed)
            n = check_rows(ds, rows, ids, order, f"aurora s{seed}")
            mse, mae, _pt, seqs = ladder(rows)
            per_seed_mse[seed], per_seed_mae[seed] = mse, mae
            seed_detail[str(seed)] = {"n_windows": n,
                                      "mse_recomputed": mse,
                                      "mae_recomputed": mae}
        mses = [per_seed_mse[s] for s in P.AURORA_SEEDS]
        maes = [per_seed_mae[s] for s in P.AURORA_SEEDS]
        mean_mse, mean_mae = float(np.mean(mses)), float(np.mean(maes))
        std_mse = float(np.std(mses, ddof=1))
        std_mae = float(np.std(maes, ddof=1))
        checks = {}
        for name, got, want in (
                ("mse_mean", mean_mse, ref["mse_seq_equal_weight_mean"]),
                ("mse_std", std_mse, ref["mse_seq_equal_weight_std"]),
                ("mae_mean", mean_mae, ref["mae_seq_equal_weight_mean"]),
                ("mae_std", std_mae, ref["mae_seq_equal_weight_std"])):
            c, ad, rd = close(got, want)
            checks[name] = {"got": got, "ref": want, "abs_diff": ad,
                            "rel_diff": rd, "ok": bool(c)}
        for s in P.AURORA_SEEDS:
            ps = ref["per_seed"][str(s)]
            for met, got in (("mse_seq_equal_weight", per_seed_mse[s]),
                             ("mae_seq_equal_weight", per_seed_mae[s])):
                c, ad, _rd = close(got, ps[met])
                checks[f"s{s}_{met}"] = {"got": got, "ref": ps[met],
                                         "abs_diff": ad, "ok": bool(c)}
        ds_ok = all(v["ok"] for v in checks.values())
        ok &= ds_ok
        rep["aurora"]["subsets"][ds] = {
            "seeds": list(P.AURORA_SEEDS), "checks": checks,
            "ok": bool(ds_ok)}
        worst = max(v["abs_diff"] for v in checks.values())
        print(f"[verify-refs] aurora {ds}: max abs diff {worst:.3e} "
              f"-> {'OK' if ds_ok else 'FAIL'}")
    return bool(ok)


def main():
    rep = {"rtol": RTOL, "atol": ATOL,
           "note": "double tolerance: |got-ref| <= atol + rtol*|ref|"}
    ok1 = verify_vts_old(rep)
    ok2 = verify_aurora(rep)
    rep["ok"] = bool(ok1 and ok2)
    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    C.dump_json(rep, os.path.join(P.RESULTS_DIR, "verify_refs.json"))
    print(f"[verify-refs] RESULT: {'PASS' if rep['ok'] else 'FAIL'} "
          f"(visionts_old={ok1}, aurora={ok2})")
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
