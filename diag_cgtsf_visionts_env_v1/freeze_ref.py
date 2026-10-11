#!/usr/bin/env python3
"""Freeze the two references into this experiment tree BEFORE anything runs
on the GPU (local machine only; deploy copies these trees verbatim).

  visionts_ref/  <- sibling EXP-016 repro_cgtsf_zeroshot_v1/
      predictions/{ds}__shardXXX.npz (+ .done), results/{ds}_metrics.json,
      results/sequence_level_{ds}.csv, results/summary.json,
      protocol.json, logs/infer_summary.json, COMMIT.txt
  aurora_ref/    <- Aurora EXP-017 (Aurora_Experiments repo, commit 3d20862a)
      aurora predictions_dt/{ds}__s{seed}__shardXXX.npz (+ .done),
      verify_dt/results/{ds}_metrics.json (+ per-seed metrics json),
      repro COMMIT.txt, protocol.json, logs_dt/infer_summary.json

Each tree gets MANIFEST.json {rel_path: sha256} plus provenance pins
(commits, tokens). Metrics shas are hard-checked against protocol.py pins
immediately — any drift stops here, before deployment."""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
MMTS = os.path.dirname(REPO_ROOT)
for p in (HERE,):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C
import protocol as P

VTS_SRC = os.path.join(REPO_ROOT, "repro_cgtsf_zeroshot_v1")
AURORA_REPO = os.path.join(MMTS, "Aurora_Experiments",
                           "repro_cgtsf_zeroshot_v1")
AURORA_VERIFY = os.path.join(MMTS, "Aurora_Experiments", "verify_dt")


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def freeze_vts():
    out = P.VTS_REF_DIR
    if os.path.isdir(out):
        sys.exit(f"{out} already exists — delete it first to re-freeze")
    files = []
    for ds in P.ORDER:
        for shard in range(P.n_shards(ds)):
            base = f"{ds}__shard{shard:03d}"
            files.append((f"predictions/{base}.npz",
                          os.path.join(VTS_SRC, "predictions", base + ".npz")))
            files.append((f"predictions/{base}.done",
                          os.path.join(VTS_SRC, "predictions",
                                       base + ".done")))
        files.append((f"results/{ds}_metrics.json",
                      os.path.join(VTS_SRC, "results", f"{ds}_metrics.json")))
        files.append((f"results/sequence_level_{ds}.csv",
                      os.path.join(VTS_SRC, "results",
                                   f"sequence_level_{ds}.csv")))
    files += [
        ("results/summary.json",
         os.path.join(VTS_SRC, "results", "summary.json")),
        ("protocol.json", os.path.join(VTS_SRC, "protocol.json")),
        ("logs/infer_summary.json",
         os.path.join(VTS_SRC, "logs", "infer_summary.json")),
        ("COMMIT.txt", os.path.join(VTS_SRC, "COMMIT.txt")),
    ]
    manifest = {"kind": "visionts_ref", "exp016_archive_commit":
                P.EXP016_ARCHIVE_COMMIT, "exp016_token": P.EXP016_TOKEN,
                "files": {}}
    for rel, src in files:
        if not os.path.isfile(src):
            sys.exit(f"freeze_vts: missing source {src}")
        dst = os.path.join(out, rel)
        copy_file(src, dst)
        manifest["files"][rel] = C.sha256_file(dst)
    for ds in P.ORDER:
        got = manifest["files"][f"results/{ds}_metrics.json"]
        if got != P.EXP016_METRICS_SHA256[ds]:
            sys.exit(f"freeze_vts: {ds}_metrics.json sha {got} != pinned "
                     f"{P.EXP016_METRICS_SHA256[ds]}")
    proto_sha = C.sha256_file(os.path.join(out, "protocol.json"))
    if proto_sha[:12] != P.EXP016_TOKEN:
        sys.exit(f"freeze_vts: EXP-016 protocol token {proto_sha[:12]} != "
                 f"{P.EXP016_TOKEN}")
    C.dump_json(manifest, os.path.join(out, "MANIFEST.json"))
    print(f"[freeze_ref] visionts_ref: {len(manifest['files'])} files, "
          f"metrics shas == pins OK, token OK")


def freeze_aurora():
    out = P.AURORA_REF_DIR
    if os.path.isdir(out):
        sys.exit(f"{out} already exists — delete it first to re-freeze")
    files = []
    for ds in P.ORDER:
        for seed in P.AURORA_SEEDS:
            for shard in range(P.n_shards(ds)):
                base = f"{ds}__s{seed}__shard{shard:03d}"
                files.append((f"predictions_dt/{base}.npz",
                              os.path.join(AURORA_REPO, "predictions_dt",
                                           base + ".npz")))
                files.append((f"predictions_dt/{base}.done",
                              os.path.join(AURORA_REPO, "predictions_dt",
                                           base + ".done")))
        files.append((f"results/{ds}_metrics.json",
                      os.path.join(AURORA_VERIFY, "results",
                                   f"{ds}_metrics.json")))
        for seed in P.AURORA_SEEDS:
            files.append((f"results/{ds}_metrics__s{seed}.json",
                          os.path.join(AURORA_VERIFY, "results",
                                       f"{ds}_metrics__s{seed}.json")))
    files += [
        ("COMMIT.txt", os.path.join(AURORA_REPO, "COMMIT.txt")),
        ("protocol.json", os.path.join(AURORA_REPO, "protocol.json")),
        ("logs/infer_summary.json",
         os.path.join(AURORA_REPO, "logs_dt", "infer_summary.json")),
    ]
    manifest = {"kind": "aurora_ref",
                "aurora_archive_commit": P.AURORA_ARCHIVE_COMMIT,
                "aurora_code_commit": P.AURORA_CODE_COMMIT,
                "aurora_token": P.AURORA_TOKEN, "seeds": list(P.AURORA_SEEDS),
                "files": {}}
    for rel, src in files:
        if not os.path.isfile(src):
            sys.exit(f"freeze_aurora: missing source {src}")
        dst = os.path.join(out, rel)
        copy_file(src, dst)
        manifest["files"][rel] = C.sha256_file(dst)
    for ds in P.ORDER:
        got = manifest["files"][f"results/{ds}_metrics.json"]
        if got != P.AURORA_METRICS_SHA256[ds]:
            sys.exit(f"freeze_aurora: {ds}_metrics.json sha {got} != pinned "
                     f"{P.AURORA_METRICS_SHA256[ds]}")
    commit = open(os.path.join(out, "COMMIT.txt")).read().strip()
    if commit != P.AURORA_CODE_COMMIT:
        sys.exit(f"freeze_aurora: Aurora COMMIT.txt (code commit) {commit} "
                 f"!= pinned {P.AURORA_CODE_COMMIT}")
    C.dump_json(manifest, os.path.join(out, "MANIFEST.json"))
    print(f"[freeze_ref] aurora_ref: {len(manifest['files'])} files, "
          f"metrics shas == pins OK, commit OK")


def main():
    freeze_vts()
    freeze_aurora()
    print("[freeze_ref] both references frozen and pinned")


if __name__ == "__main__":
    main()
