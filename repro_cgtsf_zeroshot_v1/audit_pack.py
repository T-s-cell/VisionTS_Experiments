#!/usr/bin/env python3
"""Audit pack builder. Runs on the machine holding the full experiment tree
(local after rsync-ing artifacts back from the GPU host). Includes the
actual executed source snapshot, frozen protocol, data CSVs, preflight
evidence and (for --stage final) all prediction shards + verification
results + logs. Every packed file's sha256 goes to MANIFEST.sha256.
Redaction scan (API-key / private-key / password patterns) aborts the pack
if anything matches. Output: temp/VisionTS_CGTSF_zeroshot_<stage>_<ts>.tar.gz
"""
import argparse
import glob
import os
import re
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (HERE,):
    if p not in sys.path:
        sys.path.insert(0, p)

import protocol as P

REDACTION = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "API key pattern"),
    (re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"), "private key block"),
    (re.compile("TIMESX_" + "LLM_API_KEY"), "LLM key variable"),
]


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(stage):
    files = []
    exp_py = sorted(glob.glob(os.path.join(HERE, "*.py"))) + \
        sorted(glob.glob(os.path.join(HERE, "*.sh"))) + \
        [os.path.join(HERE, n) for n in
         ("README.md", ".gitignore", "COMMIT.txt", "protocol.json")
         if os.path.isfile(os.path.join(HERE, n))]
    files += [("code/" + os.path.basename(p), p) for p in exp_py]
    for rel in P.VTS_CODE_FILES:
        p = os.path.join(REPO_ROOT, rel)
        if os.path.isfile(p):
            files.append(("code/" + rel.replace("/", "__"), p))
    for ds in P.ORDER:
        files.append((f"data/{ds}.csv", P.csv_path(ds)))
    files.append(("results/preflight_report.json",
                  os.path.join(P.RESULTS_DIR, "preflight_report.json")))
    for p in sorted(glob.glob(os.path.join(P.LOGS_DIR, "*"))):
        if os.path.isfile(p):
            files.append(("logs/" + os.path.basename(p), p))
    if stage == "final":
        for p in sorted(glob.glob(os.path.join(P.PRED_DIR, "*"))):
            if os.path.isfile(p):
                files.append(("predictions/" + os.path.basename(p), p))
        for p in sorted(glob.glob(os.path.join(P.QUARANTINE_DIR, "*"))):
            if os.path.isfile(p):
                files.append(("quarantine/" + os.path.basename(p), p))
        for p in sorted(glob.glob(os.path.join(P.RESULTS_DIR, "*"))):
            if os.path.isfile(p):
                files.append(("results/" + os.path.basename(p), p))
    seen, out = set(), []
    for arc, p in files:
        if arc not in seen and os.path.isfile(p):
            seen.add(arc)
            out.append((arc, p))
    return out


def redact_scan(items, staging):
    hits = []
    for arc, src in items:
        if os.path.getsize(src) > 200 * (1 << 20):
            continue
        try:
            with open(src, "rb") as f:
                data = f.read()
        except OSError:
            continue
        for rx, what in REDACTION:
            if rx.search(data.decode("utf-8", "ignore")):
                hits.append((arc, what))
    if hits:
        for arc, what in hits:
            print(f"REDACTION HIT: {arc}: {what}", file=sys.stderr)
        sys.exit("aborted: sensitive pattern found in pack inputs")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("preflight", "final"),
                    default="preflight")
    ap.add_argument("--out", default=os.path.join(HERE, "pack_out"))
    args = ap.parse_args()

    if args.stage == "final" and not \
            os.path.isfile(os.path.join(P.RESULTS_DIR, "verify_cache.json")):
        sys.exit("final pack needs results/verify_cache.json (run verify)")
    if not os.path.isfile(os.path.join(P.RESULTS_DIR,
                                       "preflight_report.json")):
        sys.exit("preflight_report.json missing — nothing to pack")

    items = collect(args.stage)
    redact_scan(items, args.out)
    os.makedirs(args.out, exist_ok=True)
    import time
    ts = time.strftime("%Y%m%d_%H%M")
    name = f"VisionTS_CGTSF_zeroshot_{args.stage}_{ts}.tar.gz"
    out_path = os.path.join(args.out, name)
    manifest = []
    with tarfile.open(out_path + ".tmp", "w:gz") as tf:
        for arc, src in items:
            info = tf.gettarinfo(src, arcname=f"cgtsf_zeroshot_v1/{arc}")
            info.mtime = int(os.path.getmtime(src))
            with open(src, "rb") as f:
                tf.addfile(info, f)
            manifest.append((arc, sha256_file(src)))
    os.replace(out_path + ".tmp", out_path)
    with open(out_path + ".MANIFEST.sha256", "w") as f:
        for arc, h in manifest:
            f.write(f"{h}  cgtsf_zeroshot_v1/{arc}\n")
    print(f"pack -> {out_path}")
    print(f"manifest -> {out_path}.MANIFEST.sha256 "
          f"({len(manifest)} files)")


if __name__ == "__main__":
    main()
