#!/usr/bin/env python3
"""Stage guard: a .done marker only counts as up-to-date when its recorded
sha256 payload matches the CURRENT files. A missing/empty/mismatched payload
=> stale => the caller must re-run the stage. Exit 0 = up-to-date, 1 = stale.
"""
import os
import sys

import common as C

HERE = C.HERE

# must stay in sync with run_all.sh::donefile()
MARKER = {
    "freeze": "s1_freeze",
    "verify": "s2_verify_cache",
    "sample": "s3_sample_frozen",
    "preflight": "s3_preflight",
    "fill": "s4_infer_fill",
    "assemble": "s5a_assemble",
    "evaluate": "s5b_evaluate",
    "recompute": "recompute",
    "accept": "accept",
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MARKER:
        print("usage: stage_guard.py <freeze|verify|sample|preflight|fill|assemble|"
              "evaluate|recompute|accept>", file=sys.stderr)
        sys.exit(2)
    stage = sys.argv[1]
    marker = C.state_path(MARKER[stage])
    if not os.path.isfile(marker):
        sys.exit(1)
    payload = C.load_json(marker)
    sha_map = payload.get("sha")
    if payload.get("status") != "done" or not isinstance(sha_map, dict) or not sha_map:
        print(f"[guard] {stage}: marker has no sha payload — stale", file=sys.stderr)
        sys.exit(1)
    for rel, recorded in sorted(sha_map.items()):
        f = os.path.join(HERE, rel)
        if not os.path.isfile(f):
            print(f"[guard] {stage}: {rel} missing — stale", file=sys.stderr)
            sys.exit(1)
        got = C.sha256_file(f)
        if got != recorded:
            print(f"[guard] {stage}: {rel} sha256 changed since the marker "
                  f"({got[:12]}… != {recorded[:12]}…) — stale", file=sys.stderr)
            sys.exit(1)
    # input / config / code fingerprint: recompute the CURRENT declaration
    # (stage_input_guards resolves every path incl. code md5s and old-cache
    # npz) and require the marker to carry the identical mapping.
    in_guards = payload.get("input_guards")
    if not isinstance(in_guards, dict) or not in_guards:
        print(f"[guard] {stage}: marker has no input_guards — stale", file=sys.stderr)
        sys.exit(1)
    expected = C.stage_input_guards(stage, zip_path=(C._zip_path() if stage == "freeze" else None))
    if set(expected) != set(in_guards):
        only_marked = sorted(set(in_guards) - set(expected))[:5]
        only_now = sorted(set(expected) - set(in_guards))[:5]
        print(f"[guard] {stage}: input-guard declaration changed "
              f"(marker-only {only_marked}, current-only {only_now}) — stale", file=sys.stderr)
        sys.exit(1)
    for k in sorted(expected):
        if expected[k] != in_guards[k]:
            print(f"[guard] {stage}: input {k} changed since the marker "
                  f"({str(expected[k])[:12]}… != {str(in_guards[k])[:12]}…) — stale",
                  file=sys.stderr)
            sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
