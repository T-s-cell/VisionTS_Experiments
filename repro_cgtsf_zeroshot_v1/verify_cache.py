#!/usr/bin/env python3
"""Full-integrity verification of the delivered predictions (independent
pass, run after run_infer):
  - shard payload sha256 re-verified against .done markers;
  - identity set from shards == identity set from the CSVs (no missing,
    no duplicates, exact 33,693);
  - per-window saved target bitwise == CSV Parsed target;
  - every prediction finite (any NaN/Inf = failure).
Nonzero exit on any failure. Writes results/verify_cache.json."""
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
from load_cgtsf import iter_rows


def main():
    rep = {"protocol_name": P.PROTOCOL_NAME, "subsets": {}}
    all_ok = True
    total = 0
    for ds in P.ORDER:
        cfg = P.SUBSETS[ds]
        csv_ids, csv_tgt, csv_order = {}, {}, []
        for ix, date, _h, ps in iter_rows(ds):
            key = f"{ix}|{date}"
            if key in csv_ids:
                sys.exit(f"{ds}: duplicate CSV identity {key}")
            csv_ids[key] = np.asarray(json.loads(ps), dtype=np.float64)
            csv_order.append(key)
        got_ids, got_order, tgt_bad, nonfinite, sha_bad = set(), [], 0, 0, 0
        n = 0
        for shard in range(P.n_shards(ds)):
            npz, hdr, done = P.shard_paths(ds, shard)
            if not os.path.isfile(done):
                sys.exit(f"{ds} shard {shard}: .done missing")
            d = C.load_json(done)
            if C.sha256_file(npz) != d.get("payload_sha256"):
                sha_bad += 1
                continue
            with np.load(npz) as z:
                for i in range(len(z["idx"])):
                    key = f"{int(z['idx'][i])}|{str(z['date'][i])}"
                    got_ids.add(key)
                    got_order.append(key)
                    if not np.array_equal(z["target"][i], csv_ids[key]):
                        tgt_bad += 1
                    if not np.isfinite(z["pred"][i]).all():
                        nonfinite += 1
                    n += 1
        missing = csv_ids.keys() - got_ids
        extra = got_ids - csv_ids.keys()
        ok = (not missing and not extra and tgt_bad == 0
              and nonfinite == 0 and sha_bad == 0
              and n == P.EXPECTED_WINDOWS[ds])
        all_ok &= ok
        total += n
        rep["subsets"][ds] = {
            "windows_found": n,
            "expected": P.EXPECTED_WINDOWS[ds],
            "missing": sorted(missing)[:10], "n_missing": len(missing),
            "extra": sorted(extra)[:10], "n_extra": len(extra),
            "target_bitwise_mismatches": tgt_bad,
            "pred_nonfinite": nonfinite,
            "payload_sha_failures": sha_bad,
            "ok": bool(ok),
        }
        print(f"[verify] {ds}: n={n}/{P.EXPECTED_WINDOWS[ds]} "
              f"missing={len(missing)} extra={len(extra)} "
              f"tgt_mismatch={tgt_bad} nonfinite={nonfinite} "
              f"sha_fail={sha_bad} -> {'OK' if ok else 'FAIL'}")
    rep["total_windows"] = total
    rep["total_expected"] = sum(P.EXPECTED_WINDOWS.values())
    rep["ok"] = bool(all_ok)
    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    C.dump_json(rep, os.path.join(P.RESULTS_DIR, "verify_cache.json"))
    print(f"[verify] TOTAL {total}/{rep['total_expected']} -> "
          f"{'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
