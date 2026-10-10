#!/usr/bin/env python3
"""Full-integrity verification of the delivered predictions (independent
pass, run after run_infer). Per shard ALL of the following must hold:
  - .done metadata consistent (ds / shard index / window count);
  - stored guards core set == current machine-independent fingerprints
    (protocol sha / weights sha / csv sha / code md5s) — mismatch means the
    shard was produced under a different protocol or source = FAIL;
  - stored runtime batch == preflight report chosen_batch, and stored
    preflight-report sha256 == sha256 of the local preflight report;
  - shard npz payload sha256 matches .done;
  - npz fields idx/date/pred/target/ms present, equal length == the
    shard's expected window count, pred/target shaped [n, P].
Dataset level:
  - window identity sequence from the shards == CSV row order exactly
    (no missing, no duplicates, no reordering), exact expected count;
  - per-window saved target bitwise == CSV-parsed target;
  - every prediction finite (any NaN/Inf = failure).
Nonzero exit on ANY failure. Writes results/verify_cache.json."""
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
    rep_path = os.path.join(P.RESULTS_DIR, "preflight_report.json")
    if not os.path.isfile(rep_path):
        sys.exit("HARD STOP: results/preflight_report.json missing — "
                 "cannot bind shards to the preflight execution config")
    pre = C.load_json(rep_path)
    if not pre.get("ok"):
        sys.exit("HARD STOP: preflight report ok!=true")
    chosen_batch = pre.get("chosen_batch")
    rep_sha = C.sha256_file(rep_path)
    core = C.core_guards()

    rep = {"protocol_name": P.PROTOCOL_NAME,
           "bound_batch": chosen_batch,
           "preflight_report_sha256": rep_sha,
           "subsets": {}}
    all_ok = True
    total = 0
    for ds in P.ORDER:
        cfg = P.SUBSETS[ds]
        Pl = cfg["pred"]
        csv_ids, csv_tgt, csv_order = {}, {}, []
        for ix, date, _h, ps in iter_rows(ds):
            key = f"{ix}|{date}"
            if key in csv_ids:
                sys.exit(f"{ds}: duplicate CSV identity {key}")
            csv_ids[key] = np.asarray(json.loads(ps), dtype=np.float64)
            csv_order.append(key)
        n_csv = len(csv_order)
        got_ids, got_order = set(), []
        tgt_bad, nonfinite, sha_bad = 0, 0, 0
        meta_bad, guard_bad, runtime_bad, struct_bad = 0, 0, 0, 0
        bad_shards = []
        n = 0
        for shard in range(P.n_shards(ds)):
            lo = shard * P.SHARD_SIZE
            n_exp = min(P.SHARD_SIZE, n_csv - lo)
            npz, hdr, done = P.shard_paths(ds, shard)
            if not os.path.isfile(done):
                sys.exit(f"{ds} shard {shard}: .done missing")
            d = C.load_json(done)
            shard_bad = False
            if d.get("ds") != ds or d.get("shard") != shard \
                    or d.get("n") != n_exp:
                meta_bad += 1
                shard_bad = True
            g = d.get("guards") or {}
            if g.get("core") != core:
                guard_bad += 1
                shard_bad = True
            else:
                rt = g.get("runtime") or {}
                if rt.get("batch") != chosen_batch \
                        or rt.get("preflight_report_sha256") != rep_sha:
                    runtime_bad += 1
                    shard_bad = True
            if not os.path.isfile(npz) or \
                    C.sha256_file(npz) != d.get("payload_sha256"):
                sha_bad += 1
                shard_bad = True
                bad_shards.append(shard)
                continue
            fields_ok = lens_ok = shape_ok = True
            with np.load(npz, allow_pickle=False) as z:
                need = ("idx", "date", "pred", "target", "ms")
                fields_ok = all(f in z.files for f in need)
                if fields_ok:
                    idx = z["idx"]
                    date = z["date"]
                    pred = z["pred"]
                    tgt = z["target"]
                    ms = z["ms"]
                    lens_ok = (len(idx) == len(date) == len(pred)
                               == len(tgt) == len(ms) == n_exp)
                    shape_ok = (pred.shape == (n_exp, Pl)
                                and tgt.shape == (n_exp, Pl))
                if not (fields_ok and lens_ok and shape_ok):
                    struct_bad += 1
                    shard_bad = True
                    bad_shards.append(shard)
                    continue
                for i in range(n_exp):
                    key = f"{int(idx[i])}|{str(date[i])}"
                    got_ids.add(key)
                    got_order.append(key)
                    if not np.array_equal(tgt[i], csv_ids[key]):
                        tgt_bad += 1
                    if not np.isfinite(pred[i]).all():
                        nonfinite += 1
                    n += 1
            if shard_bad:
                bad_shards.append(shard)
        missing = csv_ids.keys() - got_ids
        extra = got_ids - csv_ids.keys()
        order_ok = got_order == csv_order
        ok = (not missing and not extra and order_ok
              and tgt_bad == 0 and nonfinite == 0
              and sha_bad == 0 and meta_bad == 0 and guard_bad == 0
              and runtime_bad == 0 and struct_bad == 0
              and n == P.EXPECTED_WINDOWS[ds] == n_csv)
        all_ok &= ok
        total += n
        rep["subsets"][ds] = {
            "windows_found": n,
            "expected": P.EXPECTED_WINDOWS[ds],
            "csv_rows": n_csv,
            "missing": sorted(missing)[:10], "n_missing": len(missing),
            "extra": sorted(extra)[:10], "n_extra": len(extra),
            "identity_order_matches_csv": bool(order_ok),
            "target_bitwise_mismatches": tgt_bad,
            "pred_nonfinite": nonfinite,
            "payload_sha_failures": sha_bad,
            "done_metadata_mismatches": meta_bad,
            "core_guard_mismatches": guard_bad,
            "runtime_binding_mismatches": runtime_bad,
            "field_shape_length_failures": struct_bad,
            "bad_shards": bad_shards[:20],
            "ok": bool(ok),
        }
        print(f"[verify] {ds}: n={n}/{P.EXPECTED_WINDOWS[ds]} "
              f"missing={len(missing)} extra={len(extra)} "
              f"order_ok={order_ok} tgt_mismatch={tgt_bad} "
              f"nonfinite={nonfinite} sha_fail={sha_bad} "
              f"meta={meta_bad} guards={guard_bad} runtime={runtime_bad} "
              f"struct={struct_bad} -> {'OK' if ok else 'FAIL'}")
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
