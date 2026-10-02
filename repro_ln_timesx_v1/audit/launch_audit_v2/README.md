# Launch audit v2 — post-fix re-verification (2026-09-29)

v1 audit was REJECTED by the user with mandatory fixes. This package is the
re-verification evidence AFTER the fixes, on eta (L20, execution host).

## Fixes applied since v1 (only inside repro_ln_timesx_v1/, protocol semantics frozen)
1. aggregate.py rewritten: strict merge (19 domains/190 vars, duplicate+
   coverage+non-finite hard errors), domain equal-weight, seed std ddof=1,
   Z0/N0 direct from own npz, fallback separate. Acceptance CPU simulation
   PASSES exactly: overall MSE=130, MAE=10, seed std=0, 190 var rows/method,
   F0-Z0 delta=129 (preflight K1-K7).
2. trainer.py resume boundaries: finalize path after final checkpoint
   (no UnboundLocalError), unique attempt per process start
   (max(jsonl_max, rs.attempt)+1), SUCCESS-skip returns full metadata.
   Verified on-GPU: preflight L1-L5 (first-epoch interrupt replay,
   same-epoch double interrupt attempts [0,1,2], crash-before-SUCCESS
   finalize, SUCCESS-skip metadata, stale-row voiding).
3. REAL hash refusal: actual_fingerprints() recomputes data_cache/
   split_manifest/zip/ckpt/code; trainer refuses on declared-vs-actual
   drift; resume.pt refuses on hash/version mismatch; queue gate checks
   preflight binding + protocol; evaluate_test checks protocol + per-run
   ckpt sha vs selection.json; audit re-checks full chain incl.
   split_manifest_sha256 and preflight binding; preflight M binds the
   report to recomputed fingerprints (embedded in preflight_report.json).
4. run_all.sh stage-arg leak fixed (shift); preflight F rebuilt: real
   4x8 (mb+reg)/4 vs 32 (mean+reg) with LN perturbed OFF INIT — anchor
   gradient nonzero verified (|grad_reg|=4.75e-2), max|dgrad| 4.5e-6
   within declared rtol 1e-4 / atol 1e-6.

## Procedure (per user directive)
- Fingerprints regenerated ON ETA via prepare_data.py --force (never
  copied from local). Local protocol.json is stale by design.
- GPU preflight v2 re-run after regeneration: ALL PASS 61/61
  (preflight_report.json, stage_preflight_v2.log).

## v1 evidence
Preserved locally: audit/launch_audit/ (v1) + temp tarball
VisionTS_ln_timesx_launch_audit_20260929.tar.gz. Not deleted or altered.

## Known cosmetic deviation (disclosed)
eta has a stray top-level v1.yaml, byte-identical (md5 f315d77c...) to
configs/v1.yaml, left over from an early rsync. It is included in the eta
code fingerprint (13 files vs 12 local). Left in place AFTER preflight
PASS so the verified binding stays valid; it is a duplicate of the frozen
config and changes no behavior.

## Contents
- preflight_report.json   61/61 ALL PASS, bound fingerprints, timing/VRAM
- stage_preflight_v2.log  full check log
- protocol_eta.json       authoritative protocol (regenerated on eta)
- consistency.json        dual-host data_cache array shas, split_manifest
                          content sha, code fingerprint comparison
- configs_v1.yaml         frozen config copy
- split_counts.csv        per-domain split counts (== spec table A)
- SHA256SUMS.txt          hashes of this package
