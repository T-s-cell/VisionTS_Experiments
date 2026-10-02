# repro_zeroshot_ext — VisionTS zero-shot on TimesX

Independent extension directory. Touches nothing outside `repro_zeroshot_ext/`;
imports `visionts.*` read-only. Safe to run while the LN full-shot campaign is live.

## Protocol (frozen 2026-09-27, see splits.json)

- Task: 96-step history -> 12-step forecast, one TimesX variable at a time.
- Split per variable (sorted by prediction-start timestamp, idx tie-break):
  test = last ceil(N/5); val = most recent pre-test samples whose 12-step target
  window does not overlap any test target window, capped at 32; fallback to fixed
  P=1 if val < 8 (none trigger in this data).
- calib_std = std(ddof=0) of the first sample's 96 past values; used ONLY as the
  standardization denominator when a window's own std < 1e-8. pred/true are never
  modified; if calib_std is also degenerate the standardized metric is flagged
  undefined and the raw-scale error is kept.
- P candidates: CommodityPrice+Currency (daily) {1,5}; SearchTrend (weekly) {1,52}.
  Selection: argmin val standardized MSE, tie -> P=1.
- Model: mae_base, original ImageNet weights, r(norm_const)=c(align_const)=0.4,
  224/patch16, grey->3ch, bilinear, FP32, eval, TF32 off. Explicit in
  record_fingerprint.RUN_CONFIG.
- Metrics: MSE/MAE; sample -> variable -> domain (equal weight) -> overall
  (equal weight over 19 domains). Raw scale reported at variable level only.
  Protocol name: `TimesX-ext-protocol` — do not compare with tables built
  under other protocols.

## Leakage prevention is enforced by execution order

`splits.json` frozen first -> `infer.py` caches forward predictions for val+test
rows -> `select_period.py` reads val rows only and writes `P_selection.json`
(with fingerprints of splits/zip/code/ckpt/config) -> `evaluate_test.py` verifies
that linkage, then scores test rows for exactly two configs (fixed P=1 and the
selected P). Candidate-wise test rankings are never produced.

### Fingerprint chain (cache -> selection -> test)

- `cache/manifest.json` binds run_config, zip/splits/ckpt sha256 and the hash of
  the forward code subset (`timesx_data.py`, `infer.py`, `record_fingerprint.py`).
  Any change there invalidates cached forwards: `infer.py` refuses to mix old and
  new manifests (archive the old cache dir, then rebuild).
- `select_period.py` first verifies the cache manifest against current state and
  checks all (var,P) entries are present, then writes `P_selection.json` with the
  full fingerprint plus a hash of the selection/scoring code subset
  (`select_period.py`, `metrics_util.py`).
- `evaluate_test.py` verifies BOTH links: the cache manifest AND the
  P_selection linkage including selection-code hashes; a mere existence check of
  `P_selection.json` is never accepted.
- Frozen-file protection: `build_splits.py` refuses to overwrite an existing
  `splits.json` without `--force`; `run_all.sh freeze` skips when `splits.json`
  exists (override with `FORCE_FREEZE=1`); `select_period.py` refuses to
  overwrite a P_selection whose recorded bindings changed without `--force`
  (legacy files lacking `code_md5_selection` may regenerate once).
- `infer.py` logs `forwards_executed`, which equals the number of saved rows
  (exactly one forward per (var,P,sample); resume runs report 0).

## Stages

```bash
bash run_all.sh freeze      # recompute + verify + freeze splits.json
bash run_all.sh infer       # GPU; PAUSE sentinel + free-VRAM gate (>=9000 MiB)
bash run_all.sh select      # val-only P selection -> P_selection.json
bash run_all.sh test        # linkage-checked test metrics
bash run_all.sh aggregate   # domain/overall tables + summary.md
bash run_all.sh all         # everything in order
bash audit_ext.sh           # isolation check + tarball to /home/wlt/MMTS/temp/
```

Useful flags (stage scripts directly): `infer.py --only-var <substr> --only-p 1 5
--device cpu` for smoke runs.

## Isolation contract

Never modify: `visionts/`, `long_term_tsf/` (incl. `repro_zeroshot/`),
`repro_fullshot/`, `scripts/`, `eval_gluonts/`, `setup.py`, `requirements.txt`,
repo `README.md`, `ckpt/`, the original benchmark CSVs, or
`/home/wlt/MMTS/TimesX_Datasets.zip` (read-only). `results/pre_snapshot.txt`
records git status + locked-file md5 at task start; `audit_ext.sh` re-verifies.
