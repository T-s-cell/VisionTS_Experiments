# VisionTS environment-drift diagnostic (diag_cgtsf_visionts_env_v1)

One question: does re-running the FROZEN VisionTS EXP-016 pipeline
(commit `0932a5ac841ea399271ce88bde5b607ee0bcb831`, protocol token
`a4c01683793a`) in the Aurora final-run environment (dt: RTX 5060 Ti /
WSL2, py3.10.20, torch 2.13.0+cu129) explain the "Aurora MSE lower /
MAE higher" pattern of Aurora EXP-017 (Aurora_Experiments
`repro_cgtsf_zeroshot_v1`, commit `3d20862a35fce5854ab2bd2bc6b0e849bd67da57`)?

- The model pipeline is byte-identical to EXP-016: MAE-base fully frozen,
  eval(), seed=2021, **batch=1 FIXED** (no batch probe), FP32, fp64=False,
  TF32/AMP off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, norm=align=0.4,
  bilinear, history-only, all-zero windows kept, identity
  (dataset, Idx, Date), CSV row order = window order. The 5 visionts/
  source files are pinned to EXP-016's own md5s (any drift = hard stop).
- The environment is the ONLY intended variable. It runs in a CLONE of
  dt's `torch_50_series` env (`diag_vts_env`; original env untouched)
  with VisionTS's own deps pinned to EXP-016 versions (timm==0.9.16,
  einops==0.8.1, pillow==10.4.0); torch/torchvision/numpy stay the
  Aurora-run stack. If the EXP-016 pandas==2.0.3 pin is incompatible with
  numpy 2.x, a fixed numpy-2-compatible pandas is installed and recorded
  (pandas is not in the forward numeric path); env_report.py verifies
  pandas parses the CSVs identically to the stdlib loader.
- References are frozen and verified BEFORE any GPU work (verify_refs.py,
  rtol=atol=1e-12 double tolerance): EXP-016 predictions/metrics
  (visionts_ref/) and Aurora EXP-017 per-seed predictions/metrics
  (aurora_ref/). Aurora is NOT re-run.
- Execution: 230-window preflight (shape/finite/same-env bitwise repeat/
  weights before-after/resume drill tier-1+tier-2) -> on PASS the full
  33,693-window run starts automatically (PTF->LEU->MSPG, 512-shards,
  resumable, NaN/Inf hard stop). Cross-environment bitwise equality vs
  EXP-016 is NOT expected and NOT a gate — it is measured, not blocked on.
- Scoring: one float64 ladder (point->window->sequence-equal-weight,
  original units) for three groups — VTS-old (EXP-016 frozen), VTS-new
  (this run), Aurora (EXP-017 per seed -> mean+-std(ddof=1)). Per subset:
  env_drift = VTSnew−VTSold; same_env_gap = Aurora_mean−VTSnew;
  orig_gap = Aurora_mean−VTSold; drift/orig-gap ratio; sign verdicts;
  VTS new-vs-old prediction diff (mean/max abs + argmax window).
  Conclusion scope: whether THIS set of environment changes explains the
  current model gap — no per-component (e.g. GPU-only) attribution.

## Run order (dt, via wrapper_dt.sh)

1. `python freeze_ref.py` (local; builds visionts_ref/ + aurora_ref/)
2. `python protocol.py freeze` (local; approval token = sha256[:12])
3. deploy to dt: `~/VisionTS_Experiments/{visionts/, diag_cgtsf_visionts_env_v1/, ckpt/}`
4. `./wrapper_dt.sh env_report.py` (env diff + pandas parse check; CPU only)
5. `./wrapper_dt.sh verify_refs.py` (BOTH references; MUST pass before GPU)
6. `nohup bash watch_and_run_dt.sh <token> > logs/watcher_nohup.log 2>&1 &`
   (waits for a free GPU, preflight 230 windows, auto full run on PASS)
7. pull back predictions/ results/ logs/ -> verify_cache.py -> evaluate.py
   -> recompute_check.py (1e-12) -> audit_pack.py --stage final -> temp/

No model/package source is modified; the sibling EXP-016 tree and the
Aurora EXP-017 tree are read-only sources for the frozen references.

## Archival notes (2026-10-11, user-approved freeze)

- Protocol token `68975e5ec59d`; full run on dt (RTX 5060 Ti / WSL2,
  cloned env `diag_vts_env`): 33,693 windows in 4.0 min (batch=1 fixed,
  0 quarantined); all gates PASS (verify_refs 0.0 pre-GPU, verify_cache
  33,693/33,693, recompute max diff 0.0).
- Headline: environment drift (VTS-new − VTS-old) is ~1e-8 relative on
  every subset/metric — at most ~0.000101% of the original Aurora−VisionTS
  gap. After aligning the environment, Aurora keeps lower MSE AND higher
  MAE on all three subsets; the MSE↓MAE↑ pattern is real model behavior,
  not an environment artifact (conclusion scoped to this environment set
  and the frozen config).
- Recording caveat: in `env_report.json` some environment variables come
  from the report-collection process; the ACTUAL inference settings are
  those recorded by the preflight report, `logs/infer_summary.json`, and
  the per-shard guard records — the three agree. This is a recording-
  provenance note only; it does not affect the frozen results.
- Final audit pack:
  temp/VisionTS_CGTSF_envdiag_final_20261011_1943.tar.gz (279 files,
  sha256 daddb69198b6d1050e513509fe817d9877ac03ded1dd2a5142776667c11c31e7).
