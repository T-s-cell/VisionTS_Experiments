# EXP-040: Illness validation-rule / early-stop mechanism diagnostic

Question: does the EXP-003 Illness paper gap relate to the validation /
model-selection / early-stopping mechanism?  The EXP-003 recipe is
frozen unchanged (LN finetune, seq_len=104, label_len=48, Adam 1e-4,
bs=256, patience=3, lradj=type1, max 100 epochs); ONLY the validation
rule / model selection differs.  **Approaching paper values is NOT an
acceptance criterion** — this is a controlled diagnostic and does not
claim to recover undisclosed author details.

## Three validation rules on the SAME shared trajectory

| rule | definition | provenance |
|---|---|---|
| A | full fixed val set (518/434/350/266 samples), element-weighted MSE | EXP-003-consistent (`repro_fullshot/exp_univariate.py` `Exp_Univariate.vali`) |
| B | same full fixed set, equal-weight per-batch criterion means (partial last batch gets 1/n_batches weight) | official base `Exp_Long_Term_Forecast.vali` semantics (distortion probe) |
| C | per-epoch permutation + drop_last truncation to floor(N/256) full batches + equal-weight batch means | verified official rule (`data_factory.py:30-43`); dedicated `np.random.RandomState` — simulates the official selection rule, NOT its exact RNG interleaving (documented) |

12 tasks = H{24,36,48,60} x seeds{2021,2022,2023}; main round
`save_t40_valstop/<H>_seed<s>/` carries A,B,C; a same-environment
control round `save_t40_valstop_ctrl/<H>_seed<s>/` (A-only) on the
smoke task (24, 2021) provides the trajectory-equivalence hard gate.

## Why the extra validation cannot change training (structural + audited)

- args replayed from each config's OWN EXP-003 archived args.json
  (`save_fullshot/illness_<H>_seed<s>/args.json`); allowed delta keys
  are EXACTLY {model_id, save_dir}; seeding identical to
  `run_train.py` (random -> torch.manual_seed -> np.random.seed) BEFORE
  model construction.
- train loader via `Exp_Univariate._get_data('train')` zero-change
  (shuffle=True, drop_last=True, num_workers=10, generator=None) — the
  official global-RNG draw pattern per epoch.
- A/B share ONE forward pass over one DataLoader carrying an explicit
  dedicated `torch.Generator` (num_workers=0, sequential) — iterator
  base_seed draws touch that generator, never the global streams.
  A accumulates SSE/cnt (exactly the audited A rule); B records the
  per-batch criterion means (official CPU-offload + criterion form).
- rule C uses manual `default_collate` over the same dataset object —
  zero DataLoader RNG surface; the per-epoch sample indices are SAVED
  (`curves.npz c_idx_e<k>`) so verification replays the best epoch's
  SAME batches (never re-randomized).
- four-state RNG (python/numpy/torch-CPU/torch-CUDA) hashes are taken
  around every hook and hard-asserted unchanged in-process; the whole
  audit is re-checked by verify TG1.
- early stopping uses the REAL `tools.EarlyStopping` (patience=3,
  delta=0; ties count as improvement and re-save) with one clone per
  rule saving into `locks/<rule>/checkpoint.pth` + `valid_loss.json`;
  a locked rule receives no further calls (selection frozen).  lr
  updates use the official `adjust_learning_rate(model_optim, epoch+1,
  args)` AFTER each epoch; the loop breaks only when ALL enabled rules
  are locked (that break epoch gets no lr update — mirrors base).
  Per-epoch BOTH `train_lr_used` (epoch 1 = 1e-4, epoch 2 = 1e-4,
  epoch E>=2 = 1e-4*0.5^(E-2)) and `lr_after_update`
  (1e-4*0.5^(E-1)) are recorded and asserted (TG9).

## Files

| File | Role |
|---|---|
| `t40_common.py` | constants: 12 tasks (assert), SMOKE (24,2021)/QUEUE(11), rule defs, EXP-003/theta-log resolution, csv/ckpt sha pins, val-coverage pins, sha256-derived dedicated-stream seeds, lr formulas |
| `t40_audit.py` | step-1 audit, gates AG1-AG9 over the 12 EXP-003 archives + theta logs (AG7 reloads archived best ckpts and recomputes test metrics, rel<=1e-3, true bitwise); STOPs the line if materials are missing; writes `logs/audit_t40.json` |
| `t40_train_driver.py` | one (H, seed) per process, shared trajectory, rules by round; order-recording batch-sampler proxy (no RNG change); per-phase timing; artifacts: curves.npz / trajectory.json / phases.json / rng_audit.json / lock_events.json / fingerprint / args / SUCCESS |
| `verify_t40_fullshot.py` | gates TG1-TG10 (`--smoke` = hard smoke gate on (24,2021): TG1+TG2+TG6-10); f64 recompute 36 units; state-machine replay (real tie semantics, never first-argmin); locked-ckpt val recompute (rule C on the SAVED best-epoch indices); lr dual-record assertions; TG3 theta curves reference-only |
| `collect_t40.py` | `summary_t40.csv` (36 rows) + `RESULT_t40_illness_valstop.md` (T1 per-H / T2 overall / T3 paired diffs / T4 selection behaviour / T5 reference curves / budget) + figure |
| `run_t40_all.sh` | stages: preflight -> audit -> smoke (ctrl round + main round + `verify --smoke` hard gate) -> queue (11, skip-if-done) -> verify -> collect -> pack (<50MB, no ckpt/npy) |

## Gates

AG1 archive completeness (STOP if missing) / AG2 archived code_md5 ==
current 12-file tree / AG3 illness csv sha / AG4 frozen ckpt sha /
AG5 strict torch.load / AG6 coverage arithmetic (518/434/350/266;
official drops 6/178/94/10; anchor 178==434-256) / AG7 archived-ckpt
reload -> test metrics rel<=1e-3 + true.npy bitwise / AG8 theta-log
parse (reference) / AG9 tie-semantics replay vs archived best epoch.

TG1 four-state RNG audit 100% / TG2 **same-environment ctrl-vs-main
trajectory equality** (identical init digest; per common epoch
identical sampler-order hash AND identical post-epoch state digest;
common prefix >= 2; stop epochs may differ) / TG3 theta curves
side-by-side reference (soft) / TG4 true.npy bitwise == EXP-003
(36 units) / TG5 f64 recompute <= 1e-6 + finiteness (36) / TG6
state-machine replay == recorded events + locked-ckpt val recompute
<=1e-5 (C uses saved indices) / TG7 post-lock invariance (no events
after trigger; counter frozen; file sha + test-reload digest ==
recorded) / TG8 36-unit coverage / TG9 lr dual record (both formulas,
NaN only on the all-locked break epoch) / TG10 re-entry completeness
(read-only).

Deterministic-kernel mode (empirical fix, recorded in every
fingerprint/env): stock CUDA kernels are run-to-run nondeterministic
(atomic backward kernels), which broke TG2's BITWISE state-digest
equality between the two rounds.  Both rounds therefore pin
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic=True`,
`cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`.
Training math is unchanged; kernel selection is deterministic so the
ctrl-vs-main comparison is a true isolation proof.  This is an
eta-environment execution setting, not a recipe change (EXP-003 ran on
theta without these flags; agreement with EXP-003 is proven by AG7 and
the A-rule test agreement, not by bitwise trajectory equality).

## Budget

EXP-003 illness epochs are fast (archived runs early-stopped at epoch
6-17 of 100; patience=3).  Estimated 30-60 s/epoch x ~6-20 epochs x 12
tasks + 1 ctrl round ≈ **2-6 h wall**; refresh from smoke measurements.
Lock copies: 12 tasks x 3 rules x ~0.46 GB ≈ 17 GB (kept as the
per-rule selected models; the smoke ctrl round keeps 1 x 0.46 GB).

## Environment

eta L20, conda env `visionts` (`/dev_data/wlt/conda/envs/visionts/bin/
python`, torch 2.12.0+cu126), GPU_ID=0, repo checkout
`/dev_data/wlt/VisionTS_Experiments`, CWD contract: scripts chdir to
`long_term_tsf/` (same as EXP-003).  Branch: `repro/ltsf-t40-t41`
based on the EXP-039 closing head.  Commit SHA: recorded after the
smoke gate passes (docs-only commits afterwards if any).

## Results

(filled after queue/verify/collect)

## Delivery

(pack + sidecar sha recorded after collect; sha lives ONLY in the
sidecar and this repo's delivery record, never in the in-pack README)
