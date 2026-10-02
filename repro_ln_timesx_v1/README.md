# repro_ln_timesx_v1 — VisionTS TimesX LN fine-tuning (protocol timesx_ln_v1)

LN-only fine-tuning of VisionTS (mae_base, 96->12, P=1) on TimesX, 19 domains.
Methods: Z0 (fixed-P1 zero-shot on the NEW split), N0 (last-value),
F0 (LN + native train pool), F1 (LN + dense sliding windows), F2 (F1 + FreqMask),
F3 (F1 + LN anchor reg 0.01*sum((th-th0)^2)), F4 (F2 + F3).
F0-F4 x 19 domains x 3 seeds (2021/2022/2023) = 285 runs, each from pristine
pretrained weights. Only the 84 LayerNorm tensors (55,808 scalars) train.

## Stages (run_all.sh, fail-fast, each tees to logs/stage_<name>.log)

1. `prepare`    rebuild series/splits (hard-checks spec Appendix A), freeze
                manifests/{protocol,split_manifest,split_counts}+data_cache.npz,
                snapshot isolation baseline (audit/baseline.json)
2. `preflight`  data/LN/grad/cache/FreqMask/accum/budget/coverage checks +
                timing/VRAM measurement + real interrupt->resume test
3. `train`      serial 285-run queue (queue_train.py), domains ascending by
                dense count, PAUSE sentinel + free-VRAM gate per run
4. `freeze_selection` verify all 285 + freeze best_ft per run (val only)
5. `test`       Z0/N0 + every frozen run on the NEW 2,474-window test list
6. `aggregate`  window->var->domain->overall tables, 3-seed mean+-std
7. `audit`      baseline bidirectional diff + hash chain + 285-run recheck

## Usage

    PY=/path/to/python bash run_all.sh all          # full pipeline
    bash run_all.sh train                           # single stage
    touch PAUSE                                     # pause queue at next run

Stages must run in order; test only runs after freeze_selection (no test
metric is computed before model selection is frozen).

## Key invariants

- RNG re-derived from constant seeds every epoch (resume-safe, F1-F4 same-seed
  base sequences identical by construction; F2/F4 share the aug stream).
- Frozen backbone never wrapped in no_grad; only LN tensors carry gradients;
  frozen params verified bitwise unchanged after optimizer steps.
- Interrupted epochs are voided by attempt_id (consumers dedupe on max attempt);
  checkpoints/resume.pt written atomically (tmp + os.replace).
- Float equivalence checks use declared allclose tolerances
  (configs/v1.yaml preflight_tolerances) with recorded max error; frozen
  parameters must remain bitwise identical.
- Isolation: baseline locks static sources/data/weights/frozen results;
  co-tenant dynamic outputs are existence-tracked only.
  PYTHONDONTWRITEBYTECODE=1 everywhere (read-only reuse of sibling modules).

See VisionTS_TimesX_LN_Experiment_Plan_v1.md (repo root) for the frozen spec.
