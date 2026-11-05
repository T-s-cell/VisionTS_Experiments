# EXP-039: VisionTS paper Table 22 B-line across three seeds

Continuation of the frozen EXP-038 (`8ca98a0`, branch
`repro/ltsf-t38-t15t22`): adds All / Rand-VM (full finetune) at seeds
**2022 and 2023** — 64 new trainings (16 configs x 2 seeds x 2 arms) —
to test the cross-seed stability of the EXP-038 single-seed findings
(pretrained benefit, LN advantage, gaps vs paper). Seed 2021 is
**reused read-only** from the EXP-038 archive (32 groups, incl. the
frozen EXP-037 ETTh1_96 pair).

## Scope and reuse (frozen provenance)

| Column | Source | Seed handling |
|---|---|---|
| All (ImageNet init, full finetune) | this round (seeds 2022/2023) + EXP-038 archive (seed 2021) | measured per seed |
| Rand-VM (all weights random init, same recipe) | same | measured per seed |
| LN | EXP-003 per-seed archives (`save_fullshot/<DS>_<H>_seed<s>/`), 48 pins (`t39_common.LN48`) | per-seed measured (2021 == t38 `LN16`) |
| Freeze | EXP-033 `summary_t22.csv` (`t38_common.load_freeze16`) | single frozen inference, seed-independent |

Base recipe inherited unchanged from EXP-003/EXP-038: Table 19 L/P
(ETTh1/2 1152/24, ETTm1 2304/96, ETTm2 1152/96), bilinear r=c=0.4,
FP32, Adam 1e-4, bs 256, 1 epoch, full-val MSE selection, strict
reload then test. **Aggregation convention**: per (ds, seed) 4H
equal-weight mean -> per-seed overall -> cross-seed mean ± std
(ddof=1). Paper Table 22 anchors only at the dataset-4H-mean and
overall layers (the paper prints 3-seed means; our LN aggregate is the
mean of the same three seeds 2021/2022/2023).

## Seed-injection mechanics (why the seeds are real)

- argv is rebuilt from each config's OWN EXP-003 audited `args.json`;
  the `seed` key is skipped in the replay and appended explicitly
  (`--seed <s>`); `run_train.py` seeds
  `random -> torch.manual_seed(CPU+all CUDA) -> np.random.seed`
  BEFORE any model construction.
- Driver gates (H1-H4 observation hooks, identical to EXP-037/038):
  All init bitwise == frozen ckpt per key; Rand init differs; same
  seed => identical trainable lists / entry RNG / epoch sampler order
  / first batch across arms; **different seeds => all four RNG
  sub-states, sampler order, first batch necessarily differ**
  (hard gates BG5/BG6/BG7 cross-seed).
- `fingerprint.seed`, `fingerprint.args.seed`, `args.json` delta and
  `SUCCESS.txt.seed` are all pinned to the requested seed (BG1_seed_pinned).
- Test loader is sequential => `true.npy` bitwise == EXP-003 at any
  seed (BG2).

## Resume semantics (first launch vs re-entry)

- An empty `save_t39_full/` is expected only on first launch (preflight
  reports the resume state, does not assume it).
- A task is SKIPPED only when ALL of: `SUCCESS.txt` (readable, mse/mae,
  seed), train json (`success.mse` match + `fingerprint.seed` pin +
  `requested_seed`), `results/_/pred.npy`, `results/_/true.npy`, and
  final `checkpoints/_/checkpoint.pth` are present.
- A partial job dir is moved to `save_t39_full/.failed/<name>_<ts>`
  (scene preserved) and re-trained.

## Files

| File | Role |
|---|---|
| `t39_common.py` | constants: SEEDS/TASKS(64 assert)/SMOKE/QUEUE(60), LN48 pins + overall anchor self-check, seed-aware paths, t38/t37 provenance paths |
| `t39_train_driver.py` | one training via `repro_fullshot/run_train.main()` zero-change; hooks H1-H4; per-phase timing + GPU peaks; record `logs/train_<DS>_<H>_<arm>_seed<s>.json` |
| `verify_t39_fullshot.py` | gates BG0-BG10 over 64 arms (`--smoke` = ETTh1_96 only -> `verify_t39_smoke.json`); f64 recompute 64 records; 3-layer aggregation mean±std(ddof=1) |
| `collect_t39_fullshot.py` | `summary_t39_ett.csv` (48 rows: 16 configs x 3 seeds), `RESULT_t39_ett.md` (T1 per-H / T2 dataset 4H / T3 overall / T4 paired diffs / T5 stability / budget / gates), `fig_t39_seed_effect.png`; `--selfcheck` |
| `run_t39_all.sh` | stages: preflight (GPU/disk/code md5/data sha/ckpt sha/LN48 archives/EXP-038 provenance/64-task assert) -> smoke (4 groups + smoke verify HARD gate + budget print) -> queue (60, skip-if-done) -> verify -> collect -> pack |

## Gates (verify_t39_fullshot.py)

BG0 presence / BG1 provenance + seed pins / BG2 truth bitwise ==
EXP-003 / BG3 f64 recompute <= 1e-6 + finiteness (64 records) / BG4
init semantics + trainable lists across all 3 seeds / BG5 RNG
same-seed equal, cross-seed distinct / BG6 sampler formula + same-seed
order equal, cross-seed distinct / BG7 first batch same rules / BG8
**6 final ckpts per config (4 new + 2 seed-2021 archived):
tensor-content digests pairwise distinct (15/15 pairs), each with >=1
vision_model key differing from the frozen ImageNet weights** / BG9
LN48 pins (48 archives) + `aggregate_snapshot.csv` mean/std(ddof=1)
cross-check <= 1e-6 + LN overall anchor 0.3323544589802623 /
0.36986022753020126 + Freeze16 coverage / BG10 EXP-038 provenance
(30 t38 jsons + 2 t37 reuse).

## Budget

Pure-train estimate from EXP-038 measured s/batch (train_wall −
vali_wall; ETTh1 1.092 / ETTh2 1.217 / ETTm1 1.217 / ETTm2 1.224
s/batch; batches hourly 202/199/195/185, ETTm1 879/876/872/862, ETTm2
910/908/904/893): ~11.7 h pure training for 64 arms; + vali/test
(ETTm ~290 s/run, hourly ~100 s/run) and startup ≈ **15–16 h total
wall clock**. Refreshed from the smoke runs' measurements at launch.

## Environment and reproducibility

- eta L20, conda env `visionts` (same env as EXP-038 B-line).
- Launch: `PYTHON_B=... GPU_ID=0 nohup bash repro_t39_t22seeds/run_t39_all.sh all > repro_t39_t22seeds/logs/all.out 2>&1 &`
- Frozen ckpt sha256 `c551c37b...`; 4 ETT CSV sha pinned via each
  config's EXP-003 fingerprint; 12 training-tree files md5-checked
  against EXP-003 in preflight.
- Commit SHA (this round): `<<FILL_AT_COMMIT>>` (branch
  `repro/ltsf-t39-t22seeds`, based on EXP-038 head `8ca98a0`).
- Big artifacts stay on eta: `long_term_tsf/save_t39_full/`
  (64 job dirs: ckpt/pred/true), `logs/train_*.json` evidence is in
  the pack; the pack carries NO npy/ckpt.

## Results

<<TO BE FILLED AFTER QUEUE+VERIFY+COLLECT>>

## Delivery

- Pack: `temp/VisionTS_t39_t22seeds_<date>.tar.gz` (<50MB) + `.sha256`
  sidecar. In-pack wording: the sha in any README inside the pack is
  the sidecar value quoted as provenance; the authoritative sha is the
  sidecar file itself and this repo README's delivery record.
- EXP-038 wording fix (user correction 2026-11-05) delivered together:
  "per-config" -> dataset-4H-mean framing; gap attribution -> "the
  cause remains undetermined; this round tests cross-seed stability".
  EXP-038 packs A/B are REPACKED after that fix (new sidecars, sha
  recorded in `repro_t38_t15t22/README_t38.md`).
