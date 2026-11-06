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

Known incidents (all fixed BEFORE the queue launched; scene preserved
in `logs/FAILED.txt`): three early smoke-verify FAILs during bring-up —
driver import-order (fixed ac2bed6), BG4 trainable-list gate compared
unhashable lists via set() (fixed 38612b9), and the cross-seed gates
expected arm-differences within a seed, which is impossible by design
(arms share the seed's RNG stream; fixed d9687e4 by comparing per-seed
representatives, 3-way pairwise distinct). Final smoke after the fixes:
**147/147 gates pass (0 FAIL)** at 2026-11-05 14:24, then the queue
started (which itself added ZERO failures). The first full verify had
5 FAILs, all one class: seed-2021 archive metrics were not fed into
the aggregation summary (fixed d6a78e3); the re-run was 1095/1095.
The first pack shipped 0 of this round's train jsons (glob pattern
mismatch, fixed 56b7215 + hard count 64, b9ee00d). Post-review
hardening of the re-entry check (f7be1ed) initially mis-typed the
fingerprint provenance keys as strings — they are dict-valued maps —
so one re-entry test moved the four smoke dirs aside and retrained
them; the training is NOT bitwise deterministic across runs
(differed ~1e-5), so the ORIGINAL archived evidence was restored
verbatim (dirs from `.failed/`, the overwritten smoke train jsons and
logs restored from the delivered pack, bitwise sha-checked), the
partial rerun scene is preserved under `save_t39_full/.failed/rerunA_*`,
and the check was fixed in 7f114e3 + 525427a. Final re-entry test:
**64 tasks considered, 64 skipped as complete** (strict checks pass on
all originals).

## Budget

Pure-train estimate from EXP-038 measured s/batch (train_wall −
vali_wall; ETTh1 1.092 / ETTh2 1.217 / ETTm1 1.217 / ETTm2 1.224
s/batch; batches hourly 202/199/195/185, ETTm1 879/876/872/862, ETTm2
910/908/904/893): ~11.7 h pure training for 64 arms; + vali/test
(ETTm ~290 s/run, hourly ~100 s/run) and startup ≈ **15–16 h total
wall clock**. Smoke-measured refresh at launch (2026-11-05): ETTh1_96
this round 1.19–1.22 s/batch (in line with the estimate), remaining
queue 33,856 batches ≈ 11.3 h pure train.

## Environment and reproducibility

- eta L20, conda env `visionts` (same env as EXP-038 B-line,
  `/dev_data/wlt/conda/envs/visionts/bin/python`, torch 2.12.0+cu126).
- Actual commands used (from `long_term_tsf/` inside the repo checkout
  at `/dev_data/wlt/VisionTS_Experiments`):
  `PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python GPU_ID=0 bash repro_t39_t22seeds/run_t39_all.sh smoke`
  (2026-11-05 14:24, green), then the same with stage `queue` under
  `nohup` (output `logs/queue_all.out`); `verify`, `collect`, `pack`
  run after the queue drains.
- Frozen ckpt sha256 `c551c37b...`; 4 ETT CSV sha pinned via each
  config's EXP-003 fingerprint; 12 training-tree files md5-checked
  against EXP-003 in preflight.
- Commit SHA (this round): code-freeze head
  **`525427a2bda02a7fec5d2d55b4a46bb5ec90ea8f`** (branch
  `repro/ltsf-t39-t22seeds`, based on EXP-038 head `8ca98a0`; the
  results README is a docs-only commit on top).
- Big artifacts stay on eta: `long_term_tsf/save_t39_full/`
  (64 job dirs: ckpt/pred/true), `logs/train_*.json` evidence is in
  the pack; the pack carries NO npy/ckpt.

## Results

Gates: **1095/1095 PASS (0 FAIL)** over 64 arms (BG0-BG10) + LN48
triple anchor + Freeze16 + EXP-038 provenance; f64 recompute 64/64
coverage, all |d| <= 1e-6 (`logs/verify_t39.json`,
`logs/recompute_f64.json`).

Overall layer (4 datasets equal-weight; cross-seed mean ± std ddof=1,
n=3; paper Table 22 in parentheses):

| col | MSE | MAE |
|---|---|---|
| All | 0.4237±0.0040 (0.417) | 0.4163±0.0023 (0.414) |
| Rand-VM | 0.4733±0.0030 (0.417) | 0.4598±0.0027 (0.414) |
| LN | 0.3324±0.0006 (0.333) | 0.3699±0.0003 (0.369) |
| Freeze | 0.3595 (0.360) | 0.3750 (0.375) |

Key cross-seed findings (all direction-consistent 3/3 seeds; details
and per-seed values in `RESULT_t39_ett.md` T2-T5):

1. **Pretrained benefit is stable**: overall All − Rand per seed
   −10.00% / −11.08% / −10.40% (s2021/2022/2023); per dataset:
   ETTh1 −0.198±0.015, ETTh2 −0.018±0.003, ETTm2 −0.010±0.007 MSE,
   but **ETTm1 reversed in 3/3 seeds** (All worse than Rand
   +3.33% / +8.01% / +8.86%).
2. **LN advantage is stable**: LN − All overall per seed
   −22.37% / −21.05% / −21.23%; direction-consistent on all 4 datasets.
3. **EXP-038 anomalies persist across seeds**: ETTh1 Rand vs paper
   4H-mean deviations +0.208/+0.220/+0.193; ETTm1 All-worse-than-Rand
   3/3 seeds. Seed scatter is small (dataset-layer std <= 0.018 MSE),
   so the single-seed EXP-038 picture is cross-seed stable within
   these three seeds.
4. **Gaps vs paper remain undetermined**: All +0.0067, Rand +0.0563
   overall dMSE; the cause is not identified this round — these seeds
   only quantify its cross-seed stability.

Budget: measured pure-train 1.19–1.31 s/batch by dataset; queue of 60
ran 2026-11-05 14:26 → 2026-11-06 04:25 (~14 h wall).

## Delivery

- Pack: `temp/VisionTS_t39_t22seeds_20261106.tar.gz` (1,182,190 bytes,
  277 files: 64 train jsons, 96 LN evidence, 30 t38 + 2 t37 provenance
  jsons, verify/recompute/summary/RESULT/fig, code + README; the
  import-time deps `repro_t38_t15t22/t38_common.py` and
  `repro_t22_freeze_ln/t22_common.py` are packed at their canonical
  relative paths so the tree is importable standalone — verified) +
  `.sha256` sidecar, sha256
  **`b161a4b590e652f686209f1e43b0e82e327e988aa411f506f29c8851831a9863`**
  (double-ended verified). The authoritative sha is the sidecar file
  and this repo-side delivery record; an in-pack README lags the
  final docs commit by construction.
- Presentation caliber (per user review): Freeze is a single frozen
  inference and is shown as ONE value (no ±0.0000, labeled "single
  frozen inference"); in `summary_t39_ett.csv` the `<arm>_train_s`
  column is the RAW train_wall INCLUDING in-loop validation, and
  `<arm>_train_pure_s = train_wall − vali_wall` is the pure-training
  figure used in the report.
- Delivery also includes the REPACKED EXP-038 pair (round-3 wording):
  A `0e899c7d…`, B `6a27e298…` — recorded in
  `repro_t38_t15t22/README_t38.md`.
- EXP-038 wording fix (user correction 2026-11-05) delivered together:
  "per-config" -> dataset-4H-mean framing; gap attribution -> "the
  cause remains undetermined; this round tests cross-seed stability".
  EXP-038 packs A/B are REPACKED after that fix (new sidecars, sha
  recorded in `repro_t38_t15t22/README_t38.md`).

## Archival closing round (2026-11-07, numbers unchanged, no retraining)

Post-delivery review found the delivered CSV's `*_train_pure_s` columns
empty (root cause: `build_rows` computed `train_pure(row, arm)` before
`vali_s` was set; RESULT values were always correct — the report
recomputes on completed rows). Closing fixes (code head
**`512073d`**, this README is a docs-only commit on top):

1. `collect_t39_fullshot.py`: `vali_s` assigned before `train_pure_s`,
   with a row-level post-condition `pure == train_s − vali_s`.
2. New `check_t39_reentry.py` (single implementation behind
   `job_decide`): independent anchors vs the config's EXP-003
   fingerprint (code_md5 / data_sha256 / ckpt_sha256 dict maps),
   full config pin — whitelisted keys pinned to per-arm expected
   values (model_id=job, save_dir=abs(job dir), ft_type=full,
   vm_pretrained=1|0 by arm, gpu=0, seed=<s>), arrays ACTUALLY loaded
   (shape == EXP-003 archived, all-finite), checkpoint FILE sha256 ==
   the DELIVERED train json's `final_ckpt_sha256` FIRST, then
   torch.load(weights_only=False) + state_dict tensor digest vs a
   one-time accepted manifest (`accepted_ckpt_digests.json`,
   generated read-only, itself gated on the delivered file sha).
3. New READ-ONLY `check` stage: NEVER moves dirs, NEVER retrains (the
   archival verification deliberately does not enter the queue).

Evidence: `check` = **64/64 PASS** (`logs/reentry_check_t39.json`);
`verify` re-run = **1095/1095 (0 FAIL)**; `collect` re-run — RESULT
**byte-identical** to the delivered 20261106 pack (sha256
`6756d990…fbe87db0` matched before/after); regenerated CSV has 94/96
pure cells filled with 0 arithmetic mismatches — the 2 empty cells are
the reused ETTh1_96 seed-2021 pair (EXP-037/038 archive records carry
no timing data; source column says so).

Closing pack (supersedes the 20261106 pack; its sha `b161a4b5…` is
void from here on): `temp/VisionTS_t39_t22seeds_20261107.tar.gz`
(1,193,294 bytes, 281 files — adds check_t39_reentry.py,
accepted_ckpt_digests.json, reentry_check_t39.json) + `.sha256`
sidecar, sha256
**`e45dd9ce5c1cbc93fa7f033cfccb62c19d197fe7b31657828b3f9b9db0a22f8c`**
(double-ended verified). The pack sha lives ONLY in the sidecar and
this delivery record, never in the in-pack README (a pack cannot
contain its own hash).
