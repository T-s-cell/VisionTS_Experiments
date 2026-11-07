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

全链一次通过(2026-11-07,eta;12/12 任务 SUCCESS,共用轨迹):

- **verify TG1-TG10:1093/1093 ALL PASS**(TG2 同环境对照轮
  ctrl-vs-main 逐位轨迹等价在冒烟硬门 59/59 已证;TG4 true.npy 与
  EXP-003 逐位一致;TG5 f64 复算 ≤1e-6;TG6 144/144 状态机重放 +
  锁定 ckpt 重算;TG7 锁定后不变;TG9 lr 双记录;TG10 134/134)。
- **总体(4-H 等权,3 种子)**:A 1.8783±0.0010 / B 1.8796±0.0042 /
  C 1.8775±0.0061(论文 1.899;EXP-003 归档 1.8778 为 **12 任务 =
  4H×3种子均值**,非 seed2021 单独值 1.8762)。相对 A:B +0.0667% /
  C −0.0412%;**三规则最大两两差 0.108%(B vs C)**;T3 方向一致性
  均为 NO —— **未发现一致缩小论文差距的效果**(受控诊断观察,
  不作归因)。各规则实际停止(trigger)轮次见 RESULT T4 新列。
- **B(官方等权批均值)在 H24 偏离最大**:best epoch 2/3/5(A 为
  15/14/6),seed2021 test MSE 2.0273 vs A 1.9852(+2.1%);H60 则
  反向(B 偏好晚期 epoch 16-19,MSE 略优于 A −0.9%)。H36/48 A=B
  逐位同(同轨迹同选择 → 同一锁定 ckpt)。
- **预算**:12 任务总训练墙钟 43.0 min(H24 301 s、H36 114 s、
  H48 119 s、H60 325 s 每任务均值;36/36 规则单元全部触发锁定,
  实际锁定轮次范围 E5–E22,逐任务逐规则见 RESULT T4)。
- 曲线/图表:`logs/summary_t40.csv`(36 行)、
  `logs/fig_t40_valstop.png`、RESULT 全表;theta 归档曲线仅参考
  (TG3 并排,无阈值)。

## Delivery

- 包(当前版 v3):`temp/temp_t40_pack.tar.gz`(393,571 bytes,<50MB)
  + 同名 sidecar `.sha256` =
  `708d95632509d3b690590b52f5ce77d7f625b853b0287d9cd9c1e41f3f2ac877`
  (**包 sha 只记 sidecar 与仓库交付记录,不入包内 README**;v3 为纯
  文档重打包:README 预算行锁定轮次口径修正为实际的 E5–E22,全部
  实验工件与 v2 相同,代码冻结头不变)。
- 包内容:9 代码文件 + README + runner、logs(audit/verify/RESULT/
  csv/图/recompute)、job_records(12 任务 × curves/trajectory/phases/
  lock_events/rng_audit/args/fingerprint/SUCCESS)+ **job_records_ctrl
  (A-only 对照轮同款 8 件,TG2 证据)**;ckpt 留 eta 不入包。
- 包历史:v1 sha `aff4b4be…` 作废;v2(评审修正版)sha
  `869cacb50a354fa1db0f1f368f53a74b61de87d8b61de58ea14c54f8eff4bcc0`
  (392,729 bytes)被 v3 取代。
- 代码冻结头(完整 40 位):`7e7fa5b9146fd2ebca0170c5f4842f13f2e6cb88`
  (分支 `repro/ltsf-t40-t41`;其后均为文档提交,包内 README 滞后
  终版交付记录提交,属已知口径)。
