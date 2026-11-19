# EXP-047h 预测距离 × 误差分解(零推理补充分析)— README

日期:2026-11-19
目录:`long_term_tsf/repro_t47_horizon_analysis/`(EXP-047 的零推理补充,不修改 EXP-047 任何文件)
分支:`repro/ltsf-t47-levelshift`(EXP-047 v2 冻结头 45b7e0c 之后新增,只新增本目录)

## 依赖

- **上游依赖:EXP-047 v2 资料包** `VisionTS_t47_levelshift_20261119_v2.tar.gz`
  (sha256 `46ad7eb1ca975060cf8d2ad4b9f8b45394f80cc87cc8ded130858bdf3fd86801`)。
  本轮零推理,全部输入为 EXP-047 冻结产物;13 个输入文件的 SHA256 见
  `results/input_shas_t47h.json`(compute 第一步记录,先于一切输入读取)。
- 输入根:`long_term_tsf/repro_t47_levelshift/`(`baseline_list_t47.csv`、`manifest_t47.csv`、
  `save_t47_levelshift/baseline_<DS>.npz`、`save_t47_levelshift/pred_<m>_t47.npz`、
  `per_sample_metrics_t47.csv`、`pairs_t47.csv`、`method_diff_t47.csv`)。
- 环境:CPU;eta `t46diag`(`/dev_data/wlt/conda/envs/t46diag/bin/python`,numpy+matplotlib);
  本地 python3 仅 py_compile。

## 内容与目的

检查 H=96 全程平均是否掩盖局部问题,并区分三种不能混同的量:
①向旧水平牵引更多(D_diff)、②历史冲突误差增量更大(cost_diff)、③最终预测误差更大(gap_B)。
探索性重分析,不预设方向;不重新推理/训练/筛选窗口/加大幅度或持续时间。

- 区间:b1=[0,24)…b4=[72,96) 为同一 H=96 预测的切片(**非重跑 H=24**);
  full=[0,96) 仅作原结果核对与单列,不作第五时段。ETTh 每 band=1P、ETTm 每 band=0.25P。
- 逐窗口口径:pred f32→f64;真值 y′ = baseline npz f64 y + manifest δ(npz true_p 仅核对 ≤1e-5,
  实测 max 2.38e-07);A/B 按 (sid_t46, delta_idx, m_idx) 配对;有效性沿用冻结 pairs ok 标志;
  分解恒等式 MSE=level+centered、cost=cost_level+cost_centered(V4 实测残差 1.78e-14)。
- 汇总一律逐配对差值的 median/mean/正比例(**禁止两组中位数之差**);无显著性检验,线索级。
- centered_error 不解释为旧形状依赖;A 组只称**参照条件**;不归因 MAE。

## 文件清单

| 文件 | 作用 |
|---|---|
| `protocol_t47h.md` | 完整协议(输入/区间/公式/汇总口径/V1–V7/纪律/复现命令) |
| `t47h_common.py` | 常量、输入校验、SHA 记录、manifest/pairs 解析 |
| `compute_t47h.py` | 对齐核验 + 逐方法×配对×区间指标与分解 + 逐对照差值 |
| `aggregate_t47h.py` | 5 张汇总表(主表/ds×var/δ 方向/|δ|/s 幅度/方法自比) |
| `figures_t47h.py` | 三主图(D_diff/cost_diff/gap_B 随区间)+ fig_data npz |
| `verify_t47h.py` | V1–V7 门禁(atol=1e-9, rtol=1e-10,fail 即停) |
| `gen_report_numbers_t47h.py` | 五问数字与一页结论(全部程序生成,禁止手抄) |
| `results/` | 全部产物(见下) |

## 复现命令(eta,CPU)

```bash
PY=/dev_data/wlt/conda/envs/t46diag/bin/python
T47=/dev_data/wlt/VisionTS_Experiments/long_term_tsf/repro_t47_levelshift
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf/repro_t47_horizon_analysis
$PY compute_t47h.py    $T47 results   # ~37s,记录 input_shas
$PY aggregate_t47h.py  results
$PY figures_t47h.py    results
$PY verify_t47h.py     $T47 results   # V1–V7 ALL PASS 才继续
$PY gen_report_numbers_t47h.py results
```

注意:代码文件改动后 V7(代码 SHA 门禁)会 FAIL,须按上序重跑全链再 verify。

## 产物(results/)

- 逐配对:`pairs_horizon_t47h.csv`(82,320 行)、`method_diff_horizon_t47h.csv`(55,440 行,含失效 nan 行)
- 汇总:`table_main/var/direction/amplitude/method_t47h.csv`(150/1050/300/300/210 行)
- 图:`fig_Ddiff_band_t47h.png`、`fig_costdiff_band_t47h.png`、`fig_gapB_band_t47h.png`、`fig_data_t47h.npz`(219 数组)
- 核验与元数据:`verify_t47h.json`(all_pass=true)、`compute_meta_t47h.json`、`aggregate_meta_t47h.json`、`input_shas_t47h.json`
- 结论:`report_t47h.md`(一页,五问)、`report_numbers_t47h.json`

## V1–V7 结果(2026-11-19,eta 全链重跑后)

| 项 | 内容 | 结果 |
|---|---|---|
| V7 | 13 输入 SHA + 代码 SHA 现状比对 | PASS(13/13 match) |
| V1 | npz↔manifest↔冻结 pairs 对齐;true_p ≤1e-5;δ↔delta_mult·s | PASS(max_true=2.38e-07, max_dres=4.97e-12;ok 对 vts/moirai/autoets 5376、autoarima 336) |
| V2 | 独立重算 D/cost/MSE vs 冻结 CSV 与本轮 CSV | PASS |
| V3 | full = 四 band 算术平均 | PASS(max 7.11e-15) |
| V4 | MSE=level+centered、cost=cost_level+cost_centered | PASS(max 1.78e-14) |
| V5 | cost_diff = gap_B−gap_A 逐行;gap vs 重算 | PASS |
| V6 | 五张表 + fig_data 独立复算 | PASS(max_stat=5.00e-12, max_fig=0) |

## 一页结论摘要(详见 report_t47h.md)

- 背景:D(牵引)随预测距离**增大**,Moirai 增长更快(ETTh1 m=1 b1→b4:VTS 0.278→0.489、Moirai 0.195→0.545)→ m=1 早期(b1)VTS 牵引更强、后期(b3/b4)Moirai 更强。
- Q1 早期 vs 全程不同:ETTh1 m=1 符号翻转(b1 +0.106/72% → full −0.005/48%);ETTm2 m=1 大幅衰减(b1 +0.132 → full +0.037,b4 −0.031);ETTh2 m=1 全程反向。
- Q2 同时正仅 13/60 格;cost_only 15 格(含 ETTh2 m=1 全部 5 格)= 牵引不更强但代价更大的反例。
- Q3 误差增量主要是水平项(bias²):|cost_level_diff|≥|cost_centered_diff| 的格 55/60。
- Q4 口径区分:无条件 full gap_B 中位前三 ds 为负(VTS 优势来自参照条件 A 领先);但 cost_diff>0 的配对子集内 gap_B 中位转正(+0.06/+0.01/+0.16,gap_B<0 仅 11–46%)。
- Q5 方向 down>up、幅度 1.0>0.5;变量级反例 = ETTh2 全部 7 var;AutoETS 对照 D_diff 几乎处处正;AutoARIMA 仅原子集单独汇报。

以上均为线索级(构造变体非独立样本、无显著性检验),四种解读的对应与限定见 report_t47h.md。
