# EXP-047 补充分析协议:预测距离与误差分解(零推理)

**性质**:探索性重分析,不预设结果方向;不重新推理/训练/筛选窗口/加大幅度或持续时间。
**代码版本**:分支 `repro/ltsf-t47-levelshift` HEAD 45b7e0c(v2 分析修订后);本目录代码文件
SHA256 见 `input_shas_t47h.json` 的 `code` 块(compute 前记录)。
**依赖**:EXP-047 v2 审计包 `temp/VisionTS_t47_levelshift_20261119_v2.tar.gz`
(sha256 `46ad7eb1ca975060cf8d2ad4b9f8b45394f80cc87cc8ded130858bdf3fd86801`)内的冻结产物;
输入根目录 `repro_t47_levelshift/`,全部只读。

## 一、输入(13 个文件,SHA256 见 input_shas_t47h.json,执行前后必须一致)

baseline_list_t47.csv、manifest_t47.csv(7168 行)、save_t47_levelshift/baseline_<DS>.npz
(f64 x/y/s)、save_t47_levelshift/pred_<method>_t47.npz(4 方法)、
per_sample_metrics_t47.csv、pairs_t47.csv、method_diff_t47.csv。
主比较 VTS vs Moirai;辅助 VTS vs AutoETS;AutoARIMA 仅原有子集(ETTh1/h2 各 14 基准窗)
单独报告,不缩小主比较范围。

## 二、固定区间(同一次 H=96 预测的切片,非重新运行 H=24 模型)

| band | 步 | ETTh1/h2(P=24)周期比例 | ETTm1/m2(P=96)周期比例 |
|---|---|---|---|
| b1 | [0,24) | 1.0P | 0.25P |
| b2 | [24,48) | 1–2P | 0.25P |
| b3 | [48,72) | 2–3P | 0.25P |
| b4 | [72,96) | 3–4P | 0.25P |
| full | [0,96) | 4P | 1P |

full 仅作原结果核对与单列汇报,不作为"第五个时间段"进入主图。
m/P=1、2、4 分别报告;四个数据集分别报告,小时/分钟数据不混合解释。

## 三、逐窗口口径

真值 y′ = baseline npz 的 f64 y + manifest δ(npz f32 true_p 仅核对,不作指标真值);
预测 f32→f64。索引以 row_globals 建立,核对 item_ids=sid_t46、cond_ids;
A/B 按 (sid_t46, delta_idx, m_idx) 配对:A cond_id = delta_idx,B cond_id = 4+3·delta_idx+m_idx,
不依赖数组行序(尤其 AutoARIMA 子集)。有效性沿用冻结 pairs_t47.csv 的 ok_A/ok_B。

对每方法、每配对、每区间 I:
e = pred − y′;MSE = mean_I(e²);D = mean_I(pred_A−pred_B)/δ;cost = MSE_B − MSE_A;
bias = mean_I(e);level_error = bias²;centered_error = mean_I((e−bias)²);
cost_level = level_B − level_A;cost_centered = centered_B − centered_A。
恒等式:MSE = level + centered;cost = cost_level + cost_centered。
centered_error 称"去均值后的误差",**不能解释为旧形状依赖**。

## 四、配对差与汇总

同一 (sid, δ, m, 区间) 内:D_diff、cost_diff、gap_A = MSE_A_VTS−MSE_A_对照、gap_B、
cost_level_diff、cost_centered_diff;正值 = VTS 牵引更强/代价更大/误差更大。
恒等式 cost_diff = gap_B − gap_A;分解恒等式 cost_diff = cost_level_diff + cost_centered_diff
⇒ **均值严格可加、中位数不可加**——凡涉及分解相加的示例一律用同一配对集的均值,
并注明分解对象是方法间代价差。均值与中位数可能反向,一律并报。
两指标格点中位数同为正属**组统计**,须另报逐配对联合比例(两差值同时>0 的窗口占比),
不得据此宣称"多数窗口两指标同时成立"。
汇总一律基于逐配对差值(mean/median/正值比例),**禁止"两组中位数之差"**;
报 n_base(不同基准窗数)、n_pairs、coverage(有效配对/可能配对)。
主表:数据集 × m/P × 区间 × 对照方法;附表同口径加 ds×变量、δ 方向、|δ|/s 幅度
(方向/幅度附表仍按 ds 分列,不跨数据集合并)。保留负值与极端值,不截断、不删样本;
构造变体非独立样本,不做显著性检验。

## 五、核验(V1–V7,fail 即停,不放宽容差、不重跑模型)

V1 输入 ID/条件/覆盖率对齐;true_p vs 重建真值 max ≤1e-5。
V2 full-96 的 MSE/D/cost 与方法差复现冻结 CSV(pairs/method_diff/per_sample_metrics)。
V3 逐配对:full-96 MSE/D/cost = 四个 24 步区间算术平均。
V4 full-96 level/centered 直接重算(非四段平均);分解恒等式逐区间成立。
V5 cost_diff = gap_B − gap_A 逐配对成立。
V6 汇总表从逐配对 CSV 独立复算;图与汇总表同源。
V7 输入文件 SHA256 执行前后一致。
容差:atol=1e-9、rtol=1e-10(|a−b| ≤ atol + rtol·|b|),实测最大差记录于 verify_t47h.json。

## 六、纪律

A 组只称"参照条件",不自动称稳定场景;不归因 MAE;探索性发现不写成独立验证结论。
必须区分:代价更大但最终误差仍较低(原有优势缩小)/ 牵引不更强但代价更大(不能解释为
更依赖旧水平)/ 仅个别数据集成立(条件性线索)/ 无稳定差异(不支持"全程平均掩盖问题")。

## 七、复现命令(eta,CPU,t46diag env)

```bash
PY=/dev_data/wlt/conda/envs/t46diag/bin/python
T47=/dev_data/wlt/VisionTS_Experiments/long_term_tsf/repro_t47_levelshift
OUT=/dev_data/wlt/VisionTS_Experiments/long_term_tsf/repro_t47_horizon_analysis/results
$PY compute_t47h.py $T47 $OUT          # 对齐核验 + 逐配对逐区间 CSV + 输入 SHA 记录
$PY aggregate_t47h.py $OUT             # 主表/附表/方法表
$PY figures_t47h.py $OUT               # 3 主图 + fig_data
$PY verify_t47h.py $T47 $OUT           # V1–V7
$PY gen_report_numbers_t47h.py $OUT    # 报告数字 + 一页结论(数字程序生成)
```
