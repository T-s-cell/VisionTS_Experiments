# EXP-047 构造与指标公式(分析/推理前落盘冻结,2026-11-19)

目的:检验 VisionTS 是否被旧历史拉向旧水平;该倾向是否对应稳定场景优势与变化场景代价。只研究**水平变化**;不改周期形状、图像块、网络参数。线索级结论,不支持机制,不归因 MAE。

## 1. 基准窗口(输入历史选择,冻结于推理前)

- 源:EXP-046 `sample_list_t46.csv`(1792,start-major)+ `features_t46.csv`;
- 每 (ds×var):排除 near_constant(hist_std<1e-8)与非有限特征行,按 `recent_old` 升序,同值按 `sample_id` 升序,取前 **16** → 4×7×16 = **448**;
- 选择只用输入历史特征,不用未来与预测误差。

## 2. 成对构造(修改后的输入必须重新推理,不复用原预测)

设原始历史 `x`(L 步,标准化尺度,f64)、未来 `y`(96 步,f64)、`s = x.std(ddof=0)`(csv f64 重算,与 features_t46.csv hist_std 交叉核对 ≤1e-8):

- 水平变化量:`δ = s·d`,`d ∈ {-1, -0.5, +0.5, +1}`;
- 新水平已持续长度:`m = P·k`,`k ∈ {1, 2, 4}`;
- **A 组(参照)**:`x_A = x + δ`(整段;未额外加入水平跳变,不认定其未来一定稳定);
- **B 组(冲突)**:`x_B = x`,仅末 m 步 `+ δ`;
- 真值:`y' = y + δ`(A/B 共享);
- 恒等式:A 与 B 的末 m 步逐位相同、未来逐位相同;
- `cond_id`:A = `delta_idx`(0..3 ↔ d 序);B = `4 + 3*delta_idx + m_idx`(m_idx 0..2 ↔ k=1,2,4);`(sid_t46, cond_id)` 唯一;共 448×16 = **7168** 条件窗口;
- 行序:window-major then cond,`row_global = w_global*16 + cond_id`;
- δ=0 仅冒烟自检(两次实际推理验证一致性),不计入 16 条件。

## 3. 方法与随机控制

- 主对照:冻结 VisionTS(mae_base,norm_const=0.4/align_const=0.4,单变量 `[bs,L,1]` batch32)、冻结 Moirai(patch=32、num_samples=200、逐时刻中位数)、窗口内拟合 AutoETS(season_length=P);
- **Moirai 随机控制(按条件推理)**:每个 cond_id 单独推理,该条件内基准窗口顺序、batch 划分一致,推理前重置全部种子(random/numpy/torch/cuda = 2021)→ A/B 条件共享同一随机流;结果按 (sid_t46, cond_id) 映射回 manifest 行序;
- AutoARIMA 辅助臂:仅 ETTh1/ETTh2,每 var 基准清单前 2 窗 × 16 条件 = 448 次预测;参数/1800s 真实进程超时/记账同 EXP-046;仅辅助证据;
- 失败/超时全入 ledger;比较用双方共同成功样本。

## 4. 指标

- 每条件窗口:`pred.astype(f64)` vs `y' (f64)` 的 MSE、MAE(标准化尺度);
- 冲突代价:`cost = MSE_B − MSE_A`(同窗口、同 δ、同 m 配对);
- 旧历史牵引量:`D = mean_h(pred_A − pred_B) / δ`(有符号;D=0 两组预测均值相同;D>0 冲突组预测向旧水平偏移;保留负值与 >1,不截断);
- **方法间差**(共同有效基准窗口 = 该对两条件四方法均 ok):`ΔD = D_VTS − D_对照`、`Δcost = cost_VTS − cost_对照`;
- 聚合:先按基准窗口配对,再按 ds / var / 方向(δ 符号)/ 幅度(|δ|/s ∈ {0.5,1})汇总 median+mean+n;构造变体非独立样本,只作描述统计。

## 5. §6 原始数据复核(零推理;未来仅事后分类)

复用 EXP-046 全部 1792 窗口与预测。对每窗口(输入历史 x,真值未来 y_true):

- `a = x[L−4P : L−P].mean()`(**3P 个点的整体标量均值**,非相位模板;用户确认口径);
- `b = x[L−P : L].mean()`;`c = y_true.mean()`;`s = hist_std`;
- 预先固定两类(阈值不因结果调整):
  - **水平稳定**:`|b−a| ≤ 0.1s` 且 `|c−b| ≤ 0.1s`;
  - **水平变化后持续**:`|b−a| ≥ 0.5s` 且 `|c−b| ≤ 0.5·|b−a|`;
- 比较各方法误差(复用 EXP-046 per_sample_metrics),按 hist_std 三分位(沿用 t46 per (ds×var) tertile 口径)分层复核;如实报告各组 n,样本不足照写。

## 6. 精度与交付纪律

- 基准 `x / y / s / δ` 保存 **float64**;模型输入构造时转 float32;指标 = pred(f32→f64)vs y'(f64);
- 报告数字由 `gen_report_numbers_t47.py` 自动生成,禁止手抄;
- 保留反向结果;不把行为直接归因于 MAE 损失;VisionTS 实例归一化的平移等变性(A 组)仅作事后核验记录,不作门禁、不作机制结论。
