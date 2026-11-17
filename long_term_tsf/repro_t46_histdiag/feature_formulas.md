# EXP-046 特征公式(分析前落盘;实现见 t46_common.hist_features)

输入:标准化历史窗 x(长度 L,单变量;train [0,12K) 段 StandardScaler,ddof=0)。
预测起点 s,输入窗 = csv 行 [s−L, s),通道 c。周期 P 取各数据集默认 P(24/24/96/96)。

## 特征定义(仅由输入历史计算)

| 特征 | 公式 | 说明 |
|---|---|---|
| hist_std | `std(x)`(ddof=0) | 幅度描述量;分组结果另按 hist_std 分层复核,不作"幅度控制"结论 |
| period_corr(周期一致性) | `pearson(x[P:], x[:-P])` | 滞后 P 相关系数 |
| recent_old(近旧周期差) | `sqrt(mean((x[-P:] − phase_mean)²)) / hist_std`,phase_mean = `mean(x[-4P:-P].reshape(3,P), axis=0)` | 最后一个周期 vs 此前三个周期同相位均值 |
| trend(趋势强度) | `abs(slope) × (4P) / hist_std`,slope = 对 `x[-4P:]` 与 `arange(4P)` 的一阶线性拟合斜率 | 拟合区间内标准化幅度变化 |
| vol(波动程度) | `std(diff(x)) / hist_std`(ddof=0) | 一阶差分标准差比 |

## 近常数与除零

- `hist_std < 1e-8`(标准化尺度)→ 标记 near_constant;四个比值特征记 NaN,**不进入 tertile 分组**,计数单独报告。
- period_corr 分母为 0 → NaN 同上处理。

## 分组与分析口径(冻结)

- tertile:每个 (数据集 × 变量) 内 64 个(非近常数)样本,按该特征以 `np.quantile` 1/3、2/3 分箱为 low/mid/high。
- 逐方法误差:每格均值 MSE(标准化尺度,f64);两两差距:
  绝对 gap = MSE_VTS − MSE_对照;**相对 gap = (MSE_VTS − MSE_对照) / (MSE_VTS + MSE_对照 + 1e-12)**(正 = VisionTS 相对弱势线索)。
- 共同样本口径:两方法比较用双方均成功样本;四方法排名用四者共同成功样本;覆盖率按特征组细分报告。
- hist_std 自身作为第 5 个分组维度做同款 tertile 表,检验结果是否被窗口幅度差异解释(描述性)。
- "共同困难":每 (数据集 × 变量 × 方法) 各自 90 分位阈值;四者共同成功样本中全部方法 ≥ 各自阈值判共同困难;允许某类为空。
- 案例选取(确定性,排除近常数):明显较好 = VisionTS 在四方法中排名 1 且 MSE < 0.8×次优;明显较差 = 排名 4 且 MSE > 1.25×次差;每类每数据集取 margin 前 2,不足则如实空缺。
- 结论统一表述为"相对弱势线索";窗口重叠,1792 样本仅描述性分析,非独立证据。
