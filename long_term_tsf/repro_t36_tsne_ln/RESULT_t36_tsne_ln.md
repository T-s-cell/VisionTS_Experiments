# EXP-036 结果:VisionTS 微调 LayerNorm 前后的 t-SNE 对照(1,900 固定样本)

日期:2026-11-03 | commit `3b0658d` | 分支 repro/ltsf-t36-tsne-ln-v1 | 冻结 MAE-base(sha256 c551c37b63a221a3…)

问题:让冻结编码器与三套 EXP-003 编码器-LN 微调权重(ETTm1/Electricity/Weather,H=96、seed=2021 最佳验证 checkpoint)看**完全相同的 1,900 个输入**(EXP-002 归档抽样清单:ImageNet 1,000 + ETTm1/Weather/Electricity 各 300),局部交叠如何变化。四条件输入逐批哈希一致;仅编码器 50 个 LN 张量来自各自 checkpoint,其余参数与冻结逐位一致。

## 门禁(全部通过)

- weights/ETTm1: probe max|d|=0.000e+00 (B=256), 50/50 LN keys changed (max 5.646e-02), encoder non-LN bitwise==frozen, ckpt d2da2b4448da690e...
- weights/Electricity: probe max|d|=0.000e+00 (B=256), 50/50 LN keys changed (max 4.340e-01), encoder non-LN bitwise==frozen, ckpt 455bf02b907a7c84...
- weights/Weather: probe max|d|=0.000e+00 (B=256), 50/50 LN keys changed (max 9.789e-02), encoder non-LN bitwise==frozen, ckpt 19c02336c1f7c600...
- features: 4x (1900,54528) f32 finite; frozen bitwise == EXP-002 original (max|d|=0); per-batch input + noise hashes identical across conditions; 1,900-row order == archived meta.csv/groups
- extract: 50 spliced encoder-LN keys bitwise == source checkpoint in every LN condition; ids_keep == archived mask_ref
- t-SNE: 3 LN conditions, EXP-002 params verbatim, sklearn 1.9.1; frozen panel = archived EXP-002 coords (sha d4706007477d1278...)
- NN: 30-NN over all 1,900 features, self excluded by global row, denominator 300x30=9000 per cell; neighbour row indices archived

## 权重来源与回放

| 数据集 | 训练配置(EXP-003 args) | 首批回放 max\|d\| (≤1e-4) | 改变 LN 键 | checkpoint sha256(前 16) |
|---|---|---|---|---|
| ETTm1 | L=2304/P=96/r=0.4/c=0.4 freq=t | 0.000e+00 (B=256) | 50/50 | d2da2b4448da690e |
| Electricity | L=1152/P=24/r=0.4/c=0.4 freq=h | 0.000e+00 (B=256) | 50/50 | 455bf02b907a7c84 |
| Weather | L=576/P=144/r=1.0/c=0.7 freq=h | 0.000e+00 (B=256) | 50/50 | 19c02336c1f7c600 |

锚点(SUCCESS 未舍入):ETTm1 0.2832735478878021/0.3229347765445709、Electricity 0.12593606114387512/0.21679048240184784、Weather 0.14431872963905334/0.19301669299602509。

## 高维 30-NN ImageNet 占比(欧氏距离,分母 9,000)

| encoder | ETTm1 | Weather | Electricity |
|---|---|---|---|
| Frozen (EXP-002) | 0.0000 | 0.0000 | 0.0000 |
| ETTm1-LN | 0.0000 (+0.00 pp) | 0.0000 (+0.00 pp) | 0.0000 (+0.00 pp) |
| Electricity-LN | 0.0000 (+0.00 pp) | 0.0000 (+0.00 pp) | 0.0000 (+0.00 pp) |
| Weather-LN | 0.0000 (+0.00 pp) | 0.0000 (+0.00 pp) | 0.0000 (+0.00 pp) |

括号为相对冻结的百分点变化。邻域在**该编码器的全部 1,900 个特征**内取(排除自身,按全局行号),不限于同组。

**高维结果饱和:4 编码器 × 3 时序来源全部 12 格均为 0/9,000。**在原始 54,528 维特征空间,每个时序样本的 30 近邻全部是时序样本(冻结编码器下抽查行最近 ImageNet 距离约为最近时序距离的 1.4–3.3 倍;独立复算与 nn_neighbors.npz 一致)。模态鸿沟主导原始特征,该高维指标在本设置下不区分四套编码器,微调的变化量恒为 ±0 pp,照实记录。

## 二维面板内 30-NN ImageNet 占比(t-SNE 坐标空间,描述性)

| encoder | ETTm1 | Weather | Electricity |
|---|---|---|---|
| Frozen (EXP-002, 归档坐标) | 0.2759 | 0.1079 | 0.1306 |
| ETTm1-LN | 0.3051 (+2.92 pp) | 0.0947 (-1.32 pp) | 0.1822 (+5.17 pp) |
| Electricity-LN | 0.3032 (+2.73 pp) | 0.1266 (+1.87 pp) | 0.2293 (+9.88 pp) |
| Weather-LN | 0.2399 (-3.60 pp) | 0.1124 (+0.46 pp) | 0.1504 (+1.99 pp) |

同一 30-NN 算法用在各面板自己的 2D 坐标上,刻画图面所见;各面板独立拟合,数值跨面板仅作方向参考,不构成显著性检验,也不替代高维表。

## 判读(训练域 vs 迁移域)

高维原始特征空间四条件全部 0/9,000(模态鸿沟主导,见上);以下按各面板 2D 局部交叠(描述性,括号为相对冻结的百分点)判读。
- ETTm1-LN(训练域 ETTm1): 面板内 ImageNet 占比 上升 +2.92 pp; 迁移域 Weather -1.32 pp(下降); 迁移域 Electricity +5.17 pp(上升)。
- Electricity-LN(训练域 Electricity): 面板内 ImageNet 占比 上升 +9.88 pp; 迁移域 ETTm1 +2.73 pp(上升); 迁移域 Weather +1.87 pp(上升)。
- Weather-LN(训练域 Weather): 面板内 ImageNet 占比 上升 +0.46 pp; 迁移域 ETTm1 -3.60 pp(下降); 迁移域 Electricity +1.99 pp(上升)。

二维面板各自独立拟合(t-SNE 坐标方向/尺度跨面板不可比),只比较各面板内部的局部交叠;高维指标见表。二维观察与高维结果若不一致,以两者数值并存、不调和。交叠增加、减少或变化很小均照实呈现;预测精度的变化不由本图承担,由 EXP-003 的预测评估单独支持。

## 大件(留在 eta,不入包)

- 四条件特征:/dev_data/wlt/VisionTS_Experiments/long_term_tsf/save_t36_tsne(每份 1900x54528 f32;frozen 与 EXP-002 原件 /dev_data/wlt/VisionTS_Experiments/repro_fig7/outputs/features_1900x54528.npy 逐位一致)
- 权重:三份 EXP-003 checkpoint.pth(/dev_data/wlt/VisionTS_Experiments/long_term_tsf/save_fullshot/{ETTm1,Electricity,Weather}_96_seed2021/checkpoints/_/)+ 冻结 ckpt(/dev_data/wlt/VisionTS_Experiments/ckpt/mae_visualize_vit_base.pth,sha256 c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e)
- ImageNet 原图:/dev_data/wlt/data/ImageNet/val(50,000 张)

