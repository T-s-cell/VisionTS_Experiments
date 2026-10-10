# VisionTS × TimesX — 在 TimesX 全部原生窗口上的固定 P=1 zero-shot 补充评估

Protocol: `TimesX-all-native-P1-v1` | 数据 zip sha256 `9adb9540733c…` | 权重 sha256 `c551c37b63a2…` | r=c=0.4, 224×224, patch16, bilinear, fp32, TF32 off, seed 2021, batch 64

- 冻结: **8106** 原生窗口 / 190 变量 / 19 领域（test 1677 + val 4736 + 跨边界剔除 242 + train 区 1451）；零方差窗 4 个全部走 calib 回退
- 预检: 复推 194 窗（compare=191），c1_max=3.10e-05 (阈值 1e-05), c2_max=1.02e-05 (阈值 0.0001)，失败变量 1 个（其窗口 c1/c2 超出阈值，整变量重推）
- 缓存复用: 189 变量 / 6386 窗（S2 绑定校验+逐窗核对，来源文件 sha256 全记录）；补推理 1720 窗
- 汇总: 8106 窗（cache=6386, new=1720），每窗恰一条预测

## 总体（标准化尺度；19 域等权）

| MSE | MAE |
|---|---|
| 4.212298 | 0.856011 |

原始尺度 MSE/MAE 只在变量层报告（variable_level.csv），不跨单位平均。

## 子集对照

- **1,677 旧 test 子集（纯缓存复算）**: MSE 3.254715 / MAE 1.042067，vs 参考 3.254715/1.042067，Δ=(9.2e-08, 3.0e-07)，≤5e-7: **True**
- 1,677 子集（最终预测口径）: MSE 3.254715 / MAE 1.042067，受重推变量影响: ['shopping__black_friday_deals_96_12_4_10events']
- **2,474 旧 LN test 子集（信息性）**: MSE 3.654700 / MAE 1.049081 (cache=2225, new=249)

## 分领域（域内变量等权）

| Domain | vars | MSE | MAE |
|---|---|---|---|
| CropsAndStaples | 11 | 1.2240 | 0.7093 |
| Currency | 10 | 1.1242 | 0.7142 |
| EnergyAndFuels | 10 | 1.4980 | 0.7704 |
| LivestockAndFoodProducts | 10 | 1.2911 | 0.7189 |
| RawMaterialsAndConstruction | 10 | 1.2159 | 0.7567 |
| SpecialtyAndAdvancedMaterials | 8 | 2.9526 | 0.6145 |
| StrategicAndHighValueMaterials | 10 | 9.8844 | 0.8156 |
| arts | 10 | 2.0876 | 0.8344 |
| climate | 10 | 2.3292 | 0.8143 |
| economy | 10 | 1.6559 | 0.8637 |
| electronic_technology | 10 | 13.8178 | 1.2536 |
| finance | 10 | 3.6144 | 1.1030 |
| pets | 10 | 2.1639 | 0.8999 |
| public_health | 10 | 1.7666 | 0.8039 |
| public_policy | 11 | 20.9660 | 1.1219 |
| science | 10 | 5.4459 | 0.9734 |
| shopping | 10 | 0.8217 | 0.4957 |
| society | 10 | 2.7643 | 0.9460 |
| traffic | 10 | 3.4102 | 1.0546 |

## 验收: **ALL PASS** (16/16 项)
- PASS R1_unique_inventory_ids
- PASS R1_pred_set_equals_inventory
- PASS R1_chronology_all_past_lt_future
- PASS R1_flat_index_alignment
- PASS R2_all_finite
- PASS R2_std_defined_all
- PASS R2_pred_shape_all
- PASS R3_zero_var_count
- PASS R3_all_fallback_calib
- PASS R3_no_other_fallback
- PASS R4_cache_provenance
- PASS R4_new_provenance
- PASS R5_independent_recompute
- PASS R6_pure_cache_1677_within_5e-7
- PASS R7_totals
- PASS R7_fill_equals_set_difference

> 本实验为「在 TimesX 全部原生窗口上的固定 P=1 zero-shot 补充评估」，原 test 集实验（repro_zeroshot_ext）结论保持独立。
