# EXP-009 — TimesX 分域 LN 微调前后编码器特征 t-SNE（REPORT）

模式：**postaudit re-attestation (full 19 domains, FINAL files)** | 生成时间：2026-10-04 13:49:00 | 协议：exp009_tsne_v1 | 基线代码：dcd2425

## 方法（一句话）
对每个领域，把 1,000 张冻结 ImageNet-1k val 参考图与该领域全部原生测试窗（仅 past 96 步）的编码器特征（vision_model.norm 全 57 token 展平 43,776 维，无池化/PCA）做一次联合 t-SNE；微调前 = Z0（原始 ImageNet 预训练 MAE-base），微调后 = F0 seed2021 每域 best_ft（EXP-004 selection.json，非 fallback）。

## 逐域结果（单种子 s2021；std MSE/MAE：窗口 d 归一 → 变量平均 → 域内变量等权）

| domain | n_vars | n_win | Z0 MSE | Z0 MAE | F0_s2021 MSE | F0_s2021 MAE | ΔMSE% | best_epoch | 图 |
|---|---|---|---|---|---|---|---|---|---|
| CropsAndStaples | 11 | 188 | 1.342177 | 0.740295 | 1.105363 | 0.641821 | -17.64 | 6 | figures/CropsAndStaples/ |
| Currency | 10 | 180 | 0.971216 | 0.704397 | 0.668832 | 0.561320 | -31.13 | 10 | figures/Currency/ |
| EnergyAndFuels | 10 | 171 | 2.276056 | 0.921286 | 1.903109 | 0.792633 | -16.39 | 9 | figures/EnergyAndFuels/ |
| LivestockAndFoodProducts | 10 | 318 | 1.320066 | 0.724610 | 0.982169 | 0.606421 | -25.60 | 6 | figures/LivestockAndFoodProducts/ |
| RawMaterialsAndConstruction | 10 | 172 | 1.338255 | 0.729526 | 1.133898 | 0.677984 | -15.27 | 9 | figures/RawMaterialsAndConstruction/ |
| SpecialtyAndAdvancedMaterials | 8 | 117 | 2.050198 | 0.694168 | 1.872520 | 0.588929 | -8.67 | 4 | figures/SpecialtyAndAdvancedMaterials/ |
| StrategicAndHighValueMaterials | 10 | 178 | 2.726347 | 0.761315 | 2.429433 | 0.699524 | -10.89 | 10 | figures/StrategicAndHighValueMaterials/ |
| arts | 10 | 151 | 2.878970 | 0.938112 | 2.883697 | 0.938211 | +0.16 | 1 | figures/arts/ |
| climate | 10 | 90 | 5.511981 | 1.135626 | 5.662653 | 1.137636 | +2.73 | 10 | figures/climate/ |
| economy | 10 | 90 | 2.992216 | 1.161425 | 3.052463 | 1.160799 | +2.01 | 10 | figures/economy/ |
| electronic_technology | 10 | 90 | 8.713220 | 1.502199 | 8.358046 | 1.454406 | -4.08 | 10 | figures/electronic_technology/ |
| finance | 10 | 90 | 5.712122 | 1.516955 | 5.436808 | 1.472084 | -4.82 | 10 | figures/finance/ |
| pets | 10 | 90 | 4.470860 | 1.293532 | 4.845365 | 1.325680 | +8.38 | 10 | figures/pets/ |
| public_health | 10 | 90 | 2.804917 | 1.123759 | 2.810151 | 1.124788 | +0.19 | 1 | figures/public_health/ |
| public_policy | 11 | 99 | 3.870318 | 1.338200 | 3.863963 | 1.335718 | -0.16 | 10 | figures/public_policy/ |
| science | 10 | 90 | 6.556470 | 1.059573 | 6.514698 | 1.055280 | -0.64 | 10 | figures/science/ |
| shopping | 10 | 90 | 1.197819 | 0.651796 | 1.193347 | 0.632710 | -0.37 | 10 | figures/shopping/ |
| society | 10 | 90 | 5.438220 | 1.354871 | 5.436759 | 1.352389 | -0.03 | 10 | figures/society/ |
| traffic | 10 | 90 | 7.267871 | 1.580889 | 7.325183 | 1.584264 | +0.79 | 1 | figures/traffic/ |
| **overall（19 域等权）** | 190 | 2474 | 3.654700 | — | 3.551498 | — | -2.82 | — | — |

单种子 s2021 下 F0 优于 Z0 的领域数：**13/19**（ΔMSE% 为负 = 微调更优）。

## 结论边界（务必一并阅读）
- 事后探索性可视化：单种子（s2021）、非盲测（与 EXP-004 共享数据与划分）。
- 仅观测编码器 vision_model.norm 输出；不能将预测误差变化归因于“编码器模态距离”变化的机制。
- t-SNE 距离仅定性；前后独立拟合，绝对坐标不可跨图直接比较。
- 指标为单种子复算（Z0 已与 EXP-004 domain_summary 逐域核对一致），不代表三种子均值。

## 产物
- 主图 38 张：`figures/<domain>/{Z0,F0_s2021}.png`（300 dpi）＋ 19 张并排图 `figures/_side_by_side/`＋`index.html`。
- 坐标 38 份：`coords/<domain>__{Z0,F0_s2021}.csv`（含 sample_id/来源/域/变量/状态/坐标）与 `.params.json`（KL、n_iter、冻结参数、输入指纹）。
- 特征缓存（大文件，留在服务器）：`features/`（清单位 manifests 内、SHA256 与缓存键见 `features/_cache_index.json`）。
- 审计：`audit/audit_report.json`（完整性/检查点/特征/hook/不可变性/溯源/指标 七类检查）。

## 绘图阶段后复核（postaudit re-attestation）
- 特征提取与 t-SNE 之后，应用户要求做了两轮纯绘图改动（ImageNet 参考图点样式；每域性能排名题头），并为此扩展了审计脚本本身。
- 三处代码/配置变更均经哈希锚定：生成时文件内容先重建并双哈希验证（SHA256 == audit/baseline.json 且 MD5 == _cache_index.json 生成指纹），再以 unified diff 存档于 `audit/code_diff/`，见 `audit/postaudit_changelog.json`。
- `audit/baseline.json`、`features/_cache_index.json` 与原始 `audit/audit_report.json` 原样保留（本报告的 postaudit.preserved_records_untouched 检查项逐字节验证）。
- 本复核报告：`audit/audit_report_final.json`（覆盖最终交付文件；特征/坐标/指标检查全部按现状重算通过）。
