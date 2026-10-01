# VisionTS LN full-shot 复现最终报告（2026-10-01）

复现对象：论文 §4.3 / Appendix C.1 / Table 19/20/21 —— LN full-shot（仅微调 MAE LayerNorm，其余冻结），96 次独立训练评测 = 8 数据集 × 4 预测长度 × 3 种子（2021/2022/2023）。

## 1. 结论

- **运行完成：达标。** 96/96 SUCCESS，指纹全部匹配（数据 SHA256 + ckpt SHA256 + 代码 md5 逐任务记录于各 fingerprint.json），metrics 全为有限值。
- **数值判定：96 格中 94 格 |Δ|≤0.01（一致）；2 格偏差，均为 Illness 的 MSE（方向：复现值低于论文）。** 全部 96 格的 MAE 均 ≤0.02。整体 max|Δ| = 0.048（Illness/24 MSE）。
- 逐格判定见 `comparison.md`；原始 96 行见 `summary.csv`；(DS,pred) 3 种子 mean±std（ddof=1）见 `aggregate.csv`。

| 范围 | MSE | MAE |
|---|---|---|
| ETTh1/ETTh2/ETTm1/ETTm2（eta, L20） | 16/16 格一致 | 16/16 一致 |
| Weather / Electricity（eta, L20） | 8/8 格一致 | 8/8 一致 |
| Traffic（theta, 3090） | 4/4 格一致 | 4/4 一致 |
| Illness（theta, 3090） | 2/4 一致；24/60 两格 MSE 偏差（−0.048 / −0.035） | 4/4 ≤0.02（24/60 为基本一致 −0.013） |
| 每数据集 avg 列 | 7/8 一致；Illness avg −0.021 偏差 | 8/8 一致 |

## 2. Illness 偏差格分析（按预声明协议：记录可能原因，不重调参数）

偏差仅出现在 Illness 且仅 MSE 方向为"复现更优"（24: 1.986 vs 2.034；60: 1.875 vs 1.910）。最可能原因：

1. **Illness 是全战役中唯一使用 100 epochs + 验证早停选模的数据集**（其余 7 个数据集均 1 epoch、无选模——这 7 个全部 ≤0.006 精确吻合，强烈指向偏差源于选模环节而非训练/评测协议本身）。
2. 本复现**修正了官方验证协议**：官方 val loader 为 shuffle=True + drop_last=True 且按 batch 等权平均（illness/36 每 epoch 随机丢弃 ~41% 验证样本）；本复现为完整、固定、按元素加权的验证集 MSE 选模。更可靠的选模 → 更优 checkpoint → 更低测试 MSE，与观测方向一致。
3. 论文 Table 20 未报告 Illness 的 std（其余 7 数据集均有），论文自身对 Illness 的披露即不完整。

注：eta（L20）亦跑完 12 个 illness 冗余副本（配置/数据指纹一致），留存在 eta 服务器未纳入本报告（用户已声明跨环境对比不再需要）；如需环境稳健性旁证可随时取用。

## 3. 论文规定 / 官方实现 / 补充设定 三列对照

| 项 | 论文规定 | 官方实现 | 本复现（补充设定） |
|---|---|---|---|
| 微调范围 | 仅 LN（§4.3） | `requires_grad = ('norm' in name)`，84 张量/55,808 标量（ln_check 实证） | 同官方，逐名断言 84 名单 |
| 优化器 | Adam lr=1e-4（C.1） | Adam 1e-4（显式筛选可训练参数） | 同 |
| batch | 256 单变量样本（C.1） | 从未传 `--batch_size`（实际 32；ETT 多变量窗口 32×C） | **真 256 单变量/batch**：custom 原生单变量直接 256；ETT 用 `Dataset_UnivariateView` 包装（索引算术与 Dataset_Custom 一致） |
| epochs | 1；illness 100+早停（C.1） | 同 | 同 |
| 验证集 | 未披露 | shuffle=True/drop_last=True/batch 均值（illness/36 丢 ~41% 验证样本） | **修正：完整固定 val（shuffle=False/drop_last=False）+ 按全部元素加权 MSE** |
| 早停/lr衰减 | 未披露 | patience=3，lradj=type1（每 epoch 减半） | 同官方 |
| 测试指标空间 | 未披露 | 标准化空间（inverse=False） | 同官方 |
| 种子 | "重复三次"未给种子 | — | 2021/2022/2023 |
| r/c/L/periodicity | Table 19 + Table 6 | 8 个官方脚本逐项核对一致 | 同（配置表见计划文档） |
| Illness 数据 | 8 列 7 通道周度 | national_illness.csv | 已核验标准基准，双端 SHA256 一致 |

## 4. 战役执行记录（去重阶段，2026-09-28 批准）

- **分工**：eta（L20 单卡串行）`run_all_eta_notraffic.sh` 跑 84 个非 Traffic 任务；theta（双 3090）跑 12 Traffic（`run_all_3090_t720.sh`，09-30 07:58 收官）+ 12 illness（`run_all_3090_illness.sh` 双卡并行，09-30 21:17 收官，~2-4 分钟/任务）。跨环境对比按用户指示不做；eta 另跑完自己的 12 个 illness 冗余副本。
- **数据一致性**：8 个数据集 csv 与 ckpt 的 SHA256 双端核对一致（指纹逐任务落盘）。
- **事件**：① eta 多次遭共租户拖慢（ECL_720 最慢 ~2.2s/it、训练段 11.4h），无 OOM、无 ABORT；② ECL_720_2023 训练后 vali 静默段曾被误判卡死，00:51 确认为正常静默验证；③ 依用户指示曾将 ECL_720_2023 救援至 theta GPU1（00:14 启动），02:21 eta 先出 SUCCESS 后按用户指示终止 theta 救援、GPU 归还共租户（theta 上留有该未完成 job 的现场目录，无结果产物）；④ 全程零重启、零调参、零 SUCCESS 覆盖。
- **eta 队列收官**：10-01 03:26 `ALL JOBS COMPLETED (84)`，最后落地的 ECL_720_2023 = mse 0.1954 / mae 0.2881（论文 0.192/0.286）。

## 5. 运行环境（逐任务指纹）

- eta：Python 3.12.13 / torch 2.12.0+cu126 / NVIDIA L20（72 格）
- theta：Python 3.8.20 / torch 2.3.0+cu121 / RTX 3090 ×2（24 格）
- ckpt：mae_visualize_vit_base.pth，SHA256 c551c37b…（双端一致）

## 6. 产物清单

- 本目录（outputs/）：summary.csv（96 行）、aggregate.csv（32 行 + std）、comparison.md（逐格 Δ 与三档判定）、FINAL_REPORT.md。
- 审计包：`/home/wlt/MMTS/temp/VisionTS_fullshot_20261001_043308.tar.gz`（repro_fullshot 代码+日志+最小测试报告+队列日志+outputs；save_fullshot 96 job 轻量产物；不含 checkpoint 与 pred/true npy）。
- 完整可复现产物（checkpoint、pred/true npy）留存于两台服务器 `~/VisionTS_Experiments/long_term_tsf/save_fullshot/`。
- 最小测试报告：reports_eta/、reports_theta/ 下 ln_check_report.txt（84 张量逐名比对）与 batch_check_report.txt（8 数据集 batch 语义）。

**总判定：运行完成 96/96；数值复现达标除 Illness 两格 MSE（≤0.05 且方向为更优、原因可解释、协议修正已知）外全部一致（阈值 0.01/0.02 三档，预声明）。**
