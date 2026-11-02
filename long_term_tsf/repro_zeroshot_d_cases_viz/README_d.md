# EXP-032 审计包：VisionTS 附录 D 成功与失败案例可视化（ETTh1/OT/L=2880/H=96/P=24）

日期：2026-11-02 ｜ eta（L20 GPU）｜ 分支 `repro/ltsf-d-cases-v1`（自 EXP-030 冻结头 `cad847b`）
回放运行头 commit：`9e0ff6f995c57202d95decad4432cd6847f27c6e`（选窗复核、回放、PNG/NPZ 证据均出自此头代码的完整 `run_d_all.sh`）
首轮观察 commit：`892ac4190888e727c2f567f7eb60b8843856d193`（查图后填写 OBS）
**修订后最终报告版本 commit：`acd1cd9a6d47e150c69477832dcd4ac628c11e97`**（按评审意见限定案例结论措辞、表头改"最大绝对差"、以相对路径嵌入两图；仅改 report_d.py 并重新生成 RESULT_d_cases.md，预测与两图未重跑、无 GPU）
**EXP-031 归档版本**（本轮全部基线数组与逐窗 npz 的出处）：分支 `repro/ltsf-b4-seasonal-v1` @ `ecb5f1005a97fec7282cf31b07eb38653c404b94`（运行头 `97e658b`），审计包 `temp/VisionTS_b4_seasonal_20261102.tar.gz` sha256 `0df05a92297d0bbbdecb890146ff899636dbde5fd39401141aaaec7678337558`。

## 口径（必读）

- 本报告为**新选窗口的定性复现**：窗口由固定规则本轮重新选取，非论文原图窗口。
- 本轮胜负判据 = VisionTS 与两条周期基线（Seasonal Naïve / Seasonal Avg，EXP-031）的 OT MAE 比较；**论文图 14 的失败判据则是 VisionTS MAE 高于 MOIRAI-Large**，两者不同。
- 按用户修正 ②：预测从**原始解码图**（MAE decoder tokens 的 unpatchify）还原，**不用**展示版补全图（可见区换回输入；首列双线性插值跨过可见区边界 x=80 会取到输入像素）。同一次前向捕获：真实 MAE 输入（掩区=0；导出图 −2 仅为显示标记）、固定 patch mask（126/196，patch 5-13）、原始解码图。

## 结论

- **选窗 6 项复核全中**（成功 2046 窗下中位 win=1480、失败 61 窗下中位 win=2663、两窗三方法 MAE 表与用户冻结表逐项一致；任一不符即停，无回退分位数）：

| 案例 | n | win | 起始行(时间) | 末行(时间) | vts | naive | avg |
|---|---|---|---|---|---|---|---|
| success | 2046 | 1480 | 13000 (2017-12-24 16:00) | 13095 (2017-12-28 15:00) | 0.093802 | 0.133358 | 0.507193 |
| failure | 61 | 2663 | 14183 (2018-02-11 23:00) | 14278 (2018-02-15 22:00) | 0.237829 | 0.202828 | 0.185789 |

- **回放核验逐位一致**：①原始解码图还原 96 步 vs 同次 forward 输出 最大绝对差 = **0**（门禁 ≤1e-6）；②回放输出 vs EXP-001 冻结预测 最大绝对差 = **0**（7 变量与 OT 列，门禁 ≤1e-4）。回放 OT MAE 与冻结数组 f64 重算完全相等。评审方从证据 NPZ 的原始解码图**独立**还原 96 步，最大绝对差 2.38×10⁻⁷（成功窗）/ 1.19×10⁻⁷（失败窗），均 ≤1e-6。
- **每例观察**（详见 RESULT_d_cases.md，结论限定在曲线能支持的范围内）：成功例——VisionTS 96 步总 MAE 最低、大体延续近期压低水平与日周期相位，但第 3 个预测日 MAE 0.1567 反高于 S-Naïve 的 0.0767，并非全程最接近；S-Avg 停在全历史平均轮廓的高水平严重高估。失败例——模型低估中段抬升（第 63 步真值 −1.031 vs VisionTS −1.611），总 MAE 劣于两条基线；S-Avg 平均轮廓恰好偏高且相位对齐反而最接近。

## 包内容

| 路径 | 内容 |
|---|---|
| `repro_zeroshot_d_cases_viz/*.py, run_d_all.sh` | 5 个包脚本（d_common / select_windows_d / replay_figures_d / report_d / 驱动） |
| `repro_zeroshot_d_cases_viz/RESULT_d_cases.md` | 结果报告（口径 + 选窗表 + 核验表 + 每例观察 + 环境） |
| `repro_zeroshot_d_cases_viz/README_d.md` | 本文件 |
| `repro_zeroshot_d_cases_viz/logs/selection_verification.json` | 选窗复核全记录（计数/索引/npz MAE/冻结数组 f64 重算/时间戳/批次/运行时 md5） |
| `repro_zeroshot_d_cases_viz/logs/replay_evidence.json` | 回放证据（ver_a/ver_b、MAE 三方对照、ckpt sha256、9 个运行时文件 md5、重构口径注记） |
| `repro_zeroshot_d_cases_viz/logs/env_snapshot.txt` | eta 环境快照（python 3.12.13 / torch 2.12.0+cu126 / numpy 2.4.4 / pandas 2.3.3 / L20） |
| `save_d_cases/ETTh1_96/success_win1480.png` | 成功案例图（1×4：真实 MAE 输入｜展示版补全图｜原始解码图(红带=未来96步列[120,124))｜近336步历史+96步曲线） |
| `save_d_cases/ETTh1_96/failure_win2663.png` | 失败案例图（同版式） |
| `save_d_cases/ETTh1_96/selection_list.md` | 选窗清单（变量/行号/时间/未舍入 MAE/组内名次/批次） |
| `save_d_cases/ETTh1_96/evidence_win{1480,2663}.npz` | 每窗证据：真实输入图/展示补全图/原始解码图（各 3×224×224 f32）、patch mask 图与 tokens、means/stdev、OT 真值、三方法预测、ver_a/ver_b |

## 门禁（全部通过）

- precheck：ckpt `mae_visualize_vit_base.pth` 在盘且 447,725,789 B、sha256 `c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e`（与 EXP-001 SUCCESS 一致）；3 份逐窗 npz 与三方法 pred/true 数组在盘。
- 选窗：三 npz 按 window_index/win_start/channel_names 对齐（2785 窗、win_start[0]=11520、OT=列6）；组计数 2046/61、下中位 (n−1)//2 = 1480/2663、MAE 表与冻结表 |Δ|≤5e-5 全中，否则 SystemExit。
- 回放：seed 2021、batch_size=32 顺序 loader（shuffle=False, drop_last=False）、`models.VisionTS.Model`（构造器内加载冻结权重）；monkeypatch 仅包裹 `vm.vision_model.forward` 以捕获 (image_input, y_tokens, mask)，**未改动任何指纹文件**。

## 从零复跑命令（eta）

```bash
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf
PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python bash repro_zeroshot_d_cases_viz/run_d_all.sh
```
（选窗门禁失败即停；回放两窗在原测试批次 46/83 上重放，GPU 确定性下逐位复现，ver_a/ver_b=0。）

## 大数组（未入包，留在 eta）

- EXP-001 原周期预测/真值：`save_repro_zeroshot/ETTh1_96/results/_/{pred,true}.npy`（(2785,96,7) f32，未改动）
- EXP-031 周期基线：`save_b4_seasonal/ETTh1_96/{naive,avg}/results/_/{pred,true}.npy`；逐窗 npz：`save_b4_seasonal/ETTh1_96/{win_visionts_ETTh1_96.npz, naive/win_ETTh1_96.npz, avg/win_ETTh1_96.npz}`
- MAE 权重：`/dev_data/wlt/VisionTS_Experiments/ckpt/mae_visualize_vit_base.pth`（447MB，绝不入包）
- 本实验证据：`save_d_cases/ETTh1_96/`（PNG/NPZ/selection_list.md）
