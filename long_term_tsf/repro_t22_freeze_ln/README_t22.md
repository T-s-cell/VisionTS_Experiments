# EXP-033 审计包：VisionTS 表 22 — 完全冻结 vs 仅微调 LayerNorm（ETT 4 数据集 × H∈{96,192,336,720}）

日期：2026-11-02 ｜ eta（L20 GPU）｜ 分支 `repro/ltsf-t22-freeze-v1`（自 EXP-032 冻结头 `212d6cb`）

**问题**：量化 LN 微调本身的收益——相同 Table 19 回看长度 L、周期 P、转图参数（r=c=0.4，双线性）与测试窗口下，**Freeze**（原始 MAE-base 权重直接推理，train_epochs=0）对照 **LN**（仅微调 LayerNorm）。LN 列**复用 EXP-003**（`repro_fullshot/`，commit `b40d663`，`outputs/aggregate.csv` 3-seed 均值 seeds 2021/2022/2023），本轮只补 Freeze 推理。

## 口径

- 配置（论文 Table 19）：ETTh1 L=1152/P=24，ETTh2 L=1152/P=24，ETTm1 L=2304/P=96，ETTm2 L=1152/P=96；bs=32，seed=2021，标准化空间 MSE/MAE。
- 窗口数（用户核对 EXP-031 win_start，16 组共 **109,840** 窗）：小时 N=2881−H（2785/2689/2545/2161），首预测行 **11520**；分钟 N=11521−H（11425/11329/11185/10801），首预测行 **46080**；均与 L 无关 → 真值数组跨实验逐元素可比。
- EXP-003 真值 layout (N·7, H, 1)，按 7·窗口索引+变量索引展平（归档 `exp_univariate.py` 已核对），固定还原 `true[...,0].reshape(N,7,H).transpose(0,2,1)`，无候选分支。
- 汇总：逐数据集 = 4 个 H 等权平均，总体 = 4 数据集等权平均；**Δ = LN − Freeze**（正值 = 微调后更差）；退步项逐项列出；论文表 22 锚点只在数据集层与总体层给出，Freeze vs 论文 |差|>0.001 如实标注（不停机，EXP-030 先例）。
- LN 列自检（硬门禁）：由快照 `aggregate_snapshot.csv` 复算总体 LN = **0.33235446/0.36986023**（与用户复算一致，≤1e-6）。

## 包内容

| 路径 | 内容 |
|---|---|
| `repro_t22_freeze_ln/t22_common.py` | 共享助手：Table-19 配置表、窗口数公式、std_array/truth_recomputed（自 b4_common 移植）、EXP-003 真值固定还原、aggregate 快照加载（sha256 钉死）、LN 聚合+自检、SUCCESS 指纹解析 |
| `repro_t22_freeze_ln/run_t22_group.sh` | 单组冻结推理（与 EXP-001 `run_one.sh` 同一调用协议，仅 L/P 换 Table 19）；指纹+SUCCESS.txt，指纹一致自动跳过 |
| `repro_t22_freeze_ln/verify_t22.py` | 三模式：`--group`（新组全门禁）、`--cache`（ETTm1 逐组缓存核对，缺失/代码漂移→rerun_needed 只补跑该组；权重不一致→停）、`--first_window`（冒烟附加：首窗 L 步历史+变量顺序 bitwise） |
| `repro_t22_freeze_ln/collect_t22.py` | 三层汇总：逐 H 实测表（16 组，无论文列）、数据集层+论文锚点+Δ、总体；LN 自检硬门禁；退步清单；论文对照标注；`summary_t22.csv` + `RESULT_t22_freeze_ln.md` |
| `repro_t22_freeze_ln/run_t22_all.sh` | 驱动：env 快照、preflight（ckpt sha+大小、EXP-003 四 ds 真值在盘、快照 sha；**不要求 EXP-001 全 16 组完好**）、`min`=ETTh1_96 冒烟、`main`=其余 11 组 + ETTm1 逐组缓存核验（按需补跑）+ collect；失败写 logs/FAILED.txt 即停 |
| `repro_t22_freeze_ln/aggregate_snapshot.csv` | EXP-003 `outputs/aggregate.csv` 快照，sha256 `a1622c33b35a54c75e6d4ac5fe1b6649b3797e3234270807c91b87e280f75dcf` |
| `repro_t22_freeze_ln/RESULT_t22_freeze_ln.md` | 结果报告（运行后生成） |
| `repro_t22_freeze_ln/summary_t22.csv` | 16 组 + 4 数据集 + 总体（运行后生成） |
| `repro_t22_freeze_ln/logs/verify_t22.json` | 每组核验全记录（指纹复算/真值双 bitwise/metrics 复算/缓存状态） |

## 门禁（每组，任一失败即停）

1. SUCCESS.txt 在盘；8 个框架文件 md5 逐文件复算 == 记录；ckpt sha256 == `c551c37b…`（EXP-001 冻结权重）且 447,725,789 B。
2. 配置行：Table-19 L/P、pred_len、r=c=0.4、bs=32、train_epochs=0、seed=2021。
3. N == 预期（2881−H / 11521−H）；pred/true 形状 (N,H,7)。
4. true **双 bitwise**：== truth_recomputed(std_array)（含首预测行 11520/46080 的隐含核对）且 == EXP-003 seed2021 真值（固定 reshape 还原）。
5. metric(pred,true) 逐项 == metrics.npy。
6. 运行目录无 checkpoint 产物（只推理）。

## 从零复跑命令（eta）

```bash
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf
PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python bash repro_t22_freeze_ln/run_t22_all.sh min   # ETTh1_96 冒烟 + 首窗历史/变量序检查
PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python bash repro_t22_freeze_ln/run_t22_all.sh main  # 其余 11 组 + ETTm1 缓存核验 + 汇总
```

（冒烟通过后 main 直跑；ETTm1 四组缓存均通过时全程仅 12 组新推理。）

## 大数组（未入包，留在 eta）

- 本轮 Freeze：`save_t22_freeze/{DS}_{H}/results/_/{pred,true,metrics}.npy`
- EXP-001 缓存（ETTm1 复用）：`save_repro_zeroshot/ETTm1_{H}/results/_/`
- EXP-003 LN：`save_fullshot/{DS}_{H}_seed2021/`（48 份种子指标汇总于包内快照）
- MAE 权重：`/dev_data/wlt/VisionTS_Experiments/ckpt/mae_visualize_vit_base.pth`（447MB，绝不入包）

## 运行记录

- 本地 commit：包 `06dc599e873102be4713547ee310d8cf9ae8f5b4`（运行代码树 == 分支头 `8186258`，即包 + 冒烟期两处 verify 修复：门禁只禁 checkpoint **文件**不禁框架自建的空 `checkpoints/` 脚手架目录、loader 样本为 numpy 数组）。
- **冒烟（min）**：ETTh1_96 冻结推理 46s、峰值 1.62 GB；全门禁通过（n=2785、true 双 bitwise、metrics 复算==metrics.npy、mse=0.376013/mae=0.379026）；首窗检查通过（loader N=2785，win0 历史 rows [10368,11520) 与 std_array bitwise 一致，L=1152、7 变量顺序正确）；SMOKE_OK 2026-11-02 23:30:36。
- **main**：其余 11 组全部一次通过（每小时组 ~30-45s、分钟组 ~130-160s，峰值均 1.62 GB）；**ETTm1 四组缓存全部 cache_ok（零补跑）**，全程共 12 组新推理。
- **结果**：总体 Freeze **0.359538/0.374954**（论文 0.360/0.375，数据集层与总体层 |差| 全部 0.000）；LN 0.332354/0.369860（自检 ≤1e-6 通过）；**Δ=LN−Freeze = −0.027183 MSE / −0.005094 MAE**（微调整体有益）。退步 9 项（集中于 ETTh2 与长 H 的 MAE、H=720 的 MSE，如 ETTh2 数据集层 MAE 0.376165→0.382710）；论文对照标注 3 条（均 LN 列，≤0.0013）。
- 环境：logs/env_snapshot_{min,main}.txt（Python 3.12.13 / torch 2.12.0+cu126 / L20）。
- 审计包：`temp/VisionTS_t22_freeze_ln_20261102.tar.gz`（44 KB），sha256 `6f0af5b8374fb65714476adff6c24bbd14bb3146eea21523508b1524cb1b74b5`。
