# EXP-047 旧历史水平对预测的影响验证(level-shift)

- 分支 `repro/ltsf-t47-levelshift`,基线 = EXP-046 冻结头 **21e2fa3**(只新增,不改 t46 任何产物);不推送。
- 问题:VisionTS 是否更容易被旧历史拉向旧水平;该倾向是否对应稳定场景的优势与变化场景的代价。
- 范围:只研究**水平变化**;不改周期形状/图像块/网络参数。线索级结论,不支持机制,不归因 MAE。

## 规格(冻结于 variant_formulas.md,先于一切推理)

- 基准窗口:EXP-046 清单每 (ds×var) `recent_old` 最低 16(排除近常数/非有限)→ 448;只用输入历史。
- 成对构造:δ = s·{−1,−0.5,+0.5,+1},s = x.std(ddof=0)(csv f64 重算,↔features ≤1e-8);
  m ∈ {P,2P,4P};A = 整段+δ(参照),B = 仅末 m 步+δ(冲突);真值 y′ = y+δ(A/B 共享);
  **修改后的输入全部重新推理,不复用原预测**。16 条件(4A+12B)×448 = **7168**;
  cond_id:A = delta_idx;B = 4+3·delta_idx+m_idx;行序 window-major then cond(rg = w·16+cid)。
- 方法:冻结 VisionTS(mae_base,batch32 单变量)/ Moirai(patch32、200 采样、逐时刻中位数、
  **按条件分别推理,每次实际推理前重置种子 2021**)/ AutoETS(窗口内拟合,season_length=P);
  AutoARIMA 辅助臂仅 ETTh1/ETTh2 每 var 前 2 基准窗 ×16 = 448 次,1800s 真实进程超时,失败全入 ledger。
- 指标:MSE/MAE(f64:pred f32→f64 vs y′ f64);冲突代价 cost = MSE_B−MSE_A;
  旧历史牵引量 **D = mean_h(pred_A−pred_B)/δ**(有符号,保留负值与 >1);
  **方法间差 ΔD = D_VTS−D_对照、Δcost**(共同有效基准窗口)直接回答"是否比对照更强"。
- §6 原始复核(零推理):复用 t46 全 1792 窗;a = x[L−4P:L−P].mean()(3P 整体标量)、
  b = x[L−P:L].mean()、c = 真值未来 96 均值、s = hist_std;
  稳定 |b−a|≤0.1s ∧ |c−b|≤0.1s;变化后持续 |b−a|≥0.5s ∧ |c−b|≤0.5|b−a|;阈值预先固定。

## 冻结链(本地 2026-11-19 生成,eta 重放逐字节一致)

| 文件 | sha256(前 16) |
|---|---|
| baseline_list_t47.csv(448 窗) | `5c750c645af06fbb` |
| manifest_t47.csv(7168 行) | `94aec758b2e490ba` |
| baseline_ETTh1.npz | `d19478024b5404bd` |
| baseline_ETTh2.npz | `081b00981106f512` |
| baseline_ETTm1.npz | `420d622a171d58c6` |
| baseline_ETTm2.npz | `5aaacd20b404c96b` |

精度纪律:基准 x/y/s/δ 全程 **float64** 落盘(baseline_*.npz);模型输入构造时转 float32;
指标 = pred(f32→f64)vs y′(f64)。报告数字由 `gen_report_numbers_t47.py` 生成,禁止手抄。

## 文件

| 文件 | 作用 |
|---|---|
| t47_common.py | 常量/条件表/apply_variant/冒烟子集(import t46_common 单源复用) |
| variant_formulas.md | 构造与指标公式(分析前冻结) |
| select_t47.py → build_variants_t47.py | 基准清单冻结 → manifest + f64 基准数据 |
| infer_vts_t47.py | VisionTS(visionts env;A 组等变性事后核验,非门禁) |
| infer_moirai_t47.py | Moirai(t46diag env;按条件推理 + 种子重置) |
| infer_sfdx_t47.py | AutoETS 全量 + AutoARIMA 辅助臂(TimeoutPool 32w/1800s) |
| smoke_t47.py | 冒烟门禁 --part vts/moirai/sfdx/finalize |
| collect_metrics_t47.py / analyze_t47.py | 逐条件指标(长表)→ 配对 D/cost/ΔD/Δcost + §6 复核 |
| figures_t47.py / gen_report_numbers_t47.py | 图(真值核对 ≤1e-5)→ 报告数字自动生成 |
| verify_t47.py | G1–G10 十门禁 |
| preflight_t47.py / run_t47_all.sh | 预检 + 全流程编排(冒烟 FAIL 不续全量) |

## 运行(eta)

```bash
bash run_t47_all.sh /dev_data/wlt/VisionTS_Experiments/long_term_tsf
# VTS_PY=/dev_data/wlt/conda/envs/visionts/bin/python
# T46_PY=/dev_data/wlt/conda/envs/t46diag/bin/python
```

冒烟(ETTh1+ETTm1 各 2 基准窗 ×16 条件 = 64 行):
S1-a t43 原组织 [32,L,7] 复现 vs arr_vts_t46 ≤**1e-5**;
S1-b t47 单变量 [bs,L,1] vs S1-a ≤**1e-3**(修订记录见下节);
S2 δ=0 A/B 逐位相同 + 两次实际推理逐位一致;S3 配对恒等式;S4 id 对齐 + true_p ≤1e-5。

## S1-b 容差修订记录(2026-11-19,用户批准)

- **原门禁失败**:2026-11-19 04:15 冒烟 S1-b 实测 max=8.242e-4 > 原 5e-4,流水线按 `set -e` 中止(smoke_t47_vts.json;失败现场存档 `smoke_t47_vts_FAIL_5e-4_20261119.json`)。同轮 S1-a(t43 原组织 [32,L,7] vs arr_vts_t46)= **0.0 逐位**,S2 δ=0 逐位 0。
- **排查证据**(`diag_s1b_t47.py`/`diag_s1b2_t47.py`,JSON 随审计包):
  1. 出处:实际运行类 = `VisionTS`(`visionts.model.VisionTS`;eta 源码 `model.py` sha `2ce8c017…` = 本地 HEAD 40475fde 提交版逐字节一致,ckpt sha 核验);
  2. 通道独立:VisionTS.forward 将通道并入 batch(`b s n -> b n s` → `(b n) 1 f p`,model.py L115/119),**逐通道独立重建,无跨通道依赖**(初判"跨通道 attention"有误,已更正);实验证伪:仅改其他 6 通道内容、目标通道逐位不动 → 8/8 窗目标通道预测**逐位相同**;
  3. 组织差([32,L,7] 有效 batch 224 vs [32,L,1] batch 32)为数值效应:实例归一化 mean/stdev 与 resize 渲染逐位为 0,首个差异步骤 = **patch_embed**(第一个卷积/GEMM,1.07e-3),沿 encoder 放大(~2.8e-2 激活尺度)decoder 收缩至 8.24e-4;同输入同种子两次前向逐位一致(确定性);
  4. 影响量化(仅覆盖已测窗口:4 ds × 8 基准窗 × δ=±1.0·s × m=P,每 ds 16 对):换组织 |ΔD| max ≤ 2.4e-5、|Δcost| max ≤ 3.2e-5;表中"相对值" = 最大绝对差 ÷ 该组效应量中位数(≤ 9.2e-5 / ≤ 3.9e-4),**不是逐窗相对误差上限**。
- **修订决定**:S1-b 容差 5e-4 → **1e-3**;单变量路径保留,不改 7 通道推理;S1-a ≤1e-5 与其他规格不变。重跑仅 VisionTS 门禁,其余三方法冒烟推理与门禁结果复用。

## 结论措辞纪律

- 三问:①是否偏向旧水平(D 符号/量级);②加大 m 是否缓解;③稳定场景相对优势是否成立(§6)。
- 仅 VTS 自身 D>0 不足以断言"更依赖旧水平",必须以 ΔD/Δcost(共同有效窗)为直接证据;
- 构造变体非独立样本,只作描述统计;保留反向结果与反例;不把行为归因于 MAE。
