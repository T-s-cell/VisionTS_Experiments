# EXP-047 报告:旧历史水平对预测的影响(level-shift)

分支 `repro/ltsf-t47-levelshift`(基线 = EXP-046 冻结头 21e2fa3,只新增);产物目录
`save_t47_levelshift/`。所有引用数字由 `gen_report_numbers_t47.py` 自动生成
(`report_numbers_t47.json` / `report_tables_t47.md`),表号对应 `report_tables_t47.md`。
结论为**线索级**:构造变体非独立样本,只作描述统计;不支持机制,不归因 MAE;保留反例。

## 0. 三问三答(摘要)

| 问题 | 答案 |
|---|---|
| ① VTS 是否被旧历史拉向旧水平 | **是,但非独有**:D 中位 m=1 为 0.227(84.0% 窗 D>0);直接对比 ΔD 显示其牵引显著强于 AutoETS/ARIMA,**不强于 Moirai**(Moirai D 中位 0.257 且衰减更慢) |
| ② 加大新水平持续长度 m 是否缓解 | **是,全方法单调缓解;VTS 缓解最快**(m=1→4:0.227→0.011,约 −95%;Moirai 0.257→0.115,约 −55%) |
| ③ 稳定场景相对优势是否成立(§6) | **部分成立但方向与假设不符**:VTS 在 stable 类最优(0.113),但在 changed_persist 也最优(0.253),且相对 Moirai 的优势幅度在 changed 更大——"稳定场景特有优势"未获支持 |

## 1. 设置与运行

- 基准:EXP-046 清单每 (ds×var) recent_old 最低 16 窗 → 448;成对构造 δ=s·{−1,−0.5,+0.5,+1},
  m∈{P,2P,4P};A=整段+δ,B=仅末 m 步+δ,真值 y′=y+δ;16 条件 × 448 = 7168;
  规格冻结于 `variant_formulas.md`(先于一切推理);冻结链 sha 见 README。
- 方法:冻结 VisionTS(mae_base,单变量 [32,L,1],batch32)/ Moirai(patch32、200 采样、
  逐时刻中位数、按条件推理+每次实推前重置种子 2021)/ AutoETS(窗口内拟合,season_length=P)
  / AutoARIMA 辅助臂(仅 ETTh1/h2 每 var 前 2 窗 × 16 条件 = 448 次,1800s 真超时)。
- 覆盖率(表 Coverage):vts/moirai/autoets **7168/7168 ok**,autoarima **448/448 ok**;
  零超时、零非有限、零 worker 异常(sfdx_meta:timeout=0,nonfinite=0,other_fail=0)。
- verify 门禁 **G1–G10 ALL PASS**(`verify_t47.json`);冒烟 S1–S4 全过(§6 修订记录)。

## 2. 问题①:是否被旧历史拉向旧水平

口径:D = mean_h(pred_A−pred_B)/δ(有符号;D>0 = 冲突组 B 被拉向旧水平);
ΔD = D_VTS−D_对照 仅在两方法共同有效窗上计算(本实验全窗有效,n=1792=448×4δ)。

**VTS 自身**(表 D on common-windows,m=1 行):D 中位 0.227,84.0% 窗为正 → 存在系统性
旧水平牵引。分解:
- 方向不对称(表 direction,m=1):δ 向下 D 中位 **0.297** > δ 向上 **0.175**(Moirai 两向
  0.264/0.251 更对称);
- 幅度 0.5·s 与 1.0·s 的 D 中位几乎相同(0.226/0.227),符合 D 的尺度归一化定义。

**直接对比**(表 Method difference,overall):
- vs AutoETS:ΔD 中位 **+0.232**(m=1,83.1% 正)→ VTS 牵引显著更强;vs AutoARIMA
  +0.093(61.6% 正,n=112 辅助臂);
- **vs Moirai:ΔD 中位 m=1 −0.009(48.3% 正)、m=2 −0.096(32.6%)、m=4 −0.053(35.0%)**
  → VTS 牵引**不强于 Moirai**,大 m 时反而更弱。

横向排序(牵引强度):**Moirai ≳ VTS ≫ AutoETS(≈0,中位恒 0、D>0 率仅 ~29%)**。
"VisionTS 更容易被旧历史拉向旧水平"在与统计方法的对照中成立;与同为大规模预训练
模型的 Moirai 对照中**不成立**。

## 3. 问题②:加大 m 是否缓解

表 D on common-windows(每方法三 m 全有效基准窗;n=1792 全窗):

| method | m/P=1 | m/P=2 | m/P=4 |
|---|---|---|---|
| VTS D 中位 | 0.227 | 0.029 | 0.011 |
| Moirai D 中位 | 0.257 | 0.181 | 0.115 |
| AutoETS D 中位 | 0.000 | 0.000 | 0.000 |

- 全方法单调缓解;**VTS 缓解最快**(−95%),Moirai 仅 −55%,AutoETS 本就无牵引;
- 冲突代价同向收敛:cost 中位 VTS 0.0296→0.0028→0.0008,Moirai 0.0200→0.0104→0.0042;
- Δcost(表 Method difference):vs Moirai m=1 +0.0052(VTS 代价略高)→ m=2/4 转负
  (−0.0011/−0.0013,VTS 略低);vs AutoETS 恒正(m=1 +0.0318)。
- 解读线索:VTS 对"新水平"的适应随其持续长度迅速恢复;两种预训练模型的差异随 m 拉大。

## 4. 问题③:稳定场景相对优势是否成立(§6 复核,零推理)

复用 EXP-046 全 1792 窗的 t43/t46 冻结预测;a=x[L−4P:L−P].mean、b=x[L−P:L].mean、
c=真值未来 96 均值;s=hist_std;稳定 |b−a|≤0.1s ∧ |c−b|≤0.1s;变化后持续
|b−a|≥0.5s ∧ |c−b|≤0.5|b−a|;阈值预先固定。类别计数:**stable 153 / changed_persist 298
/ other 1341**(表 Section-6,类别 × 方法,MSE;复用 t46 各方法 ok/误差):

| method | stable(153) | changed_persist(298) | other(1341) |
|---|---|---|---|
| VTS | **0.1134** | **0.2535** | 0.3972 |
| Moirai | 0.1352 | 0.3026 | **0.3844** |
| AutoARIMA | 0.1250 | 0.3890 | 0.4917 |
| AutoETS | 0.7217 | 0.4271 | 0.9290 |

- VTS 在 stable 类**绝对最优**(0.113 < ARIMA 0.125 < Moirai 0.135);
- 但 VTS 在 changed_persist 同样最优(0.253 < Moirai 0.303),且相对 Moirai 的优势幅度
  在 changed(−0.049)**大于** stable(−0.022)→ **"stable 场景特有相对优势"方向上不成立**;
- 结合 §2:VTS 牵引不强于 Moirai,故 stable 最优**不能归因于更强的旧水平依赖**;
  Moirai 在 other 类略优(0.384 vs 0.397),是 VTS 未横扫的反例,如实保留;
- 反例:AutoETS 在 stable 类 0.722 反常偏大(其余场景亦弱),本实验不解释、不剔除;
- 类别 n 小(stable 仅 153),以上为描述统计。

## 5. 门禁与复现性

- 冒烟(ETTh1+ETTm1 各 2 基准窗 ×16 条件):S1-a t43 原组织复现 **maxdiff=0.0(逐位,
  64/64)**;S1-b 单变量路径 maxdiff=8.24e-4 < 修订容差 1e-3(§6);S2 δ=0 两次实际推理
  **逐位一致**(vts/moirai/sfdx 三方法);S3 配对恒等式、S4 id 对齐+真值 ≤1e-5 全过
  (`smoke_report_t47.json`)。
- Moirai 复现性:run07 与 run08 两次全量推理,**pred/true_p/ids 全数组逐位一致**
  (按条件种子重置协议确定性)。
- collect:真值 y′ 重算 vs npz maxdiff=2.38e-7 ≤1e-5;verify G1–G10 ALL PASS
  (含 select/build 子进程重放、构造恒等式、指标与 D/cost 重算 ≤1e-9、§6 重算、图真值核对)。
- 图(4 案例):真值 vs 数组核对 maxdiff=1.17e-7(违者中止逻辑未触发);
  汇总图 + 方法差图 + 案例图见 `fig_*.png` / `cases_png/`。

## 6. S1-b 容差修订记录(2026-11-19,用户批准)

1. **原门禁失败**:首轮冒烟 S1-b 实测 max=**8.242e-4** > 原 5e-4,流水线按 `set -e` 中止
   (存档 `smoke_t47_vts_FAIL_5e-4_20261119.json`);同轮 S1-a = 0.0 逐位。
2. **排查证据**(`diag_s1b_t47.json` / `diag_s1b2_t47.json`):
   - 出处链:实际运行类 = `visionts.model.VisionTS`;eta 源码 model.py sha `2ce8c017…`
     与本地 HEAD 40475fde 提交版逐字节一致;ckpt sha `c551c37b…` 核验;torch 2.12.0+cu126 / L20;
   - **通道独立**(用户代码判读正确;初判"跨通道 attention"有误,已更正):
     forward 将通道并入 batch(`b s n -> b n s` → `(b n) 1 f p`,model.py L115/119),
     逐通道独立重建,无跨通道依赖;实验证伪:仅改其他 6 通道内容、目标通道逐位不动
     → 8/8 窗目标通道预测**逐位相同**;
   - 组织差([32,L,7] 有效 batch 224 vs [32,L,1] batch 32)为数值效应:实例归一化与
     resize 逐位 0,**首个差异步骤 = patch_embed**(第一个卷积/GEMM,1.07e-3),沿
     encoder 放大、decoder 收缩至 8.24e-4;同输入同种子两次前向逐位一致;
   - 影响量化(**仅覆盖已测窗口:4 ds × 8 基准窗 × δ=±1.0·s × m=P,每 ds 16 对**):
     换组织 |ΔD| max ≤ 2.4e-5、|Δcost| max ≤ 3.2e-5;"相对值" = 最大绝对差 ÷ 该组效应量
     中位数(≤9.2e-5 / ≤3.9e-4),**不是逐窗相对误差上限**。
3. **决定**:容差 5e-4 → **1e-3**(`t47_common.SMOKE_TOL_NEW`,注释含完整修订记录);
   单变量路径保留,不改 7 通道推理;S1-a ≤1e-5 与其他规格不变;仅重跑 VisionTS 门禁,
   其余三方法冒烟推理与门禁复用。

## 7. 局限与措辞纪律

- 构造变体由 448 基准窗派生,非独立样本;所有汇总为描述统计,不做显著性检验;
- 构造冲突(整段 vs 仅末 m 步、δ 至 ±1.0·s)强于真实窗口的典型水平变化;§6 实测中
  VTS 在 changed_persist 仍最优,提示构造实验的冲突代价不宜直接外推到真实场景;
- AutoARIMA 为辅助臂(28 基准窗,共同窗 n=112),结论仅作参考;
- 保留反例:VTS 未在 other 类最优;AutoETS stable 反常;ΔD vs Moirai 为负;
- 不归因 MAE 预训练目标,不支持机制解释;§6 的 c 用真值未来(方法无关)。

## 8. 交付物

审计包 `temp/VisionTS_t47_levelshift_20261119.tar.gz`(<50MB,sha256 sidecar,解压逐字节
diff 已验):代码 + README + variant_formulas + 冻结清单/manifest + baseline f64 npz
(基准原始输入 448×L + 真值 448×96)+ 四方法预测 npz + 指标/配对/差分表 + ledger
(全量 7616 行)+ 冒烟/诊断 JSON(含 FAIL 存档)+ 图 + verify + report_numbers/tables +
本报告。git 提交本地,**不推送**。
