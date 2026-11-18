# EXP-046 报告:VisionTS 历史特征诊断(验证集,H=96,四方法对照)

日期:2026-11-18(**v2,评审修正版**)| 分支 `repro/ltsf-t46-histdiag`(本地,未推送)| 十门禁全过(verify_t46.json,G1–G10)

> **v2 修订记录**(2026-11-18 评审意见落实,只动分析层,推理零重跑):
> ① 修复案例筛选 bug(analyze_t46.py 曾把 `mse4[3]`/AutoARIMA 误当 VisionTS;"明显较差=0"系此错,重算后较差候选 **131** 个);
> ② per-var 方向口径显式化(每 ds 取 high−low,再跨 ds 取中位),报告数字全部由 `gen_report_numbers.py` 从表格自动生成(`report_numbers_t46.json` / `report_tables_t46.md`),不再手抄;
> ③ 新增幅度分层分析(hist_std 三分位内特征中位切分),撤回 v1"幅度不解释差距"断言;撤回 v1"换 patch 不影响 gap 方向"未验证判断;
> ④ verify 扩至十门禁(G7 特征/分箱复算、G8 方向重算、G9 案例重算、G10 幅度分层重算),审计包补入 `features_t46.csv`。

## 1. 目的与解释边界

观察 VisionTS 在哪些**输入历史特征**下相对对照方法变差,区分「VisionTS 相对弱势线索」与「样本本身难(各方法共同变难)」。**只做线索发现,不支持机制结论**:窗口重叠(同一验证集滑窗采样),1792 样本(4 ds × 64 起点 × 7 变量)非独立证据;所有结论一律表述为"相对弱势线索",不认定视觉机制缺陷。本轮完成后停止;扩展(Weather/Electricity/其他 H/干预)待用户决定。

## 2. 执行摘要(必答三问)

| 问题 | 答案 | 关键证据(全部出自自动生成表) |
|---|---|---|
| ① 差距是否随某历史特征稳定变化? | **是——`recent_old` 最稳定,`trend` 次之(高幅度层内减弱)** | recent_old:单元级 high>low 25/28(对 Moirai)/23/28(对 ETS);变量级(每 ds 取差再跨 ds 中位)对 Moirai **6/7**、对 ETS **7/7** 中位为正;trend 对 Moirai 7/7、对 ETS 6/7 |
| ② 是否跨变量? | **大体跨变量,但非全部同向(如实修正)** | recent_old 对 Moirai:HUFL +0.089 / HULL +0.084 / MUFL **−0.006**(唯一负,近零) / MULL +0.055 / LUFL +0.070 / LULL +0.141 / **OT +0.205**;对 ETS 7/7 正,OT +0.507 最强 |
| ③ 是否只是所有方法共同变难? | **主效应不是——gap 随 recent_old/trend 高分位扩大;但 VTS 确存在明显较差样本(131 候选),并非无失败** | 52 个四方法共同困难样本上 VTS 三对 rel_gap 全负(−0.029/−0.095/−0.026);较差候选 35/76/15/5(ETTh1/h2/m1/m2) |

反向线索:`vol` 方向以负为主(对 Moirai 5/7 变量中位为负、对 ETS 6/7 为负)——高波动输入上 VTS 相对更好;`period_corr` 单元级多数正但变量级仅 4/7,列为弱线索;`hist_std` 变量级 5/7 正但跨对不一致(对 ARIMA 2/7),且幅度分层显示其不是 recent_old 效应的代理(见 §6)。

## 3. 方法

| 臂 | 来源 | 说明 |
|---|---|---|
| vts | EXP-043 `arr_val.npz` 切片 | 完全冻结,零推理;真值核验 1792/1792 maxdiff ≤ 2.4e-7(容差 1e-5) |
| moirai | uni2ts 1.1.1,moirai-1.0-R-base | patch_size=32(**预设,尚未验证是否最优**;本轮未做 patch 扫描,不就此对 gap 方向作任何断言)、num_samples=200、逐时刻中位数;seed 2021(random/numpy/torch/cuda);batch 64 |
| autoets / autoarima | statsforecast 1.7.8 | 逐样本窗口内拟合,season_length=P 默认配置(签名/默认值快照进 sfdx_meta.json);单列标注,不称冻结模型 |

统一:train [0,12K) StandardScaler(ddof=0)标准化尺度;同历史 L/同变量/同起点;采样清单先于一切推理冻结(sha `c34bd9f0…`,重跑字节一致);特征公式先落盘(`feature_formulas.md`);tertile = 每 (ds×var) 内 64 非近常数样本三分位(实测近常数 0 个);相对 gap = (MSE_VTS−MSE_B)/(MSE_VTS+MSE_B+1e-12),正 = VTS 相对弱势线索。

聚合口径(显式):**单元级** = 每 (ds×var) 内 tertile low/mid/high 的 mean rel_gap;**变量级** = 每 ds 先取 (high 单元 rel_gap − low 单元 rel_gap),再跨 4 个 ds 取中位(`direction_by_var_t46.csv`);**幅度分层** = 同一 (ds×var) 内按 hist_std 三分位分层,层内按目标特征中位切分,各侧 mean rel_gap,差值跨 (ds×var) 单元取中位(`amplitude_strat_t46.csv` / `_summary_`)。

资源与实测调整:CPU 池 32 持久 worker(数值库线程=1/进程),超时 terminate+kill 真杀+重生+记账(selftest 验证);**冒烟(8 worker/300s)AutoARIMA ETTh1 4/8、ETTm1 7/8 超时 → 按"实测后调整"改为 32 worker/1800s**;AutoETS 冒烟 8/8 ok(~0.2s/样本)。

## 4. 覆盖率与失败台账(如实记录)

| 方法 | ETTh1 | ETTh2 | ETTm1 | ETTm2 | 全量 |
|---|---|---|---|---|---|
| vts / moirai / autoets | 448/448 | 448/448 | 448/448 | 448/448 | 100% |
| autoarima | 448/448(0 超时) | 448/448(0) | **74/448(374 超时)** | **49/448(399 超时)** | 1019/1792(773 超时) |

- 无 nonfinite、无 worker 异常死亡、无管线错误;pool_events = {timeout: 773},超时样本逐条入 `failure_ledger.csv`,不删样本、指标记 NaN;
- **截尾选择偏差警告**:minutely ARIMA 的 ok 子集 = "1800s 内可拟合完"的样本,与周期结构等特征相关,**非随机**;该臂的分钟级方向统计(含幅度分层行)只作受限辅助证据,不作主证据(用户已确认 74/49 结果保留、不补跑);
- AutoARIMA 单样本耗时跨度极大(hourly ok 45–284s;minutely 大多 >1800s 撞顶)。

## 5. 总体结果(MSE:own-ok / common4,自动生成)

| ds | vts | moirai | autoets | autoarima |
|---|---|---|---|---|
| ETTh1 | **0.6094** / 0.6094 | 0.6351 / 0.6351 | 1.3656 / 1.3656 | 0.7084 / 0.7084 |
| ETTh2 | 0.2499 / 0.2499 | **0.2112** / 0.2112 | 0.2642 / 0.2642 | 0.2248 / 0.2248 |
| ETTm1 | 0.3831 / 0.3277 | 0.3854 / **0.2644** | 1.4327 / 1.3330 | 0.3349 / 0.3349 (n=74) |
| ETTm2 | **0.1538** / 0.0786 | 0.1663 / 0.0789 | 0.2488 / 0.1027 | 0.0913 / 0.0913 (n=49) |
| ALL | **0.3490** / 0.4054 | 0.3495 / 0.3951 | 0.8278 / 0.8183 | 0.4390 / 0.4390 (censored) |

- 总体 VTS ≈ Moirai 打平(own-ok 0.3490 vs 0.3495;common4 上 Moirai 略优 0.3951 vs 0.4054),ETTh1 VTS 明显最优,ETTh2 VTS 落后;**不存在全面弱势,也不存在全面优势**;
- AutoETS 全 ds 最差(其两两 gap 主要反映 ETS 自身弱,解读时单列);
- AutoARIMA 的 ALL 行被截尾子集主导,不可与 VTS/Moirai 直接比较。

## 6. 三问详答

### ① 稳定变化的特征:recent_old(最强)、trend(次之,幅度依赖)

单元级方向汇总(direction_summary_t46.csv,n_cells = 有效 ds×var 单元):

| 特征 | vts_vs_moirai high>low | vts_vs_autoets | vts_vs_autoarima(受限) | 中位 Δgap(Moirai/ETS/ARIMA) | 单调(Moirai) |
|---|---|---|---|---|---|
| **recent_old** | **25/28** | **23/28** | 12/19 | +0.084 / +0.103 / +0.060 | 15/28 |
| **trend** | **23/28** | 19/28 | 13/20 | +0.051 / +0.060 / +0.033 | 8/28 |
| period_corr | 18/28 | 17/28 | 12/20 | +0.073 / +0.023 / +0.025 | 8/28 |
| vol | 10/28(反向为主) | 9/28(反向为主) | 10/21 | −0.036 / −0.091 / −0.002 | 2/28 |
| hist_std | 18/28 | 18/28 | 8/20 | +0.033 / +0.033 / −0.039 | 4/28 |

变量级(direction_by_var_t46.csv,口径见 §3):recent_old 对 Moirai **6/7**、对 ETS **7/7** 中位为正;trend 对 Moirai **7/7**、对 ETS 6/7;period_corr 仅 4/7(两对)→ 弱线索;vol 对 Moirai 5/7、对 ETS 6/7 为负。

### ② 跨变量:大体跨变量,非全部同向

recent_old 对 Moirai 唯一负向是 MUFL(−0.006,近零,2/4 ds 正),其余 6 变量为正;对 ETS 7/7 正。OT(目标变量)在两对上效应最大(+0.205 / +0.507)。trend 对 Moirai 7/7(+0.031~+0.121),对 ETS 6/7(MUFL −0.038)。

### ③ 主效应不是"共同变难",但存在明显较差样本

- 单元级 gap 随 recent_old/trend 高分位扩大(对照误差同升而 gap 不平)→ VTS 相对弱势线索;
- **幅度分层检验**(撤回 v1 断言后的正面结果):recent_old 效应在 hist_std 三个层内**均为正**(对 Moirai:low +0.018 [17/28]、mid +0.125 [18/28]、high +0.127 [21/28];对 ETS:+0.053/+0.138/+0.162)→ recent_old 线索**不是幅度幅值的代理**,但在最低幅度层明显减弱;**trend 的层内效应在高幅度层减弱**(对 Moirai:low +0.052 → high +0.010,15/28)→ trend 线索部分与幅度纠缠,如实降级;
- 52 个"四方法共同困难"样本(90 分位阈值 per ds×var×method;ETTh2 24 / ETTh1 22 / ETTm1 4 / ETTm2 2)上 VTS rel_gap 三对全为**负**(−0.029 / −0.095 / −0.026)→ 极端难样本上 VTS 无系统性额外劣势;
- `hist_std` 自身的方向跨对照不一致(对 Moirai/ETS 5/7 正、对 ARIMA 2/7)→ 不列为独立线索,仅作为分层控制维。

### 案例修正(v1 错误撤回)

明显较差 = 四者共同成功中 VTS 排名 4 且 MSE>1.25×次差(margin = VTS/次差)。**候选 131 个**(ETTh1 35 / ETTh2 76 / ETTm1 15 / ETTm2 5,与评审方独立重算一致),每 ds margin 前 2 入选 **8 例**(margin 3.5–15.8×,LULL 4/8,集中于 hourly):

| case | ds | var | sid | margin | mse_vts |
|---|---|---|---|---|---|
| worse | ETTh1 | LULL | 117 | 15.8356 | 0.2189 |
| worse | ETTh1 | LULL | 362 | 4.8234 | 2.2326 |
| worse | ETTh2 | LUFL | 648 | 11.4107 | 1.5001 |
| worse | ETTh2 | MUFL | 457 | 11.3892 | 2.3497 |
| worse | ETTm1 | LULL | 971 | 9.6300 | 3.4652 |
| worse | ETTm1 | OT | 1168 | 8.2785 | 0.1306 |
| worse | ETTm2 | LULL | 1482 | 3.7381 | 0.0169 |
| worse | ETTm2 | LUFL | 1754 | 3.5104 | 0.2075 |

明显较好(VTS 排名 1 且 <0.8×次优)仍为 8 例(margin 3.0–9.9×,LULL 4/8、OT 2/8)。全部 16 张案例图见 `cases_png/`(better_* 与 worse_*)。**v1"没有明显失败案例"的结论撤回**。

## 7. 局限与解释边界

1. 描述性分析,窗口重叠,1792 样本非独立;单验证集(ETT),单 H=96;
2. Moirai patch_size=32 为预设(本轮未做 patch 扫描);其绝对水平可能非最优,对 gap 的幅度有影响,**方向是否受影响未验证,不作断言**;
3. AutoETS 自身显著弱,其对 VTS 的 gap 只能作方向参考;
4. AutoARIMA minutely 臂截尾(74/49 ok)且 ok 子集与特征相关 → 该臂分钟级统计仅受限辅助证据;
5. 特征为输入历史的一维统计量,相关不等于机制;recent_old 高的窗口 = 近窗偏离旧周期结构,"视觉先验失配"解释只是线索级假设,本轮不验证;
6. 无稳定变化或跨对照不一致的特征(hist_std、period_corr 变量级)如实报告,不据此支持机制假设。

## 8. 复现与产物

```bash
# 环境 eta:/dev_data/wlt/conda/envs/t46diag(python3.11 + torch 2.4.1+cu121
#   + uni2ts 1.1.1 + statsforecast 1.7.8 + pyarrow 14.0.2;freeze 见 env_freeze_t46diag.txt)
# 权重:HF_HOME=/dev_data/wlt/hf_cache(model.safetensors sha256 8e3c2451…a05)
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf/repro_t46_histdiag
HF_HOME=/dev_data/wlt/hf_cache WORKERS=32 TIMEOUT=1800 \
  bash run_t46_all.sh /dev_data/wlt/VisionTS_Experiments/long_term_tsf
# 分析层(不重跑推理):analyze_t46.py → figures_t46.py → gen_report_numbers.py → verify_t46.py
```

产物清单见 `README_t46.md`(清单/公式/**features_t46.csv**/数组/指标/表/图/ledger/config/verify/报告数字全落盘);全量 wall:sfdx 段 ≈ 15.5h(AutoETS ~5min + ARIMA hourly 1.6h+0.9h + minutely 6.7h+6.8h),moirai 段 ~1min。

**十门禁**(verify_t46.json,ALL PASS):G1 清单 sha+字节重放;G2 全 1792 真值/对齐;G3 覆盖率↔台账↔npz 三方一致;G4 overall 重算 ≤1e-9;G5 有限性;G6 item_id↔起点;**G7 特征逐样本复算(≤1e-8)+ 三分位分箱计数复算;G8 direction_summary/direction_by_var 独立重算;G9 案例规则重算(id 与 margin 逐项一致);G10 幅度分层逐行重算**。报告全部数字出自 `gen_report_numbers.py` 自动生成的 `report_numbers_t46.json` / `report_tables_t46.md`。
