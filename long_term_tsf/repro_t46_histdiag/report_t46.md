# EXP-046 报告:VisionTS 历史特征诊断(验证集,H=96,四方法对照)

日期:2026-11-18 | 分支 `repro/ltsf-t46-histdiag`(本地,未推送)| 六门禁全过(verify_t46.json)

## 1. 目的与解释边界

观察 VisionTS 在哪些**输入历史特征**下相对对照方法变差,区分「VisionTS 相对弱势线索」与「样本本身难(各方法共同变难)」。**只做线索发现,不支持机制结论**:窗口重叠(同一验证集滑窗采样),1792 样本(4 ds × 64 起点 × 7 变量)非独立证据;所有结论一律表述为"相对弱势线索",不认定视觉机制缺陷。本轮完成后停止;扩展(Weather/Electricity/其他 H/干预)待用户决定。

## 2. 执行摘要(必答三问)

| 问题 | 答案 | 关键证据 |
|---|---|---|
| ① 差距是否随某历史特征稳定变化? | **是——`recent_old`(近窗 vs 旧周期段差异)最稳定,`trend` 次之** | recent_old:high>low 单元 25/28(对 Moirai)、23/28(对 ETS)、12/19(对 ARIMA 可行子集),中位 rel_gap(high−low) = +0.084 / +0.103 / +0.060;对 Moirai 单调 15/28 |
| ② 是否跨变量? | **是,recent_old 与 trend 在全部 7 变量×两个对照同向** | recent_old 对 Moirai:HUFL +0.069 / HULL +0.093 / MUFL +0.001 / MULL +0.075 / LUFL +0.017 / LULL +0.090 / **OT +0.215**(目标变量最强);trend 7/7 同向(+0.038~+0.102) |
| ③ 是否只是所有方法共同变难? | **不是——是 VisionTS 的相对弱势线索,且集中在"近窗偏离旧周期结构/趋势强"的输入;极端难样本(四方法共同困难)上 VTS 反而无额外劣势** | rel_gap 随 recent_old/trend 高分位扩大(+0.08~+0.10);52 个共同困难样本上 VTS 三对 rel_gap 全为负(−0.029/−0.095/−0.026) |

反向线索:`vol` 方向稳定相反(高波动输入上 VTS **相对更好**,对 Moirai 6/7 变量负向,中位 −0.036);`hist_std` 无稳定方向(幅度不解释 gap)。

## 3. 方法

| 臂 | 来源 | 说明 |
|---|---|---|
| vts | EXP-043 `arr_val.npz` 切片 | 完全冻结,零推理;真值核验 1792/1792 maxdiff ≤ 2.4e-7(容差 1e-5) |
| moirai | uni2ts 1.1.1,moirai-1.0-R-base | patch_size=32(**预设,尚未验证是否最优**)、num_samples=200、逐时刻中位数;seed 2021(random/numpy/torch/cuda);batch 64 |
| autoets / autoarima | statsforecast 1.7.8 | 逐样本窗口内拟合,season_length=P 默认配置(签名/默认值快照进 sfdx_meta.json);单列标注,不称冻结模型 |

统一:train [0,12K) StandardScaler(ddof=0)标准化尺度;同历史 L/同变量/同起点;采样清单先于一切推理冻结(sha `c34bd9f0…`,重跑字节一致);特征公式先落盘(`feature_formulas.md`);tertile = 每 (ds×var) 内 64 非近常数样本三分位(实测近常数 0 个);相对 gap = (MSE_VTS−MSE_B)/(MSE_VTS+MSE_B+1e-12),正 = VTS 相对弱势线索。

资源与实测调整:CPU 池 32 持久 worker(数值库线程=1/进程),超时 terminate+kill 真杀+重生+记账(selftest 验证);**冒烟(8 worker/300s)AutoARIMA ETTh1 4/8、ETTm1 7/8 超时 → 按"实测后调整"改为 32 worker/1800s**;AutoETS 冒烟 8/8 ok(~0.2s/样本)。

## 4. 覆盖率与失败台账(如实记录)

| 方法 | ETTh1 | ETTh2 | ETTm1 | ETTm2 | 全量 |
|---|---|---|---|---|---|
| vts / moirai / autoets | 448/448 | 448/448 | 448/448 | 448/448 | 100% |
| autoarima | 448/448(0 超时) | 448/448(0) | **74/448(374 超时)** | **49/448(399 超时)** | 1019/1792(773 超时) |

- 无 nonfinite、无 worker 异常死亡、无管线错误;pool_events = {timeout: 773},超时样本逐条入 `failure_ledger.csv`,不删样本、指标记 NaN;
- **截尾选择偏差警告**:minutely ARIMA 的 ok 子集 = "1800s 内可拟合完"的样本,与周期结构等特征相关,**非随机**;该臂的 minutely 方向统计(12/19、13/20 等)只作参考,不作主证据;
- AutoARIMA 单样本耗时跨度极大(hourly ok 45–284s;minutely 大多 >1800s 撞顶)。

## 5. 总体结果(MSE,own-ok 口径 / 括号内 = 四方法共同样本 common4)

| ds | vts | moirai | autoets | autoarima |
|---|---|---|---|---|
| ETTh1 | **0.6094** | 0.6351 | 1.3656 | 0.7084 |
| ETTh2 | 0.2499 | **0.2112** | 0.2642 | 0.2248 |
| ETTm1 | 0.3831(0.3277) | 0.3854(**0.2644**) | 1.4327(1.3330) | 0.3349(0.3349,n=74) |
| ETTm2 | **0.1538**(0.0786) | 0.1663(0.0789) | 0.2488(0.1027) | 0.0913(n=49) |
| ALL | **0.3490** | 0.3495 | 0.8278 | 0.4390(censored) |

- 总体 VTS ≈ Moirai 打平(0.3490 vs 0.3495),ETTh1 VTS 明显最优,ETTh2 VTS 落后;**不存在全面弱势**;
- AutoETS 全 ds 最差(其两两 gap 主要反映 ETS 自身弱,解读时单列);
- AutoARIMA 的 ALL 行被截尾子集主导,不可与 VTS/Moirai 直接比较。

## 6. 三问详答

### ① 稳定变化的特征:recent_old(最强)、trend(次之)

方向汇总(direction_summary_t46.csv,n_cells = 有效 ds×var 单元):

| 特征 | vts_vs_moirai high>low | vts_vs_autoets | vts_vs_autoarima(截尾) | 中位 Δgap(三对) | 单调(Moirai) |
|---|---|---|---|---|---|
| **recent_old** | **25/28** | **23/28** | 12/19 | **+0.084 / +0.103 / +0.060** | 15/28 |
| **trend** | **23/28** | 19/28 | 13/20 | +0.051 / +0.060 / +0.033 | 8/28 |
| period_corr | 18/28 | 17/28 | 12/20 | +0.073 / +0.023 / +0.025 | 8/28 |
| vol | 10/28(反向) | 9/28(反向) | 10/21 | −0.036 / −0.091 / −0.002 | 2/28 |
| hist_std | 18/28 | 18/28 | 8/20 | +0.033 / +0.033 / −0.039 | 4/28 |

### ② 跨变量:是

recent_old 对 Moirai/ETS 的 (high−low) 中位差在 **7/7 变量全部为正**(唯一近零:MUFL +0.001);trend 同样 7/7 正。OT(目标变量)上 recent_old 效应最大(+0.215 / +0.295)。

### ③ 不是单纯"共同变难"

- 若只是样本难,对照误差同升而 gap 应持平;实际 **gap 随 recent_old/trend 高分位显著扩大**(中位 +0.05~+0.10)→ VTS 相对弱势线索;
- 反向验证:52 个"四方法共同困难"样本(各 (ds×var×method) ok 样本 90 分位阈值,分布 ETTh2 24 / ETTh1 22 / ETTm1 4 / ETTm2 2)上,VTS rel_gap 三对全为**负**(−0.029 / −0.095 / −0.026)→ 极端难样本上 VTS 无额外劣势;
- `hist_std` 无方向 → 幅度不解释 gap(第 5 分组维仅描述性)。

## 7. 案例(规则先落盘,确定性)

明显较好(=四者共同成功中 VTS 排名 1 且 <0.8×次优):**8 例,每 ds margin 前 2**,margin 3.0–9.9×,集中在 LULL(4/8)与 OT(2/8),见图 `cases_png/case_better_*.png`。
明显较差(排名 4 且 >1.25×次差):**0 例**——与三问③一致,VTS 没有出现灾难性相对失败样本。

## 8. 局限与解释边界

1. 描述性分析,窗口重叠,1792 样本非独立;单验证集(ETT),单 H=96;
2. Moirai patch_size=32 为预设(未做 patch 扫描),其绝对水平可能非最优;VTS-vs-Moirai 的 gap 方向不受此影响,但幅度可能变;
3. AutoETS 自身显著弱,其对 VTS 的 gap 只能作方向参考;
4. AutoARIMA minutely 臂截尾(74/49 ok)且 ok 子集与特征相关 → 其方向统计仅参考;
5. 特征为输入历史的一维统计量,相关不等于机制;recent_old 高的窗口 = 近窗偏离旧周期结构,VTS 在其上变差的"视觉先验失配"解释只是线索级假设,本轮不验证;
6. 无稳定变化的特征(hist_std)如实报告,不据此支持机制假设。

## 9. 复现与产物

```bash
# 环境 eta:/dev_data/wlt/conda/envs/t46diag(python3.11 + torch 2.4.1+cu121
#   + uni2ts 1.1.1 + statsforecast 1.7.8 + pyarrow 14.0.2;freeze 见 env_freeze_t46diag.txt)
# 权重:HF_HOME=/dev_data/wlt/hf_cache(model.safetensors sha256 8e3c2451…a05)
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf/repro_t46_histdiag
HF_HOME=/dev_data/wlt/hf_cache WORKERS=32 TIMEOUT=1800 \
  bash run_t46_all.sh /dev_data/wlt/VisionTS_Experiments/long_term_tsf
```

产物清单见 `README_t46.md`(清单/公式/数组/指标/表/图/ledger/config/verify 全部落盘);全量 wall:sfdx 段 ≈ 15.5h(AutoETS ~5min + ARIMA hourly 1.6h+0.9h + minutely 6.7h+6.8h),moirai 段 ~1min。六门禁:G1 清单 sha+字节重放、G2 全 1792 真值/对齐、G3 覆盖率↔台账↔npz 三方一致、G4 表格重算 ≤1e-9、G5 有限性、G6 item_id↔起点——**全部 PASS**(`verify_t46.json`)。
