# EXP-037 README: 双试点(FRED-MD/NN5 Daily Monash 零样本 + ETTh1 All vs Rand-VM 全量训练)

分支 `repro/ltsf-t37-pilot-v1`;eta 上执行代码 = `git archive afd871f9eedaf2f1f777f84ba37c241dd108813a`,
在其上追加本 README 与三个 G4 诊断脚本入库(commit `b791be2`),均不入运行路径。
执行机 eta(`/dev_data/wlt/VisionTS_Experiments`,env `/dev_data/wlt/conda/envs/visionts/bin/python`,
python 3.12.10,torch 2.12.0+cu126,gluonts 0.17.0,numpy 2.x,pandas 2.2+,GPU L20 46GB,与租户共跑)。
上游文件(`eval_gluonts/`、`repro_fullshot/`、`visionts/`、模型源码)**零改动**;全部接入走本包
observation-only wrapper/patch,逐一记录在 `t37_monash_runner.py` docstring 与各脚本。

## 0. 一页结论

| 任务 | 状态 | 关键数字 |
|---|---|---|
| B. ETTh1 H=96 All vs Rand-VM | **28/28 门禁全绿,用户裁决采纳冻结** | All **0.446863/0.418069**,Rand-VM **0.696181/0.558270**(预训练使 MSE 比随机初始化**低 35.81%**,限定 H=96/seed=2021);LN 锚 0.347440/0.376846、Freeze 锚 0.376013/0.379026 并列 |
| A. Monash 零样本 | **13/14;NN5 7/7 用户裁决冻结数值;fred 6/7,G4 按裁决保留 FAIL——且已在作者钉定环境逐位复现锚点,差额完整归因于软件栈** | NN5 **3.513941** vs 锚 3.514067(rel 3.6e-5);fred 当前环境 **1807.417940**(rel 4.555e-2),**作者环境 1893.668241960 == 锚 1893.668241959648(rel ≈ 0)** |

**用户裁决(2026-11-04)与本轮增量**:NN5 与 ETTh1 冻结采纳;FRED-MD 保留"本环境可重复,未复现作者数值",**不批准 G4 改判**;并指出旧代码回放只换 `visionts/` 仍属当前软件环境,不足以归因。据此本轮完成:①按作者 README.md:95 钉定环境(python 3.8.18/torch 1.7.1+cu110/torchvision 0.8.2+cu110/timm 0.3.2,自建 conda env `visionts_t37author`,gluonts 0.14.4/pandas 1.5.3/numpy 1.23.5/einops 0.6.1)在同一块 L20 上重跑 FRED-MD → **MAE 1893.668241960,与作者预计算锚逐位一致(rel 0.0000e+00)**;②逐序列分解(§3.1);③修包三件(N1/N2 CSV 分名、NN5 长度三值口径、ETTh1 pred/true 先转 float64 再算误差,§5)。

## 1. 问题与口径

### A. Monash 零样本(论文 Table 15 两子集)

论文行(:406 附近):FRED-MD VisionTS MAE **1893.67** / Naive **2825.67**;NN5 Daily **3.51** / **8.26**。
周期(Table 6/B.6):FRED-MD 月频 **P=1 → L=300**;NN5 Daily 日频 **P=7 → L=1000**。
官方入口 `eval_gluonts/run.py`,flags 同 `run_monash.sh`:`--periodicity=autotune --context_len=1000
--no_periodicity_context_len=300 --batch_size 512`。冻结 MAE-Base(sha 见 §4)、双线性、r=c=0.4。
每序列仅最后一个测试窗口(官方 `dataset.py` offset=−pred_len);指标走 gluonts `evaluate_forecasts`。

**pred_len 来源(用户修正 #1 已核)**:
- NN5 Daily:原始 .tsf 携带 `forecast_horizon=56`(`logs/monash_nn5_daily_without_missing.json`
  `generation.has_forecast_horizon=true, forecast_horizon=56`;split meta_prediction_length=56;
  每序列测试标签长度实测 56)。**计划草稿中"30(回退默认)"不成立,56 为原始基准口径**。
- FRED-MD:原始 .tsf **无** forecast_horizon → 官方回退 `default_prediction_length_from_frequency`
  的频率映射 "M"→12。eta 的 pandas≥2.2 把 "M" 别名为 "ME" 导致官方 map 查不到而抛错;
  runner 安装了等值 shim(ME→12, YE→4 = 官方表值),记录于同一 json `generation.fallback_compat`。
- **context 三值分开记录(修正 #1,按用户裁决口径更正)**:nn5 Daily——**配置 1000、按周期取整
  1001(round(1000/7)*7,官方 `convert_context_len`)、实际可用/实际进入模型 735 步**(每序列
  input 长度实测恰 735,`split.input_len_*`;forecast 切片 `x[-1001:]` 在 735 长序列上取全长,
  故模型 context=735)。前轮计划草稿的"994"为取整方向笔误,作废。fred P=1 → 配置 300
  (`no_periodicity_context_len`),实际可用 716 步,实际进入模型 300。

**Naive 分母(用户修正 #4 已核)**:`t37_naive_monash.py` 用**同一** `dataset.get_dataset` 测试划分对象、
同一 `run.imputation_nan` 前向填充、同一 gluonts evaluator 与指标列表;预测为末值重复(N1,主口径;
= Woo et al. 2024/MOIRAI 报告的 Monash "Naive",Godahewa et al. 2021 基准集定义)与季节 naive
(N2,周期=autotuned P 平铺,敏感性口径)。实测:fred N1 **2825.672461** vs paper Naive 2825.67
(rel **8.7e-7**)——窗口对齐、真值、掩码全同的直接证据;nn5 N1 **8.262752** vs 8.26(rel 3.4e-4)。
N2:fred(P=1)≡N1;nn5 N2=4.351760。比值 VTS/N1 fred **0.6396**、nn5 **0.4253**(paper normalized
MAE 聚合口径下两者均 <1,方向与论文一致)。

### B. ETTh1 All vs Rand-VM(论文 Table 22 单点)

只跑 ETTh1/H=96/seed=2021,EXP-003 受审配方(`save_fullshot/ETTh1_96_seed2021/args.json` 逐项重建,
L=1152、P=24、r=c=0.4、双线性、FP32、Adam lr=1e-4、bs 256、1 epoch、全验证 MSE 选模、strict reload)。
驱动 `t37_train_driver.py` 以 `run_train.main()` **零改动**执行:argv 自 EXP-003 args.json 重建
(nargs='+' 列表展开;布尔项按 argparse 语义跳过),delta 仅 {model_id, save_dir, ft_type=full,
vm_pretrained(1|0), gpu},由 BG1 逐项核对。All=已核验 ImageNet MAE-Base 全量微调;Rand-VM=同架构
随机初始化全量训练(`--vm_pretrained 0`)。**Rand-VM 范围声明**:论文(:547)只说"同架构、全权重随机
初始化",未披露可训集合;本实验设定为"与 All 完全同配方、全部可训",RESULT 已标注。
Table 22 为 {96,192,336,720} 四长度均值,本试点单 H=96 只作方向参考,不作论文数值复现声明。

**哈希判据(用户修正 #2 已核,证据全部来自两次正式训练进程内)**:
- All 初始 vision_model 254/254 命名张量 **逐位 == 冻结 ckpt**(`vs_ckpt.verdict=ALL_KEYS_EQUAL_TO_CKPT`);
  Rand 初始 254/254 张量逐一记录且 **全部 ≠ ckpt**(verdict=RANDOM_INIT_RECORDED)。
- 两臂 canonical init sha 不同;**可训参数名单(名排序)与数量两臂完全一致**(full → 全可训)。
- BG5:VisionTS.__init__ 后与训练入口处 torch/np/python 三 RNG 状态哈希**两臂完全一致**。
- BG6:每臂恰 1 个 RandomSampler 整 epoch 迭代,51,751 个索引(7,393 窗×7 变量),
  **索引序列 sha256 两臂相同**(first20/last20 亦同)。
- BG7(辅助):首个 train batch 4 张量 sha256 两臂相同。
- BG8:两臂最终 ckpt sha256 在录且互异、均 ≠ 冻结 ckpt。
- ckpt 文件 SHA-256 单独在录(`train_<arm>.json.ckpt_sha256`,== 钉住值 c551c37b…)。

## 2. 结果

### A 表(paper Table 15 对照;证据 `logs/monash_*.json`、`logs/naive_*.json`、`save_t37_monash/naive_n{1,2}_monash_*.csv`)

| subset | pred_len(来源) | n_series | P(autotune) | VisionTS MAE(当前环境) | Naive N1 | Naive N2 | VTS/N1 | paper VTS/Naive | 作者锚 MAE |
|---|---|---|---|---|---|---|---|---|---|
| fred_md | 12(官方 freq 回退 M→12) | 107 | 1 | 1807.417940 | 2825.672461 | 2825.672461 | 0.6396 | 1893.67/2825.67 | 1893.668242 = **作者钉定环境实测值**(§3) |
| nn5_daily_without_missing | 56(原始 .tsf forecast_horizon) | 111 | 7 | **3.513941**(锚 rel 3.6e-5,G4 过,裁决冻结) | 8.262752 | 4.351761 | 0.4253 | 3.51/8.26 | 3.514067 |

autotune 事件(nn5 完整 Table 5 日频空间):P=7 **accept**(valid 3.1557)→ 30 reject(5.8451)→
365 reject(6.0132)→ 1 reject(7.0891)。fred(本环境 pandas 2.2 下月频候选塌缩为 [1],仅试 P=1
accept 1422.9744);**G4 诊断 #1 证明完整月频空间下 P=1 仍胜出**(1422.97 vs 2375.82/1923.21/2229.51),
塌缩未影响结论。耗时/峰值显存:fred 9.17 s / 6987 MB,nn5 11.13 s / 7211 MB(进程内 CUDA 统计)。

### B 表(证据 `logs/train_{all,rand}.json`、SUCCESS.txt、`logs/verify_t37.json` fullshot 节)

| 臂 | MSE | MAE | wall s | GPU 峰值 MB | paper T22 4H 均值 |
|---|---|---|---|---|---|
| **All**(ImageNet 全量微调) | **0.446863** | **0.418069** | 329.4 | 25573 | 0.534/0.470 |
| **Rand-VM**(随机初始化全量训练) | **0.696181** | **0.558270** | 329.8 | 25573 | 0.534/0.470 |
| LN 锚(EXP-003 seed2021) | 0.347440 | 0.376846 | — | — | 0.395/0.409 |
| Freeze 锚(EXP-033) | 0.376013 | 0.379026 | — | — | 0.419/0.418 |

方向结论(单 H=96/seed=2021,参考;**按用户裁决口径**:该配置下预训练使 MSE 比随机初始化
**低 35.81%**(0.446863 vs 0.696181;换向表述 Rand-VM 劣 +55.8% MSE / +33.5% MAE),支持
ImageNet 预训练权重的价值;与论文 Table 22"All≡Rand"的 4 长度均值口径不同,不能直接对照
(论文亦未披露 Rand-VM 可训范围)。All/Rand 均未达 LN 锚——单长度点不推翻 LN 的整体优势,
LN/Freeze 锚点仅并列参考)。

### B 指标的 float64 独立复算(`t37_f64_recompute.py` → `logs/recompute_f64.json`,2026-11-04)

对现存 `pred/true.npy` **先转 float64、再做全部运算**(普通 numpy,独立于 utils/metrics.py;
此前的 BG3 复算与 SUCCESS 同为 float32 路径故 0.00e+00)。结果(max |Δ| vs SUCCESS):
all MSE **0.4468631405092113**(Δ −4.13e-9)、MAE **0.4180694890654503**(Δ +7.22e-9);
rand MSE **0.6961805152025430**(Δ −7.24e-9)、MAE **0.5582699198147876**(Δ +1.85e-9);
LN 锚数组复算 MSE 0.3474401135393300 / MAE 0.3768464963232362(Δ −1.00e-8 / +4.03e-9)。
全部差异 ~1e-8 量级 = float32 舍入,与门禁容差 1e-6 相容。原始数组留在 eta 未入包。

### B 的复跑历史与跨运行方差(诚实记录)

- 早期两次 all 臂启动因 argv/枚举问题崩溃,残留目录 `save_t37_full/_failed/…_{000318,000659}` 保留。
- run1(00:03–00:18)两臂各训成一次,但因SKIP机制与 BG4 断言冲突需取证重跑,产物移至
  `save_t37_full/ETTh1_96_seed2021_{all,rand}_run1/`(SUCCESS 在录)。
- run2(正式,00:32–00:49)两臂 28/28 门禁全绿 → `B_OK`。
- **跨运行方差(run1→run2,RNG 全同由 BG5 证明)**:all MSE Δ≈**2.3e-5**
  (0.44684017→0.44686314);rand MSE 打印精度内相同、MAE Δ≈6e-8。源于 GPU 原子操作/cuDNN
  非确定性,与种子无关;单运行内 BG3 复算 0.00e+00。

## 3. G4 锚点偏差:FRED-MD 4.56% —— 已完整归因于软件栈(证据 `logs/diag_*.log`、`logs/fred_series_*.npz/json`、五个 `t37_diag_*.py`)

按计划政策"超差即停、先核对数据与口径、不调宽容度",fred G4 FAIL 后先做了四重排除诊断:

1. **完整周期搜索不变局**(`diag_fred_psweep.log`):官方 Table 5 月频全空间 {12,6,3,1} 手动跑验证集,
   P=1 仍胜(1422.97 vs 2375.82/1923.21/2229.51)→ autotune 选择正确;测试 P=12 反而 4651.12。
2. **推理确定性**(`diag_fred_psweep.log`):P=1 测试预报逐位重放一致(determinism rerun True)
   → 非运行间抖动。
3. **精度无关**(`diag_fp_sweep.log`):fp64 1807.417940 vs fp32 1807.418212(rel ~1.5e-7)
   → fp 精度解释不了 4.56%。
4. **代码时代无关**(`diag_oldcode_repro.log`):把 visionts 包回放到 `67466d3`(precomputed monash.csv
   与 run.py 最后一次同 commit 的时代)`git archive` 重建后重放 → fred MAE **1807.417940070353**
   与当前代码**逐位相同**;nn5 亦同(3.513941406127)。(**用户正确指出局限:该回放只换 `visionts/`,
   仍在当前软件环境,只能证明代码时代稳定性,不能归因。**)

**决定性对照(用户裁决后执行,`t37_diag_fred_author_env.py`)**:按作者 README.md:95 钉定
`python==3.8.18, torch==1.7.1, torchvision==0.8.2, timm==0.3.2`(eta 新建 conda env
`visionts_t37author`;gluonts 0.14.4、pandas 1.5.3、numpy 1.23.5、einops 0.6.1、pillow 10.0.1、
huggingface_hub 0.17.3;torch 1.7.1+cu110 经 PTX JIT 在 L20 上运行)。同一权重、同一 107 序列
(两环境真值逐位相等已证)、最后 300 步输入、12 步真值、P=1、bs512、fp64:

| 环境 | MAE(f64) | vs 锚 1893.668241959648 |
|---|---|---|
| 当前(visionts env,torch 2.12.0+cu126) | 1807.417940070 | rel **4.5547e-02** |
| **作者钉定(visionts_t37author)** | **1893.668241960** | **rel 0.0000e+00(MSE 228818192.977 亦同,逐位复现)** |

→ **作者锚点 = 作者钉定软件栈在该 GPU 上的数值;4.56% 差额全部来自 torch 版本数值差异,
与 GPU 硬件、代码、数据、评估口径均无关。** 按 2026-11-04 裁决,G4 门禁维持原阈值不改判:
当前环境 fred 记"本环境可重复,未复现作者数值"。

### 3.1 逐序列分解(`t37_diag_fred_series_compare.py` → `logs/fred_series_compare.json`)

- 真值两环境**逐位相等**(数据路径同一性);预报逐位不同(0/107 序列相同)。
- 总差额 +86.2503(1893.668−1807.418):作者环境在 **55** 条序列上更差、**52** 条上更好。
- **序列 56 主导**:其自身 MAE 128,836.75(当前)→ 143,167.66(作者),差 **+14,330.91**,
  单条即把总体 MAE 抬高 **+133.9**(=14330.91/107),超过总差额 +86.25;其余序列净回抵 −47.7。
- 主要回抵项(作者环境更好):序列 50(−2,695.79)、3(−1,595.24)、27(−667.07)、24/28/35/19 等。
- 即:torch 2.12 在量纲最大的序列 56 上"偶然更好",在多数中小序列上略差;净效应 −4.56%。

### 3.2 原门槛修订建议(未采纳,存档)

此前(诊断 1–4 后)曾建议 G4 改为"口径链全过 + 锚点降级为 ENV_STALE_ANCHOR 记录项";用户裁决
**不批准改判**,维持原验收阈值。本节保留仅为审计完整性。

## 4. 数据与权重版本指纹

| 对象 | sha256 | 字节 |
|---|---|---|
| 冻结 MAE-Base `ckpt/mae_visualize_vit_base.pth` | `c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e` | 447,725,789 |
| Monash fred_md 原始 zip(zenodo 下载) | `305c0edd2b5e97159c6339be4990ea97fdf86772c1edef2a0dfc836bf29f45c3` | 169,107 |
| Monash nn5_daily_without_missing 原始 zip | `14d504384a97f7730a4b2595da9f35f77ee82b1974ffcf515eb17d607f7a07bb` | 289,779 |
| gluonts 解包缓存(eta `~/.gluonts`,4 文件 sha 在 monash_*.json data_version) | 见 json | ~1.2 MB |
| ETTh1.csv | fingerprint data_sha256(EXP-003 同值,BG1 门禁) | — |

代码指纹:B 两臂 fingerprint code_md5 == EXP-003 eta fingerprint(12 文件 md5,BG1);
A 无 fingerprint 文件,以分支 commit + env 快照(`logs/env_snapshot.txt`)+ 零上游改动为准。

作者钉定环境(`visionts_t37author`,G4 决定性对照用,eta `/dev_data/wlt/conda/envs/`):
python 3.8.18 / torch 1.7.1+cu110 / torchvision 0.8.2+cu110 / timm 0.3.2 / gluonts 0.14.4 /
pandas 1.5.3 / numpy 1.23.5 / einops 0.6.1 / pillow 10.0.1 / huggingface_hub 0.17.3
(构建日志 `eta:/dev_data/wlt/t37_author_logs/build_env.out`;版本号依据仓库 README.md:95;
gluonts/pandas/numpy 等为 2024 年 Monash 评测时代合理搭配,作者未明示——已在上句如实注明)。

## 5. 运行命令(eta,复跑)

```bash
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf
PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python GPU_ID=0 \
  bash repro_t37_pilot/run_t37_all.sh min     # A 全链(两子集+naive+verify-A)
PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python GPU_ID=0 \
  bash repro_t37_pilot/run_t37_all.sh main    # B 两训练+verify-B+collect
# G4 诊断(GPU,单跑):
/dev_data/wlt/conda/envs/visionts/bin/python repro_t37_pilot/logs/t37_diag_fred.py
/dev_data/wlt/conda/envs/visionts/bin/python repro_t37_pilot/logs/t37_diag_fp.py
/dev_data/wlt/conda/envs/visionts/bin/python repro_t37_pilot/logs/t37_diag_old.py
# G4 作者环境对照(逐序列;两环境各跑一次,tag 区分):
cd repro_t37_pilot
/dev_data/wlt/conda/envs/visionts/bin/python t37_diag_fred_author_env.py --tag current
HOME=/dev_data/wlt/t37_author_home /dev_data/wlt/conda/envs/visionts_t37author/bin/python \
  t37_diag_fred_author_env.py --tag author
/dev_data/wlt/conda/envs/visionts/bin/python t37_diag_fred_series_compare.py
# 修包工具(CPU):naive N1/N2 分名导出 + pred/true f64 复算
/dev_data/wlt/conda/envs/visionts/bin/python t37_export_naive_csv.py
/dev_data/wlt/conda/envs/visionts/bin/python t37_f64_recompute.py
# 汇总(本地 CPU):python collect_t37.py(--selfcheck 为无副作用自测)
```

注意:B 重跑前须移走已有 SUCCESS 产物目录(run_train SKIP 机制);A 重跑须删 `~/.gluonts`
对应缓存子目录才会重新触发下载取证(probe_recipe 仅在 regenerate 时触发)。

## 6. eta 大件路径与指纹(不入审计包)

| 路径 | 内容 | 大小 |
|---|---|---|
| `/dev_data/wlt/VisionTS_Experiments/long_term_tsf/save_t37_full/ETTh1_96_seed2021_all/` | 正式 All 臂(checkpoint.pth sha256 见 train_all.json final_ckpt_sha256) | 442 MB |
| `…/save_t37_full/ETTh1_96_seed2021_rand/` | 正式 Rand 臂 | 442 MB |
| `…/save_t37_full/ETTh1_96_seed2021_{all,rand}_run1/` | run1 产物(跨运行方差证据) | 442 MB ×2 |
| `…/save_t37_full/_failed/` | 两次早期崩溃残留(时间戳命名) | — |
| `…/save_t37_monash/` | 两子集 visionts 官方 CSV + naive N1/N2 分名 CSV(小件,已入包) | ~几 KB |
| `/dev_data/wlt/.gluonts/` | Monash 解包缓存 | 1.2 MB |
| `/dev_data/wlt/t37_author_home/.gluonts/` | 作者环境隔离缓存(fred_md) | ~0.5 MB |
| `/dev_data/wlt/conda/envs/visionts_t37author/` | 作者钉定环境(约 6 GB;版本指纹见 §4) | — |
| `…/repro_t37_pilot/logs/` | 全部 json/log/npz 证据(小件,已入包) | — |

## 7. 扩跑预算(按用户要求按序列数/批次数估算,非简单 ×27)

- **Monash**:VisionTS 推理按序列分批(fred 107 序列 9.17 s ≈ 0.086 s/序列;nn5 111 序列
  11.13 s ≈ 0.100 s/序列)。单序列成本随 context、horizon、周期批次数增长,量级 0.05–0.5 s/序列;
  剩余 27 子集按各 subset 序列数(Monash 128K 页面)累计,粗估总 <1.5 h GPU。autotune 每候选
  一次验证预报,候选 ≤4,已含在 per-subset 经验里(nn5 4 候选共 11 s)。
- **ETTh1 全量训练**:329 s/臂(202 train batch × bs256 + 每 epoch 全验证 + 2,785 窗测试),
  1 epoch 配置;每新增 (dataset,H) 配置 ≈ 5.5 min,seed 数×长度数线性放大;4 长度×7 数据集×
  2 臂全表 ≈ 56 臂 ≈ 5.2 h(不含 data provider 差异)。
- 附加:Long-term TSF 大上下文数据集(H=720)训练时间随 epoch 内 batch 数近线性。

## 8. 门禁清单(与 `logs/verify_t37.json` 一致)

- **A(13/14,`monash` 节)**:fred/nn5 各 7 门(G1 pred_len、G2 来源、G3 autotune P、G4 锚点、
  G5 数据版本、G6 naive 对齐、G7 比值);唯一 FAIL = fred G4(§3;阈值按 2026-11-04 裁决维持,
  偏差已完整归因于软件栈,作者钉定环境逐位复现锚点)。
- **B(28/28,`fullshot` 节)**:BG1×4(指纹/args delta)、BG2×3(真值逐位==EXP-003)、
  BG3×2(复算 vs SUCCESS ≤1e-6)、BG4×6(init 证据)、BG5×2(RNG)、BG6×3(样本序)、
  BG7×1(首批)、BG8×1(终 ckpt)、BG9×2(LN/Freeze 锚)。
- 冻结状态(2026-11-04 用户裁决):**NN5 7/7 数值冻结;ETTh1 单点对照采纳冻结(限 H=96/seed=2021);
  FRED-MD 6/7 保留 FAIL(本环境可重复,未复现作者数值;归因证据 §3)。A_OK 不出。**

## 9. 交付物

- 审计包 `temp/VisionTS_t37_pilot_20261104.tar.gz`(v2,含本轮作者环境对照与修包产物;
  <50 MB,不含 npy/权重/Monash 原始数据)+ sha256 sidecar(v2 sha 以 `temp/*.sha256` 与交付消息为准)。
- RESULT:`logs/RESULT_t37_pilot.md` + `logs/summary_t37.csv`(collect_t37.py 生成,每个数字溯源)。
- 本地分支 `repro/ltsf-t37-pilot-v1`;运行代码 = `afd871f9…` 的 git archive,后续 commit
  (G4 诊断/README/修包/作者环境对照)不入运行路径;**推送与扩跑待用户复审本包后进行**。
