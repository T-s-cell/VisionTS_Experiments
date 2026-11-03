# EXP-037 README: 双试点(FRED-MD/NN5 Daily Monash 零样本 + ETTh1 All vs Rand-VM 全量训练)

分支 `repro/ltsf-t37-pilot-v1`;eta 上执行代码 = `git archive afd871f9eedaf2f1f777f84ba37c241dd108813a`,
本仓库头 commit `b791be2`(在其上追加本 README 与三个 G4 诊断脚本,均不入运行路径)。
执行机 eta(`/dev_data/wlt/VisionTS_Experiments`,env `/dev_data/wlt/conda/envs/visionts/bin/python`,
python 3.12.10,torch 2.12.0+cu126,gluonts 0.17.0,numpy 2.x,pandas 2.2+,GPU L20 46GB,与租户共跑)。
上游文件(`eval_gluonts/`、`repro_fullshot/`、`visionts/`、模型源码)**零改动**;全部接入走本包
observation-only wrapper/patch,逐一记录在 `t37_monash_runner.py` docstring 与各脚本。

## 0. 一页结论

| 任务 | 状态 | 关键数字 |
|---|---|---|
| B. ETTh1 H=96 All vs Rand-VM | **28/28 门禁全绿,收官** | All **0.446863/0.418069**,Rand-VM **0.696181/0.558270**(Rand 劣 +55.8% MSE);LN 锚 0.347440/0.376846、Freeze 锚 0.376013/0.379026 并列 |
| A. Monash 零样本 | **13/14**(唯一 FAIL = fred G4 锚点,已排除法定根因,**待用户裁决门槛修订**) | NN5 **3.513941** vs 锚 3.514067(rel 3.6e-5)**全过**;fred **1807.417940** vs 锚 1893.668242(rel 4.555e-2)超差;两子集 Naive N1 与 paper Naive rel ≤ 3.4e-4 |

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
- **context 三值分开记录(修正 #1)**:fred P=1 → context=300(`no_periodicity_context_len`);
  nn5 P=7 → 官方 `convert_context_len` 四舍五入到周期倍数 = round(1000/7)*7 = **1001**
  (计划草稿 994 为取整方向笔误,以代码为准);窗口实际可用历史(input 长度)两子集分别为
  716 与 735(`split.input_len_*`,均 ≥ context,无截断)。

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

### A 表(paper Table 15 对照;证据 `logs/monash_*.json`、`logs/naive_*.json`、`save_t37_monash/*.csv`)

| subset | pred_len(来源) | n_series | P(autotune) | VisionTS MAE | Naive N1 | Naive N2 | VTS/N1 | paper VTS/Naive | 作者锚 MAE(rel diff) |
|---|---|---|---|---|---|---|---|---|---|
| fred_md | 12(官方 freq 回退 M→12) | 107 | 1 | 1807.417940 | 2825.672461 | 2825.672461 | 0.6396 | 1893.67/2825.67 | 1893.668242(**+4.56%** G4) |
| nn5_daily_without_missing | 56(原始 .tsf forecast_horizon) | 111 | 7 | 3.513941 | 8.262752 | 4.351761 | 0.4253 | 3.51/8.26 | 3.514067(**3.6e-5**,G4 过) |

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

方向结论(单 H=96,参考):Rand-VM 劣于 All **+55.8% MSE / +33.5% MAE**,支持 ImageNet 预训练
权重的价值;与论文 Table 22"All≡Rand"的 4 长度均值口径不同,不能直接对照(论文亦未披露 Rand-VM
可训范围)。All/Rand 均未达 LN 锚——单长度点不推翻 LN 的整体优势,LN/Freeze 锚点仅并列参考。

### B 的复跑历史与跨运行方差(诚实记录)

- 早期两次 all 臂启动因 argv/枚举问题崩溃,残留目录 `save_t37_full/_failed/…_{000318,000659}` 保留。
- run1(00:03–00:18)两臂各训成一次,但因SKIP机制与 BG4 断言冲突需取证重跑,产物移至
  `save_t37_full/ETTh1_96_seed2021_{all,rand}_run1/`(SUCCESS 在录)。
- run2(正式,00:32–00:49)两臂 28/28 门禁全绿 → `B_OK`。
- **跨运行方差(run1→run2,RNG 全同由 BG5 证明)**:all MSE Δ≈**2.3e-5**
  (0.44684017→0.44686314);rand MSE 打印精度内相同、MAE Δ≈6e-8。源于 GPU 原子操作/cuDNN
  非确定性,与种子无关;单运行内 BG3 复算 0.00e+00。

## 3. G4 锚点偏差:FRED-MD 4.56% 的排除法根因(证据 `logs/diag_*.log` + 三个 `t37_diag_*.py`)

按计划政策"超差即停、先核对数据与口径、不调宽容度",fred G4 FAIL 后做了四重诊断:

1. **完整周期搜索不变局**(`diag_fred_psweep.log`):官方 Table 5 月频全空间 {12,6,3,1} 手动跑验证集,
   P=1 仍胜(1422.97 vs 2375.82/1923.21/2229.51)→ autotune 选择正确;测试 P=12 反而 4651.12。
2. **推理确定性**(`diag_fred_psweep.log`):P=1 测试预报逐位重放一致(determinism rerun True)
   → 非运行间抖动。
3. **精度无关**(`diag_fp_sweep.log`):fp64 1807.417940 vs fp32 1807.418212(rel ~1.5e-7)
   → fp 精度解释不了 4.56%。
4. **代码时代无关**(`diag_oldcode_repro.log`):把 visionts 包回放到 `67466d3`(precomputed monash.csv
   与 run.py 最后一次同 commit 的时代)`git archive` 重建后重放 → fred MAE **1807.417940070353**
   与当前代码**逐位相同**;nn5 亦同(3.513941406127)。锚点在**任何代码状态**下都不可达。

**旁证**:同 harness 下 nn5 锚 rel 仅 3.6e-5(管线/数据/口径正确的强证据);N1 vs paper Naive
fred rel 8.7e-7(窗口对齐正确);git 考古确认锚点行与当前 run.py 同 commit。
**结论:作者预计算锚点对作者运行时环境(torch/GPU kernel 数值)不可移植,被 FRED-MD 的异质量纲放大**
(NN5 同尺度下同源漂移仅 3.6e-5)。

**待用户裁决的门槛修订建议**(未实施,verify_t37_monash.py 维持原 G4 FAIL):
G4 改为两层——(a) 口径链全过(数据 sha/pred_len 来源/序列数/P/N1 对齐/确定性/旧代码逐位/fp 排除
全部在录且 PASS);(b) 锚点比对降级为记录项,verdict=`ENV_STALE_ANCHOR`,实测 rel(−4.56%)与全部
诊断证据并列进 RESULT。批准后重跑 verify-A 出 A_OK;不批准则 A 保持 13/14(fred 锚点未复现,
其余 13 门全过),A_OK 不出。

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
| `…/save_t37_monash/` | 两子集 visionts/naive 官方 CSV(小件,已入包) | ~几 KB |
| `/dev_data/wlt/.gluonts/` | Monash 解包缓存 | 1.2 MB |
| `…/repro_t37_pilot/logs/` | 全部 json/log 证据(小件,已入包) | — |

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
  G5 数据版本、G6 naive 对齐、G7 比值);唯一 FAIL = fred G4(§3)。
- **B(28/28,`fullshot` 节)**:BG1×4(指纹/args delta)、BG2×3(真值逐位==EXP-003)、
  BG3×2(复算 vs SUCCESS ≤1e-6)、BG4×6(init 证据)、BG5×2(RNG)、BG6×3(样本序)、
  BG7×1(首批)、BG8×1(终 ckpt)、BG9×2(LN/Freeze 锚)。

## 9. 交付物

- 审计包 `temp/VisionTS_t37_pilot_20261104.tar.gz`(<50 MB,不含 npy/权重/Monash 原始数据)+ sha256 sidecar。
- RESULT:`logs/RESULT_t37_pilot.md` + `logs/summary_t37.csv`(collect_t37.py 生成,每个数字溯源)。
- 本地分支 `repro/ltsf-t37-pilot-v1` 头 commit `b791be2`(运行代码 = `afd871f9…` 的 git archive);**推送与扩跑待用户复审本包后进行**。
