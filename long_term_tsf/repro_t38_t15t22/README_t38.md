# EXP-038 README:Monash 图 5/表 15 全量(作者钉定环境,29 论文行)+ 表 22 四 ETT All vs Rand-VM(seed 2021)

分支 `repro/ltsf-t38-t15t22`(自 EXP-037 冻结头 `f115434`);eta 执行代码 =
`git archive` 于本 README 所引 commit(见 §8 运行记录);上游文件(`eval_gluonts/`、
`repro_fullshot/`、`visionts/`、模型源码、EXP-037 一切产物)**零改动**;全部接入走
observation-only wrapper/patch,逐一记录在 `t38_monash_runner.py` docstring 与各脚本。

双环境(与 EXP-037 一致):
- **A 线主表环境 = 作者钉定** `visionts_t37author`(python 3.8.20 / torch 1.7.1+cu110 /
  gluonts 0.14.4 / pandas 1.5.3 / numpy 1.23.5 / einops 0.6.1,`HOME=/dev_data/wlt/t37_author_home`
  — 每条 A 命令显式 export,防默认 HOME 缓存泄漏与假下载计时;同一块 L20,下载与推理计时均诚实记录)。
- **B 线环境 = 当前** `visionts`(python 3.12.13 / torch 2.12.0+cu126 / gluonts 0.17.0,
  与 EXP-003/033/037 B 侧锚点同环境)。

## 0. 一页结论

| 任务 | 状态 | 关键数字 |
|---|---|---|
| A. Monash 表 15 全量(29 论文行) | **完成,门禁 300/302 全过(0 硬失败;2 项为 CIF G6 设计内 SKIP)** | 30 子集锚 rel 全部 ≤1e-3(最差 sunspot 7.879e-04);29 比值几何均值 **0.7367** vs 论文 0.729(rel 1.05% < 2% WARN 线);CIF 池化行 vs 池化锚 rel 2.05e-04 |
| B. 表 22 四 ETT(All vs Rand-VM,seed 2021) | **完成,门禁 432/432 全过(0 FAIL)** | 30 臂 sampler 长度全部 == 公式;4H 平均层 LN 列最大 \|Δ\| 0.00237(ETTm1)、Freeze 列最大 \|Δ\| 0.00049(ETTm2);Overall LN 0.3325/0.3700 vs 论文 0.333/0.369,Freeze 0.3595/0.3750 vs 0.360/0.375;All 列 Overall 0.4283 vs 论文 0.417(+0.011);Rand-VM(单种子全量重训设定)Overall 0.4759;All-vs-Rand(相对 Rand)ETTh1 低 24.54%、ETTh2 低 4.24%、ETTm2 低 1.54%,**ETTm1 高 3.33%**(单种子下预训练收益不单调) |

## 复审修订(2026-11-05,commit `85fbe63f2db229f58139b87d30b613ef4d984520`;二轮收尾 `dc3c253`+`9acaada`;打包态代码头 = `9acaadae643af0f509cbceeff39c2082e234401a`,其后仅本 README 文档性提交)

按用户复审 4 项缺项逐项落地,全部基于既有产物复算/修正,不重训
(另含打包修复 `6357345`:pack_a 文件清单先追加 90 个指标 CSV 再做
nv==90 硬检查,修首轮 "metric CSVs in pack = 0" 的检查顺序错误):

1. **BG3 真 float64 复算**:旧路径在 float32 数组上调 metric 后仅转型结果;
   修正为 pred/true **先 astype(float64) 再计算**,覆盖新增 30 组(verify BG3)
   与复用 2 组(新增 BG3_metrics_f64,t37 归档 pred/true 同法);核验记录落
   `logs/recompute_f64.json`(32 条,含 mse/mae_f64、SUCCESS 值、Δ)并并入
   verify_t38.json fullshot 节。注意:SUCCESS 由官方代码在 fp32 下计算,f64
   复算允许 ≤1e-6(METRIC_TOL)级差异。
2. **Monash 总分差异分析**:RESULT_t38_monash.md 新增"Geomean print-precision
   analysis"段——论文印刷值复算 geomean = **0.728579(→0.729)**,未舍入实测
   0.736658(rel 1.05%);主因 Traffic Hourly 2dp 印刷(0.01/0.03)使该行
   印刷比仅一位有效数字。未舍入实测汇总保持为主报告数字。
3. **ETT 报告修正**:T1 计时列改为 train\* = train_wall − vali_wall(官方
   train() 在 epoch 内调 vali,原列重复计时);gpu_peak 改为三相峰值 + overall
   max(原 driver 单值只反映末次 test 相);预算表不再把复用 ETTh1_96 两臂列为
   待跑;新增 Paper-difference analysis 段(All 0.4283/Rand 0.4759 vs 论文
   0.417;LN 在 16/16 config MSE 上同时优于 All 与 Rand)。
4. **审计包补齐**:包 A 补 `save_t38_monash/` 90 个原始指标 CSV
   (visionts_monash_<ds> + naive_n1/n2;旧包因路径缺前缀静默漏收,现打包时
   硬校验 ==90);包 B 补复用两组原始证据(t37 train json/SUCCESS/fingerprint)
   与 `recompute_f64.json`;两包均含本 README;打包 sha 以 sidecar 为准。

### 二轮收尾(2026-11-05 复审通过后,冻结前;`dc3c253`+`9acaada`)

1. **归因口径收紧**:Paper-difference analysis 中"单种子波动""Rand-VM 训练
   设定差异"明确标注为**待验证解释(candidate explanations, NOT verified)**;
   删除"LN/Freeze 因引用冻结值而必然匹配论文(by construction)"的表述,
   改为观测事实:4H 层 max |ΔMSE| LN **0.00237**、Freeze **0.00049**。
2. **残留数字修正**:门禁 **432/432**(原 430 漏计复用对 BG3_metrics_f64
   两门);LN 数据集层最大 Δ 0.00237(ETTm1,原误记 0.0009);ETTm1 All
   相对 Rand **高 3.33%**(分母统一改为 Rand);预算表 s/batch 改纯训练
   (train_wall − vali_wall,与 T1 train\* 同口径);RESULT 门数行改为动态
   n/n(432/432)。
3. **补传校验文件**:两包重打后,`.sha256` sidecar 与包同放 `temp/` 一并交付。

## 1. A 线口径(Monash Table 15 / 图 5)

- 官方入口 `eval_gluonts/run.py` `evaluate()` 进程内重放,flags 同 `run_monash.sh`:
  `--periodicity=autotune --context_len=1000 --no_periodicity_context_len=300 --batch_size 512`;
  冻结 MAE-Base(sha `c551c37b…`)、双线性、r=c=0.4、fp64。
- **重放双硬门**(先行):FRED-MD(rel ≤1e-3 且 N1 四舍五入 == 2825.67)与 NN5 Daily
  (同 8.26);两门全过才继续队列。
- 30 个代码子集按官方顺序(tourism_monthly 先行核对月频候选 {12,6,3,1}→P=12),
  每子集独立 runner+naive;单子集失败 → FAILED.txt 记录 + 重试 + 队列继续。
- **pred_len 独立溯源(用户修正 #1)**:每子集记录原始 .tsf forecast_horizon / 官方频率
  fallback(M→12、W-SUN→8 等,pandas≥2.2 别名 shim 在录)/ 官方硬覆盖(aus_elec=336、
  pedestrian=24)/ CIF 特例(:83-86 按后缀过滤并强制 meta)/ M4 recipe 常量
  (gluonts datasets.py partial `prediction_length`,Monthly 18 / Weekly 13 / Daily 14 /
  Hourly 48)/ zenodo 候选塌缩;label 窗口长度逐一实测 == meta(证据在 monash_<ds>.json)。
  4 个钉定子集(fred 12 / nn5_daily 56 / cif_2016_6 6 / cif_2016_12 12)为硬门,其余证据。
- **P(表 6)autotune 全事件在录**;accepted vs 表 6 期望(EXPECTED_P 30 项)逐项对照,
  replay 两子集硬门,其余 flag+异常清单;绝不事后改期望值。
- **锚点**:作者未舍入 MAE 全 30 行(`eval_gluonts/precomputed_results/monash.csv`,
  逐字抄录于 t38_common.AUTHOR_ANCHOR),rel ≤1e-3;超差保留该行 + 异常清单,不调宽容度。
- **N1 主分母**(Woo et al. 2024 / Godahewa et al. 2021 的 Monash "Naive" = 末值重复,
  与 VisionTS 同 harness、同划分、同 imputation);N2 季节 naive(P=autotuned)敏感性。
  N1 四舍五入须 == 论文印刷值(replay 两子集硬门,其余 flag;CIF 无直接行 → 池化行 G9)。
- **CIF 2016 池化(用户修正 #2)**:官方把 CIF 拆 cif_2016_6(15 条,H=6)+ cif_2016_12
  (57 条,H=12);论文一行 = 两子集按序列数加权池化。运行时核验 n6/n12、池化 VTS 与
  池化 N1;池化 VTS vs 池化未舍入锚 rel ≤1e-3 为硬门;论文印刷 570907.24 与池化锚
  570907.2287 的末位不一致(锚四舍五入应为 570907.23)= 论文侧印刷口径问题,记录为
  documented flag,不作硬门。
- **几何均值 G10**:29 个 VTS/N1 比值(28 子集直接 + CIF 池化行)全部有效才计算,
  vs 论文 0.729,>2% WARN;否则报告覆盖数(coverage)。
- **图 5 对照**:`logs/fig5_monash_ratio.png`(每行实测比值 vs 论文引值,log 轴,0.729 参考线)。
- **Bitcoin**:按论文科学计数法印刷的有效数字位做 round-match(quantum 取印刷串最小位)。

### 阶段异常记录(队列期,诚实留痕)

- monash_m3_monthly:zenodo 下载中断(ContentTooShortError)→ 重试。
- monash_m3_other:runner probe 读 `meta.frequency` 未防御(该子集 TSF 元数据无此属性)
  → `getattr` 修复(commit `5aca36f`)→ 重试。
- m4_monthly:runner 进程内调 `run.evaluate` 未包 `torch.no_grad()`(官方 main 在
  run.py:214 有)→ 4.8 万序列 × 94 batch 激活图累积 42.7 GiB → CUDA OOM;包 `no_grad`
  修复(commit `18798a6`;no_grad 不改变前向数值,已完成子集结果不受影响)→ 重试。
- m4_daily / m4_hourly:GitHub 连接失败(RemoteDisconnected / Connection timed out)→ 重试。
- m4_weekly:GitHub 下载 stall 后自愈完成(668.5 s,锚 rel 1.205e-05 PASS),但 pred_rule
  溯源未识别 M4 recipe 常量 → 补 `m4_recipe_constant` 规则(commit `0d76a7b`)→ 重试刷新归因。
- GitHub/zenodo 传输 stall 由看门狗处置(240 s rchar 零增长 + CPU 近零 → 杀 runner,
  队列记 FAILED 继续;GPU 推理阶段 CPU 高不误杀,`logs/watchdog_a.log` 全程留痕)。

## 2. B 线口径(Table 22 四 ETT)

- 15 config × {All, Rand-VM} = 30 组新训练 + 复用 EXP-037 冻结 ETTh1_96 两臂
  (All 0.4468631446361542/0.4180694818496704;Rand 0.696180522441864/0.5582699179649353);
  pilot = ETTm1_96 两臂(计入 30,先跑,分钟级实测显存/耗时入预算表)。
- 当前 env(visionts);表 19 配置:L/P = ETTh1 1152/24、ETTh2 1152/24、ETTm1 2304/96、
  ETTm2 1152/96;双线性、r=c=0.4、FP32、Adam 1e-4、bs 256/drop_last=True、1 epoch、
  全验证集 MSE 选模、strict reload 后测试;seed=2021。
- argv 自各 config 的 EXP-003 args.json 重建 + delta {model_id, save_dir, ft_type,
  vm_pretrained, gpu};driver 拒绝已存在 SUCCESS 的目录。
- 门禁(BG):BG1 12 文件 code md5 == EXP-003 fingerprint + 4 CSV sha == 各自 config
  fingerprint + ckpt sha;BG2 true.npy 形状/字节公式 + **逐位 == EXP-003**;
  BG3 pred/true **先转 float64** 复算 vs SUCCESS ≤1e-6;BG4 init 逐键 bitwise
  (All 全等 ckpt / Rand 全不等,总体 init hash 两臂互异);BG5 RNG 态两臂一致;
  BG6 全 epoch 采样索引序列 sha 两臂一致 + 恰 1 次 + 长度 == 公式
  `7*(12*month_rows−L−h+1)`;BG7 首批张量 hash 两臂一致;BG8 终 ckpt sha 在录;
  BG9 LN16 锚(逐字 EXP-003 seed2021 SUCCESS 16 值)/ Freeze16(运行时读
  summary_t22.csv freeze 列,ETTm1 注明 exp001_cache 来源);BG10 复用对证据。
- 聚合:每数据集 4H 等权均(MSE/MAE × All/Rand/LN/Freeze)→ 4 数据集等权 →
  与论文表 22 在 **4H 平均层**对照;**单种子 + Rand-VM 全量训练为本次对照设定**(声明)。
- 分相计时/显存:train/vali/test 三相分别 wall + CUDA peak(instance-attr wrapper,
  reset_peak 边界;Exp_Univariate 每 epoch 恰 1 次 vali 调用核验)。

## 3. 运行命令(eta,复跑)

```bash
cd /dev_data/wlt/VisionTS_Experiments
PYTHON_A=/dev_data/wlt/conda/envs/visionts_t37author/bin/python \
AUTHOR_HOME=/dev_data/wlt/t37_author_home \
PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python GPU_ID=0 \
  bash long_term_tsf/repro_t38_t15t22/run_t38_all.sh preflight_a a_replay   # A 重放硬门
# ... a_all 跑全链(重放→队列→verify→collect→包A);b_all 跑 B 全链
# 失败子集重试(备份/回滚保护):
bash long_term_tsf/repro_t38_t15t22/retry_failed_a.sh <ds...>
# 自检(本地无 GPU):python collect_t38_monash.py --selfcheck / collect_t38_fullshot.py --selfcheck
```

## 4. 数据与权重版本指纹

| 对象 | sha256 / 证据 | 在录位置 |
|---|---|---|
| 冻结 MAE-Base ckpt | `c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e`(447,725,789 B) | preflight_a |
| TSF 子集原始 zip | zenodo bytes+sha256(逐子集) | monash_<ds>.json generation |
| M4 四子集 | gluonts `_m4` generator(GitHub M4-methods)provenance + cache diff | 同上 |
| gluonts 解包缓存 | n_files_new_or_changed + split 证据 | monash_<ds>.json data_version |
| ETT 四 CSV | sha256 == 各 config EXP-003 fingerprint | preflight_b + verify BG1 |

## 5. eta 大件路径(不入审计包)

| 路径 | 内容 |
|---|---|
| `/dev_data/wlt/VisionTS_Experiments/long_term_tsf/save_t38_monash/` | 每子集最终 forecast 矩阵 npz + 官方/naive CSV |
| `/dev_data/wlt/VisionTS_Experiments/long_term_tsf/save_t38_full/` | 30 组训练产物(权重 ckpt、pred/true npy、SUCCESS/fingerprint) |
| `/dev_data/wlt/VisionTS_Experiments/long_term_tsf/repro_t38_t15t22/logs/` | 全部 json/log/CSV/图证据(小件已入包) |

## 6. 门禁清单(与 `logs/verify_t38.json` 一致)

- **A(monash 节)**:G0 覆盖 30;G1/G2 pred_len(4 pin 硬,其余证据);G3 P vs 表 6
  (replay 硬,其余 flag);G4 锚 rel ≤1e-3(replay 硬,其余超差=保留行+排查);
  G5 数据版本(硬);G6 N1 rounding(replay 硬,CIF SKIP);G7 比值有限(硬);
  G8 三值 context(配置/取整/实际,含截断说明);G9 CIF 池化(权重+池化锚 rel 硬,
  论文印刷位 documented flag);G10 几何均值(报告+2% WARN);G11 相位计时残差 <1%(硬)。
- **B(fullshot 节)**:BG0-BG10 见 §2;聚合对照表 22(4H 平均层)。

## 7. 交付物

- 包 A `temp/VisionTS_t38_monash_t15_20261105.tar.gz` + sha256 sidecar;
- 包 B `temp/VisionTS_t38_ett_t22_20261105.tar.gz`(不含 npy/ckpt)+ sidecar;
  二轮修订后重打,终版大小与 sha256 见同目录 `.sha256` sidecar及本仓
  README 交付记录(打包内 README 以 sidecar 口径自洽);
- `logs/summary_t38_monash.csv` / `logs/RESULT_t38_monash.md`(含异常清单与 EXP-037
  现环境 FRED-MD 1807.417940 + G4 FAIL 的单独保留段);
- `logs/summary_t38_ett.csv` / `logs/RESULT_t38_ett.md`(含预算表);
- 图 5 对照 `logs/fig5_monash_ratio.png`。
- **已冻结,分支已推送**(用户 2026-11-05 复审数值验收合格并批准推送;
  增加种子仍属后续可选工作,未排期)。

## 8. 运行记录(执行代码 commit)

- A 线 runner 修订链:`5aca36f`(probe frequency getattr)→ `18798a6`(run.evaluate 包
  torch.no_grad,官方 main 即如此)→ `0d76a7b`(M4 pred_len 归因 = recipe 常量)
  → `044b1d9`(model_init 桶改为 t0→download 入口)→ `fd0cfc5`(M4 本地镜像直读:
  与 GitHub 同 URL 的 8 个 CSV 经网络镜像 ghfast.top 于 2026-11-04 下载,本地校验行数
  [Monthly 48000/Weekly 359/Daily 4227/Hourly 414] 与聚合 md5 后 scp 至
  eta `/dev_data/wlt/m4_mirror`,runner 在 recipe 执行期临时重定向 pd.read_csv 并逐文件
  记录 bytes+sha256)→ `1fc241c`(CUDA 上下文初始化移出计时窗;eval 桶延伸到
  evaluate() 返回,覆盖官方后处理)→ `3fa8c5e`(inference 桶 = 末次 forecast 起点 →
  evaluate_forecasts 入口,覆盖官方逐序列 SampleForecast 构建;timing 存 ph_t_raw)。
  另 `7b79bb9`(a_collect fig5 环境回退)、`a124b18`(gpu_free_gb 单位)、
  `a021cbc`(t38_common 补 REPO_ROOT 进 sys.path,修 B 线 visionts 导入)。
  全程逐次 md5 比对热部署。
- 数值确定性:跨遍重跑全部子集 MAE 逐位一致(如 m4_weekly 404.238371、weather
  2.062314、kdd/traffic_weekly/rideshare/hospital/temp_rain 锚 rel 0.000e+00),
  runner 修订只影响计时/归因簿记,不改前向数学。
- GitHub 直连在 eta 长时间 stall(2026-11-04 21:40-23:30),M4 直连重试 7 次仅
  m4_weekly/m4_monthly/hourly 于窗口期成功;经用户同意改为本地镜像同步(见上)。
  看门狗(v2:rchar 240s 零增长 + CPU 近零 → 杀 runner,队列记 FAILED 继续)全程留痕
  `logs/watchdog_a.log`,GPU 推理相 CPU 高不误杀。
- 队列执行:pass1(21:40)27/30;m3_monthly/m3_other/sunspot 重试修复;pass2(23:16)
  M4 3/4;m4_daily 经镜像于 pass3(23:55)完成 → 30/30。
- G11 相位残差修复链:pass1 残差 2.5-40%(`.to(device)` 与 CUDA 上下文初始化未入桶)
  → `044b1d9`+`1fc241c` → 1.4-2.6%;pass4 `3fa8c5e` 把"末次预报+SampleForecast 对象
  构建"区间并入 inference 桶并记录原始相位时间戳(`ph_t_raw`)→ 残差全部 ≈0
  (residual_frac 0.0000,G11 全过)。
- pass4(2026-11-05 01:42 启动):M4 经本地镜像分钟级完成;28/30 于 02:34 完成,
  sunspot/us_births 遇 zenodo 504 → `retry_failed_a.sh` 重试成功(fail=0)。
- 最终 verify:**302 门 300 过 / 0 硬失败 / 0 flag**(2 项非过 = CIF G6 设计内 SKIP);
  geomean 0.736658 vs 0.729(rel 0.0105,无 WARN);CIF 池化 VTS 570790.05 vs
  池化锚 rel 2.05e-04(硬门过);论文印刷位 round_match=False 为 documented flag。
- 异常清单 22 项:21 项 context_truncation(短序列子集实际喂入 context < 取整期望,
  作者侧同样截断,文档化不设门)+ 1 项 cif_paper_print(论文印刷末位问题,§1)。
- 图 5:`logs/fig5_monash_ratio.png`(29 行,log 轴,0.729 参考线;作者 env 无
  matplotlib,collect 在当前 env 下重跑生成,collect 为纯 json 后处理与 env 无关)。
- 包 A:`temp/VisionTS_t38_monash_t15_20261105.tar.gz`(<50MB;二轮修订后重打,
  首版 sha 023a66a7…、一轮修订版 3454a34f… 均已作废;终版大小与 sha256 以
  同目录 `.sha256` sidecar 为准,本地/eta 双侧校验一致)。
- B 线运行记录:
- B 线执行:b_all 于 03:17 启动;首启因 t38_common 漏 REPO_ROOT(visionts 导入失败,
  `a021cbc` 修复)重启于 03:20;pilot ETTm1_96 双臂(计入 30)MSE all 0.35602/rand
  0.38267,sampler 225127 == 公式,单臂 wall 1361/1354 s;队列 04:06-10:03 顺序
  ETTh1 192/336/720 → ETTh2 96-720 → ETTm2 96-720 → ETTm1 192/336/720,每 config
  先 all 后 rand,30/30 无失败。
- 实测预算(纯训练 s/batch = train_wall − vali_wall,二轮修正口径):
  ETTh1 1.092、ETTh2 1.217、ETTm1 1.217、ETTm2 1.224
  (RESULT_t38_ett.md 预算表;T1 含 30 臂 train/vali/test 三相 wall 与分相 GPU 峰值)。
  全程 ~6.7 h,远低于 22-24 h 保守预算。
- 最终 b_verify:**432/432 门全过(0 FAIL)**——BG1 指纹 4 件、BG2 true.npy 逐位
  == EXP-003、BG3 float64 复算 ≤1e-6(30 新组)+ 复用对 BG3_metrics_f64 两门、
  BG4 init 逐键(All 全等 ckpt/Rand 全不等、两臂 init hash 互异)、BG5 RNG 态、
  BG6 采样索引 sha 两臂一致+恰 1 次+长度==公式、BG7 首批张量一致、BG8 终 ckpt sha、
  BG9 LN16==EXP-003 SUCCESS 逐字/Freeze16、BG10 复用对证据;复用对 ETTh1_96
  (All 0.4469/0.4181,Rand 0.6962/0.5583)入表。
- 聚合(4H 等权,论文表 22 对照层):LN 数据集层最大 |Δ| 0.00237(ETTm1)/
  Freeze 最大 |Δ| 0.00049(ETTm2),均为观测结果而非构造恒等;
  Overall All 0.4283/0.4190(论文 0.417/0.414)、Rand 0.4759/0.4615、LN 0.3325/0.3700
  (0.333/0.369)、Freeze 0.3595/0.3750(0.360/0.375)。**声明**:单种子(2021)+
  Rand-VM 全量训练为本次对照设定;论文 LN 列为 3-seed 均值不可 3dp 比较;
  All/Rand 与论文的差距归因(单种子散布、Rand-VM 可训练集未披露)为**待验证解释**。
- 包 B:`temp/VisionTS_t38_ett_t22_20261105.tar.gz`(<50MB,不含 npy/ckpt;
  二轮修订后重打,终版大小与 sha256 以同目录 `.sha256` sidecar 为准,
  本地/eta 双侧校验一致)。
