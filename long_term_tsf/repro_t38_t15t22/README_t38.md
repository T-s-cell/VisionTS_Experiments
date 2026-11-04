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
| A. Monash 表 15 全量(29 论文行) | «TBD-A» | «TBD-A-geomean» |
| B. 表 22 四 ETT(All vs Rand-VM,seed 2021) | «TBD-B» | «TBD-B-headline» |

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

- 包 A `temp/VisionTS_t38_monash_t15_<日期>.tar.gz`(<50 MB)+ sha256 sidecar;
- 包 B `temp/VisionTS_t38_ett_t22_<日期>.tar.gz`(<50 MB,不含 npy/ckpt)+ sidecar;
- `logs/summary_t38_monash.csv` / `logs/RESULT_t38_monash.md`(含异常清单与 EXP-037
  现环境 FRED-MD 1807.417940 + G4 FAIL 的单独保留段);
- `logs/summary_t38_ett.csv` / `logs/RESULT_t38_ett.md`(含预算表);
- 图 5 对照 `logs/fig5_monash_ratio.png`。
- 本地提交;**推送与增加种子待用户复审(本轮不推送)**。

## 8. 运行记录(执行代码 commit)

- A 线 runner 修订链:`5aca36f`(probe frequency getattr)→ `18798a6`(run.evaluate 包
  torch.no_grad,官方 main 即如此)→ `0d76a7b`(M4 pred_len 归因 = recipe 常量)
  → `044b1d9`(model_init 桶改为 t0→download 入口)→ `fd0cfc5`(M4 本地镜像直读:
  与 GitHub 同 URL 的 8 个 CSV 经网络镜像 ghfast.top 于 2026-11-04 下载,本地校验行数
  [Monthly 48000/Weekly 359/Daily 4227/Hourly 414] 与聚合 md5 后 scp 至
  eta `/dev_data/wlt/m4_mirror`,runner 在 recipe 执行期临时重定向 pd.read_csv 并逐文件
  记录 bytes+sha256)→ `1fc241c`(CUDA 上下文初始化移出计时窗;eval 桶延伸到
  evaluate() 返回,覆盖官方后处理)。全程逐次 md5 比对热部署。
- 数值确定性:跨遍重跑全部子集 MAE 逐位一致(如 m4_weekly 404.238371、weather
  2.062314、kdd/traffic_weekly/rideshare/hospital/temp_rain 锚 rel 0.000e+00),
  runner 修订只影响计时/归因簿记,不改前向数学。
- GitHub 直连在 eta 长时间 stall(2026-11-04 21:40-23:30),M4 直连重试 7 次仅
  m4_weekly/m4_monthly/hourly 于窗口期成功;经用户同意改为本地镜像同步(见上)。
  看门狗(v2:rchar 240s 零增长 + CPU 近零 → 杀 runner,队列记 FAILED 继续)全程留痕
  `logs/watchdog_a.log`,GPU 推理相 CPU 高不误杀。
- 队列执行:pass1(21:40)27/30;m3_monthly/m3_other/sunspot 重试修复;pass2(23:16)
  M4 3/4;m4_daily 经镜像于 pass3(23:55)完成 → 30/30。
- «TBD-LOG-A»(A 线收尾补:最终 verify/collect 摘要、包 sha)
- «TBD-LOG-B»(B 线收尾补:pilot 实测、30 组耗时/峰值、verify/collect 摘要)
