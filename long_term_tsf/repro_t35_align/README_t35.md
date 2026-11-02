# EXP-035 审计包:VisionTS 附录 B.9 图 9(c) — ETTm1 对齐系数 c 敏感性(零样本,H∈{96,192,336,720} × c∈{0.2..1.0},36 配置)

日期:2026-11-03 | eta(L20 GPU)| 分支 `repro/ltsf-t35-align-v1`(自 EXP-034 冻结头 `cd063cf`)

- **运行提交**(32 组推理与全部 verify 门禁的执行树,eta 上以该 commit 的 git archive 部署):**`b570bf0280b70c0e38d0e9af8e1955834ca2d8ba`**
- **归档脚本提交**(本包全部脚本的最终状态;其后脚本再无改动):**`e3e63e403b885aceb097ffaeabbd6d286172f3a5`**
- 提交链:158e866 首包 → 65e6d79 冒烟 dtype 修复 → b570bf0 fp_ok 行尾修复 → 688c2eb 出图容错 → e3e63e4 RESULT 行号文案修正 → aeaec65 结果落库(README/RESULT/两表/对照图/verify json/环境快照,脚本不变)

**问题**:固定其余设置,只扫 `--align_const`,实测可见图像宽度(可见 patch 列数 npi)对零样本 MSE 的影响,复现论文图 9(c) 的 ETTm1 子图。沿用 EXP-001 零样本协议与冻结 MAE-base:ETTm1、L=2304、P=96、r=0.4、双线性插值、原始图像方向、features=M、bs=32、seed=2021、train_epochs=0。

## 口径与复用

- **c=0.4 列整体复用** EXP-001 的 4 组(`save_repro_zeroshot/ETTm1_{H}/`,EXP-031 已核验),本轮零重跑;**仅新增 32 组推理** `--align_const {C}`,输出 `save_t35_align/c{C}/ETTm1_{H}/`。
- align_const 接线零框架改动:`run.py:108`(float 默认 0.4)→ `models/VisionTS.py:22`(update_config)→ `visionts/model.py` `npi=int(input_ratio×num_patch×c)`(0→1),一值驱动全部几何(input 224×w_in、output (P, w_out));8 个框架文件 md5 与 EXP-001 冻结记录逐文件一致。H=336/720 有 48 步右填充(pad_right=48)。
- 窗口数(与 L/c 无关):N=11521−H=**11425/11329/11185/10801**,首预测绝对行 46080,窗 0 历史 [43776,46080)、未来真值 [46080,46080+H);32 组共 **357,920** 窗(8×(11425+11329+11185+10801));真值跨实验逐元素可比(每组 true 双 bitwise:== std_array 重算 且 == EXP-001 同 H true;基线组第二条走 EXP-003 seed2021 固定 reshape 还原)。
- 汇总:先 4 个 H 等权平均(不按窗口数加权),全部用未舍入 `metrics.npy`;**Δ = 该 c − c=0.4 同 H**。
- 论文锚点(趋势 ONLY,原图无逐点数值):`audit_supplement/paper/VisionTS_paper_full.md` — B.9 L424、"MSE averaged on four prediction lengths" L426(等权 4H)、图 9 caption L494、(c) ETTm1 子图 L483-484、n=floor(c·N·L/(L+H)) 与 c=0.4 默认 L68、"around 0.4" L134。原图:`audit_supplement/paper/images/7cfc7191….jpg`(sha256 `eede61c804fc6bdf4f611fbfc15f24eb7bcd51e4a577d965712d510a1740682d`,包内副本 `paper/fig9c_panel.jpg`)。

## 包内容

| 路径 | 内容 |
|---|---|
| `repro_t35_align/t35_common.py` | 共享助手:ETTM1 配置、c 规范化(`%.1f`)、窗口数/首行公式、`expected_geometry(h,c)`(c 参数化精确 update_config 公式)、std_array/truth_recomputed、基线钉死自检值(0.373517170548439/0.37162137776613235,硬门禁 ≤1e-6)、EXP-031 快照加载(sha256 钉死 b454fb9a…) |
| `repro_t35_align/run_t35_group.sh` | 单组零样本推理(run_one.sh 同协议 + `--align_const`);指纹 config 行含 `align_const=$C` token(c 以 %.1f 规范化);组日志唯一写入方;`--fingerprint` 模式 |
| `repro_t35_align/verify_t35.py` | 四模式(合并 `logs/verify_t35.json`):`--baseline H`(EXP-001 全门禁+EXP-031 逐组对齐 ≤1e-12+c=0.4 探针)、`--group H C`(align_const token、true 双 bitwise、全有限、pred 逐位 ≠ c=0.4、metrics 复算一致+c 探针)、`--first_window H C`(窗 0 历史+未来真值 bitwise+变量序)、`--smoke_hash 96 0.6`(双协议首批逐位相同、双路回放各自 bitwise、钉死几何对 5→8/224×80→224×128/96×67→96×42 硬断言、三份 SHA-256 且预测哈希互异) |
| `repro_t35_align/collect_t35.py` | 门禁(4 基线+32 新组+**first_window/smoke_hash 必查项**+几何独立复算+自检 ≤1e-6)+ 36 行逐配置表 + 9 行四 H 等权表 + 趋势判读(只最低点/升降区间)+ fig9c_compare.png(`--fig` 可单独重生成;出图失败不阻塞数值,但 S6 完成需 png)+ `--selfcheck` 冒烟前基线硬门禁 |
| `repro_t35_align/run_t35_all.sh` | 驱动:preflight(ckpt sha+大小、4 组 EXP-001 在盘、EXP-003 真值、快照 sha)、启动清残留 FAILED.txt;`min`=4 组基线门禁→自检→ETTm1_96 c0.6 run+verify+first_window+smoke_hash→SMOKE_OK;`main`=4 基线复核先行→32 组 run+verify(fp_ok SKIP:指纹含 align_const token/L/P+三数组;冒烟组跳跑仍验证)→collect;失败写 logs/FAILED.txt 即停 |
| `repro_t35_align/exp031_visionts_column_snapshot.csv` | EXP-031 `summary_b4_seasonal.csv` VisionTS 列快照(sha256 b454fb9a3fc4da15930c9934e117827a2bcdc8f3c8f7f17e2622445aaaedafbf) |
| `repro_t35_align/summary_t35.csv` | 45 行(per_config 36 + h_mean 9),六位小数 |
| `repro_t35_align/RESULT_t35_align.md` | 结果报告(门禁清单/主表/逐 H 四表/趋势判读/证据链),commit 由本地 collect 落章 |
| `repro_t35_align/fig9c_compare.png` | 实测曲线(x=c,y=四 H 等权 MSE)与论文图 9(c) 原图并排 |
| `repro_t35_align/logs/verify_t35.json` | 36 组核验全记录(指纹复算/真值双 bitwise/有限值/pred≠c0.4/探针实测几何)+ first_window + smoke_hash(三份 SHA-256) |
| `repro_t35_align/logs/*.log`、`logs/{min_driver,min_driver2,main_driver,collect_main}.log`、`logs/env_snapshot_{min,main}.txt` | 32 份组日志、三轮驱动日志(main_driver.log 为出图失败的原始记录,保留)、collect 成功出图运行日志、环境快照(仅入包,*.log 不进 git) |
| `paper/fig9c_panel.jpg` | 论文原图副本(sha256 见上) |

## 门禁(全部通过)

1. **4 组基线门禁**(min 与 main 各执行一遍,任一失败即停):SUCCESS 在盘、config 符合 L=2304/P=96 + 0.4/0.4/32/0/2021、当前 ckpt sha256 == 钉死 `c551c37b…`(447,725,789 B)== 记录值、8 文件 md5 逐文件复算一致、运行目录无权重文件、N == 11521−H、true 双 bitwise(vs std_array 与 EXP-003 还原)、metric 复算 == metrics.npy、**EXP-031 VisionTS 列逐组对齐 ≤1e-12**、探针首批 vs `pred.npy[:32]` bitwise。
2. **c=0.4 四 H 等权自检(硬门禁)**:实测 == 用户全精度复算 0.373517170548439/0.37162137776613235,差 **0.00e+00/0.00e+00**。
3. **32 组新组门禁**:config 含 `align_const=$C`、true 双 bitwise(vs std_array 与 EXP-001 同 H true)、pred/true 全有限、metrics 复算一致、**pred.npy 逐位 ≠ c=0.4**(防 align_const 静默失效,max|d| 1.629–3.910)、探针(c 参数化几何 == 公式,collect 内独立复算;两处 Resize bilinear/antialias=False;首批 vs 落盘逐位)。
4. **冒烟证据(collect 必查项,硬门禁)**:`first_window_96_0.6` — 窗 0 历史 [43776,46080) 与未来真值 [46080,46176)(seq_y 末 H 步)对 std_array 逐位一致、7 变量序正确;`smoke_hash_96_0.6` — 两协议首批输入逐位相同、两路回放各自与落盘 pred.npy[:32] 逐位一致、钉死几何对 npi **5→8**/输入 **224×80→224×128**/输出 **96×67→96×42**、输入/两路预测三份 SHA-256 在录且两路预测哈希互异(in `c1c77195db68…`、pred c0.4 `aae12a21bffa…`、pred c0.6 `861ce7ac65bf…`)。
5. **几何跨配置一致性**:36 组期望几何 == 精确公式(H=336/720 pad_right=48),探针实测尺寸/npi 与期望一致。

## 从零复跑命令(eta)

```bash
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf
PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python bash repro_t35_align/run_t35_all.sh min   # 4 基线门禁+自检+ETTm1_96 c0.6 冒烟
PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python bash repro_t35_align/run_t35_all.sh main  # 4 基线复核 + 32 组 + collect
```

## 运行记录

- min(第 1 轮,02:57–03:04):4 基线门禁全绿、自检 0.00e+00,c0.6 组与 first_window 过;`--smoke_hash` 在"回放输入 == 共享首批"误报失败——forward 返回 float32(CUDA)而共享首批是 float64 原批,跨 dtype `array_equal` 在 float64 精度位上必假;`65e6d79` 统一 f64→f32 后比较。
- min(第 2 轮,03:06–03:12):全绿,SMOKE_OK(fp_ok 的 align_const token 在 SUCCESS 行尾导致本轮未 SKIP、重跑一次 c0.6,`b570bf0` 改 ERE 行尾匹配;不影响任何数值)。
- main(03:12–04:52):4 基线复核 → 32 组新推理全过(单组 105–259 s,峰值显存 1.62–1.81 GB,32 组合计 357,920 窗)+ collect 数值表落盘;出图因论文原图 jpg 未部署到 eta 抛 FileNotFoundError(仅 catch 了 ImportError),`688c2eb` 补兜 OSError 并上传 jpg 后 eta 重跑 collect 出图成功(成功运行日志 `logs/collect_main.log`,原失败日志 `logs/main_driver.log` 按审计惯例保留)。环境 `logs/env_snapshot_{min,main}.txt`(Python 3.12.13 / torch 2.12.0+cu126 / L20)。
- RESULT 内 commit 由本地 collect 落章(eta 非 git 树):拉回 36 份 metrics.npy+verify json 后本地复算,与 eta 侧数值逐位一致。

## 结果摘要(详见 RESULT_t35_align.md)

- **实测最低点 c=0.4**(四 H 等权 MSE 0.373517/MAE 0.371621)——与论文正文 "around 0.4" 一致。
- 区间走向(未舍入):0.2→0.3→0.4 连降;0.4→0.5 微升(+0.0032)、**0.5→0.6 跳升(+0.0306)**;0.6→0.7 回落(−0.0095);0.7→0.8→0.9 再升,**平均 MSE 全程最高在 c=0.9(0.407817),c=0.8 次高(0.407671),c=0.6 为局部高点(0.407349)**;0.9→1.0 微落(−0.0047)。
- 锯齿形态与 npi 整数量化一致(相邻 c 可见列数跳变/持平),论文原图同样非单调;对照图(fig9c_compare.png)两曲线逐段同构:0.4 谷、0.6 峰、0.7 回落、0.8–0.9 高位、1.0 微落——**图 9(c) 趋势复现**。
- c 越小越差但幅度温和(0.2 仅 +0.0111);c≥0.6 整体劣化 ~+0.03 MSE(+8–9%);逐 H 层 c=0.6 相对 c=0.4 的 ΔMSE 为 **H96 +0.027715、H192 +0.061071、H336 +0.045059、H720 +0.001480**(H720 仅略有上升,H192 劣化最重);全部 64 个 Δ 项中仅 2 项轻微为负(H96 c0.3 MAE −0.0005、H720 c0.3 MAE −0.0008),MSE 无一为负——c=0.4 的最优地位在逐 H 层同样稳固(见 RESULT 逐 H 表)。

## 大数组(留在 eta,不入包)

- 本轮新组:`save_t35_align/c{C}/ETTm1_{H}/results/_/{pred,true,metrics}.npy`
- c=0.4 基线(EXP-001 复用):`save_repro_zeroshot/ETTm1_{H}/results/_/`
- EXP-003 真值(基线第二 bitwise 路):`save_fullshot/ETTm1_{H}_seed2021/results/_/true.npy`
- MAE 权重:`/dev_data/wlt/VisionTS_Experiments/ckpt/mae_visualize_vit_base.pth`(447MB,绝不入包)
- 论文原图在 eta 的副本:`/dev_data/wlt/VisionTS_Experiments/audit_supplement/paper/images/7cfc7191….jpg`
