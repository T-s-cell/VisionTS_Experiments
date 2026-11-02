# EXP-034 审计包:VisionTS 附录 B.8 表 18 — 插值策略对照(双线性 vs 双三次 vs 最近邻,零样本,ETT 4 数据集 × H∈{96,192,336,720})

日期:2026-11-03 | eta(L20 GPU)| 分支 `repro/ltsf-t18-interp-v1`(自 EXP-033 冻结头 `5ce5590`)| 运行头 commit **`75a1a1b6e9ae3bd7bb2a422557d8ca794fdbef31`**

**问题**:固定原始图像方向,检验把双线性插值换成双三次或最近邻后零样本误差如何变化。沿用 EXP-001 零样本配置(论文表 8 回看:ETTh1 2880/24、ETTh2 1728/24、ETTm1 2304/96、ETTm2 4032/96;r=c=0.4、bs=32、seed=2021、train_epochs=0、冻结 MAE-base)。

## 口径与复用

- **双线性列整体复用** EXP-001 的 16 组(`save_repro_zeroshot/{DS}_{H}/`,EXP-031 已核验),本轮零重跑;**仅新增 32 组推理** `--interpolation bicubic|nearest`,输出在 `save_t18_interp/{interp}/{DS}_{H}/`。
- 插值接线零框架改动:`run.py:106`(--interpolation,默认 bilinear)→ `models/VisionTS.py:22`(update_config)→ `visionts/model.py:84-92`,**一值同驱 input_resize 与 output_resize**;`visionts/util.py:29-35` safe_resize 用 torchvision Resize 且 antialias=False。8 个框架文件 md5 与 EXP-001 冻结记录逐文件一致。
- 窗口数(与 L 无关):小时 N=2881−H、首预测绝对行 11520;分钟 N=11521−H、首行 46080;16 组共 **109,840** 窗;真值跨实验逐元素可比(每组 true 双 bitwise:== std_array 重算 且 == EXP-003 seed2021 固定 reshape 还原/EXP-001 true)。
- 汇总:先 4 个 H 等权平均、再 4 个数据集等权平均,全部用未舍入 `metrics.npy`;**Δ = 变体 − 双线性**。
- 论文锚点转录自 `audit_supplement/paper/VisionTS_paper_full.md`(Table 18,L454),**三张独立常量表**(逐 H / 数据集 avg 行 / Average 行),代码不做跨层反算断言——论文均值由未舍入值平均(如 ETTh1 nearest MAE 逐 H 印刷值平均 0.44675 vs 论文印 0.446)。

## 包内容

| 路径 | 内容 |
|---|---|
| `repro_t18_interp/t18_common.py` | 共享助手:Table-8 配置表、窗口数公式、std_array/truth_recomputed(自 t22_common 移植)、expected_geometry(精确 update_config 公式)、三张论文锚点表、EXP-031 快照加载(sha256 钉死)+ 双线性总体自检值(0.34439535439014435/0.37042006477713585,硬门禁 ≤1e-6) |
| `repro_t18_interp/run_t18_group.sh` | 单组零样本推理(run_one.sh 同协议 + `--interpolation`);指纹 config 行含 interpolation token;组日志**唯一写入方**;`--fingerprint` 模式 |
| `repro_t18_interp/verify_t18.py` | 三模式(合并 `logs/verify_t18.json`):`--bilinear DS H`(EXP-001 全门禁 + EXP-031 逐组对齐 ≤1e-12 + 探针首批 vs pred.npy[:32] bitwise;任一失败硬停无补跑)、`--group DS H INTERP`(config 含 interpolation、true 双 bitwise、pred 逐位 ≠ 双线性硬门禁、探针 bitwise)、`--first_window`(窗 0 历史/变量序 bitwise) |
| `repro_t18_interp/collect_t18.py` | 48 记录全绿门禁 + 跨插值几何一致性 + 自检 ≤1e-6 + **126 项论文对照**(96 逐 H + 24 数据集 + 6 总体,超 0.001 单列不停机)+ Δ<0 反例清单 + ETTm2_720 nearest 反例重现判定;`summary_t18.csv` + `RESULT_t18_interp.md` |
| `repro_t18_interp/run_t18_all.sh` | 驱动:preflight(ckpt sha+大小、16 组 EXP-001 在盘、4 ds EXP-003 真值、快照 sha)、启动清残留 FAILED.txt;`min`=ETTh1_96 冒烟(双线性门禁+双变体+首窗×2);`main`=16 组双线性门禁**先行**(任一失败停)→ 30 组 run+verify(增强 SKIP:指纹+interpolation/L/P+三数组)→ collect;失败写 logs/FAILED.txt 即停 |
| `repro_t18_interp/exp031_visionts_column_snapshot.csv` | EXP-031 `summary_b4_seasonal.csv` 的 VisionTS 列快照,sha256 `b454fb9a3fc4da15930c9934e117827a2bcdc8f3c8f7f17e2622445aaaedafbf` |
| `repro_t18_interp/RESULT_t18_interp.md` | 结果报告(总体/数据集/逐 H 三层 + Δ + 论文对照 + 反例) |
| `repro_t18_interp/summary_t18.csv` | 63 行(per_H 48 + dataset 12 + overall 3),六位小数 |
| `repro_t18_interp/logs/verify_t18.json` | 48 组核验全记录(指纹复算/真值双 bitwise/插值探针实测/几何)+ 首窗记录 |
| `repro_t18_interp/logs/*.log`、`logs/main_driver.log`、`logs/env_snapshot_{min,main}.txt` | 32 份组日志、驱动日志、环境快照(仅入包,*.log 不进 git) |

## 门禁(全部通过)

1. **16 组双线性先行门禁**(main Stage 1,任一失败即停):SUCCESS 在盘、config 符合 Table-8 L/P + 0.4/0.4/32/0/2021、当前 ckpt sha256 == 钉死 `c551c37b…`(447,725,789 B)== 记录值、8 文件 md5 逐文件复算一致、运行目录无 checkpoint 文件、N == 2881−H/11521−H、true 双 bitwise、metric 复算 == metrics.npy、**EXP-031 原周期 VisionTS 列逐组对齐 ≤1e-12**、探针首批 vs `pred.npy[:32]` bitwise。
2. **32 组新组门禁**:以上除 EXP-003 真值路改为 EXP-001 true 外,另加 config 含 `interpolation=$INTERP`、**pred.npy 逐位 ≠ 双线性**(防插值静默失效)、探针实测两处 Resize 的模式/尺寸/antialias(ETTh1_96 三种模式均为 5 个可见 patch 列、输入 224×80、输出 24×336;探针协议与产出数组同路径,首批逐位一致)。
3. **跨插值几何一致性**:每 (ds,H) 三种插值的期望几何 == 精确公式、实际尺寸与 num_patch_input 相同(几何与插值无关)。
4. **双线性总体自检(硬门禁)**:实测 == 用户全精度复算 0.34439535439014435/0.37042006477713585,差 **0.00e+00/0.00e+00**。
5. **126 项论文对照**:超 0.001 共 **0 项**——表 18 逐 H、数据集、总体三层全部复现(最大 abs diff 见 RESULT 各表,总体层 ≤0.0004)。

## 从零复跑命令(eta)

```bash
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf
PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python bash repro_t18_interp/run_t18_all.sh min   # ETTh1_96 冒烟(双线性门禁+双变体+首窗)
PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python bash repro_t18_interp/run_t18_all.sh main  # 16 双线性门禁先行 + 其余 30 组 + collect
```

## 运行记录

- 本地 commit:包 **`75a1a1b6e9ae3bd7bb2a422557d8ca794fdbef31`**(链 5ce5590 → da539c4 首包 → 75a1a1b 探针 vm_ckpt 修复;RESULT 内 commit 号由本地 collect 重算落章)。
- 冒烟(min,01:12-01:15):ETTh1_96 双变体全过——bicubic 52s、nearest 53s、峰值 1.63 GB;三列实测 0.352750/0.383280、0.350846/0.382844、0.426214/0.423525 与论文 .353/.383、.351/.383、.426/.424 三位小数逐格吻合;首窗 [8640,11520) bitwise ×2。
- main(01:20-02:22):16 组双线性门禁全绿 → 30 组新推理(小时组 45-55s、分钟组 155-165s,峰值 1.62-1.63 GB)全过,32 组验证 + collect ~1 小时收官;环境 logs/env_snapshot_{min,main}.txt。
- 首轮 min 曾因探针 `vm_ckpt` 误传文件路径而非目录(`VisionTS.__init__` 拼接 `<file>/<file>` → isfile False → 误入下载器 makedirs 撞文件)失败一次,`75a1a1b` 修复后重跑通过;驱动启动清残留 FAILED.txt 的修复为收尾提交。

## 结果摘要(详见 RESULT_t18_interp.md)

- 总体:bilinear 0.344395/0.370420;**bicubic +0.0074 MSE/+0.0030 MAE(+2.14%/+0.80%);nearest +0.0284/+0.0202(+8.25%/+5.46%)**——插值越简单误差越大,双三次与双线性接近(小时数据集上 bicubic 逐 H 普遍微优),最近邻显著劣化。
- 数据集层:bicubic 仅 ETTm1 明显劣化(+0.0199 MSE),ETTh1 −0.0008 微优;nearest 全部数据集劣化。
- **8 项逐 H 反例(Δ<0)**:bicubic 7 项(ETTh1 全部 4 个 H、ETTh2_336/720、ETTm2_720),nearest 1 项——**论文局部反例 ETTm2/H720 nearest(MSE 0.334 < 0.343)本轮实测重现:ΔMSE −0.0088、ΔMAE −0.0013**。

## 大数组(留在 eta,不入包)

- 本轮新组:`save_t18_interp/{bicubic,nearest}/{DS}_{H}/results/_/{pred,true,metrics}.npy`
- 双线性基线(EXP-001 复用):`save_repro_zeroshot/{DS}_{H}/results/_/`
- EXP-003 真值(第二 bitwise 路):`save_fullshot/{DS}_{H}_seed2021/results/_/true.npy`
- MAE 权重:`/dev_data/wlt/VisionTS_Experiments/ckpt/mae_visualize_vit_base.pth`(447MB,绝不入包)
