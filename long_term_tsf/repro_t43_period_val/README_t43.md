# EXP-043:VisionTS 周期 P 的验证集选择实验(ETT 四数据集)

**目标**:检验论文 A.2 的声明——P 从表 5 候选集(Hourly {24,168,1},15T {96,672,1})按**验证集**选出(最终 P=24/96,表 6)。复现世系(EXP-001 默认 P zero-shot、EXP-030 P=1)从未产生过验证分数,本轮补齐:48 个验证配置(4 数据集 × 3 候选 P × 4 H)→ 固定规则选 P → 三臂测试对照(验证选 P / 论文默认 P / P=1)。**验收标准 = 流程与数值自洽;选 P 是否与论文一致、测试是否改善,均如实报告,不是验收标准。**

冻结分支:`repro/ltsf-t43-period-val`,自 e662ed5(t42 已推送头)。代码头 COMMIT_SHA **`781bc86df07a38cb69fa637347bf6d32584c62b0`**(281c6e5 主体 → fdeb67e 冒烟修复 → 5054aca 解析/指纹卫生 → b7938f1 collect 常量 → 781bc86 tie 三计数措辞;verify_t43/collect_t43 不在任何运行指纹内,t43_common/val_pass/infer_one_t43 与运行时逐字节一致,md5 a4812b52 本地==eta 复核)。

## 1. 配置(与 EXP-001 逐项一致,仅 --periodicity 变动)

| 项 | 值 |
|---|---|
| 数据集 / L(ctx) | ETTh1 2880、ETTh2 1728、ETTm1 2304、ETTm2 4032(`run_one.sh` 配置表) |
| H | 96 / 192 / 336 / 720 |
| 权重 | 冻结 MAE-base `mae_visualize_vit_base.pth`,sha256 `c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e`(447,725,789 B) |
| 其余 | r=c=0.4(`--norm_const/--align_const`)、bilinear、bs=32、train_epochs=0、seed=2021、features=M、`--inverse` 不开(论文标准化口径) |
| 候选 P | ETTh1/2:{24,168,1};ETTm1/2:{96,672,1}(paper_full.md L348-354,表 5) |
| 复用 | 默认 P 臂 = EXP-001 `save_repro_zeroshot/`(16 组,P=24/96);P=1 臂 = EXP-030 `save_a2p1_zeroshot/`(16 组);只读,复用前 VG7/VG8 三层再核(配置指纹/真值逐位/指标复算) |

测试运行 `run.py` 调用与 EXP-001 **逐字节相同**,仅 `--periodicity` 注入;观测层 `infer_one_t43.py`(承 EXP-030 `infer_one_p1.py` 机制)记录每次 `update_config` 几何与 2D 分段,框架零改动。

## 2. 填充处理(统一公式;手写表仅供说明,门禁一律公式重算)

官方 `visionts/model.py update_config`:

- `pad_left = (-L) % P`,`pad_right = (-H) % P`(Python 取模语义;等价于官方 `P - x%P if x%P else 0` 分支)
- `num_patch_input = int(((pad_left+L)/(pad_left+L+pad_right+H)) × 14 × 0.4)`,min 1;`num_patch_output = 14 − num_patch_input`
- `scale_x = (pad_left+L) // P / int(224 × num_patch_input/14)`;输入 resize `(224, int(224×npi/14))`,输出 resize `(P, round(224×scale_x))`;`mask_ratio = 1 − npi/14`
- 左 pad replicate(L118),右 pad 后按 pred_len 截断(L145)

说明性 pad_right 表(**供阅读,verify 不消费手写表**):

| P \ H | 96 | 192 | 336 | 720 |
|---|---|---|---|---|
| P=168(ETTh,L≠H 整除) | 72 | 144 | 0 | 120 |
| P=96(ETTm,L 整除) | 0 | 0 | **48** | **48** |
| P=672(ETTm) | 576 | 480 | 336 | 624 |
| P=24 / P=1 | 0 | 0 | 0 | 0 |

pad_left:P=168 时 ETTh1=144、ETTh2=120;P=672 时 ETTm1=384、ETTm2=0;P=24/96/1 时 0(24 整除全部 ETTh 的 L 与 H;96 整除 ETTm 的 L;H336/H720 对 P=96 余 48)。**评审修正(2026-11-11):P=96 时 H=336、720 的 pad_right 均为 48(此前手写表误为 0,已按统一公式纠正)**;门禁 VG10 从公式重算并与实测 `update_config` 记录断言一致,不信任任何手写表。

## 3. 数据划分(显式定义,评审修正 #2)

- 月块行数 **K**:hourly K=720,15T 分钟 K=2880(`30×24×(4?)`)。
- 训练段 `[0,12K)`(StandardScaler 仅拟合该段);**验证目标段 `[12K,16K)`**;测试目标段 `[16K,20K)`。
- 验证 split 的输入从 `12K−L` 起(官方 border),窗口 i 的目标 = 绝对行 `[12K+i, 12K+i+H)`;N = 4K−H+1(hourly 2785/2689/2545/2161,minutely 11425/11329/11185/10801)。
- **不以窗口数相同作为划分正确的依据**:每配置显式记录 split 字段(K/val_start/val_end),目标真值与确定性重算(`truth_from_std`:f64 读 CSV → 训练段 StandardScaler → f32 → 滑窗)**逐位核对**(VG4),数据集对象上断言 `len(data_x) == 4K+L`。

## 4. 验证驱动(为何绕过官方 val 路径)

`data_provider(args,'val')` 为 shuffle=True / drop_last=True 且 `vali()` 按 batch 平均 MSE(data_factory.py L26-35);zero-shot(train_epochs=0)从不运行 `vali()`——**零样本世系中验证分数不存在**。故 `val_pass.py` 直接实例化官方 `Dataset_ETT_{hour,minute}`(flag='val',border/scaler 与官方逐语句一致)+ DataLoader(**官方 test loader 语义**:bs=32、shuffle=False、drop_last=False、num_workers=10),forward 循环**逐语句镜像 `Exp_Long_Term_Forecast.test` L203-258**(dec_inp 构造、f_dim、无 inverse、无 amp)。

- SSE/SAE **float64 累加**(评审修正 #3);`mse_from_sse = SSE/cnt` 未舍入入 JSON,选参用未舍入值。
- `--crosspath`(冒烟):同一验证批次经原推理路径循环体(exp.test L203-258 逐语句)二次前向,断言输出逐位相等(冒烟对照,评审修正 #3)。
- 测试侧冒烟:ETTh1/H96/P24 经 `run_one_t43.sh` 新跑,与 EXP-001 归档组 pred/true 逐位对照(同代码+同数据+同 GPU+同形状 → 逐位,承 t41 同形状重跑结论)。

## 5. 选参规则(补充设定,先固定;先验收后选,评审修正 #3)

每数据集选使 **4 个 H 等权平均验证 MSE(float64,未舍入)** 最小的 P;并列优先论文 P,再取较小 P。该聚合规则为**本轮补充设定**(论文只说按验证集选,未规定聚合方式),在冒烟前冻结于此。

流程硬序:**S3 48 配置全部通过验收门(VG3 窗口数/VG4 真值逐位/VG5 有限性/VG6 指标复算/VG10 几何公式)→ S3.5 `verify --valfull` 全绿 → S4 才生成 ledger**(`selection.json` + `val_table.csv` + `LEDGER_OK`,sha256 链)。此后任何测试指纹嵌入 `ledger_sha256`;**禁止据测试结果重选**;VG9 以 mtime+sha 双证先选后测。

## 6. 测试对照

三臂:验证选 P / 论文默认 P(EXP-001)/ P=1(EXP-030)。逐 H、数据集 4H 等权均值、总体 16 组等权,报告 MSE/MAE 及相对默认 P 的绝对/百分比变化。选 P 臂与默认或 P=1 重合的 (ds,H) 直接复用对应归档(仍过 VG6/VG7/VG8),其余新跑(每数据集 ≤4 组,总计 ≤16)。**评审修正 #4:无任何"接近论文值"硬门**——与论文表 9 的偏差仅由 EXP-001 既有 summary 记录并在此引用,全部对照作为观察报告。

## 7. 门禁(verify_t43.py;执行前全定义)

| 门 | 内容 |
|---|---|
| VG1 | 代码/权重完整性:8 框架文件 md5 == EXP-030 世系指纹(preflight P6 现场核过);ckpt sha+字节 |
| VG2 | preflight 清单:GPU≥20GB free、磁盘≥50GB、EXP-001 16 组 + EXP-030 16 组资产在位 |
| VG3 | 窗口数闭式:48 配置 == 4K−H+1 |
| VG4 | 真值逐位:val 目标 == truth_from_std([12K,16K));测试目标 == 同管线([16K,20K));EXP-001 true.npy 逐位交叉验证重算管线本身 |
| VG5 | 有限性:全部 val/test pred/true finite |
| VG6 | 指标复算:f64 vs 归档 ≤1e-6(新建/复用组同);metric() 与 SSE 口径一致;**不含论文对照** |
| VG7 | EXP-001 复用再核:配置指纹(周期=论文 P/seed/bs/epochs/norm/align)+ ckpt sha + 框架 md5 + 真值 + 指标 |
| VG8 | EXP-030 复用再核:同 VG7(周期=1)+ 与 EXP-030 验收 metrics json 逐位 |
| VG9 | ledger 时序:验收 json、LEDGER_OK mtime 先于全部新测试 SUCCESS;sha 链一致 |
| VG10 | 填充几何:update_config 实测 == 公式重算(48 val + 全部新 test) |
| VG11 | 选参 sanity:均值从 val json 重算 ≤1e-12;argmin+并列规则可追溯 |
| VG12 | 重入安全:SUCCESS.txt 指纹不符 → ABORT(exit 4)不覆盖(负例测试) |
| VG13 | 冒烟一致性:6 验证单元(ETTh1/H96、ETTm1/H96 × 3 P)+ crosspath 逐位 + 测试路径冒烟与 EXP-001 逐位 |

## 8. 预算与磁盘(评审修正 #5)

- 验证:12 个 (ds,P) 进程 × 4 H ≈ 37-151s/组(EXP-030 实测同量级)→ GPU ≈75-90 min;测试新组 ≤16 ≈30-40 min;冒烟 ≈10-15 min;合计 **GPU ≈2-2.5h,墙钟 ≈3-4h**。
- 磁盘:48 组验证 pred/true 未压缩 **≈6.05 GB** + 新测试 ≤16 组 **≈2.02 GB**(npz 压缩存储,实际更小);预检门槛 ≥50GB free 维持(留足余量)。
- eta:`/dev_data` 27TB free,L20 45GB free(部署时点);顺序执行,共租授权 free−4GB>峰值。

## 9. 环境

eta(ssh wlt@222.201.145.32):`/dev_data/wlt/conda/envs/visionts/bin/python`,torch 2.12.0+cu126;框架 md5 与 EXP-030 世系指纹现场一致(preflight P6);数据 CSV sha256 现算记录,由真值逐位门约束。部署为 tar(CWD 契约 = `long_term_tsf/`)。

## 10. 运行

```
PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python bash repro_t43_period_val/run_t43_all.sh
```

stages:S0 env → S1 preflight → S2 冒烟(6 验证单元 crosspath + 测试路径冒烟 + verify --smoke/--reentry;**全过自动继续,无需确认**)→ S3 val 12 进程 → S3.5 verify --valfull(硬门)→ S4 ledger → S5 新测试(ledger 绑定)→ S6 verify --all → S7 collect → S8 pack(<50MB + sidecar;pred/true npy、arr_val.npz 留 eta)。

## 11. 交付记录(2026-11-12)

- **结果**:48 验证配置全部过验收门(valfull 528/528);四个数据集验证选 P = **24/24/96/96,与论文表 6 全部一致**(均唯一 float64 极小值,无并列;余量最小 ETTh2 0.0097,P=168 次之)。选 P 臂与默认 P 臂逐项相同(同归档),总体 16 组 MSE 0.344395/MAE 0.370420;P=1 参照臂 0.570011/0.501725(+65.5% MSE,与 EXP-030 冻结结论一致)。
- **门禁**:verify --all(VG3-VG13,7 模式)**982/982 ALL PASS**;冒烟 VG13 76/76 + 重入负例 3/3;S5 新组 0(选 P 臂全部复用 EXP-001 归档,复用臂经 VG7 三层再核)。
- eta 大件:`/dev_data/wlt/VisionTS_Experiments/long_term_tsf/save_t43_period_val_val/`(48 组 arr_val.npz,压缩后 ≪6.05GB 未压缩口径)、`save_t43_period_val_smoke/`(测试路径冒烟,SUCCESS/shape_record/metrics);`save_t43_period_val/` 为空(选 P 与论文全一致,无新测试组)。
- 审计包:`~/MMTS/temp/VisionTS_t43_period_val_20261112.tar.gz` + `.sha256` sidecar(<50MB:代码+README+logs+selection/ledger+val json+SUCCESS/shape_record/metrics;不含 pred/true npy 与 arr_val.npz)。同名 v1 `8e5865b0…` 已被 v2 `845bb577…` 取代并作废(tie 三计数措辞修正);包 sha 只记 sidecar 与交付记录,不入包内 README。
- 本地提交,不推送。
