# EXP-045: VisionTS MAE-Base / Large / Huge 零样本对照(预测效果 + 计算开销)

**状态:已完成(2026-11-13 收官,待复审)。** 分支 `repro/ltsf-t45-mae-scale`(自世系推送头 2660784);目录 `long_term_tsf/repro_t45_mae_scale/`;eta 部署,CWD 契约 = `long_term_tsf/`;**本地提交,不推送**;模型源码与既有实验结果只读。

## 1. 问题与动机

论文表 2 / 表 17(附录 B.7)比较三个 MAE 变体(Base 112M / Large 330M / Huge 657M),结论"三者相近,更大模型略有下降"。既有世系只用 mae_base;**Large/Huge 从未复现**。本轮在同一冻结协议下仅换 `--vm_arch`,零样本仅推理,报告:

- **效果**:MSE/MAE 三层(逐 (ds,H) / 每 ds 四 H 等权 / 四 ds 等权 ETT-only 总体),Δ=变体−Base,论文表 17 锚点差值(report-only);
- **开销**:层 (a) 每组完整运行 wall_time_s(SUCCESS.txt);层 (b) 同一批次 CUDA 同步微基准 + 峰值显存(timing_t45.py)。

**不根据测试结果调参;禁止以分类权重或仅编码器权重替代;任何未解释的缺失键/形状不匹配即停。**

## 2. 冻结口径

- 范围:表 17 的 ETT 四数据集子集,H∈{96,192,336,720};L/P=表 8/表 6(ETTh1 2880/24、ETTh2 1728/24、ETTm1 2304/96、ETTm2 4032/96);r=c=0.4;双线性;原始图像方向;features=M;bs=32;seed=2021;train_epochs=0;无 inverse(论文标准化口径);其余参数沿用 EXP-001。
- Base 16 组 = EXP-001 归档复用(read-only,`save_repro_zeroshot/`);新增 Large/Huge 各 16 组 = 32 组;无联合网格、无选参。
- **Base 总体锚点(硬门)**:16 组自算总体 == `0.34439535439014435 / 0.37042006477713585`(≤1e-6;EXP-018 collect_t18.py:75-76 钉值;MG7/MG11/collect 三处把关)。它是每个 Δ 的分母。
- 论文对照纪律:表 17 锚点 3 位小数、**仅引用 ETT 逐格**(修正 #5:底行六数据集总体数字 OCR 错位风险——0.34617 与表 2 Huge 总体 ~0.315 不符——**全轮不引用**,ETT-only 总体自算);任何"接近论文"都不是门。

## 3. 来源与权重(修正 #4 相关)

- 官方入口:`run.py --vm_arch {mae_base,mae_large,mae_huge}`(run.py:103,无 choices 限制);权重经 `visionts/model.py MAE_ARCH`(:16-20)映射 `mae_vit_{base,large,huge}_patch{16,16,14}`,`torch.load(key='model')` + **strict=True** 全量加载(:34-44)。
- 权重来源:`https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_{base,large,huge}.pth`(官方 MAE repo 指定 URL)。
- **TLS 事件与最终下载方式(2026-11-13)**:初检时该 CDN leaf 证书 `*.fbaipublicfiles.com`(签发者 DigiCert Global G2 TLS RSA SHA256 2020 CA1)已于 2026-11-11 过期,curl(OpenSSL)与 wget(GnuTLS)一致正确拒验——属服务器端证书过期,非本机 CA 陈旧;曾预备"跳过校验+四件套绑定"方案并经用户裁决,但**同日复检发现服务器已换发新证书(notBefore 2026-10-09 → notAfter 2027-04-25),正常 HTTPS 校验通过,最终以标准证书校验 curl 直接下载,未使用任何跳过校验手段**(修正 #4 满足)。内容完整性仍由四件套绑定:
  1. 官方 URL 出处(本节);
  2. 实测 sha256 + 字节数写入 `t45_common.py CKPTS`,preflight 硬门(base==EXP-001 归档值 `c551c37b…ade0e`,447,725,789 B);
  3. 官方入口路径 strict 全量加载(P2b,任何缺失键/形状不匹配即停);
  4. 参数量 vs 论文表 17 头部 112/330/657M(±2%,P2b)。
- 架构事实(源码核实):Base/Large patch16、num_patch=14;Huge patch14、num_patch=16;embed/depth/heads = 768/12/12、1024/24/16、1280/32/16,decoder 均为 512/8/16;各架构原生 sin-cos 位置编码(无插值);确定性掩码(noise=mask buffer,前 npi 列可见)。

## 4. 目录与日志隔离(修正 #2)

- 目录单一用途:**`save_t45_base_replay/`**(仅 2 组默认重放)、**`save_t45_large/`**、**`save_t45_huge/`**;**没有 `save_t45_base/`**(16 组真实 Base 全部复用 EXP-001)。
- 每组独立日志,名含 ds/H/arch:`logs/t45_<base|large|huge>_<DS>_<H>.log`。
- run.py setting 不含 vm_arch(run.py:143/172)→ model_id 亦嵌入架构:`VisionTS_<DS>_<H>_<short>`。

## 5. 几何(修正 #3)

`update_config`(visionts/model.py:60-98):完整 224×224 图像,patch 网格 `num_patch×num_patch`(Base/Large 14×14,Huge 16×16),**前 npi 列 patch 可见**;`npi = int(((L+pad_left)/(L+pad_left+pad_right+H))×num_patch×0.4)`,clamp 最小 1;`input_resize` size 元组 = **(224, patch_size×npi)——第二元是可见图像的宽度像素**,Base/Large 宽 16×npi(patch16 整除)、Huge 宽 14×npi(patch14 整除);有效 mask 比例 = 1−npi/num_patch;`output_resize` = (P, int(round(224×scale_x)))。静态断言(import 时):npi_raw≥1、ctx≤16K、pad_left=0、**宽度==patch×npi 且被 patch 整除**。几何检查对象 = 实际宽度、patch 网格、可见列数(不是"固定高 224 的整除性")。

示例(ETTh1/H96,P=24,pad 全 0):Base npi=int(0.96774×14×0.4)=5(宽 80/可见 5 列,mask 9/14);Huge npi=int(0.96774×16×0.4)=6(宽 84/可见 6 列,mask 10/16)。预检产出 `logs/geometry_table_t45.csv`(48 行 = 16 (ds,H) × 3 arch 公式期望)。

## 6. 门禁 MG1-MG11(执行顺序见 §9,修正 #1)

| 门 | 内容 | 运行点 |
|---|---|---|
| MG1 | 8 框架 md5 == EXP-030 世系指纹;3 ckpt sha256+字节(base==EXP-001 归档值) | preflight P2 + verify --all |
| MG2 | preflight 清单(GPU≥20GB free/磁盘≥50GB/CSV sha/EXP-001 16 组资产/env)+ geometry_table_t45.csv 静态断言 | S1 |
| MG3 | 窗口数闭式:48 配置 pred/true shape==(N,H,7),N=4K−H+1(架构无关) | --reuse/--smoke/--new |
| MG4 | 真值逐位:32 新组 true.npy == truth_from_std('test')(按 (ds,H) 缓存)== EXP-001 true.npy | --smoke/--new |
| MG5 | 有限性:全部 pred/true finite | 同上 |
| MG6 | f64 cast-then-recompute vs metrics.npy ≤1e-6;日志 log_metrics 三源互证 ≤1e-12 | 同上 |
| MG7 | EXP-001 复用再核(配置/ckpt/框架/真值/f64)+ **Base 总体锚点门 ≤1e-6** | S1b(--reuse,先于冒烟) |
| MG8 | 参数生效与几何:init 记录 arch;update_config 实测 == 公式(ds,H,P,L,patch/num_patch 按 arch,含 norm_const=0.4);宽度==patch×npi;Base 与 Large 几何记录全等(同 patch 族)、Huge 不同(公式预测) | --smoke/--new |
| MG9 | 冒烟 6 组:2 组 Base 重放 pred/true 与 EXP-001 **逐位**;4 组新架构(Large/Huge × ETTh1/H96、ETTm2/H720)真值/几何/参数生效/f64 全查;架构生效性(Large≠Base、Huge≠Large 逐位不等) | S2 |
| MG10 | 重入安全:skip-if-done 指纹不符 → ABORT exit 4 不覆盖(负例,沙箱目录) | --all |
| MG11 | **最终门(修正 #1:消费 timing+collect 产物,最后跑)**:means_t45.json vs metrics.npy ≤1e-12(48 行);Base 总体锚点复检;timing 48 行完整(预热/同步/inference_mode/eval/顺序释放字段齐全) | S7 |

逐组验收:`verify_t45.py --group <short> <DS> <H>` 在每 group 跑完立即执行(run_group),失败即停保留现场;记录 `logs/group_accept.jsonl`;--all 校验 34 项(32 新 + 2 重放)全覆盖。

## 7. 冒烟组成(S2,6 组,硬门后自动全量)

| 组 | 配置 | 检查重点 |
|---|---|---|
| 1-2 | Base 默认重放 ETTh1/H96、ETTm1/H96(`save_t45_base_replay/`) | 与 EXP-001 归档 pred/true **逐位**;链路端到端(hourly+minutely) |
| 3-4 | Large × ETTh1/H96、Large × ETTm2/H720 | strict 权重生效;patch16 几何;轻+重耗时预算发现 |
| 5-6 | Huge × ETTh1/H96、Huge × ETTm2/720 | **Huge patch14/num_patch16 几何(npi=6 vs Base 5)、宽 14npi 整除**;最重新组前置 |

## 8. 开销比较口径

- **层 (a)**:完整运行 wall_time_s(含数据读取+推理+保存,来自 SUCCESS.txt)。Base 复用行为 EXP-001 旧环境历史值,**仅参照,不与新模型耗时比较**;本轮 Large/Huge 组同环境直接可比。
- **层 (b)**:`timing_t45.py` 微基准(公平三方对比):每 (ds,H) 用官方 test loader 首个 bs=32 批次(同一张量)喂三个架构;fp32;`model.eval()` + `torch.inference_mode()`(修正 #6,无梯度);预热 3 次 → CUDA 同步 → 重置峰值显存 → 5 次计时(`perf_counter` 包裹 [同步;前向;同步]);`max_memory_allocated` 记峰值;**架构顺序加载、`del`+`empty_cache()` 释放后下一个**(无跨模型残留);计时区仅含 GPU 常驻批次上的前向——不含 loader IO/模型加载/保存;模型加载单独记 `load_s`。

## 9. 运行顺序(修正 #1)

```
S0 env → S1 preflight(含 strict 加载+参数量)→ S1b verify --reuse(MG7)
→ S2 冒烟 6 组 + verify --smoke(硬门,过→自动继续)
→ S3 主队列 32 新组(skip-if-done 断点续跑;每 group 逐组验收)
→ S4 verify --all(MG1-MG10+MG11a,硬门)
→ S5 timing → S6 collect(Base 总体锚点硬门)→ S7 verify --mg11(最终门)
→ S8 pack(<50MB + sha256 sidecar,/dev_data/wlt 暂存后拉回 temp/)
```

复跑命令:

```bash
ssh wlt@222.201.145.32
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf
PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python \
  GPU_ID=0 nohup bash repro_t45_mae_scale/run_t45_all.sh \
  > repro_t45_mae_scale/logs/driver.log 2>&1 &
```

skip-if-done 全队列生效:中断后重跑自动跳过已完成组;SUCCESS.txt 指纹不符 → ABORT exit 4 不覆盖。

## 10. 权重 sha256

| arch | 文件 | sha256 | bytes | 来源 |
|---|---|---|---|---|
| base | mae_visualize_vit_base.pth | `c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e` | 447,725,789 | EXP-001 归档(eta 在位) |
| large | mae_visualize_vit_large.pth | `d0072183ec21f3b3e50874a957254ea4d03c2d7255d0ad47aebb85a868a6bdb6` | 1,318,315,181 | 官方 URL,2026-11-13,标准 HTTPS 校验(见 §3 披露;与 HTTP content-length 一致) |
| huge | mae_visualize_vit_huge.pth | `49ca3f48bfde7cd51feb4f6e63fd1b859e5bc11824730426a124bcda301f037e` | 2,628,481,101 | 同上 |

## 11. 预算(实测回填 2026-11-13)

- 逐组 wall_time_s(层 a,SUCCESS.txt):Base 重放 41s(ETTh1/H96)/146s(ETTm1/H96);Large hourly 53-85s、minutely 246-312s/组;Huge hourly 115-220s、minutely 504-734s/组。S2 冒烟 + S3 主队列总墙钟 ≈ 2h40m(12:45→15:25)。
- S5 timing 实测 ≈ 80min——瓶颈是每配置重建官方 test loader 与首批(H 数据集分钟级 CSV + 万级窗口展开,~5min/配置);前向微基准本身仅 24 次前向/配置。verify --all 36s(669 查);collect ~1min;MG11 <1s。
- 磁盘实测:save_t45_base_replay 73M + save_t45_large 1.9G + save_t45_huge 1.9G(eta 数组合计 ≈ 3.9G)+ ckpt 3 件 3.9GiB;审计包 572KB。

## 12. env / commit SHA(回填区)

- eta:wlt@222.201.145.32,L20 46GB,`PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python`(torch 2.12.0+cu126),CWD=/dev_data/wlt/VisionTS_Experiments/long_term_tsf;顺序单卡,共租授权 free−4GB>峰值。
- 本轮 commit 世系(本地分支 `repro/ltsf-t45-mae-scale`,自 2660784):脚手架 `a6410f3` → run_one_t45.sh 修复 `6c03e62`/`ea74067` → verify/collect 几何修复 `d08eaf5` → timing/collect 热修 `69dadb5`(运行代码终头)→ 交付记录 `2346b9b` → 归档收尾(仅 GB→GiB 标签文案 + README,collect 纯重算,无推理/计时)。infer_one_t45.py/t45_common.py 自 `a6410f3` 起未再改动(运行指纹稳定);verify/collect/timing/driver/README 按 §12 规则热修。
- 框架 md5 == EXP-030 世系指纹(与 t43/t44 同一 8 文件);`infer_one_t45.py` / `t45_common.py` 入运行指纹,S1 冻结;verify/collect/timing/driver/README 可热修不触指纹;ckpt 与框架文件只读。

## 13. 交付记录(2026-11-13 回填+收尾)

- **执行过程说明**:S5 timing 在 driver4.log 中成功(48 行 json/csv/图);S6 首次因图代码解包 bug 报 COLLECT FAIL(driver4.log 原样保留作历史);collect 热修(`69dadb5`)后于 2026-11-13 16:08 **人工顺序完成 S6 collect(锚点硬门 PASS)→ S7 verify --mg11(7/7)→ S8 打包**,未重跑任何推理或计时;logs/verify_t45_mg11.json 为最终成功运行产物。
- 审计包:`temp/VisionTS_t45_mae_scale_20261113.tar.gz`(<50MB;不含 pred/true npy 与权重)+ 外置 `.sha256`。谱系:v1 `505cd9b0…` 作废(RESULT TLS 溯源段为"跳过校验"旧文案,与实际"证书续期后标准校验下载"不符);v2 `dba4e5fd…` 作废(GB→GiB 标签修正前);**v3 终版,sha256 以外置 `.sha256` sidecar 与交付说明为准**。
- 数组留存 eta(wlt@222.201.145.32):`/dev_data/wlt/VisionTS_Experiments/long_term_tsf/{save_t45_base_replay(73M), save_t45_large(1.9G), save_t45_huge(1.9G)}`;权重三件 `/dev_data/wlt/VisionTS_Experiments/ckpt/`。
- 结果已出:三层表 + Δ + 论文表 17 锚点差值(report-only,48/48 格 |Δ_paper|≤0.0005)+ 开销两层(层 a wall_time_s;层 b 同批微基准 base 0.360s/1.63GiB、large 0.740s/2.45GiB、huge 1.713s/4.09GiB);ETT-only 总体 Base 0.34439535/0.37042006(锚点精确复现)、Large +0.00149/+0.00062、Huge +0.00494/+0.00145——与论文"三者相近、更大模型略有下降"方向一致;逐格局部反例按保留(RESULT_t45_mae_scale.md)。
- 门禁:verify --all 669/669(含 MG10 沙箱负例 exit 4)、verify --mg11 7/7、collect Base 总体锚点硬门 PASS;逐组验收 34/34(group_accept.jsonl)。
- **本地提交,不推送,待复审;memory 已更新。**
