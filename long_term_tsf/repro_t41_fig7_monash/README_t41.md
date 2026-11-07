# EXP-041: 补齐论文图 7 — Monash t-SNE 行

论文 Fig.7(MAE 编码器 2 维 t-SNE:ImageNet + Monash + Weather +
Electricity + ETTm1)在 EXP-002 复现时缺 Monash 来源(只有 1,900 =
ImageNet 1000 + ETTm1/Weather/Electricity 各 300)。EXP-041 以 EXP-002
机制补齐 Monash 300 行,联合重拟合 2,200×54528。**冻结资产零重采样、
零重提取**:EXP-002 的 sampling_lists 与 features_1900x54528.npy(文件
sha `5ac9bbd2…` 钉定)原样复用;EXP-002 的 1,900 行坐标不外推,而是整
体重进拟合。

## 记号(冻结)

| 记号 | 含义 | 来源 |
|---|---|---|
| H | 预测长度 | EXP-038 归档 `pred_rule.expected` × gluonts metadata 交叉核对(不一致 fail-stop) |
| P | 选出的周期 | EXP-038 归档 `autotune.accepted_p`(缺则 fail-stop;Table 6 的 EXPECTED_P 仅作对照记录) |
| L_config | 配置上下文 | P==1 → 300,否则 round(1000/P)×P(官方 convert_context_len 公式) |
| L_eff | 逐样本实际输入历史 | min(L_config, N−H);官方短历史保留(NN5 Daily 735 类),记 `short_history` |

## 采样(sample_monash.py,author 环境)

- 29 论文行 × floor(300/29)=10 = 290;余 10 给容量最大的 10 行(并列按
  PAPER_T15_ORDER 序)。容量 = Σ_series 每序列最大互不重叠窗口数
  `(N_in−L_eff)//L_eff + 1`。
- 缺额重分配链:容量 < 配额的行把缺口按"余量降序、并列按行序"转给
  其他行,迭代至不动点;仍不足则如实报错停线。
- CIF_2016_POOLED 行从 cif_2016_6/cif_2016_12 确定性 50/50 拆分(本论
  实际 5+5;仅当某子集连自身目标都画不满时才把真缺口溢给兄弟子集,
  日志显式区分)。
- 每行候选 = 逐序列 stride-1 窗 [s, s+L_eff),end ≤ N_in;随机贪心
  (seed 2021 sha256 派生)接受同序列互不重叠窗;不足时**丢弃该子集全部
  贪心选中**,改用确定性 stride-L_eff 装箱重画(装箱必达 max_windows
  容量,重分配配额以此为准;记 fill_method);CIF 不足溢给兄弟子集。
- 缺值 = 官方 `imputation_nan` 前向填充,逐窗记录原始 NaN 数。
- 逐子集与 EXP-038 归档 split 统计(n_series/min/median/max)硬核对
  ——缓存身份证据。
- gluonts 序列取自 t38 同款缓存(HOME=/dev_data/wlt/t37_author_home,
  author 环境 visionts_t37author,py3.8.20),regenerate=False 读缓存。

## 特征提取(extract_features_monash.py,visionts 环境)

- 前置硬门:冻结 1900 特征文件 sha == `5ac9bbd2…`,先于一切模型工作。
- 逐几何组 (subset, L_eff, H, P) 调 `update_config`;**掩码与几何解耦**:
  mask_ref.json 的 ids_keep_order(左 5 列 70 patches,行主序)编码为
  固定参考噪声张量(可见=0/其余=1)直接传入 forward_encoder,mask_ratio
  钉定 0.6428571343421936(9/14);逐批 assert argsort(noise)[:70] ==
  mask_ref ids。Monash 各子集 num_patch_input 在 3-5 浮动
  (input_ratio×14×0.4),结构化 model.mask 不可跨几何复用——此即统一
  左 5 列掩码无法从几何推导的陷阱,参考噪声张量绕开。
- build_ts_image 逐字复刻 VisionTS.forward 步骤 1-4(repro_fig7/common.py);
  keep = CLS + 70 可见 patch → (300, 54528) float32;有限性 assert。

## t-SNE / 30-NN

- `run_tsne_joint.py`:vstack(冻结1900, monash300) →
  **np.array_equal(merged[:1900], 冻结数组) 逐元素核对**(文件 sha 与数
  组字节不混比);参数 = EXP-002 run_tsne.py:23-27 逐字
  (perplexity=30, lr=200, init='pca', ee=12, max_iter=1000, euclidean,
  无预降维);rs=2021 主图 **拟合并复拟合逐位一致**(FG5)→
  coords×2 + rs2022/2023 稳定性 → coords×2 + params json。
- `nn_metric_t41.py`:4 个时序源 × 冻结编码器单元;2200×2200 float64
  欧氏;fill_diagonal inf + assert 自排除;分子 = 该源每行 30-NN(全体
  2200 行)中 ImageNet 邻居数,分母 = n_source×30 = 9000;邻居索引存
  npz 供独立复算。

## 门禁

preflight(部署 md5 / sklearn 1.9.* 钉定 / 冻结特征 sha / sampling_lists
在位+sha 记录 / mask_ref / t38 归档 30 份探测)→ **smoke 硬门**
(3 行×2 样本端到端:sample→extract→(1900+6) 点试拟合(含复拟合逐位)→
NN sanity→verify --smoke)→ queue(全 300 采样+提取 + rs2021×2 +
rs2022/2023 拟合 + 30-NN)→ verify(FG1-FG7)→ collect → pack
(<50MB,不含特征 npy / windows npz / 邻居 npz;coords ~35KB/个入包)。

FG1 冻结资产未动(特征 sha + sampling_lists sha 与 preflight 记录一致
+ ckpt sha)/ FG2 manifest 完整一致(300、配额=重分配后配额、H×metadata、
P、L_eff=min(L_config,N−H)、end≤N_in、同序列不重叠、short_history/
NaN 记账)/ FG3 输入哈希 + **批量不变性探针(前 32 样本两路重提取:
分组批量路 == 归档特征逐位(err 0.0,硬门);单样本 batch=1 路与
分组批量存在形状相关偏差(max abs 0.017219,记
single_matches_grouped=false,定位与影响见"批量差异定位与影响"
一节),计数与最大误差入 verify json)** + 机制记录 /
FG4 五组跨距不重叠且并集=[0,2200) / FG5 array_equal 1900 行核对 + 同
rs 复拟合 coords 逐位同 / FG6 30-NN 自排除 + 分母 9000 + 数值复算一致 /
FG7 产物齐全(coords×4、params、nn_neighbors.npz、图、nn 表、RESULT)。

## 论文对照锚

- 原文:`audit_supplement/paper/VisionTS_paper_full.md` L128(图题)/
  L136(ETTm1 右上、Monash 左下且与 ImageNet 显著分离、ImageNet 集中
  而时序散布)。
- 原图资产:`audit_supplement/paper/images/50bcc728….jpg`
  (sha `b9dc92e1…`,76,419 bytes);对照图为定性并排,无数值阈值。
- **补充设定披露(冻结)**:Monash 采样规则为本轮补充设定;统一左 5 列
  掩码是可视化设定,异于原预测几何(逐子集 update_config 的自身掩码);
  2D 交叠仅定性声明,不作预测/因果推断;与原图不一致处如实保留。

## Environment

eta L20;visionts env(`/dev_data/wlt/conda/envs/visionts/bin/python`,
torch 2.12.0+cu126,sklearn 钉定 1.9.*)为提取/t-SNE/NN 环境;author
环境(`visionts_t37author`,py3.8.20,gluonts 0.14.4)仅采样;GPU_ID=0,
repo checkout `/dev_data/wlt/VisionTS_Experiments`;CWD 无契约(脚本
自带 sys.path 引导)。t38 归档(30 份 `logs/monash_<ds>.json`)只在
eta,`t38_logs_base()` 以 fred_md.json 为标记解析。

## Results

全链一次通过(2026-11-07,eta):

- **采样**:300 样本/29 行;重分配 `us_births→m4_monthly n=3`
  (us_births 容量恰 7 = 重分配后配额,装箱重画 7/7);CIF 按设计
  50/50 拆分 = cif_2016_6 5 窗 + cif_2016_12 5 窗(**无缺额**;
  早期队列日志里 "short by 5 (spillover)" 是把设计的后半拆分误记成
  溢出,脚本日志口径已修正);short_history=175、NaN 窗 0、
  packing 重画 7 窗;68 个几何组。
- **提取**:gates(冻结 sha/ckpt sha/mask_ref)全过;300×54528
  float32;FG3 **批量不变性探针**:前 32 样本(k 序,跨全部几何组)
  按两条路重提取——分组批量路(归档提取调度同款)与归档特征
  **逐位一致(err 0.0)**;单样本 batch=1 路与分组批量存在形状相关
  偏差(max abs 0.017219 / RMS 相对 4.16%,记录观察;定位与下游
  影响见下节);实际样本数/单样本重提取次数/最大误差记录在
  verify json `probe_reextract`。
- **t-SNE(联合 2,200×54528)**:sklearn 1.9.1,rs2021 13.3 s 且
  复拟合 coords **逐位一致**;rs2022 12.6 s / rs2023 13.1 s;合并前
  1,900 冻结行 array_equal 逐元素核对过。
- **30-NN(冻结编码器,54528 维 float64)**:ETTm1 0/9000、
  Weather 0/9000、Electricity 0/9000(与 EXP-036 饱和一致)、
  **Monash 4/9000 = 0.000444**。
- **verify FG1-FG7:42/42 ALL PASS**;
  RESULT/logs 在包内,特征 npy / windows npz 留 eta,
  **nn_neighbors.npz 已入包**(可从包内独立复算 30-NN 表)。
- **图**:`fig41_tsne_monash.png` 两行:顶部 = 本轮 rs2021 联合
  2,200 点 + 论文原图(sha `b9dc92e1…` 资产,76,419 bytes,含
  Monash 共 2,200 行);底部 = rs2021/2022/2023 **三个独立面板,
  各自展示全部五类来源**——不同次拟合坐标未经对齐、不互相叠加,
  稳定性仅逐面板定性判读簇结构;定性对照,无数值阈值。

## 批量差异定位与影响(评审修正 v3,probe_batch_diff_t41.py)

针对评审阻塞项(verify 记录 single_matches_grouped=false 但原文写
"两路均逐位一致"的矛盾;差异归因缺证据),新增独立探针:比较两条路的
**输入图像 / 可见 patch 顺序 / 逐级编码器输出**,给出误差判据与下游
影响证据。**不宣称跨 batch 形状逐位一致**;探针证明:差异出现的位置、
每条路各自逐位可复现、30-NN 与 t-SNE 结论不受影响。工件:
`save_t41_monash/probe_batch_diff.json`(入包)+
`logs/fig41_probe_batch_diff.png`(入包);B=1 特征 / nn_neighbors_b1.npz /
tsne_coords_2200x2_b1.npy 留 eta。

- **结构性事实(调和全部此前观察)**:300 样本分 68 个几何组,其中
  **28 组恰含 1 个样本** → 这 28 行归档特征本身就是 B=1 提取产物;
  归档矩阵混合了两种核路径(272 行多行批量 + 28 行 B=1)。FG3 探针的
  分组路之所以逐位:多行组在 B≥2 下组成稳定,单行组平凡逐位。
- **定位(GPU,前 32 样本,k 序)**:输入图像逐位一致;首个分歧在
  **patch_embed**(Conv2d,max abs 1.9e-3),沿 12 个 block 单调增长
  (峰值 max abs 0.0541 @ blk10),最终特征 max abs 0.017219 —— 与
  verify 记录值逐位重复(同形状完全确定)。
- **每形状内部确定性**:分组路重跑逐位同;B=1 路重跑逐位同;多行
  小批量组成检查(3 行 probe 子集批)与归档**逐位一致**;分组路 vs
  归档 = 多行探针行 22/22 + 单行组探针行 10/10 全逐位。
- **机制**:batch 形状相关的核选择(B=1 与 B≥2 走不同 GEMM/attention
  核;每个形状内部逐位可复现,多行组成彼此逐位相等)。CPU 交叉核对
  (同权重同代码,秩编码掩码噪声):patch_embed 形状不变,首个分歧在
  **blk00**,block 内 max abs ~3.4e-5 —— 效应两种设备都存在、幅度随
  设备不同:**系统性形状依赖,非运行间不确定,非 GPU 故障**。
- **误差判据(逐样本 rel L2,B=1 vs 归档;全部 300 行 B=1 重提取)**:
  中位 0.0609% / p95 0.1405% / 最大 0.2446%(k=203);多行组行
  (272)中位 0.0695% / 最大 0.2446%;**单行组行 28/28 逐位**;
  总体 max abs 0.0228。
- **下游影响 — 30-NN 分子不变**:ETTm1 0→0、Weather 0→0、
  Electricity 0→0、Monash **4→4**;逐源邻居集重叠 ≥99.9556%;
  4 个 Monash→ImageNet 邻居对在两矩阵中为**相同行**。
- **下游影响 — t-SNE rs2021 在 B=1 矩阵上重拟合(EXP-002 参数逐字)**:
  逐源 2D 30-NN ImageNet 占比最大移动 **0.79 pp**(ETTm1 +0.70 /
  Weather −0.61 / Electricity −0.79 / Monash +0.57);簇结构目视不变
  (探针图并排);2D 交叠保持仅定性。
- **结论**:CONCLUSION ROBUST —— 批量形状效应不改变 30-NN 表与
  t-SNE 结构;归档调度(分组批量)自洽,无需重提取。

## Delivery

- 包(当前版 v3):`temp/temp_t41_pack.tar.gz`(2,306,946 bytes,
  <50MB)+ 同名 sidecar `.sha256` =
  `974ad773a280c7340efeb1e6e0965231cedbee6e4d7b854aab5fcac33d67ba7a`
  (**包 sha 只记 sidecar 与仓库交付记录,不入包内 README**;v3 在 v2
  基础上新增 probe_batch_diff.json / fig41_probe_batch_diff.png /
  nn_table_b1.csv / nn_neighbors_b1.npz 入包并刷新 RESULT/README,
  代码与归档工件不变)。
- 包历史:v1 sha `3655a114…` 作废;v2(评审修正版)sha
  `d0d8a729cc62ec2c5110ba76ef62dd7d39388caf6abadb320fbd5b3375bca682`
  (1,856,990 bytes)被 v3 取代。
- 代码冻结头(完整 40 位):`038aa48f5359ccef82a22380d92ac011af1f8641`
  (分支 `repro/ltsf-t40-t41`;含探针脚本与本文档修正;包内 README
  滞后终版交付记录提交,属已知口径)。
