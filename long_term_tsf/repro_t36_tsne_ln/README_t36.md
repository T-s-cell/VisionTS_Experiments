# EXP-036:VisionTS 微调 LayerNorm 前后的 t-SNE 对照(1,900 固定样本)

日期:2026-11-03 | 分支 `repro/ltsf-t36-tsne-ln-v1` | 证据/落章 commit `3b0658d1550c798b56186947bd2c1da2727984fc`
(代码 52069fc + 驱动修复 58c266a/5a82e44/04aeb0a + 判读口径 b9942cc + eta 证据 3b0658d;RESULT 由此头落章)

## 问题

PPT 第 5 页右图(EXP-002,commit `741987306686e4aa1c8ebdf00e4a2c787efe8994`,目录 `repro_fig7/`)是冻结 MAE 编码器对 1,900 个样本的 t-SNE。本轮回答:**看完全相同的输入,微调后编码器(仅编码器 LayerNorm 可训)的局部交叠如何变化?** 四条件各出一张 1,900 点 t-SNE(左上复用 EXP-002 归档坐标),外加高维 30-NN ImageNet 占比表。

## 四条件与权重来源

| 条件 | 编码器权重 | checkpoint sha256(前 16) |
|---|---|---|
| frozen | 冻结 MAE-base(`ckpt/mae_visualize_vit_base.pth`,447,725,789 B,sha256 `c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e`) | c551c37b63a221a3 |
| ettm1_ln | EXP-003(commit `b40d663`)ETTm1 H=96 seed=2021 最佳验证 checkpoint | d2da2b4448da690e |
| ecl_ln | EXP-003 Electricity H=96 seed=2021 | 455bf02b907a7c84 |
| weather_ln | EXP-003 Weather H=96 seed=2021 | 19c02336c1f7c600 |

LN 拼接口径:EXP-003 以 `ft_type='ln'` 训练了编码器+解码器共 84 个 LN 张量;本实验只取**编码器 50 个**(`vision_model.blocks.{0..11}.norm1/norm2.{weight,bias}` 48 + `vision_model.norm.{weight,bias}` 2)拼入冻结主干;解码器与所有非 LN 参数一概不动,特征抽取只走编码器。

各来源训练配置(EXP-003 args.json,如实列出;Weather 与第 5 页 P=144/r=1.0/c=0.7 配置不同,系 EXP-003 自身设定):

- ETTm1:L=2304/P=96/r=0.4/c=0.4,freq=t
- Electricity:L=1152/P=24/r=0.4/c=0.4,freq=h
- Weather:L=576/P=144/**r=1.0/c=0.7**,freq=h
- 共同:lr=1e-4,train_epochs=1,bs=256,patience=3,bilinear,seed=2021,label_len=48;选模=EarlyStopping 全验证 MSE,test 前严格 `load_state_dict` 重载

锚点(SUCCESS 未舍入值,verify 逐项核对一致):ETTm1 0.2832735478878021/0.3229347765445709;Electricity 0.12593606114387512/0.21679048240184784;Weather 0.14431872963905334/0.19301669299602509。

## 1,900 行口径

行序 = `GROUPS=['ImageNet','ETTm1','Weather','Electricity']`:ImageNet [0,1000)、ETTm1 [1000,1300)、Weather [1300,1600)、Electricity [1600,1900)。清单**归档即真,未重抽**:`repro_fig7/outputs/sampling_lists/{ImageNet,ETTm1,Weather,Electricity}.json` + `manifest.json`;行序由 `meta_rows_from_lists()` 从四清单重建并与归档 `meta.csv`(1,901 行)逐行断言一致,`groups.npy` 顺序与组段断言一致。

## 门禁结果(全部通过,详见 `RESULT_t36_tsne_ln.md` 与 `logs/verify_t36.json`)

1. **权重回放**:三份 EXP-003 checkpoint 在各自原训练配置下严格重载,首批(256)预测 vs 落盘 `pred.npy[:256]` **max|Δ|=0.000e+00**(阈值 ≤1e-4);
2. **拼接核验**:每条件 50/50 编码器 LN 键在来源 checkpoint 中齐全、拼入后与来源逐位相等;50 键全部相对冻结发生变化(逐键 max|Δ| 在 verify json);**编码器非 LN 张量(101 个,参数+buffer)与冻结逐位一致**;
3. **输入固定**:四条件逐批输入张量 + 掩码 noise SHA-256 完全相同(`input_hashes_*.json`);ids_keep 与归档 `mask_ref.json` 断言相等;
4. **特征核验**:frozen 特征与 EXP-002 原件**逐位一致**(新抽 sha256 `5ac9bbd2…` == 原件 `5ac9bbd2…`,np.array_equal max|Δ|=0);四份特征均 (1900,54528) f32 全有限;
5. **t-SNE**:三套 LN 条件用 EXP-002 原参数(sklearn 1.9.1,`perplexity=30, learning_rate=200, init='pca', early_exaggeration=12, max_iter=1000, random_state=2021, metric='euclidean'`,输入原始 54,528 维,无预降维);frozen 面板直接复用归档坐标(sha256 `d4706007477d1278…`),未重跑。

## 结果摘要(照实)

- **高维 30-NN ImageNet 占比(分母 9,000):4 编码器 × 3 时序来源全部 12 格 = 0/9,000。** 原始 54,528 维空间模态鸿沟主导(冻结下抽查行最近 ImageNet 距离约为最近时序距离的 1.4–3.3 倍,独立复算一致),该指标在本设置下饱和,不区分四套编码器(±0 pp)。
- **二维面板内 30-NN ImageNet 占比(描述性,相对冻结的百分点)**:ETTm1-LN 面板训练域 +2.92 pp / Electricity +5.17 pp / Weather −1.32 pp;Electricity-LN 面板训练域 +9.88 pp / ETTm1 +2.73 pp / Weather +1.87 pp;Weather-LN 面板训练域 +0.46 pp / ETTm1 −3.60 pp / Electricity +1.99 pp。各面板独立拟合,数值仅作面板内方向参考。

## 运行命令

```bash
# eta,visionts 环境,CWD 任意(驱动自定位),数据路径按 CWD=long_term_tsf 相对解析
PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python GPU_ID=0 \
  bash repro_t36_tsne_ln/run_t36_all.sh min    # 权重三连核对+冒烟(frozen/ettm1_ln+t-SNE)
PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python GPU_ID=0 \
  bash repro_t36_tsne_ln/run_t36_all.sh main   # 其余两条件+NN 表+2x2 图+collect
# 本地落章(拉回证据件后)
python collect_t36.py --stamp <证据 commit sha>
```

## 大件(留在 eta,不入包;路径与指纹)

| 件 | 路径(eta) | sha256 |
|---|---|---|
| 原件特征(EXP-002) | `/dev_data/wlt/VisionTS_Experiments/repro_fig7/outputs/features_1900x54528.npy` | 5ac9bbd232b80f569eedc0e030eddcfaa60c029b48cc92e36ae0c433ff9fe48a |
| 归档冻结坐标 | `/dev_data/wlt/VisionTS_Experiments/repro_fig7/outputs/tsne_coords_1900x2.npy` | d4706007477d127897bbc0bbd1dbe712de2ec942ba13e737e59cc80a22fd9f2e |
| frozen 特征(本轮) | `/dev_data/wlt/VisionTS_Experiments/long_term_tsf/save_t36_tsne/features_frozen.npy` | 5ac9bbd232b80f569eedc0e030eddcfaa60c029b48cc92e36ae0c433ff9fe48a(与原件逐位同) |
| ettm1_ln 特征 | `…/save_t36_tsne/features_ettm1_ln.npy` | 45a731366184e3a3e97fd0eaab9992234febbddfa3cc61dce2d4cb79e7ea2f29 |
| ecl_ln 特征 | `…/save_t36_tsne/features_ecl_ln.npy` | 2194888a4a1cae9f9ad01b1af7a3ceb671bcbf91d77cb62327b25826458db7cf |
| weather_ln 特征 | `…/save_t36_tsne/features_weather_ln.npy` | 419d116eb64f8066235b993ba7a8a3f1198b2546f294eeeb8bfadc5b1edff74f |
| EXP-003 ETTm1 ckpt | `/dev_data/wlt/VisionTS_Experiments/long_term_tsf/save_fullshot/ETTm1_96_seed2021/checkpoints/_/checkpoint.pth` | d2da2b4448da690e850cf641521ec855e4b6bbdfc515915b2d62f76badcb8c2b |
| EXP-003 Electricity ckpt | `…/save_fullshot/Electricity_96_seed2021/checkpoints/_/checkpoint.pth` | 455bf02b907a7c84a5cc8b7905dcdfaf04a5dcc95ff09e230b3a7226ce4c6e61 |
| EXP-003 Weather ckpt | `…/save_fullshot/Weather_96_seed2021/checkpoints/_/checkpoint.pth` | 19c02336c1f7c60086e63c798c5bc308b0ae5e4fc139b6c7c08637dccf655b34 |
| 冻结 ckpt | `/dev_data/wlt/VisionTS_Experiments/ckpt/mae_visualize_vit_base.pth` | c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e |
| ImageNet 原图 | `/dev_data/wlt/data/ImageNet/val`(50,000 张 `ILSVRC2012_val_*.JPEG`) | 目录级,未逐张哈希 |

## 环境与包内容

eta:Python 3.12.13 / numpy 2.4.4 / torch 2.12.0+cu126 / scikit-learn **1.9.1**(与 EXP-002 env.json 一致;快照 `logs/env_snapshot_{min,main}.txt`)。

审计包:`temp/VisionTS_t36_tsne_ln_20261103.tar.gz`——脚本、logs(nohup/env/verify json)、SMOKE 标记、输入哈希、特征 sha sidecar、3 套 2D 坐标、NN 表/邻居行号/summary、`summary_t36.csv`、`RESULT_t36_tsne_ln.md`、图 PNG/PDF,以及 `exp002_refs/`(归档 meta.csv、groups.npy、抽样清单、tsne_params.json、mask_ref.json、sanity_report.txt、冻结坐标副本——包内 `python collect_t36.py --stamp <sha>` 可独立复验全部门禁);**不含** features npy(4×414 MB)、任何 checkpoint、ImageNet 原图。包 sha 见同名 `.sha256` sidecar(不内嵌于本 README)。
