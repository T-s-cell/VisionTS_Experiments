# EXP-042:论文表 22 剩余微调方式(Bias / MLP / Attention)

补齐论文表 22 微调方式中尚未复现的 **Bias / MLP / Attention** 三种,
使用**原始 MAE-base 预训练权重**(vm_pretrained=1,非随机初始化;
init 与冻结 ckpt 逐 key 逐位相等为硬门 VG3)。LN(EXP-003,48 份逐种子
归档)/ All-Rand-VM(EXP-038 seed-2021 + EXP-039 seeds 2022/2023)/
Freeze(EXP-033 Freeze16)**复用归档,不重训**。任务量 = ETT 四数据集 ×
H={96,192,336,720} × seed={2021,2022,2023} × 3 方式 = **144 次训练**
(冒烟 9 = ETTh1/96 × 3 种子 × 3 方式;queue 135)。

**复现性验收 = 流程与数值自洽;接近论文值不是验收标准。**论文锚只在
数据集层与总体层对照,ETTm2 Bias-LN 反例方向与 1st 计数只作当前一轮
配方的观察记录。

## 配置(冻结,EXP-003 配方逐字)

r=c=0.4、bilinear、FP32、Adam 1e-4、bs=256、**train_epochs=1**、
patience=3;完整固定验证集、元素加权 MSE 选择;test 前原码强制 best
ckpt strict 重载。L/P:ETTh1 1152/24,ETTh2 1152/24,ETTm1 2304/96,
ETTm2 1152/96(t38_common.py:287-291)。MAE ckpt sha `c551c37b…`
(447,725,789 B)。种子 2021/2022/2023 显式传入 run_train(白名单
delta 内钉定)。

## 训练入口(零改动 + 观测钩子,EXP-039 同款)

- 入口 = 仓库根 `repro_fullshot/run_train.py` + `Exp_Univariate`
  (全数据集统一,vali = 完整固定验证集元素加权 MSE,
  exp_univariate.py:79-103;train() 后 test() 前**原码强制**
  torch.load + load_state_dict(strict 默认 True)重载 best ckpt,
  run_train.py:225-230)。t42 驱动 `import run_train; run_train.main()`
  **零改动调用**;argv 自该配置 EXP-003 args.json 重放,白名单 delta
  **恰好 {model_id, save_dir, ft_type, gpu, seed}**,且
  {model_id, save_dir, ft_type, seed} 必须出现、vm_pretrained==1
  (不进白名单,相等即合规)。
- skip-if-done = run_train.py:189-202 原生语义(指纹匹配 SKIP;
  异常 **ABORT rc=3,旧现场保留不覆盖**,半成品目录由 runner 移入
  `.failed/` 后重训)。fingerprint-mismatch ABORT 语义由 VG1 的
  EXP-003 code md5 12 文件钉定(行为同一)。
- 观测钩子(承 t39,t42 增量):`visionts.model.VisionTS.__init__`
  wrapper 记 canonical sha / **trainable_names/count/numel/n_params_
  total** / rng / **vs_ckpt 逐 key 相等断言(三臂均要求 init 全 key ==
  冻结 ckpt)**;训练后 `post_frozen_check`:final ckpt 从磁盘重载,
  **全部冻结 key 逐位不变(驱动内硬断言)**,可训练 key 变化数记录;
  Exp_Univariate 计时 / RandomSampler 逐 epoch 顺序哈希 / 首批张量
  摘要同 t39。

## 微调范围(口径差异记录,冻结)

官方实现 `visionts/model.py:26,46-57`:
- `bias` → `'bias' in n`:全部名称含 bias 的参数 = attn.qkv/attn.proj、
  mlp.fc1/fc2、decoder_pred 等 **Linear bias + 全部 LayerNorm bias**
  (norm1/norm2/enc norm/decoder_norm)+ **patch_embed.proj.bias
  (Conv2d)**;不含任何权重矩阵与 pos_embed/cls_token/mask_token。
- `mlp` → `'.mlp.' in n`;`attn` → `'.attn.' in n`(各自权重+bias)。
- 三者均**编码器+解码器全覆盖**;pos_embed/decoder_pos_embed 构造即
  requires_grad=False。
- **口径差异**:论文 C.3(L555)称 Bias=BitFit("bias term of all the
  linear layers"),官方代码实际更宽(含 LayerNorm bias 与 Conv2d
  bias)。**采用官方代码口径**,只记录、不自行扩大/收窄。
- preflight 阶段 `t42_scope_audit.py` 在真实模型实例上做
  **实际 requires_grad 集合 vs 官方谓词重推集合相等断言**(每方式),
  输出名称/数量/占比/覆盖声明/冻结项清单 + `frozen_digest.json`
  (冻结 ckpt 逐 key sha,driver/verify 共用);verify VG2 三方核对
  (driver 实际 == audit == verify 独立重推)。

## 门禁(VG1-VG10,执行前全定义)

VG1 144 指纹(code/data/ckpt == 该配置 EXP-003;delta 白名单恰观测;
vm_pretrained==1;种子四路钉定)/ VG2 范围三方一致 + bias 含
patch_embed.proj.bias + 编码器解码器覆盖声明 / VG3 init 逐 key ==
冻结 ckpt(144)/ VG4 **冻结参数训练后不变:final ckpt 磁盘重载,全部
冻结 key 逐位 == frozen_digest(硬门);可训练 key ≥1 变化(记录)** +
文件 sha == 交付记录 / VG5 **真值逐位 == EXP-003:48 三元组 × 3 方式 =
144 全查** / VG6 pred/true 形状+有限性 + f64 复算 MSE/MAE ≤1e-6(144)/
VG7 重入检查**覆盖全部 144 任务**(SUCCESS+json 钉定+锚点+配置全钉
+数组实载+ckpt 文件 sha)/ VG8 参考值可载(LN48==归档、t39 64、
t38 32、Freeze16)/ VG9 queue 后 144 全 SKIP(无未完/可覆盖现场;
ABORT 语义由 VG1 code md5 钉定)/ VG10 产物齐全。
`--smoke` = 9 单元同构子集(VG8/VG9 除外)。

## 流程(run_t42_all.sh:preflight → smoke → queue → verify → collect → pack)

preflight(12 文件 code_md5 / 4 CSV sha / ckpt sha+字节 / **EXP-003
归档 48 份全查** / LN48 钉值 vs 归档 / t39 64 + t38 32 json / Freeze16 /
scope audit / GPU / **磁盘 ≥100GB 硬门 + 按剩余任务实测需求
(0.6 GiB/任务)+20GB 余量**)→ **smoke 硬门**(9 组跑完 →
verify --smoke 全过才放行;**未过不进 queue,现场保留**)→ queue
(135,skip-if-done,nohup 断点续)→ verify(VG1-VG10)→ collect →
pack(<50MB,不含 ckpt/npy;+ sidecar,**包 sha 只记 sidecar 与仓库
交付记录,不入包内 README**)。

## Environment

eta L20;visionts env(`/dev_data/wlt/conda/envs/visionts/bin/python`,
torch 2.12.0+cu126);GPU_ID=0;repo checkout
`/dev_data/wlt/VisionTS_Experiments`;CWD 契约 = long_term_tsf(同
EXP-003,脚本自带 sys.path 引导)。大件(144 ckpt ≈64GB + results
数组 ≈18GB)留 eta:`long_term_tsf/save_t42_fttypes/`。

## 预算(冒烟后按实测刷新)

预估:纯训练 ≈26.2h + vali/test ≈7.8h + 开销 ≈1h → 墙钟 ≈35-36h;
冒烟 9 组 ≈51-55 min。**待冒烟实测刷新。**

## 代码冻结头(冒烟过后回填完整 40 位)

`1cd1805f0920472b3e1fedd460149cfb100440bb`(分支 `repro/ltsf-t42-fttypes`)。
= 0096521(seed-delta 白名单修正)+ 1cd1805(verify VG4 冻结/可训练 key
划分修正,纯审计侧);训练代码自 0096521 未变。冒烟 9/9 +
verify --smoke 143/143 全过后回填(2026-11-10)。

## Delivery(占位,pack 后回填)

- 包:待回填(命名 `VisionTS_t42_fttypes_<日期>.tar.gz`)+ sidecar。
- 包历史:无(v1 即首版)。
