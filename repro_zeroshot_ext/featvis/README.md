# featvis — TimesX zero-shot 编码器特征可视化(最好/中/最差 vs ImageNet)

事后诊断性探索:比较预测表现最好/中等/最差变量的测试窗口编码特征与 ImageNet
参考特征的分布关系。不预设"越接近 ImageNet 预测越好";不用于重选超参。

## 口径(冻结)

- 主配置 = 固定 P=1 zero-shot(ctx=96→pred=12, r=c=0.4, mae_base, 冻结 ckpt);
  变量筛选与特征提取均用 P=1(筛选基于 `../results/test_variable_level.csv` 的
  std_mse/std_mae 升序排名均值;组=rank 1–6 / 93–98 / 185–190;名单在看图前冻结,
  见 `selection/selected_vars.csv`,共 117 个测试窗口)。
- 特征 = MAE 编码器最后 LayerNorm 输出,**排除 CLS,对 56 个可见 patch token 取均值
  → 768 维**。提取通过运行时 hook 挂在**生产 forward** 上(`vision_model.norm`),
  归一化/重排/插值/padding/mask 与真实预测完全同路;02 阶段断言 hook 图像与独立构图
  逐位一致、hook 特征与独立 forward_encoder 特征逐位一致。
- mask 完全确定:P=1 → 每行最左 4/14 列可见(56 patch, ratio 5/7);批量调用时
  `model.mask` 按 [B,196] 扩展(生产同款 noise)。
- ImageNet 参考:ILSVRC2012 val **随机子集 2000 张**(文件名排序后
  `default_rng(2021)` 无放回抽取;**未按类别分层**——两台机器均无 ground truth)。
  预处理 = MAE 自然图像口径(Resize256 bicubic + CenterCrop224 + ImageNet mean/std);
  TimesX 保留生产预处理(无 ImageNet norm)。两者分别记录。
  - **主对照**:同批 2000 张施加与 TimesX P=1 相同的可见 patch 位置与数量。
  - **补充对照**:同批 2000 张、无 mask(mask_ratio=0,显式 arange noise=自然序)。
- 降维:行 L2 归一化 → 一次 PCA(50,fit on 全部)→ 一次联合 t-SNE
  (sklearn,seed=2021,perplexity=30,lr=200,init=pca,max_iter=1000)。
  三幅面板 = 同一坐标集按行过滤;ImageNet 背景点三图逐点相同;变量颜色固定
  (`tsne/color_map.csv`)。
- **主图与补充图分别拟合:两套绝对坐标不可直接比较**,只能观察分布关系是否一致,
  不得把跨套位置变化解读为特征移动。稳定性:seed{2022,2023}、perplexity{20,50}
  各一次(内部各自联合),全部存档不挑图。
- 高维检验(主定量证据):全部 190 变量 1677 测试窗口只提取一次特征,117 个绘图窗口
  按 sample_id 取子集(与高维分析同源);降维前 L2 768 维空间每窗口到 2000 张
  masked 参考的最近 10 邻平均余弦距离 → 按变量均值 → 190 点;
  **Spearman 主 / Pearson 补充**;p 值未处理领域内变量依赖,仅探索性参考;
  重叠测试窗口非独立样本;t-SNE 二维距离不作正式距离指标。

## 阶段

```bash
bash run_all.sh all   # 00 选变量 → 01 清单 → 02 sanity(GPU) → 03 特征(GPU)
                      # → 04 高维 → 05 t-SNE → 06 绘图
```

- 02/03 遵循 L20 共跑规则:PAUSE 哨兵 + 空闲显存 ≥9000MiB 闸门,OOM 即停。
- 与旧预测缓存对比采用**预先声明**的容差(`common.PRED_RTOL=1e-4`,相对每行预测尺度),
  记录最大绝对误差与归一化误差;超出即失败查因,不自动放宽。同设备/同批/同配置重复
  前向要求逐位一致。

## 产物

`selection/selected_vars.csv`、`lists/{timesx_windows,all_windows}.csv`、
`lists/imagenet_manifest.json`、`features/*.npz+meta.csv`、`tsne/coords_*.csv+
params.json+color_map.csv`、`highdim/{window_distances.csv,var_correlations.*,frequency_split.md}`、
`figures/`(main×4 + supp×4 + sens×4 + representative_windows)、`logs/sanity_report.json`。

## 隔离

仅本目录新增;`visionts/`、`repro_fig7/`(只读复用机制)、冻结件与 cache 只读。
