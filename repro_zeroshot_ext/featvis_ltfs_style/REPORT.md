# LTFS 式 flatten 对照实验报告(featvis_ltfs_style)

## 目的

检验 featvis 主实验观察到的 TimesX/ImageNet 分离是否依赖可视化处理方式:改用旧 Long-term TSF 实验(repro_fig7)的特征口径(含 CLS 全 token flatten,无池化/无 L2/无预 PCA,直接 t-SNE)重画同一批 117 窗口,并用同样本的池化流程作对照。

## 口径

- 同一批编码器输出两种特征:main=flatten 43776(含 CLS);ctrl=排除 CLS → 56 patch 均值 → L2 → PCA50。两组 t-SNE 超参完全相同(perplexity=30, lr=200, init='pca', early_exaggeration=12, max_iter=1000, seed=2021, metric='euclidean'),各自独立联合拟合,**绝对坐标不可跨套比较**。
- 预测配置与 featvis 冻结口径一致(固定 P=1,96→12,r=c=0.4,mae_base,TF32 关闭);117 窗口名单与 featvis 完全一致。
- ImageNet 参考复用旧 LTFS 实验的 1000 张清单(sha256 前缀 84a885e90f8fcd68),顺序原样,mask 换成本轮 56 可见 patch;每个 batch(含尾部不足 64 的 batch)均断言可见 token **顺序**逐元素一致。
- 预测核验:全部被前向行与冻结 cache 比较,最大相对误差 0.00e+00(声明 rtol≤0.0001)。

## 结果

- **main**: TimesX→最近 ImageNet 二维距离 median=3.92, min=0.95;ImageNet 内部最近邻(300 抽样)median=0.72。(见图 figures/main_triple.png)
- **ctrl**: TimesX→最近 ImageNet 二维距离 median=7.08, min=1.35;ImageNet 内部最近邻(300 抽样)median=0.77。(见图 figures/ctrl_triple.png)
- 分离/混合是否随流程改变:对比两套图内 TimesX 簇与 ImageNet 云的相对关系;t-SNE 二维距离仅作描述,不是正式距离指标。

## 结论限定(必读)

- 两组**同时**改变了:展平 vs 池化、CLS 保留与否、L2 归一化、预先 PCA。两套结果的任何差异只能归因于**整套处理流程**,不能仅凭这两组把变化单独归因于平均池化。
- 本轮已对齐可视化流程,但与旧 Long-term 实验仍存在原生差异:**数据集(TimesX vs LTFS)、原生窗口、P 配置、可见 patch 数量(56 vs 70)、样本比例(旧联合拟合 = 1000 张 ImageNet + 900 个时序点,每数据集 300;本轮 = 1000 张 ImageNet + 117 个时序点)**。
- 本轮为事后诊断性探索:观察到的是分布关系,不得解释为采样频率或预测性能差异的因果证据。
