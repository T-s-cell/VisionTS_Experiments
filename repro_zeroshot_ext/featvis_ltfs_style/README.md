# featvis_ltfs_style — TimesX 可视化对照:LTFS 式 flatten+t-SNE

事后诊断性探索:检验 featvis 主实验观察到的 TimesX/ImageNet **分离现象是否依赖
可视化处理方式**。复用旧 Long-term TSF 实验(repro_fig7)的特征口径(含 CLS 全
token flatten、无池化、无 L2、无预 PCA、直接 t-SNE)重画同一批 117 个测试窗口,
并加一个"同样本池化流程对照"。

## 口径(冻结)

- 预测配置 = featvis 冻结口径:固定 P=1(ctx=96→pred=12,r=c=0.4,mae_base,
  sha 前缀 c551c37b),FP32/eval/no_grad,TF32(matmul+cudnn)关闭(复用
  `featvis/common.py::load_model`)。
- 变量名单 = featvis 原样:最好/中间/最差各 6 变量、全部 117 测试窗口
  (`00` 断言与 `featvis/lists/timesx_windows.csv` 集合一致),不重新排名。
- ImageNet 参考 = **旧 LTFS 实验的 1000 张清单原顺序**(只读
  `repro_fig7/outputs/sampling_lists/ImageNet.json`,sha256 入档),预处理
  Resize256 bicubic + CenterCrop224 + ImageNet mean/std;mask 换成本轮 56 可见
  patch(noise 扩展至实际 batch)。
- 特征 = encoder 最终 LayerNorm 输出,同一批输出两种:
  - **main(LTFS 式)**:`[B,57,768] → flatten → [B,43776]`,含 CLS,不做池化/
    L2/标准化/预 PCA(init='pca' 仅作 t-SNE 初始化,不是 PCA 降维)。
  - **ctrl(池化式)**:排除 CLS → 56 patch 均值 → L2 → PCA(50) → t-SNE。
- t-SNE(两套完全一致):perplexity=30, lr=200, init='pca',
  early_exaggeration=12, max_iter=1000, random_state=2021, metric='euclidean'。
  两套**各自独立联合拟合,绝对坐标不可跨套比较**;每套三面板从同一坐标集行过滤
  (ImageNet 背景/范围/比例逐点一致);图例置于图外;变量颜色 tab20。
- **token 顺序**:展平前对**每个 batch**(TimesX 与 ImageNet、含尾部不足 64 的
  batch)断言 ids_keep **顺序逐元素一致**(运行时包装 forward_encoder 捕获
  ids_restore,不改核心源码),顺序数组入档 `logs/`。
- 预测核验:全部被前向行与冻结 cache 比较,声明 rtol=1e-4(相对每行预测尺度),
  超限即停;同配置重复前向要求逐位一致。

## 阶段

```bash
bash run_all.sh all   # 00 清单(CPU) → 01 sanity(GPU) → 02 特征(GPU)
                      # → 03 t-SNE+绘图+报告(CPU)
```

- 01/02 遵循 L20 共跑规则:PAUSE 哨兵 + 空闲显存 ≥9000MiB 闸门,OOM 即停;
  不使用双 3090;03 纯 CPU。

## 产物

`lists/{timesx_windows.csv,imagenet_list.json}`、`features/{flatten,pooled}_1117.npz
+meta.csv`、`tsne/{coords_main,coords_ctrl}.csv+params.json`、
`figures/{main,ctrl}_{best,middle,worst,triple}.png`、`logs/`(sanity_report.json、
extract_env.json、token_order.json)、`REPORT.md`。

## 结论限定

两组**同时**改变了展平/池化、CLS 保留、L2、预先 PCA——差异只能归因于**整套处理
流程**,不能单独归因于平均池化。本轮对齐了可视化流程,但与旧 LTFS 实验仍存在
原生差异:数据集、原生窗口、P 配置、可见 patch 数(56 vs 70)、样本比例(旧
联合拟合 = 1000 ImageNet + 900 时序点(每数据集 300);本轮 = 1000 + 117)。
观察到的是分布关系,不得解释为采样频率或预测性能差异的因果证据。

## 隔离

仅本目录新增;`featvis/`、`repro_fig7/`(只读清单)、`visionts/`、冻结件与
cache 只读;不修改任何既有文件。
