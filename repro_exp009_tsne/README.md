# EXP-009 — TimesX 分域 LN 微调前后编码器特征 t-SNE

事后探索性可视化：19 个领域 × 2 个状态（微调前 Z0 = 原始 ImageNet 预训练
MAE-base；微调后 F0 seed2021 每域 best_ft）× 与 1,000 张冻结 ImageNet-1k val
参考图做联合 t-SNE → **38 张主图**。代码基于 EXP-004（commit dcd2425），
全部旧目录只读复用。

## 运行（eta）

```bash
cd repro_exp009_tsne
# 小批（climate 域端到端）：
./run_all.sh extract --domains climate
EXP009_TSNE_JOBS=1 ./run_all.sh tsne --domains climate
./run_all.sh figures --domains climate
./run_all.sh audit --domains climate
# 全量（小批通过后）：
./run_all.sh all                 # EXP009_TSNE_JOBS 按小试实测设置
```

- GPU 闸门：extract 前检查 PAUSE 哨兵 + 空闲显存 ≥ 6 GiB；失败即停，
  不触碰 co-tenant、不退 CPU、不改 batch。
- 断点续跑：特征按缓存键（代码+复用源码+MAE 权重+data_cache+配置+两清单+
  对应模型权重）复用；t-SNE 按 fit_key（冻结参数+两份特征 SHA256）复用；
  figures/audit 幂等重跑。

## 阶段

| 脚本 | 作用 |
|---|---|
| `01_extract_features.py` | GPU 特征：Z0 ImageNet(1000)+Z0 全 2,474 TimesX 窗一次；每域 F0 ImageNet+TimesX。每模型首批判：hook 前后预测逐位相等 + 捕获特征 ≡ direct forward_encoder（逐位）+ kept-token 顺序断言 |
| `02_run_tsne.py` | 38 次独立联合拟合（指令冻结的 10 个 TSNE 参数；random_state=2021）；无池化/L2/PCA |
| `03_make_figures.py` | 38 主图（300dpi，ImageNet 浅灰、TimesX 按变量 tab20 固定色前后一致，图例在外，标注域/状态/seed/epoch/点数，脚注）+ 并排图 + index.html |
| `04_audit_report.py` | 七类检查 → `audit/audit_report.json` + `REPORT.md`（逐域一行，单种子指标：窗口 d 归一 → 变量平均 → 域内变量等权；Z0 与 EXP-004 domain_summary 逐域核对） |

## 关键口径

- 检查点绑定两级：.ln.pt **整文件** SHA256 == selection.json `selected_ckpt_sha256`；
  load 后 84 个 LN 张量名称/形状/值逐项核对（逐位）。
- 特征 = `vision_model.norm` 输出 [B,57,768]（CLS 保留）展平 43,776 维。
- TimesX 走真实 `model.forward`；ImageNet 走 direct `forward_encoder`（同一
  确定性 mask，56 可见 patch，token 顺序断言）。
- 不重新划分数据、不密集化、不按误差筛选；仅 past 96 步进编码器。

## 结论边界

事后可视化、单种子（s2021）、非盲测；仅观测编码器 norm 输出，不能将预测误差
变化归因于"编码器模态距离"变化的机制；t-SNE 距离仅定性；前后独立拟合故绝对
坐标不可跨图比较。
