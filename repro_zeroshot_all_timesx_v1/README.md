# repro_zeroshot_all_timesx_v1

**VisionTS 在 TimesX 全部原生窗口上的固定 P=1 zero-shot 补充评估**
（"VisionTS on all TimesX native windows — fixed P=1 zero-shot supplementary evaluation"）

把旧 zero-shot 实验（`../repro_zeroshot_ext`，仅 test/val 6,413 窗）的范围扩展到
TimesX **全部 8,106 个原生窗口**（190 变量 / 19 领域，past 96 → horizon 12，含旧划分
train 区 1,451 窗与跨边界剔除 242 窗），为后续 Aurora 单模态实验提供统一的数据清单、
输入数组与评分基准。旧 test 集实验的结论保持独立，不受本实验影响。

## 协议要点（冻结，唯一来源 `protocol.py`）

- 数据 `TimesX_Datasets.zip` sha256 `9adb9540…88bcc`；窗口身份 `(var_key, sample_id)`，
  sample_id 为字符串；排序：变量按 zip 枚举序、窗口按 `(future_start, idx)`。
- 模型 mae_base / 原始 ImageNet 权重（sha256 `c551c37b…ade0e`）/ P=1 / r=c=0.4 /
  224×224 / patch16 / bilinear / FP32 / TF32 双关 / eval 无梯度 / 种子 2021 / batch 64。
  不训练、不选超参。
- 评分：每窗 `d = std(past96, ddof=0)`；`d < 1e-8` 回退该变量最早原生窗 past96 的 std
  （= 旧 splits.json calib_std）；仍退化 → `std_defined=False` 显式标记。
  `d` 与 `denom_source` 冻结进 frozen_arrays.npz（S1 产出，assemble 交叉核对）。
  `std-MSE = mean(((pred-target)/d)^2)`，`std-MAE = mean(|pred-target|/d)`。
  聚合：窗口均值 → 变量 → 域内变量等权 → 19 域等权；原始尺度只到变量层。
- **缓存复用双判据（2026-10-10 冻结）**：受检窗同时满足
  ①`max|new-old| / max(1e-8, max|old|) ≤ 1e-5` ②`max|new-old| / d ≤ 1e-4`，
  预测有限；超差变量整变量重推理，阈值绝不放宽。此为缓存兼容性工程容差，
  非性能显著性标准。
- 补跑集合 = 全部窗口 − **旧缓存 npz 实际覆盖窗口键集**（集合差，无重复计数；
  变量被判超差时移除其全部复用资格）。S3 结论以 `forward_fingerprint`
  （zip/冻结数组/ckpt/代码/配置/env）写入 reuse_final.json，S4 启动重算比对，
  不一致拒绝执行（S3/S4 须同机同 env 连续执行）。
- 1,677 旧 test 子集的**纯缓存复算**必须复现 3.254715 / 1.042067（≤5e-7）；
  若变量被重推，最终口径子集指标合法漂移，报告差值与原因。

## 目录结构

```
protocol.py           冻结常量（协议唯一来源）
common.py             路径/哈希/状态/指纹（旧目录只读 import + md5 记录）
freeze_full.py        S1 冻结：inventory.json + frozen_arrays.npz + vars.json
verify_cache.py       S2 旧缓存校验（绑定旧 splits.json 指纹 + 逐窗核对）→ reuse_plan.json
preflight.py          S3 抽样名单冻结 + 复推双判据 → preflight_report.json + reuse_final.json
infer_fill.py         S4 补推理（集合差）→ cache_all/
assemble.py           S5a 合并逐窗预测（含来源）→ results/preds.jsonl.gz
evaluate.py           S5b 三级汇总 + 1,677/2,474 子集
recompute_check.py    独立复算（不 import 管线代码，公式重写；任何偏差 exit 1）
stage_guard.py        阶段标记守卫（.done 内 sha256 复核，漂移即重跑）
verify_acceptance.py  S5c 验收 R1–R7
make_report.py        生成 results/REPORT.md
run_all.sh / audit_all.sh
frozen/               冻结产物（inventory / arrays / vars / freeze_report）
cache_all/            新推理 npz + manifest（指纹绑定，支持按变量断点续跑）
results/              preds、指标、子集、验收、REPORT.md
```

## 复用与隔离

- `../repro_zeroshot_ext/` **只读**：timesx_data.py / record_fingerprint.py /
  metrics_util.py / infer.py 路径 import，全部 md5 记入 manifest 与冻结报告；
  旧 cache/ 的 P=1 npz 只读复用（每文件 sha256 记录于 reuse_plan.json）。
- `../repro_ln_timesx_v1/manifests/split_manifest.json` 只读引用（2,474 窗子集清单）。

## 运行

见 `RUN_COMMANDS.md`。执行序：冻结 → 预检 → 补齐 → 汇总 → 验收 → 打包。
