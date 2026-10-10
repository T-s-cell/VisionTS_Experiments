# repro_cgtsf_zeroshot_v1 — VisionTS × CGTSF 单模态 zero-shot

实验编号 cgtsf_zeroshot_v1。VisionTS MAE-base（全冻结 `finetune_type='none'`）
在 CGTSF（ChatTime, AAAI 2025，Context-Guided Time Series Forecasting 数据集）
三个子集上的探索性 zero-shot 评估；仅 `Hist` 单变量输入，`Text` 不用；
数据 revision `f85b3fa3a8b3f1a2adbc6edf49f82b7d9d1758c9`。

- 配置：seed=2021、fp32、TF32/AMP 关、`norm_const=align_const=0.4`、bilinear；
  PTF 120→24（P=24）、LEU 240→48（P=48）、MSPG 480→96（P=96）；
  原生 CSV 窗口，身份 `(dataset, Idx, Date)`，保留全零历史窗。
- 评分：float64 原量纲，阶梯 = 点→窗→序列等权；三子集分别报告，
  **无跨子集原量纲总分**；不与 ChatTime 论文测试集主表直接比较。
- 确定性边界：逐位一致仅要求"同环境+同 batch+同窗序"（重复推理与续跑）；
  大 batch 与 batch=1 不逐位一致 ⇒ 回退 batch=1，不算失败；无跨 GPU 逐位验收。
- 模型状态门：除 ckpt 文件 sha256（`c551c37b…`）外，每子集 update_config 后
  对内存中参数+buffers 取哈希，推理前后必须一致。
- 分片异常分级：协议/源码/权重/数据指纹不匹配 ⇒ 停止报告；仅分片损坏或
  未写完 ⇒ 隔离（`quarantine/`）重算并记录原因。

## 运行（zeta，GPU1）

```bash
python protocol.py freeze                      # 生成 protocol.json + 令牌
bash wrapper_zeta.sh preflight.py --i-approve-frozen-protocol <12hex>
# 预检 PASS -> 审计包 -> 用户放行 ->
bash wrapper_zeta.sh run_infer.py --i-approve-frozen-protocol <12hex>
bash wrapper_zeta.sh evaluate.py
bash wrapper_zeta.sh verify_cache.py
python recompute_check.py
python audit_pack.py --stage final
```

## 文件

| 文件 | 作用 |
|---|---|
| `protocol.py`/`protocol.json` | 冻结协议（哈希/配置/令牌 = sha256 前 12 hex） |
| `common.py` | 哈希、原子写、内存模型状态摘要 |
| `load_cgtsf.py` | CSV→窗表（身份门/长度门/有限性门/全零窗统计/230 窗选取） |
| `preflight.py` | 230 窗预检（几何/有限性/权重与模型状态/重复逐位/roundtrip/batch 试探/耗时显存） |
| `run_infer.py` | 全量分片推理（严格续跑+异常分级+NaN/Inf 即停） |
| `evaluate.py` | float64 阶梯聚合（点→窗→序列等权，固定顺序） |
| `verify_cache.py` | 全量完整性（无漏无重/target==CSV 逐位/pred 有限/shard sha） |
| `recompute_check.py` | 独立复算（固定顺序，rtol=atol=1e-12，报最大差异） |
| `audit_pack.py` | 审计包（源码快照+数据+证据+脱敏终检→tar.gz+MANIFEST） |
| `wrapper_zeta.sh` | env 固定（GPU/CUBLAS/HASHSEED） |
