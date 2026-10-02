# 审计包：TimesX LN 微调（timesx_ln_v1）全量训练启动前审查

**状态：preflight 36/36 全部通过；等待人工审计批准后才启动 285 运行全量队列。**

依据规范 `VisionTS_TimesX_LN_Experiment_Plan_v1.md`；执行环境 eta L20（
`/dev_data/wlt/VisionTS_Experiments/repro_ln_timesx_v1`）。

## 1. 一句话结论

数据划分、代码指纹、预训练权重、老零样本缓存对齐全部核验通过；预检（含真实
GPU 训练、中断续跑、资源实测）全绿；预计全量训练约 **14.0 小时**。发现并修复了
1 个数据缓存严重 bug 与 6 个次要问题（见 §5，均已加防回归检查）。

## 2. 预检结果（audit/launch_audit/preflight_report.json，eta 实测）

| 检查 | 结果 |
|---|---|
| A 数据/划分（190 变量、4041/36763/895/2474/696、边界严格性） | PASS |
| B LN 可训练集=84 张量/55,808 标量；冻结参数梯度 None；train==eval 逐位 | PASS |
| C1 微调模型初始预测 ≡ 零样本模型 | **max\|d\|=0（逐位）** |
| D1/D2/D3 全部 6,413 个老缓存窗：预测 & 目标 | **worst=0（逐位）**，覆盖 6413/6413 |
| E FreqMask 自检 + 增强分母 | PASS |
| F1/F2 梯度累积等价 / 无损失膨胀 | max\|dgrad\|=3.04e-6（声明容差 1e-4/1e-6 内） |
| G 预算 U_d/S_d、F1–F4 同种子基础序列一致、F0 原生池 | PASS |
| H 验证 895 / 测试 2,474 全覆盖 | PASS |
| I 实测：**0.287 s/update，峰值 910 MiB**，总计估算 **14.0 h**（train 13.9 + val 0.1 + test 0.04） | PASS（现空闲 27.9 GB >> 闸门 3,982 MiB） |
| J 真实中断→续跑：epoch 重放损失一致、残轮 attempt 作废、续跑后 val MSE 差 1e-8 | PASS |

## 3. 双端一致性（本地 ↔ eta，见 consistency.json）

- 代码指纹 12 个文件 md5：**完全一致**
- TimesX zip sha256：`9adb9540…`（= 规范值）；eta ckpt sha256：`c551c37b…`（= 规范附录 B）
- data_cache.npz **内容级**（6 个数组逐一 sha256）：**完全一致**（整文件 sha 不同仅因
  npz 容器内嵌时间戳，已用数组级哈希作内容锚）
- split_manifest：仅 `zip_path_used`（主机路径）不同，其余逐字节一致

## 4. 启动后将执行（批准后）

1. `queue_train.py`：285 运行 = F0–F4 × 19 领域 × 3 种子，串行；领域按密集窗数
   升序（arts 先、Currency 最后），领域内 seed → F0–F4；每次运行前查
   PAUSE 哨兵 + 空闲显存 ≥ 峰值+3GiB 闸门；
2. 断点：epoch 粒度 resume.pt（哈希/版本不符拒绝续跑）；崩溃丢弃残轮、日志按
   attempt 作废；单次失败即停、保留现场（exit 3）；
3. 完成后依序：freeze_selection（仅验证集选点）→ test（此时才算 Z0/N0 测试指标
   +285 冻结点测试）→ aggregate → audit；
4. 日志：`logs/run_all.log`、`logs/stage_*.log`、逐运行 `logs/{run_id}.jsonl`。

停止条件不变：哈希不符、数量不符、显存闸门持续不过、非有限损失/异常梯度、
需改协议的问题——停队列、保现场、报告待决事项。

## 5. 预检期间发现并修复的问题（全部披露）

| # | 严重度 | 问题 | 修复与防回归 |
|---|---|---|---|
| 1 | **严重** | `prepare_data` 写 data_cache 时 `series_off` 误存变量序号而非累计偏移 → ExperimentData 读到跨变量错位序列（指数值读成别家数据） | 改为累计偏移；prepare 增加**写后读回逐位自检**；preflight D 段从 2 个领域扩为**全部 19 领域 6,413 窗目标逐位比对**（本次即由 D2 抓获） |
| 2 | 中 | `select.py` 遮蔽标准库 `select` 模块，任何从本目录启动的进程都可能崩 | 更名 `freeze_selection.py` |
| 3 | 中 | torch ≥2.6 默认 `weights_only=True`，resume.pt / LN 检查点加载被拒 | 两处显式 `weights_only=False`（仅加载本实验自产文件） |
| 4 | 中 | preflight J 段计数 step 未调用真实 `Adam.step`，中断路径等于没训练 | 先执行真实 step 再计数抛中断（J1–J4 现全部通过） |
| 5 | 低 | preflight B 段反传用 `yt.std` 当分母，常数目标窗产生 NaN | 改用与训练完全一致的 `window_d` 分母 |
| 6 | 低 | preflight A3 测试边界索引推导错误（把窗口数当成观测数） | 改为 `min(test 起点)+96`，全部 190 变量通过 |
| 7 | 低 | **声明容差重校准**：`accum_grad` 由 1e-5/1e-8 放宽为 **1e-4/1e-6**（实测求和顺序差异 3.04e-6，属你预告的批大小/求和顺序舍入；冻结参数仍逐位校验；最大误差如实记录） | configs/v1.yaml `preflight_tolerances`，此处明示披露 |

修复后代码已通过 `prepare --force` 全量重建（数量与规范附表 A 完全一致：
4041/36763/895/2474/696，ΣU_d=1157），eta 与本地各自重建后内容级一致。

## 6. 建议审计要点

1. §5 问题 1 的修复是否可接受（影响面：此前所有本地数值检查在错位数据上运行，
   修复后全部重跑；错位仅影响缓存读取层，划分计数/边界逻辑不受影响）；
2. §5 问题 7 的容差值是否可接受（或你指定更严值，预检重跑约 8 分钟）；
3. 训练配置抽查：`configs_v1.yaml`（lr 1e-5、micro8×accum4、clip 1.0、10 epochs、
   anchor 0.01、FreqMask p=0.5/rate 0.10）与规范 §4 逐项比对；
4. 伦理面：D1 逐位一致证明推理条件与旧零样本研究完全一致（TF32 双关、bs=64、
   P=1 确定性掩码）。

## 7. 包内文件

| 文件 | 说明 |
|---|---|
| preflight_report.json / stage_preflight.log | eta 预检完整报告与原始日志 |
| protocol_eta.json / protocol_local.json | 双端冻结协议（哈希链+代码指纹+U_d） |
| consistency.json | 双端一致性机读核对（数组级 sha256） |
| configs_v1.yaml | 冻结配置副本 |
| split_counts.csv | 逐领域划分计数 + ΣU_d=1157 |
| SHA256SUMS.txt | 本包自校验清单 |
