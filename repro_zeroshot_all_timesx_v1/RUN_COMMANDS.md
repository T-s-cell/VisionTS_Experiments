# RUN_COMMANDS — repro_zeroshot_all_timesx_v1

本地 = WSL2（`/home/wlt/miniconda3/envs/aurora_test/bin/python`，numpy 2.2.6 / torch 2.12.1 CPU）。
GPU = theta（双 RTX 3090，优先）或 eta（L20，备用）。

## 阶段 1：本地冻结与校验（2026-10-10 修复后重跑完成）

```bash
cd /home/wlt/MMTS/VisionTS_Experiments/repro_zeroshot_all_timesx_v1
PY=/home/wlt/miniconda3/envs/aurora_test/bin/python
$PY freeze_full.py            # S1: 8,106 窗冻结（frozen_arrays.npz 含 d + denom_source）
$PY verify_cache.py           # S2: 旧缓存 190×P1 绑定+逐窗核对 → reuse_plan.json
$PY preflight.py --sample-only  # 冻结抽样名单 preflight_sample.json（194 窗）
```

状态与续跑：`run_all.sh` 用 `.state/<stage>.done` 记录，标记内含**输出 sha256 +
input_guards（该阶段的关键输入、上游报告、代码 md5，如 fill 含 190 个旧 npz、
protocol/infer_fill 等 6 个代码文件）**，由 `stage_guard.py` 逐项重算比对——
产物、输入或代码任一漂移即自动重跑；强制重跑用 `FORCE_<STAGE大写>=1`
（如 `FORCE_FILL=1 bash run_all.sh fill`）。

S4 读文件来源：复用**判定**取自 reuse_final.json（仅 verdict）；旧缓存**文件名与
sha256** 取自 reuse_plan.json（S2）——两文件字段分离，键缺失即硬停。

## 阶段 2：GPU 部署（theta 优先，被占切 eta）

**S3→S4 绑定**：preflight `--run` 把 `forward_fingerprint()`（zip/冻结数组/
ckpt/代码/配置/env）写入 reuse_final.json；infer_fill 启动时重算同指纹并比对，
不一致即拒跑。因此 **S3 与 S4 必须在同一主机、同一 conda env、同一 tmux 会话内
连续执行**；中途换机/换 env 必须重跑 S3。

```bash
# 0) 门检：存在 used<=512MiB 卡则用之；否则共跑判定 free-4096MiB>~3.5GB；再不行切 eta
ssh theta "nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv"

# 1) 同步（含旧 cache 只读副本 / frozen / 代码；ckpt 与 zip 若已在服务器且 sha 合法则跳过）
rsync -av --exclude '.state' --exclude 'cache_all' --exclude 'results' --exclude 'logs' \
  /home/wlt/MMTS/VisionTS_Experiments/repro_zeroshot_all_timesx_v1/ \
  theta:~/VisionTS_Experiments/repro_zeroshot_all_timesx_v1/
rsync -av /home/wlt/MMTS/VisionTS_Experiments/repro_zeroshot_ext/cache/ \
  theta:~/VisionTS_Experiments/repro_zeroshot_ext/cache/   # 只读副本
# zip: TimesX_Datasets.zip -> 服务器（infer_fill/preflight 经 timesx_data.find_zip 定位）
# ckpt: VisionTS_Experiments/ckpt/mae_visualize_vit_base.pth（sha256 必须 = c551c37b…）

# 2) 服务器执行（tmux；TF32 双关已写死在代码并在运行时读回断言）
ssh theta
tmux new -s vz_allfill_theta
conda activate <torch2.12.0+cu126 env>   # theta: timemixer 克隆 env；fallback: visionts env(torch 2.3, 先 1-var smoke)
cd ~/VisionTS_Experiments/repro_zeroshot_all_timesx_v1
CUDA_VISIBLE_DEVICES=<idle_gpu> python preflight.py --run      # S3（含 ckpt sha 断言 + 指纹绑定）
CUDA_VISIBLE_DEVICES=<idle_gpu> python infer_fill.py           # S4（先校验 S3 绑定，再做集合差补推理）
# smoke（可选先行）: python infer_fill.py --only-var <某小变量前缀> 确认前向链路；
#                    --only-var 不写 .state 标记，不影响正式计数

# 3) 回传
rsync -av theta:~/VisionTS_Experiments/repro_zeroshot_all_timesx_v1/{cache_all,preflight_report.json,reuse_final.json,.state} \
  /home/wlt/MMTS/VisionTS_Experiments/repro_zeroshot_all_timesx_v1/
```

## 阶段 3：本地汇总与验收

```bash
bash run_all.sh post           # assemble → evaluate → recompute → accept
python make_report.py
bash audit_all.sh --final      # 审计包 → /home/wlt/MMTS/temp/（断言 <50M）
```

- assemble 合并时**重新哈希**每个被引用 npz（旧 cache 与 cache_all）并与记录值比对；
  同时断言补跑集合恰为「全部窗口 − 实际缓存窗口」的集合差。
- recompute_check.py 独立复算：d 偏差、target 与冻结数组一致性、身份/覆盖、
  变量→域→总体自底向上汇总，任何偏差即 exit 1（不再"错误也通过"）。

## 断点续跑

- `cache_all/manifest.json` 指纹绑定 zip/frozen 数组/ckpt/代码/配置；指纹不符 → 拒绝混跑。
- `reuse_final.json.forward_binding` 绑定 S3 时的同一指纹；S4 启动即校验。
- 每变量 entry 校验：文件存在 + 完整键集 + sha256 与 manifest 一致 + n/sample_ids 匹配，
  否则该变量整变量重推。
- 阶段级：`bash run_all.sh <stage>` 经 stage_guard 校验标记内 sha256 后才跳过；
  `FORCE_<STAGE>=1` 无条件重跑。
