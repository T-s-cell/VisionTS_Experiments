#!/bin/bash
# 严格串行复现队列。用法: bash run_all.sh [min|main]   (默认 main)
#   min : 仅 ETTh1_96 -> save_min_test/   (最小测试)
#   main: 6 数据集 x 4 pred_len -> save_repro_zeroshot/   (正式复现)
# 续跑: 已完成且 SUCCESS.txt 指纹与当前一致的 run 自动跳过; 指纹不一致则报错停止。
# 失败/OOM: 立即终止整个队列, 写 logs/FAILED.txt。不自动改 batch/精度/参数。
set -u

MODE=${1:-main}
GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}

REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$REPRO_DIR/.." && pwd)"
cd "$ROOT"
mkdir -p "$REPRO_DIR/logs"

LOG="$REPRO_DIR/logs/run_all_${MODE}_$(date +%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

if [ "$MODE" = "min" ]; then
  QUEUE="ETTh1:96"
  SAVE_BASE=save_min_test
else
  QUEUE="ETTh1:96 ETTh1:192 ETTh1:336 ETTh1:720 ETTh2:96 ETTh2:192 ETTh2:336 ETTh2:720 ETTm1:96 ETTm1:192 ETTm1:336 ETTm1:720 ETTm2:96 ETTm2:192 ETTm2:336 ETTm2:720 Weather:96 Weather:192 Weather:336 Weather:720 Electricity:96 Electricity:192 Electricity:336 Electricity:720"
  SAVE_BASE=save_repro_zeroshot
fi

# ---- 环境快照 ----
ENV_TXT="$REPRO_DIR/logs/env_snapshot_${MODE}.txt"
{
  echo "=== $(date +%F\ %T) mode=$MODE gpu=$GPU_ID ==="
  echo "--- python ---"; "$PYTHON" --version 2>&1
  echo "--- key packages ---"
  "$PYTHON" - <<'EOF'
import torch, torchvision, numpy
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("numpy", numpy.__version__)
try:
    import timm; print("timm", timm.__version__)
except Exception as e:
    print("timm import failed:", e)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
EOF
  echo "--- nvidia-smi ---"; nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv,noheader
  echo "--- MAE ckpt ---"
  CKPT="$ROOT/../ckpt/mae_visualize_vit_base.pth"
  if [ -f "$CKPT" ]; then ls -lh "$CKPT"; sha256sum "$CKPT"; else echo "MISSING: $CKPT"; fi
  echo "--- data protocol note ---"
  echo "ETT split: 12/4/4 months (total 20); custom (Weather/Electricity): 0.7/0.1/0.2; StandardScaler fit on train only; metrics in normalized space."
  if [ "$MODE" = "main" ]; then "$PYTHON" -m pip freeze; fi
} > "$ENV_TXT" 2>&1
echo "[run_all] env snapshot -> $ENV_TXT"
grep -E "torch|timm|device|MISSING|months" "$ENV_TXT" | head -8

# ---- 串行队列 ----
FAIL=0
for ITEM in $QUEUE; do
  DS=${ITEM%%:*}
  PRED=${ITEM##*:}
  RUN_DIR="$ROOT/$SAVE_BASE/${DS}_${PRED}"

  if [ -f "$RUN_DIR/SUCCESS.txt" ]; then
    CURRENT=$(bash "$REPRO_DIR/run_one.sh" --fingerprint "$DS" "$PRED")
    if diff <(grep -vE 'wall_time|nvidia|log_metrics|finished|Peak' "$RUN_DIR/SUCCESS.txt") \
            <(echo "$CURRENT") > /dev/null; then
      echo "[run_all] SKIP ${DS}_${PRED} (done, fingerprint match)"
      continue
    else
      echo "[run_all] ABORT: ${DS}_${PRED} has SUCCESS.txt but fingerprint changed (code/ckpt/config drift)."
      echo "          Inspect $RUN_DIR/SUCCESS.txt, re-evaluate, then delete it intentionally to force rerun."
      FAIL=9
      break
    fi
  fi

  echo "[run_all] RUN  ${DS}_${PRED}"
  bash "$REPRO_DIR/run_one.sh" "$DS" "$PRED" "$SAVE_BASE"
  RC=$?
  if [ "$RC" -ne 0 ]; then
    {
      echo "time=$(date +%F\ %T)  config=${DS}_${PRED}  exit=$RC"
      echo "log: $REPRO_DIR/logs/${DS}_${PRED}.log"
      echo "--- log tail ---"
      tail -50 "$REPRO_DIR/logs/${DS}_${PRED}.log"
    } >> "$REPRO_DIR/logs/FAILED.txt"
    echo "[run_all] QUEUE STOPPED on ${DS}_${PRED} (exit=$RC). See logs/FAILED.txt"
    echo "[run_all] OOM 恢复方式: 等显存释放 (nvidia-smi 确认) 后重跑本脚本, 已完成任务自动跳过。"
    FAIL=1
    break
  fi
done

if [ "$FAIL" -eq 0 ]; then
  echo "[run_all] ALL DONE ($MODE): $(echo $QUEUE | wc -w) runs. Collect results:"
  echo "  PYTHON=$PYTHON $PYTHON $REPRO_DIR/collect_results.py --save_base $SAVE_BASE"
fi
exit $FAIL
