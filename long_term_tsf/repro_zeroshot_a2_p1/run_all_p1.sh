#!/bin/bash
# EXP-030 P=1 zero-shot queue (16 groups = 4 ETT x 4 pred_lens).
# 用法: bash run_all_p1.sh [min|main]   (默认 main)
#   min : preflight + ETTh1_96 冒烟 + probes 三断言
#   main: preflight + 要求 ETTh1_96/PROBES_OK.txt 后跑完全部 16 组(已完成且指纹一致者跳过)
# 失败/OOM: 立即终止队列, 写 logs/FAILED.txt; 不自动改参。
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

QUEUE="ETTh1:96 ETTh1:192 ETTh1:336 ETTh1:720 ETTh2:96 ETTh2:192 ETTh2:336 ETTh2:720 ETTm1:96 ETTm1:192 ETTm1:336 ETTm1:720 ETTm2:96 ETTm2:192 ETTm2:336 ETTm2:720"
if [ "$MODE" = "min" ]; then
  QUEUE="ETTh1:96"
fi
SAVE_BASE=save_a2p1_zeroshot
BASE_SAVE_BASE=save_repro_zeroshot

# ---- 预检(基线在盘 + 指纹 + 权重 sha + 基线指标复算) ----
echo "[run_all_p1] PREFLIGHT start"
$PYTHON "$REPRO_DIR/preflight.py"
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "[run_all_p1] PREFLIGHT FAILED (exit=$RC), abort."
  exit "$RC"
fi

# ---- 环境快照 ----
ENV_TXT="$REPRO_DIR/logs/env_snapshot_${MODE}.txt"
{
  echo "=== $(date +%F\ %T) mode=$MODE gpu=$GPU_ID ==="
  echo "--- python ---"; "$PYTHON" --version 2>&1
  echo "--- key packages ---"
  "$PYTHON" - <<'EOF'
import torch, numpy
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
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
  echo "--- baseline EXP-001 env ---"
  grep -E "^torch |^numpy|^timm|Python" "$REPRO_DIR/../repro_zeroshot/logs/env_snapshot_main.txt" 2>/dev/null | head -5 || true
} > "$ENV_TXT" 2>&1
echo "[run_all_p1] env snapshot -> $ENV_TXT"
grep -E "torch|timm|device|MISSING" "$ENV_TXT" | head -6

# ---- 串行队列 ----
FAIL=0
for ITEM in $QUEUE; do
  DS=${ITEM%%:*}
  PRED=${ITEM##*:}
  RUN_DIR="$ROOT/$SAVE_BASE/${DS}_${PRED}"
  BASE_DIR="$ROOT/$BASE_SAVE_BASE/${DS}_${PRED}"

  if [ ! -f "$BASE_DIR/SUCCESS.txt" ]; then
    echo "[run_all_p1] ABORT: baseline $BASE_DIR/SUCCESS.txt missing (need EXP-001 outputs)."
    FAIL=8; break
  fi

  if [ -f "$RUN_DIR/SUCCESS.txt" ]; then
    CURRENT=$(bash "$REPRO_DIR/run_one_p1.sh" --fingerprint "$DS" "$PRED")
    if diff <(grep -vE 'wall_time|nvidia|log_metrics|finished|Peak|shape_record' "$RUN_DIR/SUCCESS.txt") \
            <(echo "$CURRENT") > /dev/null; then
      echo "[run_all_p1] SKIP ${DS}_${PRED} (done, fingerprint match)"
    else
      echo "[run_all_p1] ABORT: ${DS}_${PRED} has SUCCESS.txt but fingerprint changed."
      FAIL=9
      break
    fi
  else
    if [ "$MODE" = "main" ] && [ ! -f "$ROOT/$SAVE_BASE/ETTh1_96/PROBES_OK.txt" ]; then
      echo "[run_all_p1] ABORT: smoke gate — $SAVE_BASE/ETTh1_96/PROBES_OK.txt missing."
      echo "              run: bash run_all_p1.sh min  (then probes_p1.py), before main."
      FAIL=7; break
    fi
    echo "[run_all_p1] RUN  ${DS}_${PRED}"
    bash "$REPRO_DIR/run_one_p1.sh" "$DS" "$PRED" "$SAVE_BASE"
    RC=$?
    if [ "$RC" -ne 0 ]; then
      {
        echo "time=$(date +%F\ %T)  config=${DS}_${PRED}  exit=$RC"
        echo "log: $REPRO_DIR/logs/${DS}_${PRED}.log"
        echo "--- log tail ---"
        tail -50 "$REPRO_DIR/logs/${DS}_${PRED}.log"
      } >> "$REPRO_DIR/logs/FAILED.txt"
      echo "[run_all_p1] QUEUE STOPPED on ${DS}_${PRED} (exit=$RC). See logs/FAILED.txt"
      FAIL=1
      break
    fi
  fi

  # 每组跑完(或跳过)即做窗口/形状核验并写 PROBES_OK.txt
  if [ ! -f "$RUN_DIR/PROBES_OK.txt" ]; then
    echo "[run_all_p1] PROBES ${DS}_${PRED}"
    $PYTHON "$REPRO_DIR/probes_p1.py" --group "$DS" "$PRED"
    RC=$?
    if [ "$RC" -ne 0 ]; then
      echo "[run_all_p1] PROBES FAILED on ${DS}_${PRED} (exit=$RC). QUEUE STOPPED."
      FAIL=2; break
    fi
  fi
done

if [ "$FAIL" -eq 0 ]; then
  echo "[run_all_p1] ALL DONE ($MODE): $(echo $QUEUE | wc -w) runs. Next:"
  echo "  $PYTHON $REPRO_DIR/post_eval.py --all"
  echo "  $PYTHON $REPRO_DIR/collect_summary.py"
fi
exit $FAIL
