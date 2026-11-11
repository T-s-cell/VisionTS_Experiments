#!/bin/bash
# EXP-043: single zero-shot test run with a candidate periodicity P,
# mirroring EXP-030's run_one_p1.sh (which mirrors EXP-001 run_one.sh).
# The run.py invocation is byte-identical to EXP-001's except --periodicity.
# 用法: bash run_one_t43.sh <DATASET> <PRED_LEN> <SAVE_DIR> <PERIOD>
#       bash run_one_t43.sh --fingerprint <DATASET> <PRED_LEN> <PERIOD>
# 环境变量: GPU_ID (默认 0), PYTHON, LEDGER_REQUIRED (默认 1; 冒烟设 0)
# Ledger gate (frozen): LEDGER_REQUIRED=1 requires logs/LEDGER_OK; the
# ledger sha256 is embedded in the fingerprint (accept-before-select).
# 成功判定: 进程退出码 0 + metrics/pred/true 齐全 + 日志解析出有限 mse/mae
#           + shape_record 含 update_config 与 segmentation 记录。
# skip-if-done: SUCCESS.txt 指纹一致 -> 跳过; 不一致 -> ABORT(不覆盖)。
set -u

FPMODE=0
if [ "${1:-}" = "--fingerprint" ]; then FPMODE=1; shift; fi
DS=$1
PRED=$2
if [ "$FPMODE" -eq 1 ]; then SAVE_DIR="."; PERIOD=$3; else SAVE_DIR=$3; PERIOD=$4; fi
GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
LEDGER_REQUIRED=${LEDGER_REQUIRED:-1}

REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$REPRO_DIR/.." && pwd)"
cd "$ROOT"

case "$DS" in
  ETTh1)       CTX=2880; ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTh1.csv; DATA=ETTh1 ;;
  ETTh2)       CTX=1728; ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTh2.csv; DATA=ETTh2 ;;
  ETTm1)       CTX=2304; ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTm1.csv; DATA=ETTm1 ;;
  ETTm2)       CTX=4032; ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTm2.csv; DATA=ETTm2 ;;
  *) echo "[run_one_t43] Unknown dataset: $DS"; exit 2 ;;
esac

CKPT_DIR="../ckpt/"
CKPT_FILE="$ROOT/$CKPT_DIR/mae_visualize_vit_base.pth"

LEDGER_FILE="$REPRO_DIR/logs/LEDGER_OK"
if [ "$LEDGER_REQUIRED" -eq 1 ]; then
  if [ ! -f "$LEDGER_FILE" ]; then
    echo "[run_one_t43] ABORT: LEDGER_OK missing (accept-before-select frozen)"
    exit 5
  fi
  LEDGER_SHA=$(sha256sum "$LEDGER_FILE" | cut -d' ' -f1)
else
  LEDGER_SHA="none"
fi

fingerprint() {
  {
    echo "dataset=$DS pred_len=$PRED seq_len=$CTX periodicity=$PERIOD arch=mae_base"
    echo "norm_const=0.4 align_const=0.4 batch_size=32 train_epochs=0 seed=2021"
    echo "root_path=$ROOT_PATH data_path=$DATA_PATH data=$DATA features=M"
    echo "ckpt_sha256=$(sha256sum "$CKPT_FILE" | cut -d' ' -f1)"
    for f in run.py exp/exp_long_term_forecasting.py exp/exp_basic.py \
             models/VisionTS.py data_provider/data_factory.py data_provider/data_loader.py \
             ../visionts/model.py ../visionts/models_mae.py; do
      echo "md5:$f=$(md5sum "$f" | cut -d' ' -f1)"
    done
    for f in repro_t43_period_val/infer_one_t43.py \
             repro_t43_period_val/t43_common.py \
             repro_t43_period_val/val_pass.py; do
      echo "md5:$f=$(md5sum "$f" | cut -d' ' -f1)"
    done
    echo "ledger_sha256=$LEDGER_SHA"
  }
}

if [ "$FPMODE" -eq 1 ]; then
  fingerprint
  exit 0
fi

RUN_DIR="$ROOT/$SAVE_DIR/${DS}_${PRED}"
LOG="$REPRO_DIR/logs/${DS}_${PRED}_P${PERIOD}.log"
mkdir -p "$REPRO_DIR/logs" "$RUN_DIR"

# ---- skip-if-done: SUCCESS.txt 指纹一致 -> 跳过; 不一致 -> ABORT 不覆盖 ----
if [ -f "$RUN_DIR/SUCCESS.txt" ]; then
  if diff -q <(fingerprint) <(grep -v -e '^wall_time_s=' -e '^Peak GPU' \
        -e '^nvidia_smi_device_peak_mib' -e '^log_metrics=' -e '^shape_record=' \
        -e '^finished=' "$RUN_DIR/SUCCESS.txt") >/dev/null 2>&1; then
    echo "[run_one_t43] $DS pred=$PRED P=$PERIOD: SUCCESS fingerprint match -> skip"
    exit 0
  fi
  echo "[run_one_t43] ABORT: $RUN_DIR/SUCCESS.txt fingerprint mismatch (no overwrite)"
  exit 4
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
GPU_POLL_FILE="$REPRO_DIR/logs/.gpu_poll_$$"
( while true; do nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "$GPU_POLL_FILE"; sleep 5; done ) &
GPU_POLL_PID=$!
trap 'kill $GPU_POLL_PID 2>/dev/null; rm -f "$GPU_POLL_FILE"' EXIT

echo "[run_one_t43] $DS pred=$PRED ctx=$CTX period=$PERIOD save=$SAVE_DIR gpu=$GPU_ID start=$(date +%F\ %T)"
START_TS=$(date +%s)

$PYTHON -u "$REPRO_DIR/infer_one_t43.py" \
  --shape_log "$RUN_DIR/shape_record.jsonl" \
  -- \
  --task_name long_term_forecast \
  --is_training 1 \
  --model VisionTS \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --save_dir "$SAVE_DIR/${DS}_${PRED}" \
  --model_id "VisionTS_${DS}_${PRED}_P${PERIOD}" \
  --data "$DATA" \
  --features M \
  --train_epochs 0 \
  --vm_arch mae_base \
  --vm_ckpt "$CKPT_DIR" \
  --seq_len "$CTX" \
  --periodicity "$PERIOD" \
  --pred_len "$PRED" \
  --norm_const 0.4 \
  --align_const 0.4 \
  > "$LOG" 2>&1
RC=$?

kill $GPU_POLL_PID 2>/dev/null
wait $GPU_POLL_PID 2>/dev/null
trap - EXIT

ELAPSED=$(( $(date +%s) - START_TS ))
NVIDIA_PEAK=$(sort -n "$GPU_POLL_FILE" 2>/dev/null | tail -1)
echo "[run_one_t43] exit_code=$RC wall_time=${ELAPSED}s" | tee -a "$LOG"

METRIC_DIR="$RUN_DIR/results/_"
MSE_LOG=$(grep -oE 'mse:[0-9]+\.[0-9]+, mae:[0-9]+\.[0-9]+' "$LOG" | tail -1)
SHAPE_OK=0
N_UC=0
if [ -f "$RUN_DIR/shape_record.jsonl" ]; then
  N_UC=$(grep -c '"type": "update_config"' "$RUN_DIR/shape_record.jsonl" 2>/dev/null || echo 0)
  N_SEG=$(grep -c '"type": "segmentation"' "$RUN_DIR/shape_record.jsonl" 2>/dev/null || echo 0)
  [ "${N_UC:-0}" -ge 1 ] && [ "${N_SEG:-0}" -gt 0 ] && SHAPE_OK=1
fi
OUT_OK=0
if [ -f "$METRIC_DIR/metrics.npy" ] && [ -f "$METRIC_DIR/pred.npy" ] \
   && [ -f "$METRIC_DIR/true.npy" ] && [ -n "$MSE_LOG" ] && [ "$SHAPE_OK" -eq 1 ]; then
  OUT_OK=1
fi

if [ "$RC" -ne 0 ]; then
  echo "[run_one_t43] FAILED (exit=$RC), see $LOG"
  exit "$RC"
fi
if [ "$OUT_OK" -ne 1 ]; then
  echo "[run_one_t43] INCOMPLETE: outputs/shape_record missing or metrics not parseable, see $LOG"
  exit 3
fi

PROJ_PEAK=$(grep -oE 'Peak GPU memory \(this process\): [0-9.]+ GB' "$LOG" | tail -1)
{
  fingerprint
  echo "wall_time_s=$ELAPSED"
  echo "$PROJ_PEAK"
  echo "nvidia_smi_device_peak_mib(aux, whole-GPU incl. other procs)=${NVIDIA_PEAK:-n/a}"
  echo "log_metrics=$MSE_LOG"
  echo "shape_record=shape_record.jsonl update_config_calls=$N_UC"
  echo "finished=$(date +%F\ %T)"
} > "$RUN_DIR/SUCCESS.txt"

echo "[run_one_t43] SUCCESS ${DS}_${PRED} P${PERIOD} $MSE_LOG time=${ELAPSED}s"
