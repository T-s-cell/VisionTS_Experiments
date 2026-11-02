#!/bin/bash
# EXP-035: 单组 ETTm1 零样本推理, 对齐系数 c 可切换 (train_epochs=0, 冻结 MAE-base)。
# 用法: bash run_t35_group.sh <PRED_LEN> <ALIGN_C> [SAVE_BASE]
#       bash run_t35_group.sh --fingerprint <PRED_LEN> <ALIGN_C>
# 环境变量: GPU_ID (默认 0), PYTHON (默认 python)
# 与 repro_zeroshot/run_one.sh (EXP-001) 同一调用协议 (ETTm1 表 8: L=2304,
# P=96, r=0.4), 仅 --align_const 取扫描值 (run.py:108 float ->
# models/VisionTS.py:22 -> visionts/model.py npi=int(ratio*14*c))。
# 插值保持默认 bilinear; 原始图像方向。c 以 %.1f 规范化 (0.6 -> "0.6")。
set -u

FPMODE=0
if [ "${1:-}" = "--fingerprint" ]; then FPMODE=1; shift; fi
PRED=$1
C=$(python3 -c 'import sys; print("%.1f" % float(sys.argv[1]))' "$2")
if [ "$FPMODE" -eq 1 ]; then SAVE_BASE="save_t35_align"; else SAVE_BASE=${3:-save_t35_align}; fi
GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CTX=2304; PERIOD=96
ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTm1.csv; DATA=ETTm1

CKPT_DIR="../ckpt/"
CKPT_FILE="$ROOT/$CKPT_DIR/mae_visualize_vit_base.pth"

fingerprint() {
  {
    echo "dataset=ETTm1 pred_len=$PRED seq_len=$CTX periodicity=$PERIOD arch=mae_base align_const=$C"
    echo "norm_const=0.4 batch_size=32 train_epochs=0 seed=2021"
    echo "root_path=$ROOT_PATH data_path=$DATA_PATH data=$DATA features=M"
    echo "ckpt_sha256=$(sha256sum "$CKPT_FILE" | cut -d' ' -f1)"
    for f in run.py exp/exp_long_term_forecasting.py exp/exp_basic.py \
             models/VisionTS.py data_provider/data_factory.py data_provider/data_loader.py \
             ../visionts/model.py ../visionts/models_mae.py; do
      echo "md5:$f=$(md5sum "$f" | cut -d' ' -f1)"
    done
  }
}

if [ "$FPMODE" -eq 1 ]; then
  fingerprint
  exit 0
fi

SAVE_DIR="$SAVE_BASE/c$C"
RUN_DIR="$ROOT/$SAVE_DIR/ETTm1_${PRED}"
LOG="$REPRO_DIR/logs/ETTm1_${PRED}_c${C}.log"
mkdir -p "$REPRO_DIR/logs" "$RUN_DIR"

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "[run_t35_group] ETTm1 pred=$PRED align=$C ctx=$CTX period=$PERIOD save=$SAVE_DIR gpu=$GPU_ID start=$(date +%F\ %T)"
START_TS=$(date +%s)

$PYTHON -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model VisionTS \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --save_dir "$SAVE_DIR/ETTm1_${PRED}" \
  --model_id "VisionTS_ETTm1_${PRED}_c${C}" \
  --data "$DATA" \
  --features M \
  --train_epochs 0 \
  --vm_arch mae_base \
  --vm_ckpt "$CKPT_DIR" \
  --seq_len "$CTX" \
  --periodicity "$PERIOD" \
  --pred_len "$PRED" \
  --norm_const 0.4 \
  --align_const "$C" \
  > "$LOG" 2>&1
RC=$?

ELAPSED=$(( $(date +%s) - START_TS ))
echo "[run_t35_group] exit_code=$RC wall_time=${ELAPSED}s" | tee -a "$LOG"

METRIC_DIR="$RUN_DIR/results/_"
MSE_LOG=$(grep -oE 'mse:[0-9]+\.[0-9]+, mae:[0-9]+\.[0-9]+' "$LOG" | tail -1)
METRICS_OK=0
if [ -f "$METRIC_DIR/metrics.npy" ] && [ -f "$METRIC_DIR/pred.npy" ] && [ -f "$METRIC_DIR/true.npy" ] \
   && [ -n "$MSE_LOG" ]; then
  METRICS_OK=1
fi

if [ "$RC" -ne 0 ]; then
  echo "[run_t35_group] FAILED (exit=$RC), see $LOG"
  exit "$RC"
fi
if [ "$METRICS_OK" -ne 1 ]; then
  echo "[run_t35_group] INCOMPLETE: outputs missing, see $LOG"
  exit 3
fi

{
  fingerprint
  echo "wall_time_s=$ELAPSED"
  echo "run_date=$(date +%F\ %T)"
} > "$RUN_DIR/SUCCESS.txt"

echo "[run_t35_group] OK ETTm1 pred=$PRED align=$C -> $RUN_DIR"
