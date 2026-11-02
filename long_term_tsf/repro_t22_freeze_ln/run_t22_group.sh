#!/bin/bash
# EXP-033: 单组 Table-19 冻结推理 (完全冻结, train_epochs=0, 原始 MAE-base 权重)。
# 用法: bash run_t22_group.sh <DATASET> <PRED_LEN> <SAVE_DIR>
#       bash run_t22_group.sh --fingerprint <DATASET> <PRED_LEN>  (仅输出配置指纹)
# 环境变量: GPU_ID (默认 0), PYTHON (默认 python)
# 与 repro_zeroshot/run_one.sh (EXP-001) 同一调用协议, 仅 L/P 换为论文 Table 19:
#   ETTh1 1152/24, ETTh2 1152/24, ETTm1 2304/96, ETTm2 1152/96; r=c=0.4。
set -u

FPMODE=0
if [ "${1:-}" = "--fingerprint" ]; then FPMODE=1; shift; fi
DS=$1
PRED=$2
if [ "$FPMODE" -eq 1 ]; then SAVE_DIR="."; else SAVE_DIR=$3; fi
GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# 论文 Table 19 参数表
case "$DS" in
  ETTh1) CTX=1152; PERIOD=24;  ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTh1.csv; DATA=ETTh1 ;;
  ETTh2) CTX=1152; PERIOD=24;  ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTh2.csv; DATA=ETTh2 ;;
  ETTm1) CTX=2304; PERIOD=96;  ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTm1.csv; DATA=ETTm1 ;;
  ETTm2) CTX=1152; PERIOD=96;  ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTm2.csv; DATA=ETTm2 ;;
  *) echo "[run_t22_group] Unknown dataset: $DS"; exit 2 ;;
esac

CKPT_DIR="../ckpt/"
CKPT_FILE="$ROOT/$CKPT_DIR/mae_visualize_vit_base.pth"

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
  }
}

if [ "$FPMODE" -eq 1 ]; then
  fingerprint
  exit 0
fi

RUN_DIR="$ROOT/$SAVE_DIR/${DS}_${PRED}"
LOG="$REPRO_DIR/logs/${DS}_${PRED}.log"
mkdir -p "$REPRO_DIR/logs" "$RUN_DIR"

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "[run_t22_group] $DS pred=$PRED ctx=$CTX period=$PERIOD save=$SAVE_DIR gpu=$GPU_ID start=$(date +%F\ %T)"
START_TS=$(date +%s)

$PYTHON -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model VisionTS \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --save_dir "$SAVE_DIR/${DS}_${PRED}" \
  --model_id "VisionTS_${DS}_${PRED}" \
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

ELAPSED=$(( $(date +%s) - START_TS ))
echo "[run_t22_group] exit_code=$RC wall_time=${ELAPSED}s" | tee -a "$LOG"

METRIC_DIR="$RUN_DIR/results/_"
MSE_LOG=$(grep -oE 'mse:[0-9]+\.[0-9]+, mae:[0-9]+\.[0-9]+' "$LOG" | tail -1)
METRICS_OK=0
if [ -f "$METRIC_DIR/metrics.npy" ] && [ -f "$METRIC_DIR/pred.npy" ] && [ -f "$METRIC_DIR/true.npy" ] \
   && [ -n "$MSE_LOG" ]; then
  METRICS_OK=1
fi

if [ "$RC" -ne 0 ]; then
  echo "[run_t22_group] FAILED (exit=$RC), see $LOG"
  exit "$RC"
fi
if [ "$METRICS_OK" -ne 1 ]; then
  echo "[run_t22_group] INCOMPLETE: outputs missing, see $LOG"
  exit 3
fi

{
  fingerprint
  echo "wall_time_s=$ELAPSED"
  echo "run_date=$(date +%F\ %T)"
} > "$RUN_DIR/SUCCESS.txt"

echo "[run_t22_group] OK $DS pred=$PRED -> $RUN_DIR"
