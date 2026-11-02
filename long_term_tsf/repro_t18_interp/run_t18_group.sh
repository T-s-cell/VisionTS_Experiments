#!/bin/bash
# EXP-034: 单组 Table-8 零样本推理, 插值可切换 (train_epochs=0, 冻结 MAE-base)。
# 用法: bash run_t18_group.sh <DATASET> <PRED_LEN> <INTERP> [SAVE_BASE]
#       bash run_t18_group.sh --fingerprint <DATASET> <PRED_LEN> <INTERP>
# 环境变量: GPU_ID (默认 0), PYTHON (默认 python)
# 与 repro_zeroshot/run_one.sh (EXP-001) 同一调用协议 (表 8: ETTh1 2880/24,
# ETTh2 1728/24, ETTm1 2304/96, ETTm2 4032/96; r=c=0.4), 仅新增
# --interpolation (run.py:106 -> models/VisionTS.py:22 -> visionts/model.py:84-92
# 一值同时驱动 input_resize 与 output_resize; 双线性列整体复用 EXP-001)。
set -u

FPMODE=0
if [ "${1:-}" = "--fingerprint" ]; then FPMODE=1; shift; fi
DS=$1
PRED=$2
INTERP=$3
case "$INTERP" in
  bilinear|bicubic|nearest) ;;
  *) echo "[run_t18_group] Bad interpolation: $INTERP (need bilinear|bicubic|nearest)"; exit 2 ;;
esac
if [ "$FPMODE" -eq 1 ]; then SAVE_BASE="save_t18_interp"; else SAVE_BASE=${4:-save_t18_interp}; fi
GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# 论文 Table 8 参数表 (EXP-001 case 表)
case "$DS" in
  ETTh1) CTX=2880; PERIOD=24;  ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTh1.csv; DATA=ETTh1 ;;
  ETTh2) CTX=1728; PERIOD=24;  ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTh2.csv; DATA=ETTh2 ;;
  ETTm1) CTX=2304; PERIOD=96;  ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTm1.csv; DATA=ETTm1 ;;
  ETTm2) CTX=4032; PERIOD=96;  ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTm2.csv; DATA=ETTm2 ;;
  *) echo "[run_t18_group] Unknown dataset: $DS"; exit 2 ;;
esac

CKPT_DIR="../ckpt/"
CKPT_FILE="$ROOT/$CKPT_DIR/mae_visualize_vit_base.pth"

fingerprint() {
  {
    echo "dataset=$DS pred_len=$PRED seq_len=$CTX periodicity=$PERIOD arch=mae_base interpolation=$INTERP"
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

SAVE_DIR="$SAVE_BASE/$INTERP"
RUN_DIR="$ROOT/$SAVE_DIR/${DS}_${PRED}"
LOG="$REPRO_DIR/logs/${DS}_${PRED}_${INTERP}.log"
mkdir -p "$REPRO_DIR/logs" "$RUN_DIR"

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "[run_t18_group] $DS pred=$PRED interp=$INTERP ctx=$CTX period=$PERIOD save=$SAVE_DIR gpu=$GPU_ID start=$(date +%F\ %T)"
START_TS=$(date +%s)

$PYTHON -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model VisionTS \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --save_dir "$SAVE_DIR/${DS}_${PRED}" \
  --model_id "VisionTS_${DS}_${PRED}_${INTERP}" \
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
  --interpolation "$INTERP" \
  > "$LOG" 2>&1
RC=$?

ELAPSED=$(( $(date +%s) - START_TS ))
echo "[run_t18_group] exit_code=$RC wall_time=${ELAPSED}s" | tee -a "$LOG"

METRIC_DIR="$RUN_DIR/results/_"
MSE_LOG=$(grep -oE 'mse:[0-9]+\.[0-9]+, mae:[0-9]+\.[0-9]+' "$LOG" | tail -1)
METRICS_OK=0
if [ -f "$METRIC_DIR/metrics.npy" ] && [ -f "$METRIC_DIR/pred.npy" ] && [ -f "$METRIC_DIR/true.npy" ] \
   && [ -n "$MSE_LOG" ]; then
  METRICS_OK=1
fi

if [ "$RC" -ne 0 ]; then
  echo "[run_t18_group] FAILED (exit=$RC), see $LOG"
  exit "$RC"
fi
if [ "$METRICS_OK" -ne 1 ]; then
  echo "[run_t18_group] INCOMPLETE: outputs missing, see $LOG"
  exit 3
fi

{
  fingerprint
  echo "wall_time_s=$ELAPSED"
  echo "run_date=$(date +%F\ %T)"
} > "$RUN_DIR/SUCCESS.txt"

echo "[run_t18_group] OK $DS pred=$PRED interp=$INTERP -> $RUN_DIR"
