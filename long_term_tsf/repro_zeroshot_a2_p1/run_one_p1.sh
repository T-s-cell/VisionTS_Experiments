#!/bin/bash
# EXP-030: single P=1 zero-shot run, mirroring EXP-001 run_one.sh.
# Only delta vs EXP-001: periodicity=1 (imaging period), save dir, and the
# launcher records the real-path segmentation shape (infer_one_p1.py).
# 用法: bash run_one_p1.sh <DATASET> <PRED_LEN> <SAVE_DIR>
#       bash run_one_p1.sh --fingerprint <DATASET> <PRED_LEN>
# 环境变量: GPU_ID (默认 0), PYTHON (默认 python)
set -u

FPMODE=0
if [ "${1:-}" = "--fingerprint" ]; then FPMODE=1; shift; fi
DS=$1
PRED=$2
if [ "$FPMODE" -eq 1 ]; then SAVE_DIR="."; else SAVE_DIR=$3; fi
GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
PERIOD=1   # <-- the only experimental change vs EXP-001

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

case "$DS" in
  ETTh1)       CTX=2880; ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTh1.csv; DATA=ETTh1 ;;
  ETTh2)       CTX=1728; ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTh2.csv; DATA=ETTh2 ;;
  ETTm1)       CTX=2304; ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTm1.csv; DATA=ETTm1 ;;
  ETTm2)       CTX=4032; ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTm2.csv; DATA=ETTm2 ;;
  *) echo "[run_one_p1] Unknown dataset: $DS"; exit 2 ;;
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
    for f in repro_zeroshot_a2_p1/infer_one_p1.py repro_zeroshot_a2_p1/a2p1_common.py; do
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
GPU_POLL_FILE="$REPRO_DIR/logs/.gpu_poll_$$"
( while true; do nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "$GPU_POLL_FILE"; sleep 5; done ) &
GPU_POLL_PID=$!
trap 'kill $GPU_POLL_PID 2>/dev/null; rm -f "$GPU_POLL_FILE"' EXIT

echo "[run_one_p1] $DS pred=$PRED ctx=$CTX period=$PERIOD save=$SAVE_DIR gpu=$GPU_ID start=$(date +%F\ %T)"
START_TS=$(date +%s)

$PYTHON -u "$REPRO_DIR/infer_one_p1.py" \
  --shape_log "$RUN_DIR/shape_record.jsonl" \
  -- \
  --task_name long_term_forecast \
  --is_training 1 \
  --model VisionTS \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --save_dir "$SAVE_DIR/${DS}_${PRED}" \
  --model_id "VisionTS_${DS}_${PRED}_P1" \
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
echo "[run_one_p1] exit_code=$RC wall_time=${ELAPSED}s" | tee -a "$LOG"

METRIC_DIR="$RUN_DIR/results/_"
METRICS_OK=0
MSE_LOG=$(grep -oE 'mse:[0-9]+\.[0-9]+, mae:[0-9]+\.[0-9]+' "$LOG" | tail -1)
SHAPE_OK=0
if [ -f "$RUN_DIR/shape_record.jsonl" ]; then
  N_SEG=$(grep -c '"type": "segmentation"' "$RUN_DIR/shape_record.jsonl" 2>/dev/null || echo 0)
  [ "${N_SEG:-0}" -gt 0 ] && SHAPE_OK=1
fi
if [ -f "$METRIC_DIR/metrics.npy" ] && [ -f "$METRIC_DIR/pred.npy" ] && [ -f "$METRIC_DIR/true.npy" ] \
   && [ -n "$MSE_LOG" ] && [ "$SHAPE_OK" -eq 1 ]; then
  METRICS_OK=1
fi

if [ "$RC" -ne 0 ]; then
  echo "[run_one_p1] FAILED (exit=$RC), see $LOG"
  exit "$RC"
fi
if [ "$METRICS_OK" -ne 1 ]; then
  echo "[run_one_p1] INCOMPLETE: outputs/shape_record missing or metrics not parseable, see $LOG"
  exit 3
fi

PROJ_PEAK=$(grep -oE 'Peak GPU memory \(this process\): [0-9.]+ GB' "$LOG" | tail -1)
{
  fingerprint
  echo "wall_time_s=$ELAPSED"
  echo "$PROJ_PEAK"
  echo "nvidia_smi_device_peak_mib(aux, whole-GPU incl. other procs)=${NVIDIA_PEAK:-n/a}"
  echo "log_metrics=$MSE_LOG"
  echo "shape_record=shape_record.jsonl seg_batches=$N_SEG"
  echo "finished=$(date +%F\ %T)"
} > "$RUN_DIR/SUCCESS.txt"

echo "[run_one_p1] SUCCESS ${DS}_${PRED} $MSE_LOG $PROJ_PEAK time=${ELAPSED}s"
