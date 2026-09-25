#!/bin/bash
# 单次 zero-shot 运行封装。
# 用法: bash run_one.sh <DATASET> <PRED_LEN> <SAVE_DIR>
#       bash run_one.sh --fingerprint <DATASET> <PRED_LEN>   (仅输出配置指纹, 供续跑核对)
# 环境变量: GPU_ID (默认 0), PYTHON (默认 python)
# 成功判定(三者缺一不可): 进程退出码 0 + metrics/pred/true 三文件齐全 + 日志解析出有限 mse/mae。
# 成功后写 <SAVE_DIR>/<DS>_<PRED>/SUCCESS.txt (含配置指纹, 供续跑核对)。
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

# 参数表: 与 scripts/vision_ts_zeroshot/ 官方脚本逐项一致 (论文 Table 8), 此处显式维护, 不 source 官方脚本
case "$DS" in
  ETTh1)       CTX=2880; PERIOD=24;  ROOT_PATH=./dataset/ETT-small/;    DATA_PATH=ETTh1.csv;       DATA=ETTh1 ;;
  ETTh2)       CTX=1728; PERIOD=24;  ROOT_PATH=./dataset/ETT-small/;    DATA_PATH=ETTh2.csv;       DATA=ETTh2 ;;
  ETTm1)       CTX=2304; PERIOD=96;  ROOT_PATH=./dataset/ETT-small/;    DATA_PATH=ETTm1.csv;       DATA=ETTm1 ;;
  ETTm2)       CTX=4032; PERIOD=96;  ROOT_PATH=./dataset/ETT-small/;    DATA_PATH=ETTm2.csv;       DATA=ETTm2 ;;
  Weather)     CTX=4032; PERIOD=144; ROOT_PATH=./dataset/weather/;      DATA_PATH=weather.csv;     DATA=custom ;;
  Electricity) CTX=2880; PERIOD=24;  ROOT_PATH=./dataset/electricity/;  DATA_PATH=electricity.csv; DATA=custom ;;
  *) echo "[run_one] Unknown dataset: $DS"; exit 2 ;;
esac

CKPT_DIR="../ckpt/"
CKPT_FILE="$ROOT/$CKPT_DIR/mae_visualize_vit_base.pth"

# ---- 配置指纹: 数据参数 + 权重 SHA256 + 关键代码 md5 (续跑一致性核对依据) ----
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

# ---- 辅助监控: 整卡显存采样 (含其他进程占用, 仅参考); 主口径为进程内 torch 峰值 ----
export CUDA_VISIBLE_DEVICES=$GPU_ID
GPU_POLL_FILE="$REPRO_DIR/logs/.gpu_poll_$$"
( while true; do nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "$GPU_POLL_FILE"; sleep 5; done ) &
GPU_POLL_PID=$!
trap 'kill $GPU_POLL_PID 2>/dev/null; rm -f "$GPU_POLL_FILE"' EXIT

echo "[run_one] $DS pred=$PRED ctx=$CTX period=$PERIOD save=$SAVE_DIR gpu=$GPU_ID start=$(date +%F\ %T)"
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

kill $GPU_POLL_PID 2>/dev/null
wait $GPU_POLL_PID 2>/dev/null
trap - EXIT

ELAPSED=$(( $(date +%s) - START_TS ))
NVIDIA_PEAK=$(sort -n "$GPU_POLL_FILE" 2>/dev/null | tail -1)
echo "[run_one] exit_code=$RC wall_time=${ELAPSED}s" | tee -a "$LOG"

# ---- 成功判定与标记 ----
METRIC_DIR="$RUN_DIR/results/_"
METRICS_OK=0
MSE_LOG=$(grep -oE 'mse:[0-9]+\.[0-9]+, mae:[0-9]+\.[0-9]+' "$LOG" | tail -1)
if [ -f "$METRIC_DIR/metrics.npy" ] && [ -f "$METRIC_DIR/pred.npy" ] && [ -f "$METRIC_DIR/true.npy" ] \
   && [ -n "$MSE_LOG" ]; then
  METRICS_OK=1
fi

if [ "$RC" -ne 0 ]; then
  echo "[run_one] FAILED (exit=$RC), see $LOG"
  exit "$RC"
fi
if [ "$METRICS_OK" -ne 1 ]; then
  echo "[run_one] INCOMPLETE: outputs missing or metrics not finite, see $LOG"
  exit 3
fi

PROJ_PEAK=$(grep -oE 'Peak GPU memory \(this process\): [0-9.]+ GB' "$LOG" | tail -1)
{
  fingerprint
  echo "wall_time_s=$ELAPSED"
  echo "$PROJ_PEAK"
  echo "nvidia_smi_device_peak_mib(aux, whole-GPU incl. other procs)=${NVIDIA_PEAK:-n/a}"
  echo "log_metrics=$MSE_LOG"
  echo "finished=$(date +%F\ %T)"
} > "$RUN_DIR/SUCCESS.txt"

echo "[run_one] SUCCESS ${DS}_${PRED} $MSE_LOG $PROJ_PEAK time=${ELAPSED}s"
