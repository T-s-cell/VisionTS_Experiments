#!/bin/bash
# EXP-045: single zero-shot test run for one MAE variant (arch), mirroring
# t44's run_one_t44.sh (which mirrors EXP-001 run_one.sh).  The run.py
# invocation is byte-identical to EXP-001's except --vm_arch; everything
# else (r=c=0.4, L=table 8, P=table 6, bs=32, seed=2021, epochs=0) stays
# the frozen EXP-001 protocol.
# 用法: bash run_one_t45.sh <DATASET> <PRED_LEN> <ARCH_SHORT: base|large|huge>
#       bash run_one_t45.sh --fingerprint <DATASET> <PRED_LEN> <ARCH_SHORT>
# 环境变量: GPU_ID (默认 0), PYTHON
# Output isolation (review correction #2): dirs are single-purpose --
#   base  -> save_t45_base_replay/   (the ONLY base runs this round are the
#                                     2 smoke replays; the 16 real Base
#                                     groups are EXP-001 reuse, read-only)
#   large -> save_t45_large/         huge -> save_t45_huge/
# and the per-group log name embeds ds/H/arch: logs/t45_<arch>_<DS>_<H>.log
# 成功判定: 进程退出码 0 + metrics/pred/true 齐全 + 日志解析出 mse/mae
#           + shape_record 含 update_config 与 segmentation 记录。
# skip-if-done: SUCCESS.txt 指纹一致 -> 跳过; 不一致 -> ABORT(不覆盖)。
set -u

FPMODE=0
if [ "${1:-}" = "--fingerprint" ]; then FPMODE=1; shift; fi
DS=$1
PRED=$2
SHORT=$3
if [ "$FPMODE" -eq 1 ]; then SAVE_DIR_REL="."; else SAVE_DIR_REL=$4; fi
GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}

case "$SHORT" in
  base)  ARCH=mae_base;  ARCH_CKPT=mae_visualize_vit_base.pth ;;
  large) ARCH=mae_large; ARCH_CKPT=mae_visualize_vit_large.pth ;;
  huge)  ARCH=mae_huge;  ARCH_CKPT=mae_visualize_vit_huge.pth ;;
  *) echo "[run_one_t45] Unknown arch short: $SHORT (base|large|huge)"; exit 2 ;;
esac

REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$REPRO_DIR/.." && pwd)"
cd "$ROOT"

case "$DS" in
  ETTh1)       ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTh1.csv; DATA=ETTh1; PERIOD=24; CTX=2880 ;;
  ETTh2)       ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTh2.csv; DATA=ETTh2; PERIOD=24; CTX=1728 ;;
  ETTm1)       ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTm1.csv; DATA=ETTm1; PERIOD=96; CTX=2304 ;;
  ETTm2)       ROOT_PATH=./dataset/ETT-small/; DATA_PATH=ETTm2.csv; DATA=ETTm2; PERIOD=96; CTX=4032 ;;
  *) echo "[run_one_t45] Unknown dataset: $DS"; exit 2 ;;
esac

CKPT_DIR="../ckpt/"
CKPT_FILE="$ROOT/$CKPT_DIR/$ARCH_CKPT"

# save parent is a FUNCTION OF THE ARCH ONLY (correction #2: single-purpose
# dirs; base this round == replays only)
case "$SHORT" in
  base)  PARENT=save_t45_base_replay ;;
  large) PARENT=save_t45_large ;;
  huge)  PARENT=save_t45_huge ;;
esac

fingerprint() {
  {
    echo "dataset=$DS pred_len=$PRED seq_len=$CTX periodicity=$PERIOD arch=$ARCH"
    echo "norm_const=0.4 align_const=0.4 batch_size=32 train_epochs=0 seed=2021"
    echo "root_path=$ROOT_PATH data_path=$DATA_PATH data=$DATA features=M"
    echo "ckpt_sha256=$(sha256sum "$CKPT_FILE" | cut -d' ' -f1)"
    for f in run.py exp/exp_long_term_forecasting.py exp/exp_basic.py \
             models/VisionTS.py data_provider/data_factory.py data_provider/data_loader.py \
             ../visionts/model.py ../visionts/models_mae.py; do
      echo "md5:$f=$(md5sum "$f" | cut -d' ' -f1)"
    done
    for f in repro_t45_mae_scale/infer_one_t45.py \
             repro_t45_mae_scale/t45_common.py; do
      echo "md5:$f=$(md5sum "$f" | cut -d' ' -f1)"
    done
  }
}

if [ "$FPMODE" -eq 1 ]; then
  fingerprint
  exit 0
fi

if [ -n "${SAVE_DIR_REL:-}" ]; then :; else SAVE_DIR_REL="$PARENT/${DS}_${PRED}"; fi
case "$SAVE_DIR_REL" in
  "$PARENT"/"$DS"_"$PRED") : ;;  # dirname must be <single-purpose parent>/<DS>_<PRED>
  *) echo "[run_one_t45] ABORT: SAVE_DIR_REL '$SAVE_DIR_REL' does not match '$PARENT/${DS}_${PRED}'"; exit 2 ;;
esac

RUN_DIR="$ROOT/$SAVE_DIR_REL"
LOG="$REPRO_DIR/logs/t45_${SHORT}_${DS}_${PRED}.log"
mkdir -p "$REPRO_DIR/logs" "$RUN_DIR"

# ---- skip-if-done: SUCCESS.txt 指纹一致 -> 跳过; 不一致 -> ABORT 不覆盖 ----
if [ -f "$RUN_DIR/SUCCESS.txt" ]; then
  if diff -q <(fingerprint) <(grep -v -e '^wall_time_s=' -e '^Peak GPU' \
        -e '^nvidia_smi_device_peak_mib' -e '^log_metrics=' -e '^shape_record=' \
        -e '^finished=' "$RUN_DIR/SUCCESS.txt") >/dev/null 2>&1; then
    echo "[run_one_t45] $SAVE_DIR_REL: SUCCESS fingerprint match -> skip"
    exit 0
  fi
  echo "[run_one_t45] ABORT: $RUN_DIR/SUCCESS.txt fingerprint mismatch (no overwrite)"
  exit 4
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
GPU_POLL_FILE="$REPRO_DIR/logs/.gpu_poll_$$"
( while true; do nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "$GPU_POLL_FILE"; sleep 5; done ) &
GPU_POLL_PID=$!
trap 'kill $GPU_POLL_PID 2>/dev/null; rm -f "$GPU_POLL_FILE"' EXIT

echo "[run_one_t45] $SAVE_DIR_REL arch=$ARCH gpu=$GPU_ID start=$(date +%F\ %T)"
START_TS=$(date +%s)

$PYTHON -u "$REPRO_DIR/infer_one_t45.py" \
  --shape_log "$RUN_DIR/shape_record.jsonl" \
  -- \
  --task_name long_term_forecast \
  --is_training 1 \
  --model VisionTS \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --save_dir "$SAVE_DIR_REL" \
  --model_id "VisionTS_${DS}_${PRED}_${SHORT}" \
  --data "$DATA" \
  --features M \
  --train_epochs 0 \
  --vm_arch "$ARCH" \
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
echo "[run_one_t45] exit_code=$RC wall_time=${ELAPSED}s" | tee -a "$LOG"

METRIC_DIR="$RUN_DIR/results/_"
MSE_LOG=$(grep -oE 'mse:[0-9]+\.[0-9]+, mae:[0-9]+\.[0-9]+' "$LOG" | tail -1)
SHAPE_OK=0
N_UC=0
if [ -f "$RUN_DIR/shape_record.jsonl" ]; then
  N_UC=$(grep -c '"type": "update_config"' "$RUN_DIR/shape_record.jsonl" 2>/dev/null)
  N_SEG=$(grep -c '"type": "segmentation"' "$RUN_DIR/shape_record.jsonl" 2>/dev/null)
  [ -z "${N_UC:-}" ] && N_UC=0
  [ -z "${N_SEG:-}" ] && N_SEG=0
  [ "${N_UC:-0}" -ge 1 ] && [ "${N_SEG:-0}" -gt 0 ] && SHAPE_OK=1
fi
OUT_OK=0
if [ -f "$METRIC_DIR/metrics.npy" ] && [ -f "$METRIC_DIR/pred.npy" ] \
   && [ -f "$METRIC_DIR/true.npy" ] && [ -n "$MSE_LOG" ] && [ "$SHAPE_OK" -eq 1 ]; then
  OUT_OK=1
fi

if [ "$RC" -ne 0 ]; then
  echo "[run_one_t45] FAILED (exit=$RC), see $LOG"
  exit "$RC"
fi
if [ "$OUT_OK" -ne 1 ]; then
  echo "[run_one_t45] INCOMPLETE: outputs/shape_record missing or metrics not parseable, see $LOG"
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

echo "[run_one_t45] SUCCESS $SAVE_DIR_REL $MSE_LOG time=${ELAPSED}s"
