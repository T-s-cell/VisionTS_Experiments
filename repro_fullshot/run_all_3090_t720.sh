#!/usr/bin/env bash
# VisionTS LN full-shot: theta de-overlap continuation (user-approved 2026-09-28).
# Runs ONLY the remaining Traffic_720 jobs after the original workers pause on
# their PAUSE_w{W} sentinels (placed after their last Traffic_336 job).
# Independent sentinel namespace: PAUSE720_w{W}. Same VRAM yield gate.
# Usage: bash run_all_3090_t720.sh <worker_id 0|1>   (CWD-independent)
set -u

W=${1:?usage: run_all_3090_t720.sh <0|1>}
case "$W" in 0|1) ;; *) echo "worker id must be 0 or 1" >&2; exit 2;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/dev_data/wlt/conda/envs/visionts/bin/python
LIST="$REPO/repro_fullshot/split_theta/w${W}_t720.txt"
LOG_DIR="$REPO/repro_fullshot/logs_theta/w${W}"
QUEUE_LOG="$REPO/repro_fullshot/queue_status_theta_w${W}_t720.txt"

cd "$REPO/long_term_tsf" || { echo "ERROR: missing $REPO/long_term_tsf" >&2; exit 2; }
[[ -f "$LIST" ]] || { echo "ERROR: missing $LIST" >&2; exit 2; }
mkdir -p "$LOG_DIR"

declare -A DATA=( [Traffic]=custom )
declare -A ROOT=( [Traffic]=./dataset/traffic/ )
declare -A FILE=( [Traffic]=traffic.csv )
declare -A FREQ=( [Traffic]=h )
declare -A SEQ=( [Traffic]=1152 )
declare -A PERIOD=( [Traffic]=24 )
declare -A NORMC=( [Traffic]=0.4 )
declare -A ALIGNC=( [Traffic]=0.4 )
declare -A EPOCHS=( [Traffic]=1 )

say() { echo "$*"; echo "$*" >> "$QUEUE_LOG"; }

MIN_FREE_MB=22700
wait_turn() {
  while true; do
    if [ -f "$REPO/repro_fullshot/PAUSE720_w${W}" ]; then
      say "[QUEUE] $(date '+%F %T') PAUSE720 sentinel present, waiting 60s"
      sleep 60
      continue
    fi
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$W")
    if [ "${free_mb:-0}" -ge "$MIN_FREE_MB" ]; then
      return 0
    fi
    say "[QUEUE] $(date '+%F %T') GPU $W busy (free ${free_mb}MiB < ${MIN_FREE_MB}MiB), yield & retry in 60s"
    sleep 60
  done
}

say "===== theta t720 continuation worker $W start $(date '+%F %T') ====="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader >> "$QUEUE_LOG"
df -h . >> "$QUEUE_LOG"

total_done=0
while read -r job; do
  [[ -z "$job" || "$job" == \#* ]] && continue
  mid="${job%_seed*}"; seed="${job##*_seed}"
  ds="${mid%_*}"; pred="${mid##*_}"
  log="$LOG_DIR/$job.log"
  wait_turn
  say "[QUEUE] $(date '+%F %T') start $job (gpu $W)"
  "$PY" "$REPO/repro_fullshot/run_train.py" \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTS \
    --model_id "$job" \
    --data "${DATA[$ds]}" \
    --root_path "${ROOT[$ds]}" \
    --data_path "${FILE[$ds]}" \
    --features M \
    --target OT \
    --freq "${FREQ[$ds]}" \
    --seq_len "${SEQ[$ds]}" \
    --label_len 48 \
    --pred_len "$pred" \
    --train_epochs "${EPOCHS[$ds]}" \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --patience 3 \
    --lradj type1 \
    --num_workers 10 \
    --vm_pretrained 1 \
    --vm_ckpt ../ckpt/ \
    --vm_arch mae_base \
    --ft_type ln \
    --periodicity "${PERIOD[$ds]}" \
    --interpolation bilinear \
    --norm_const "${NORMC[$ds]}" \
    --align_const "${ALIGNC[$ds]}" \
    --seed "$seed" \
    --save_dir ./save_fullshot/"$job" \
    --checkpoints ./checkpoints/ \
    --gpu "$W" \
    > "$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    say "[QUEUE] $(date '+%F %T') DONE $job"
    total_done=$((total_done + 1))
  else
    say "[QUEUE] $(date '+%F %T') ABORT rc=$rc job=$job log=$log (scene kept, worker $W stopped)"
    exit "$rc"
  fi
done < "$LIST"

say "[QUEUE] $(date '+%F %T') ALL JOBS COMPLETED t720 worker $W (done/skip this invocation: $total_done)"
