#!/usr/bin/env bash
# VisionTS LN full-shot: two-worker queue for theta (2x RTX 3090 24GB).
#
# Same 96 jobs as the eta L20 queue, split into two balanced lists
# (repro_fullshot/split_theta/w{0,1}.txt, LPT by iteration count ~638k each).
# One worker per GPU: worker id == gpu id.
#
# Usage: bash run_all_3090.sh <worker_id 0|1>   (CWD-independent)
# Exit codes mirror run_all.sh via run_train.py:
#   exit 0   per job -> done or clean resume-skip; 3 -> fingerprint mismatch abort;
#   other    -> failure (incl. OOM) -> abort this worker immediately, keep scene.
# Per-job log:    ../repro_fullshot/logs_theta/w<W>/<JOB>.log
# Queue-level log: ../repro_fullshot/queue_status_theta_w<W>.txt
set -u

W=${1:?usage: run_all_3090.sh <0|1>}
case "$W" in 0|1) ;; *) echo "worker id must be 0 or 1" >&2; exit 2;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/dev_data/wlt/conda/envs/visionts/bin/python
LIST="$REPO/repro_fullshot/split_theta/w${W}.txt"
LOG_DIR="$REPO/repro_fullshot/logs_theta/w${W}"
QUEUE_LOG="$REPO/repro_fullshot/queue_status_theta_w${W}.txt"

cd "$REPO/long_term_tsf" || { echo "ERROR: missing $REPO/long_term_tsf" >&2; exit 2; }
[[ -f "$LIST" ]] || { echo "ERROR: missing $LIST" >&2; exit 2; }
mkdir -p "$LOG_DIR"

declare -A DATA=( [ETTh1]=ETTh1 [ETTh2]=ETTh2 [ETTm1]=ETTm1 [ETTm2]=ETTm2
                  [Weather]=custom [Electricity]=custom [Traffic]=custom [illness]=custom )
declare -A ROOT=( [ETTh1]=./dataset/ETT-small/ [ETTh2]=./dataset/ETT-small/
                  [ETTm1]=./dataset/ETT-small/ [ETTm2]=./dataset/ETT-small/
                  [Weather]=./dataset/weather/ [Electricity]=./dataset/electricity/
                  [Traffic]=./dataset/traffic/ [illness]=./dataset/illness/ )
declare -A FILE=( [ETTh1]=ETTh1.csv [ETTh2]=ETTh2.csv [ETTm1]=ETTm1.csv [ETTm2]=ETTm2.csv
                  [Weather]=weather.csv [Electricity]=electricity.csv [Traffic]=traffic.csv
                  [illness]=national_illness.csv )
declare -A FREQ=( [ETTh1]=h [ETTh2]=h [ETTm1]=t [ETTm2]=t
                  [Weather]=h [Electricity]=h [Traffic]=h [illness]=h )
declare -A SEQ=( [ETTh1]=1152 [ETTh2]=1152 [ETTm1]=2304 [ETTm2]=1152
                 [Weather]=576 [Electricity]=1152 [Traffic]=1152 [illness]=104 )
declare -A PERIOD=( [ETTh1]=24 [ETTh2]=24 [ETTm1]=96 [ETTm2]=96
                    [Weather]=144 [Electricity]=24 [Traffic]=24 [illness]=52 )
declare -A NORMC=( [ETTh1]=0.4 [ETTh2]=0.4 [ETTm1]=0.4 [ETTm2]=0.4
                   [Weather]=1.0 [Electricity]=0.4 [Traffic]=0.4 [illness]=1.0 )
declare -A ALIGNC=( [ETTh1]=0.4 [ETTh2]=0.4 [ETTm1]=0.4 [ETTm2]=0.4
                    [Weather]=0.7 [Electricity]=0.4 [Traffic]=0.4 [illness]=0.4 )
declare -A EPOCHS=( [ETTh1]=1 [ETTh2]=1 [ETTm1]=1 [ETTm2]=1
                    [Weather]=1 [Electricity]=1 [Traffic]=1 [illness]=100 )

say() { echo "$*"; echo "$*" >> "$QUEUE_LOG"; }

# Yield policy (per user 2026-09-26): before EVERY job, block while
#  - a pause sentinel exists (manual yield to other tenants), or
#  - GPU free VRAM < MIN_FREE_MB (4GB safety margin + largest measured job peak
#    18.16GiB -> 22700MiB). Co-run only when the margin still fits alongside
#    whatever the other tenant is running.
MIN_FREE_MB=22700
wait_turn() {
  while true; do
    if [ -f "$REPO/repro_fullshot/PAUSE_w${W}" ]; then
      say "[QUEUE] $(date '+%F %T') PAUSE sentinel present, waiting 60s"
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

say "===== theta worker $W queue start $(date '+%F %T') ====="
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

say "[QUEUE] $(date '+%F %T') ALL JOBS COMPLETED worker $W (done/skip this invocation: $total_done)"
