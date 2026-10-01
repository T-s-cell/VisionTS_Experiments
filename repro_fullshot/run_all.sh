#!/usr/bin/env bash
# VisionTS LN full-shot: 96-job serial queue (8 datasets x 4 pred_lens x 3 seeds).
#
# CWD must be long_term_tsf (root_path values are './dataset/...').
# run_train.py owns all job bookkeeping (fingerprint / resume-skip / archive of
# failed dirs / SUCCESS marker / exit codes):
#   exit 0   -> job done, or clean resume-skip (SUCCESS + matching fingerprint)
#   exit 3   -> SUCCESS exists but fingerprint mismatch -> abort queue, keep scene
#   other    -> failure (incl. OOM) -> abort queue immediately, keep scene
# Per-job stdout/stderr: ../repro_fullshot/logs/<JOB>.log
# Queue-level log:       ../repro_fullshot/queue_status.txt
set -u

if [[ $(basename "$PWD") != "long_term_tsf" ]]; then
  echo "ERROR: run_all.sh must run with CWD=long_term_tsf (got $PWD)" >&2
  exit 2
fi

PY=/dev_data/wlt/conda/envs/visionts/bin/python
LOG_DIR=../repro_fullshot/logs
QUEUE_LOG=../repro_fullshot/queue_status.txt
mkdir -p "$LOG_DIR"

ORDER=(ETTh1 ETTh2 ETTm2 ETTm1 Weather Electricity Traffic illness)
declare -A JOBPREF=( [ETTh1]=ETTh1 [ETTh2]=ETTh2 [ETTm1]=ETTm1 [ETTm2]=ETTm2
                     [Weather]=Weather [Electricity]=Electricity [Traffic]=Traffic [illness]=illness )
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

preds_of() {
  case "$1" in
    illness) echo "24 36 48 60" ;;
    *)       echo "96 192 336 720" ;;
  esac
}

say() { echo "$*"; echo "$*" >> "$QUEUE_LOG"; }

say "===== queue start $(date '+%F %T') ====="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader >> "$QUEUE_LOG"
df -h . >> "$QUEUE_LOG"

total_done=0
for ds in "${ORDER[@]}"; do
  for pred in $(preds_of "$ds"); do
    for seed in 2021 2022 2023; do
      job="${JOBPREF[$ds]}_${pred}_seed${seed}"
      log="$LOG_DIR/$job.log"
      say "[QUEUE] $(date '+%F %T') start $job"
      "$PY" ../repro_fullshot/run_train.py \
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
        --gpu 0 \
        > "$log" 2>&1
      rc=$?
      if [ "$rc" -eq 0 ]; then
        say "[QUEUE] $(date '+%F %T') DONE $job"
        total_done=$((total_done + 1))
      else
        say "[QUEUE] $(date '+%F %T') ABORT rc=$rc job=$job log=$log (scene kept, queue stopped)"
        exit "$rc"
      fi
    done
  done
done

say "[QUEUE] $(date '+%F %T') ALL JOBS COMPLETED (done/skip this invocation: $total_done)"
