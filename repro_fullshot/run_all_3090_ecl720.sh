#!/usr/bin/env bash
# VisionTS LN full-shot: last-cell rescue on theta (user-approved 2026-10-01).
# eta's Electricity_720_seed2023 hung in vali/test after training finished
# (log frozen since 00:06), blocking its queue; this reruns the single job
# on theta's idle GPU1. Config per Table 19/C.1: custom, seq_len 1152,
# periodicity 24, r=0.4, c=0.4, 1 epoch, batch 256. electricity.csv
# sha256 7e45845d... identical on both servers (verified 2026-10-01).
# Fresh sentinel namespace PAUSEECL_w1 (stale PAUSE*/PAUSEILL* ignored).
# VRAM gate 22000MiB: ECL_720 peak ~21.4GB, GPU1 free 22671MiB at launch.
# Usage: bash run_all_3090_ecl720.sh <worker_id>   (CWD-independent)
set -u

W=${1:?usage: run_all_3090_ecl720.sh <0|1>}
case "$W" in 0|1) ;; *) echo "worker id must be 0 or 1" >&2; exit 2;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/dev_data/wlt/conda/envs/visionts/bin/python
LIST="$REPO/repro_fullshot/split_theta/w${W}_ecl720.txt"
LOG_DIR="$REPO/repro_fullshot/logs_theta/w${W}"
QUEUE_LOG="$REPO/repro_fullshot/queue_status_theta_w${W}_ecl720.txt"

cd "$REPO/long_term_tsf" || { echo "ERROR: missing $REPO/long_term_tsf" >&2; exit 2; }
[[ -f "$LIST" ]] || { echo "ERROR: missing $LIST" >&2; exit 2; }
mkdir -p "$LOG_DIR"

say() { echo "$*"; echo "$*" >> "$QUEUE_LOG"; }

MIN_FREE_MB=22000
wait_turn() {
  while true; do
    if [ -f "$REPO/repro_fullshot/PAUSEECL_w${W}" ]; then
      say "[QUEUE] $(date '+%F %T') PAUSEECL sentinel present, waiting 60s"
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

say "===== theta ECL_720 rescue worker $W start $(date '+%F %T') ====="
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
    --data custom \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --features M \
    --target OT \
    --freq h \
    --seq_len 1152 \
    --label_len 48 \
    --pred_len "$pred" \
    --train_epochs 1 \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --patience 3 \
    --lradj type1 \
    --num_workers 10 \
    --vm_pretrained 1 \
    --vm_ckpt ../ckpt/ \
    --vm_arch mae_base \
    --ft_type ln \
    --periodicity 24 \
    --interpolation bilinear \
    --norm_const 0.4 \
    --align_const 0.4 \
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

say "[QUEUE] $(date '+%F %T') ALL JOBS COMPLETED ecl720 rescue worker $W (done/skip this invocation: $total_done)"
