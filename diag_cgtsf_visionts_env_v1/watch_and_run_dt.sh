#!/usr/bin/env bash
# GPU-wait + preflight + automatic full run (no separate go-ahead).
# Usage: nohup bash watch_and_run_dt.sh <approval_token> > logs/watcher_nohup.log 2>&1 &
# Requires logs/ to exist BEFORE starting (nohup redirection happens first).
set -uo pipefail
SUBDIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="/usr/lib/wsl/lib:$PATH"
TOKEN="${1:?usage: watch_and_run_dt.sh <approval_token>}"
LOG="$SUBDIR/logs/watcher.log"

mkdir -p "$SUBDIR/logs"
gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | head -1
}

echo "[watcher] start $(date '+%F %T') token=$TOKEN" | tee -a "$LOG"
# Wait until the GPU has >= 13000 MiB free (MAE-base needs well under that;
# the threshold only avoids starting next to another big job).
while true; do
  FREE="$(gpu_free_mib)"
  echo "[watcher] $(date '+%F %T') GPU free ${FREE} MiB" | tee -a "$LOG"
  if [ -n "$FREE" ] && [ "$FREE" -ge 13000 ] 2>/dev/null; then
    break
  fi
  sleep 120
done

echo "[watcher] GPU idle enough -> preflight $(date '+%F %T')" | tee -a "$LOG"
if bash "$SUBDIR/wrapper_dt.sh" preflight.py \
    --i-approve-frozen-protocol "$TOKEN" >> "$LOG" 2>&1; then
  echo "[watcher] preflight PASS -> full run $(date '+%F %T')" | tee -a "$LOG"
  bash "$SUBDIR/wrapper_dt.sh" run_infer.py \
    --i-approve-frozen-protocol "$TOKEN" >> "$LOG" 2>&1
  RC=$?
  echo "[watcher] run_infer exit=$RC $(date '+%F %T')" | tee -a "$LOG"
  exit $RC
else
  echo "[watcher] preflight FAILED -> no full run $(date '+%F %T')" \
    | tee -a "$LOG"
  exit 1
fi
