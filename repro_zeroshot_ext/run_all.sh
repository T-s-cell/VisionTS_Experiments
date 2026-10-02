#!/usr/bin/env bash
# Stage driver for repro_zeroshot_ext. Usage:
#   bash run_all.sh <freeze|infer|select|test|aggregate|all>
# Isolation rules:
#   - PAUSE sentinel: if file `PAUSE` exists in this dir, all stages wait.
#   - GPU gate: `infer` waits until free VRAM on $GPU >= MIN_FREE_MIB (default 9000).
#   - OOM => the python stage exits non-zero and the driver aborts (no retry).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p logs

PY="${PY:-}"
if [ -z "$PY" ]; then
  for c in /dev_data/wlt/conda/envs/visionts/bin/python /opt/miniconda3/envs/visionts/bin/python python3; do
    if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
echo "python: $PY"
GPU="${GPU:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-9000}"
STAGE="${1:-all}"
LOG="logs/${STAGE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

wait_pause() {
  while [ -f "$HERE/PAUSE" ]; do
    echo "[run_all] PAUSE sentinel present, waiting 60s ($(date -Is))"
    sleep 60
  done
}

wait_gpu() {
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1)
    if [ -n "$FREE" ] && [ "$FREE" -ge "$MIN_FREE_MIB" ]; then
      echo "[run_all] GPU $GPU free ${FREE}MiB >= ${MIN_FREE_MIB}MiB, proceeding"
      return 0
    fi
    echo "[run_all] GPU $GPU free ${FREE:-unknown}MiB < ${MIN_FREE_MIB}MiB, waiting 60s ($(date -Is))"
    sleep 60
  done
}

run_stage() {
  case "$1" in
    freeze)    if [ -f splits.json ] && [ "${FORCE_FREEZE:-0}" != "1" ]; then
                 echo "[run_all] splits.json exists (frozen) — skipping freeze stage (set FORCE_FREEZE=1 to override)"
               else
                 "$PY" build_splits.py
               fi ;;
    infer)     wait_pause; wait_gpu
               { "$PY" -c "import torch,sys;print('torch',torch.__version__,file=sys.stderr)"; } 2>/dev/null
               nvidia-smi -i "$GPU" > "logs/env_snapshot_$(date +%Y%m%d_%H%M%S).txt" 2>&1 || true
               CUDA_VISIBLE_DEVICES="$GPU" "$PY" infer.py --device cuda "${@:2}" ;;
    select)    "$PY" select_period.py ;;
    test)      "$PY" evaluate_test.py ;;
    aggregate) "$PY" aggregate.py ;;
    *) echo "unknown stage $1"; exit 2 ;;
  esac
}

wait_pause
if [ "$STAGE" = "all" ]; then
  for s in freeze infer select test aggregate; do
    echo "[run_all] === stage $s ==="
    run_stage "$s" "${@:2}" || { echo "[run_all] stage $s FAILED" | tee logs/FAILED.txt; exit 1; }
  done
else
  run_stage "$STAGE" "${@:2}" || { echo "[run_all] stage $STAGE FAILED" | tee logs/FAILED.txt; exit 1; }
fi
echo "[run_all] stage(s) completed OK"
