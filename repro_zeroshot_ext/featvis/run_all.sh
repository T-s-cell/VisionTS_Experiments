#!/usr/bin/env bash
# featvis stage driver. Usage: bash run_all.sh <00|01|02|03|04|05|06|all>
#   00-01,04-06: CPU; 02-03: GPU (PAUSE sentinel + free-VRAM gate, L20 only).
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
LOG="logs/stage_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

wait_pause() {
  while [ -f "$HERE/PAUSE" ]; do
    echo "[featvis] PAUSE sentinel present, waiting 60s ($(date -Is))"
    sleep 60
  done
}
wait_gpu() {
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1)
    if [ -n "$FREE" ] && [ "$FREE" -ge "$MIN_FREE_MIB" ]; then
      echo "[featvis] GPU $GPU free ${FREE}MiB >= ${MIN_FREE_MIB}MiB, proceeding"
      return 0
    fi
    echo "[featvis] GPU $GPU free ${FREE:-unknown}MiB < ${MIN_FREE_MIB}MiB, waiting 60s ($(date -Is))"
    sleep 60
  done
}

run_stage() {
  case "$1" in
    00) "$PY" 00_select_vars.py ;;
    01) "$PY" 01_make_sampling_lists.py ;;
    02) wait_pause; wait_gpu
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" 02_sanity_check.py --device cuda ;;
    03) wait_pause; wait_gpu
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" 03_extract_features.py --device cuda ;;
    04) "$PY" 04_highdim_analysis.py ;;
    05) "$PY" 05_run_tsne.py ;;
    06) "$PY" 06_plot_fig.py ;;
    *) echo "unknown stage $1"; exit 2 ;;
  esac
}

if [ "$STAGE" = "all" ]; then
  for s in 00 01 02 03 04 05 06; do
    echo "[featvis] === stage $s ==="
    run_stage "$s" || { echo "[featvis] stage $s FAILED" | tee logs/FAILED.txt; exit 1; }
  done
else
  run_stage "$STAGE" || { echo "[featvis] stage $STAGE FAILED" | tee logs/FAILED.txt; exit 1; }
fi
echo "[featvis] stage(s) completed OK"
