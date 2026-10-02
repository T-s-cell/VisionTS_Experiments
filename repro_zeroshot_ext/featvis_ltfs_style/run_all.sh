#!/usr/bin/env bash
# featvis_ltfs_style stage driver. Usage: bash run_all.sh <00|01|02|03|all>
#   00,03: CPU; 01-02: GPU (PAUSE sentinel + free-VRAM gate, L20 only).
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
    echo "[ltfs_style] PAUSE sentinel present, waiting 60s ($(date -Is))"
    sleep 60
  done
}
wait_gpu() {
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1)
    if [ -n "$FREE" ] && [ "$FREE" -ge "$MIN_FREE_MIB" ]; then
      echo "[ltfs_style] GPU $GPU free ${FREE}MiB >= ${MIN_FREE_MIB}MiB, proceeding"
      return 0
    fi
    echo "[ltfs_style] GPU $GPU free ${FREE:-unknown}MiB < ${MIN_FREE_MIB}MiB, waiting 60s ($(date -Is))"
    sleep 60
  done
}

run_stage() {
  case "$1" in
    00) "$PY" 00_make_lists.py ;;
    01) wait_pause; wait_gpu
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" 01_sanity_check.py --device cuda ;;
    02) wait_pause; wait_gpu
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" 02_extract_features.py --device cuda ;;
    03) "$PY" 03_tsne_plot.py ;;
    *) echo "unknown stage $1"; exit 2 ;;
  esac
}

if [ "$STAGE" = "all" ]; then
  for s in 00 01 02 03; do
    echo "[ltfs_style] === stage $s ==="
    run_stage "$s" || { echo "[ltfs_style] stage $s FAILED" | tee logs/FAILED.txt; exit 1; }
  done
else
  run_stage "$STAGE" || { echo "[ltfs_style] stage $STAGE FAILED" | tee logs/FAILED.txt; exit 1; }
fi
echo "[ltfs_style] stage(s) completed OK"
