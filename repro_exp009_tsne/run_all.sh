#!/usr/bin/env bash
# EXP-009 pipeline: extract | tsne | figures | audit | all
#   extract  — GPU features (Z0 once + F0 per domain; PAUSE + VRAM gate)
#   tsne     — 38 joint fits; EXP009_TSNE_JOBS picks parallelism (from the
#              small-trial measured per-process RSS/CPU; workers are
#              single-threaded)
#   figures  — 38 main figures + side-by-side + index.html
#   audit    — 7-check audit_report.json + REPORT.md
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/dev_data/wlt/conda/envs/visionts/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
STAGE="${1:-all}"
[ "$#" -gt 0 ] && shift
TSNE_JOBS="${EXP009_TSNE_JOBS:-1}"

E9OPTS="import importlib.util as iu, sys; s=iu.spec_from_file_location('exp009_common','$HERE/common.py'); m=iu.module_from_spec(s); sys.modules['exp009_common']=m; s.loader.exec_module(m)"

gpu_gate() {
  if [ -f "$HERE/PAUSE" ]; then
    echo "[run_all] PAUSE sentinel present — remove $HERE/PAUSE to continue"
    exit 3
  fi
  local FREE NEED
  FREE=$("$PY" -c "$E9OPTS; print(m.gpu_free_mib())")
  NEED=$("$PY" -c "$E9OPTS; print(int(m.CFG['resources']['gpu_gate_min_free_gib'])*1024)")
  echo "[run_all] GPU gate: free ${FREE} MiB / need ${NEED} MiB"
  if [ "$FREE" -lt "$NEED" ]; then
    echo "[run_all] insufficient free VRAM — aborting (never touch co-tenant)"
    exit 4
  fi
}

run_stage() {
  local name="$1"; shift
  mkdir -p "$HERE/logs"
  echo "[run_all] == stage: $name =="
  "$@" 2>&1 | tee -a "$HERE/logs/stage_${name}.log"
}

case "$STAGE" in
  extract)
    gpu_gate
    run_stage extract "$PY" "$HERE/01_extract_features.py" "$@" ;;
  tsne)
    run_stage tsne "$PY" "$HERE/02_run_tsne.py" --jobs "$TSNE_JOBS" "$@" ;;
  figures)
    run_stage figures "$PY" "$HERE/03_make_figures.py" "$@" ;;
  audit)
    run_stage audit "$PY" "$HERE/04_audit_report.py" "$@" ;;
  all)
    gpu_gate
    run_stage extract "$PY" "$HERE/01_extract_features.py" "$@"
    run_stage tsne "$PY" "$HERE/02_run_tsne.py" --jobs "$TSNE_JOBS"
    run_stage figures "$PY" "$HERE/03_make_figures.py"
    run_stage audit "$PY" "$HERE/04_audit_report.py" ;;
  *)
    echo "usage: run_all.sh [extract|tsne|figures|audit|all] [stage args]"
    exit 2 ;;
esac
echo "[run_all] stage '$STAGE' done"
