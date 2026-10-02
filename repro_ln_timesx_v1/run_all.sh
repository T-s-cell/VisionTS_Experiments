#!/usr/bin/env bash
# Stage driver: prepare|preflight|train|freeze_selection|test|aggregate|audit|all
# Fail-fast; every stage tees to logs/stage_<name>.log. GPU stages wait for
# free VRAM >= required (preflight peak + 3 GiB once known; 8 GiB for
# preflight itself) and honor the PAUSE sentinel file in this directory.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PY:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export GPU="${GPU:-0}"
STAGE="${1:-all}"
[ "$#" -gt 0 ] && shift
mkdir -p logs

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/run_all.log; }

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" \
    2>/dev/null | head -1 | tr -d ' '
}

# wait until PAUSE sentinel is gone and free VRAM >= $1 MiB
gpu_gate() {
  local need="$1"
  while true; do
    if [ -f PAUSE ]; then
      log "gpu_gate: PAUSE sentinel present; waiting"
      sleep 60
      continue
    fi
    local free
    free="$(gpu_free_mb)"
    if [ -n "$free" ] && [ "$free" -ge "$need" ]; then
      log "gpu_gate: free=${free}MiB >= need=${need}MiB, proceed"
      return 0
    fi
    log "gpu_gate: free=${free:-unknown}MiB < need=${need}MiB; waiting 60s"
    sleep 60
  done
}

required_for_test() {
  "$PY" - <<'EOF'
import json, os
p = "audit/preflight_report.json"
if os.path.isfile(p):
    pre = json.load(open(p))
    print(int(pre["peak_vram_mb"]) + 3072)
else:
    print(8192)
EOF
}

run_stage() {
  local name="$1"; shift
  log "=== stage $name start: $*"
  "$@" 2>&1 | tee "logs/stage_${name}.log"
  local rc=${PIPESTATUS[0]}
  log "=== stage $name rc=$rc"
  if [ "$rc" -ne 0 ]; then
    log "STOPPING: stage $name failed (rc=$rc); scene preserved"
    exit "$rc"
  fi
}

case "$STAGE" in
  prepare)
    run_stage prepare "$PY" prepare_data.py "$@" ;;
  preflight)
    gpu_gate 8192
    run_stage preflight "$PY" preflight.py ;;
  train)
    run_stage train "$PY" queue_train.py ;;
  freeze_selection)
    run_stage freeze_selection "$PY" freeze_selection.py ;;
  test)
    NEED="$(required_for_test)"; gpu_gate "$NEED"
    run_stage test "$PY" evaluate_test.py ;;
  aggregate)
    run_stage aggregate "$PY" aggregate.py ;;
  audit)
    run_stage audit "$PY" audit.py ;;
  all)
    gpu_gate 8192
    run_stage prepare "$PY" prepare_data.py
    run_stage preflight "$PY" preflight.py
    run_stage train "$PY" queue_train.py
    run_stage freeze_selection "$PY" freeze_selection.py
    NEED="$(required_for_test)"; gpu_gate "$NEED"
    run_stage test "$PY" evaluate_test.py
    run_stage aggregate "$PY" aggregate.py
    run_stage audit "$PY" audit.py
    log "ALL STAGES COMPLETE"
    ;;
  *)
    echo "usage: $0 {prepare|preflight|train|freeze_selection|test|aggregate|audit|all} [-- extra-args]" >&2
    exit 2 ;;
esac
