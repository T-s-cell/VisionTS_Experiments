#!/usr/bin/env bash
# Stage runner. Usage: bash run_all.sh <stage> [stage...]
#   freeze | verify | sample | preflight | fill | assemble | evaluate | recompute | accept
#   post  = assemble evaluate recompute accept ;  all = freeze verify sample
# Local stages: freeze verify sample assemble evaluate recompute accept
# GPU stages (server): preflight fill
# Re-run: FORCE_<STAGE_UPPER>=1 bash run_all.sh <stage>   e.g. FORCE_FILL=1 bash run_all.sh fill
# A .done marker only skips the stage when stage_guard.py re-verifies its recorded
# output sha256s; any drift => the stage re-runs automatically.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY="${PY:-python3}"

donefile() {
  case "$1" in
    freeze)    echo .state/s1_freeze.done ;;
    verify)    echo .state/s2_verify_cache.done ;;
    sample)    echo .state/s3_sample_frozen.done ;;
    preflight) echo .state/s3_preflight.done ;;
    fill)      echo .state/s4_infer_fill.done ;;
    assemble)  echo .state/s5a_assemble.done ;;
    evaluate)  echo .state/s5b_evaluate.done ;;
    recompute) echo .state/recompute.done ;;
    accept)    echo .state/accept.done ;;
    *) echo ".state/$1.done" ;;
  esac
}

force_flag() {
  # stage name -> the FORCE_<UPPER_WITH_UNDERSCORES> env var name
  echo "FORCE_$(echo "$1" | tr '[:lower:]-' '[:upper:]_')"
}

run_stage() {
  local s="$1" fv force marker overwrite=""
  fv=$(force_flag "$s")
  force="${!fv:-0}"
  marker=$(donefile "$s")
  if [ -f "$marker" ]; then
    if [ "$force" != "1" ] && $PY stage_guard.py "$s" >/dev/null 2>&1; then
      echo "[run_all] $s: up-to-date ($marker; set $fv=1 to redo)"
      return 0
    fi
    if [ "$force" = "1" ]; then
      echo "[run_all] $s: forced re-run ($fv=1)"
    else
      echo "[run_all] $s: stale state ($marker failed guard) — re-running"
    fi
    overwrite="--force"
    rm -f "$marker"
  fi
  case "$s" in
    freeze)    $PY freeze_full.py $overwrite ;;
    verify)    $PY verify_cache.py $overwrite ;;
    sample)    $PY preflight.py --sample-only ;;
    preflight) $PY preflight.py --run ;;
    fill)      $PY infer_fill.py ;;
    assemble)  $PY assemble.py $overwrite ;;
    evaluate)  $PY evaluate.py ;;
    recompute) $PY recompute_check.py ;;
    accept)    $PY verify_acceptance.py ;;
    *) echo "unknown stage: $s" >&2; exit 2 ;;
  esac
  echo "[run_all] $s: OK"
}

if [ "$#" -eq 0 ]; then set -- all; fi
for s in "$@"; do
  case "$s" in
    all)
      for st in freeze verify sample; do run_stage "$st"; done
      echo "[run_all] local pre-exec stages complete. Deploy to GPU host -> preflight+fill;" \
           "after rsync back -> bash run_all.sh post" ;;
    post) for st in assemble evaluate recompute accept; do run_stage "$st"; done ;;
    *) run_stage "$s" ;;
  esac
done
