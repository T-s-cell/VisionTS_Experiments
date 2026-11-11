#!/bin/bash
# EXP-043 driver: S0 env -> S1 preflight -> S2 smoke (6 val units with
# crosspath + test-path smoke + verify --smoke/--reentry) -> S3 val queue
# (12 processes, 48 configs) -> S3.5 verify --valfull (accept-before-select,
# HARD) -> S4 select (ledger) -> S5 new test runs (selected arm, ledger-bound)
# -> S6 verify --all -> S7 collect -> S8 pack.
#
# Frozen: smoke passing auto-continues (no extra confirmation round); the
# ledger is generated ONLY after the 48-config acceptance; no paper-value
# gate exists anywhere.
# 用法: PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python bash run_t43_all.sh [stage...]
# 环境变量: GPU_ID (默认 0), PYTHON_B (必填), SKIP_TO (调试用)
set -eu

REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$REPRO_DIR/.." && pwd)"
cd "$ROOT"
GPU_ID=${GPU_ID:-0}
: "${PYTHON_B:?Set PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python}"
export PYTHON="$PYTHON_B"
STAGES=${@:-S0 S1 S2 S3 S35 S4 S5 S6 S7 S8}

have() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac }

log() { echo "[run_t43_all $(date +%F\ %T)] $*"; }

if have S0; then
  log "S0 env snapshot"
  mkdir -p "$REPRO_DIR/logs"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv \
    | tee "$REPRO_DIR/logs/gpu_snapshot_t43.txt"
fi

if have S1; then
  log "S1 preflight (hard)"
  "$PYTHON_B" "$REPRO_DIR/preflight_t43.py"
fi

if have S2; then
  log "S2 smoke: 6 val units (H96) with crosspath"
  export CUDA_VISIBLE_DEVICES=$GPU_ID
  for spec in "ETTh1 24" "ETTh1 168" "ETTh1 1" "ETTm1 96" "ETTm1 672" "ETTm1 1"; do
    set -- $spec
    log "  val smoke $1 P=$2 H=96 (crosspath)"
    "$PYTHON_B" "$REPRO_DIR/val_pass.py" --ds "$1" --P "$2" --H 96 --crosspath
  done
  log "S2 test-path smoke run: ETTh1/H96/P24 -> save_t43_period_val_smoke"
  LEDGER_REQUIRED=0 bash "$REPRO_DIR/run_one_t43.sh" ETTh1 96 \
    save_t43_period_val_smoke 24
  log "S2 verify --smoke (hard gate VG13)"
  "$PYTHON_B" "$REPRO_DIR/verify_t43.py" --smoke
  log "S2 verify --reentry (negative skip-if-done, VG12)"
  "$PYTHON_B" "$REPRO_DIR/verify_t43.py" --reentry
  log "S2 smoke PASSED -> auto-continue (no confirmation round)"
fi

if have S3; then
  log "S3 validation queue: 12 (ds,P) processes x 4 horizons (48 configs)"
  export CUDA_VISIBLE_DEVICES=$GPU_ID
  for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
    case $ds in
      ETTh1|ETTh2) PS="24 168 1";;
      *)           PS="96 672 1";;
    esac
    for P in $PS; do
      log "  val $ds P=$P"
      "$PYTHON_B" "$REPRO_DIR/val_pass.py" --ds "$ds" --P "$P"
    done
  done
fi

if have S35; then
  log "S3.5 verify --valfull: accept all 48 BEFORE selection (hard)"
  "$PYTHON_B" "$REPRO_DIR/verify_t43.py" --valfull
fi

if have S4; then
  log "S4 selection ledger (requires valfull acceptance)"
  "$PYTHON_B" "$REPRO_DIR/select_t43.py"
  cp "$REPRO_DIR/logs/LEDGER_OK" "$REPRO_DIR/logs/LEDGER_OK.snapshot"
fi

if have S5; then
  log "S5 new test runs for the selected arm (ledger-bound)"
  export CUDA_VISIBLE_DEVICES=$GPU_ID
  NEW_GROUPS=$("$PYTHON_B" - <<'EOF'
import sys
sys.path.insert(0, "repro_t43_period_val")
import t43_common as X
sel = X.load_json(X.LEDGER_JSON)
for ds in X.ETT4:
    P = sel["per_ds"][ds]["chosen_P"]
    if P in (X.PAPER_P[ds], 1):
        continue
    for h in X.HORIZONS:
        print(ds, h, P)
EOF
)
  NNEW=$(echo "$NEW_GROUPS" | grep -c . || true)
  log "  new groups: $NNEW (selected arm reuses archives elsewhere)"
  if [ "$NNEW" -gt 0 ]; then
    echo "$NEW_GROUPS" | while read -r ds h P; do
      log "  test $ds H=$h P=$P"
      bash "$REPRO_DIR/run_one_t43.sh" "$ds" "$h" save_t43_period_val "$P"
    done
  fi
fi

if have S6; then
  log "S6 verify --all (VG3-VG13)"
  "$PYTHON_B" "$REPRO_DIR/verify_t43.py" --all
fi

if have S7; then
  log "S7 collect"
  "$PYTHON_B" "$REPRO_DIR/collect_t43.py"
fi

if have S8; then
  log "S8 pack (<50MB: code + logs + SUCCESS/shape_record/metrics; no "
  log "   pred/true npy, no arr_val.npz — large arrays stay on eta)"
  PACK=/dev_data/wlt/temp_t43_pack.tar.gz
  mkdir -p /dev_data/wlt
  tar -czf "$PACK" \
    --exclude='__pycache__' \
    --exclude='repro_t43_period_val/logs/.gpu_poll_*' \
    --exclude='save_t43_period_val/*/results/_/pred.npy' \
    --exclude='save_t43_period_val/*/results/_/true.npy' \
    --exclude='save_t43_period_val_smoke/*/results/_/pred.npy' \
    --exclude='save_t43_period_val_smoke/*/results/_/true.npy' \
    --exclude='save_t43_period_val_smoke/_reentry_neg' \
    --exclude='save_t43_period_val_val' \
    repro_t43_period_val \
    save_t43_period_val \
    save_t43_period_val_smoke
  SIZE=$(stat -c%s "$PACK")
  log "  pack size: $SIZE bytes"
  if [ "$SIZE" -ge $((50 * 1024 * 1024)) ]; then
    echo "PACK TOO LARGE ($SIZE >= 50MiB)" >&2; exit 2
  fi
  sha256sum "$PACK" | tee "$PACK.sha256"
  log "  pack: $PACK (+ sidecar .sha256)"
fi

log "requested stages done: $STAGES"
