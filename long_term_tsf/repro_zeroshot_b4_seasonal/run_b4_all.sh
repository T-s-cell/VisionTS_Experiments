#!/bin/bash
# EXP-031 seasonal-baseline queue (CPU only, no GPU/CUDA).
# 用法: bash run_b4_all.sh [min|main]   (默认 main)
#   min : preflight + ETTh1_96 冒烟 + 手算核对(HANDCHECK_OK)
#   main: preflight + 要求 ETTh1_96/HANDCHECK_OK.txt 后跑完全部 16 组,
#         ETTm1_336 完成后追加手算抽查(H=336 非 P=96 整数倍路径)
# 已跑组且本实验 md5 指纹一致者跳过。失败: 写 logs/FAILED.txt 并终止。
set -u

MODE=${1:-main}
PYTHON=${PYTHON:-python}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # long_term_tsf/
REPRO_DIR="$ROOT/repro_zeroshot_b4_seasonal"
BASE_DIR="$ROOT/save_repro_zeroshot"
SAVE_BASE="$ROOT/save_b4_seasonal"
LOGDIR="$REPRO_DIR/logs"
mkdir -p "$LOGDIR"
cd "$ROOT"

HORIZONS="96 192 336 720"
DATASETS="ETTh1 ETTh2 ETTm1 ETTm2"

env_snapshot() {
  "$PYTHON" - > "$1" <<'EOF'
import platform, sys
import numpy, pandas, sklearn
print("Python", platform.python_version())
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("executable", sys.executable)
EOF
  echo "--- pip freeze ---" >> "$1"
  "$PYTHON" -m pip freeze 2>/dev/null | sort >> "$1"
}

fp_ok() {  # fp_ok <DS> <H> <method>: SUCCESS.txt exists and own-md5 lines match tree
  local S="$SAVE_BASE/$1_$2/$3/SUCCESS.txt"
  [ -f "$S" ] || return 1
  local line f want cur
  while IFS= read -r line; do
    f=${line#md5:}; f=${f%%=*}; want=${line##*=}
    cur=$(md5sum "$ROOT/$f" 2>/dev/null | cut -d' ' -f1)
    [ "$cur" = "$want" ] || return 1
  done < <(grep '^md5:' "$S")
  return 0
}

run_group() {  # run_group <DS> <H>
  local ds=$1 h=$2
  if fp_ok "$ds" "$h" naive && fp_ok "$ds" "$h" avg; then
    echo "[run_b4_all] SKIP ${ds}_${h} (done, fingerprint match)"
    return 0
  fi
  echo "[run_b4_all] RUN ${ds}_${h}"
  "$PYTHON" "$REPRO_DIR/run_b4.py" --group "$ds" "$h" \
    > "$LOGDIR/${ds}_${h}.log" 2>&1
  local rc=$?
  tail -3 "$LOGDIR/${ds}_${h}.log"
  if [ $rc -ne 0 ] || [ ! -f "$SAVE_BASE/${ds}_${h}/naive/SUCCESS.txt" ] \
     || [ ! -f "$SAVE_BASE/${ds}_${h}/avg/SUCCESS.txt" ] \
     || [ ! -f "$SAVE_BASE/${ds}_${h}/probes_evidence.json" ]; then
    echo "[run_b4_all] FAILED ${ds}_${h} (rc=$rc)" | tee "$LOGDIR/FAILED.txt"
    exit 1
  fi
  echo "[run_b4_all] OK ${ds}_${h}"
}

echo "=== EXP-031 run_b4_all mode=$MODE (CPU) ==="
env_snapshot "$LOGDIR/env_snapshot_$MODE.txt"
"$PYTHON" "$REPRO_DIR/preflight_b4.py" || { echo "[run_b4_all] preflight FAILED"; exit 1; }

if [ "$MODE" = "min" ]; then
  run_group ETTh1 96
  "$PYTHON" "$REPRO_DIR/run_b4.py" --hand_check ETTh1 96 1000 2>&1 | tee "$LOGDIR/handcheck_ETTh1_96.log"
  [ -f "$SAVE_BASE/ETTh1_96/HANDCHECK_OK.txt" ] || { echo "[run_b4_all] hand check FAILED"; exit 1; }
  echo "[run_b4_all] MIN DONE: ETTh1_96 smoke + hand check OK. Next: bash $0 main"
  exit 0
fi

# main: smoke gate
if [ ! -f "$SAVE_BASE/ETTh1_96/HANDCHECK_OK.txt" ]; then
  echo "[run_b4_all] ABORT: run 'bash $0 min' first (need ETTh1_96/HANDCHECK_OK.txt)"
  exit 1
fi

for ds in $DATASETS; do
  for h in $HORIZONS; do
    run_group "$ds" "$h"
    if [ "$ds" = "ETTm1" ] && [ "$h" = "336" ]; then
      "$PYTHON" "$REPRO_DIR/run_b4.py" --hand_check ETTm1 336 2000 2>&1 \
        | tee "$LOGDIR/handcheck_ETTm1_336.log"
      [ -f "$SAVE_BASE/ETTm1_336/HANDCHECK_OK.txt" ] \
        || { echo "[run_b4_all] ETTm1_336 hand check FAILED"; exit 1; }
    fi
  done
done

echo "[run_b4_all] ALL DONE (main): 16 groups. Next:"
echo "  $PYTHON $REPRO_DIR/collect_summary_b4.py"
