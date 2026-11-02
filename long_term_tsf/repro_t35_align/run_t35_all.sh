#!/bin/bash
# EXP-035 queue: B.9 Figure 9(c) align-constant c sweep on ETTm1 (zero-shot;
# the c=0.4 column = EXP-001 reuse, no rerun).
# 用法: bash run_t35_all.sh [min|main]   (默认 main)
#   min : preflight + 4 组 c=0.4 基线门禁 + 四 H 等权自检 ≤1e-6(硬门禁)
#         + ETTm1_96 c=0.6 run+verify(探针)+ first_window + smoke_hash
#         + SMOKE_OK 标记
#   main: 要求冒烟通过 -> 4 组基线门禁复核先行(任一失败即停,无补跑路径)
#         -> 32 组新 c run+verify(指纹一致跳过,SKIP 另核对 align_const/L/P
#         与三份数组) -> collect(必查 first_window/smoke_hash 证据)
# 失败: 写 logs/FAILED.txt 并终止。组日志唯一写入方是 run_t35_group.sh 自身。
set -u

MODE=${1:-main}
PYTHON=${PYTHON:-python}
GPU_ID=${GPU_ID:-0}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$REPRO_DIR/logs"
mkdir -p "$LOGDIR"
rm -f "$LOGDIR/FAILED.txt"  # stale marker from a previous aborted run must not masquerade as this run's failure
cd "$ROOT"

HORIZONS="96 192 336 720"
VARIANT_C="0.2 0.3 0.5 0.6 0.7 0.8 0.9 1.0"
CKPT="$ROOT/../ckpt/mae_visualize_vit_base.pth"

env_snapshot() {
  "$PYTHON" - > "$1" <<'EOF'
import platform, sys
import numpy, pandas, sklearn, torch
print("Python", platform.python_version())
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("executable", sys.executable)
EOF
  echo "--- pip freeze ---" >> "$1"
  "$PYTHON" -m pip freeze 2>/dev/null | sort >> "$1"
}

fail_exit() {
  echo "[run_t35_all] FAILED: $1" | tee "$LOGDIR/FAILED.txt"
  exit 1
}

fp_ok() {  # fp_ok <H> <C>: 增强SKIP — SUCCESS 在盘、config 含 align_const/L/P/
           # pred_len、md5+ckpt 行与当前树一致、三份数组齐全
  local h=$1 c=$2
  local S="$ROOT/save_t35_align/c$c/ETTm1_$h/SUCCESS.txt"
  [ -f "$S" ] || return 1
  grep -Eq "align_const=$c( |\$)" "$S" || return 1
  grep -q "seq_len=2304 " "$S" || return 1
  grep -q "periodicity=96 " "$S" || return 1
  grep -q "pred_len=$h " "$S" || return 1
  grep -q "norm_const=0.4 " "$S" || return 1
  grep -q "train_epochs=0 " "$S" || return 1
  grep -q "seed=2021" "$S" || return 1
  local line f want cur
  while IFS= read -r line; do
    case "$line" in
      md5:*)
        f=${line#md5:}; f=${f%%=*}; want=${line##*=}
        cur=$(md5sum "$ROOT/$f" 2>/dev/null | cut -d' ' -f1)
        [ "$cur" = "$want" ] || return 1 ;;
      ckpt_sha256=*)
        want=${line#ckpt_sha256=}
        cur=$(sha256sum "$CKPT" | cut -d' ' -f1)
        [ "$cur" = "$want" ] || return 1 ;;
    esac
  done < "$S"
  local RD="$ROOT/save_t35_align/c$c/ETTm1_$h/results/_"
  [ -f "$RD/pred.npy" ] && [ -f "$RD/true.npy" ] && [ -f "$RD/metrics.npy" ]
}

run_group() {  # run_group <H> <C>
  local h=$1 c=$2
  if fp_ok "$h" "$c"; then
    echo "[run_t35_all] SKIP ETTm1_${h} c=$c (done, fingerprint+config+arrays match)"
  else
    echo "[run_t35_all] RUN ETTm1_${h} c=$c"
    GPU_ID=$GPU_ID bash "$REPRO_DIR/run_t35_group.sh" "$h" "$c" \
      || fail_exit "ETTm1_${h} c=$c run (see $LOGDIR/ETTm1_${h}_c$c.log)"
    tail -2 "$LOGDIR/ETTm1_${h}_c$c.log"
  fi
  "$PYTHON" "$REPRO_DIR/verify_t35.py" --group "$h" "$c" \
    || fail_exit "ETTm1_${h} c=$c verify"
}

gate_baseline() {  # 只读门禁(含探针), 无补跑路径
  "$PYTHON" "$REPRO_DIR/verify_t35.py" --baseline "$1" \
    || fail_exit "baseline gate ETTm1_$1 (EXP-001 drift/missing -> stop, see hint)"
}

preflight() {
  [ -f "$CKPT" ] || fail_exit "ckpt missing: $CKPT"
  "$PYTHON" - "$CKPT" <<'EOF' || fail_exit "ckpt sha256/size != pinned EXP-001 weights"
import sys, os
p = sys.argv[1]
import hashlib
h = hashlib.sha256()
with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
want = "c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e"
assert os.path.getsize(p) == 447725789 and h.hexdigest() == want, \
    (os.path.getsize(p), h.hexdigest())
EOF
  for h in $HORIZONS; do
    D="$ROOT/save_repro_zeroshot/ETTm1_$h"
    [ -f "$D/SUCCESS.txt" ] || fail_exit "EXP-001 SUCCESS missing: $D"
    for f in pred true metrics; do
      [ -f "$D/results/_/$f.npy" ] || fail_exit "EXP-001 $f.npy missing: $D"
    done
    T="$ROOT/save_fullshot/ETTm1_${h}_seed2021/results/_/true.npy"
    [ -f "$T" ] || fail_exit "EXP-003 truth missing: $T"
  done
  "$PYTHON" - "$REPRO_DIR/exp031_visionts_column_snapshot.csv" <<'EOF' || fail_exit "exp031 snapshot sha mismatch"
import hashlib, sys
want = "b454fb9a3fc4da15930c9934e117827a2bcdc8f3c8f7f17e2622445aaaedafbf"
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
assert h.hexdigest() == want, h.hexdigest()
EOF
  echo "[run_t35_all] preflight OK (ckpt / 4 EXP-001 ETTm1 groups / EXP-003 truths / exp031 snapshot)"
}

echo "=== EXP-035 run_t35_all mode=$MODE (GPU) ==="
env_snapshot "$LOGDIR/env_snapshot_$MODE.txt"
preflight

SMOKE_MARK="$ROOT/save_t35_align/SMOKE_OK.txt"

if [ "$MODE" = "min" ]; then
  for h in $HORIZONS; do
    gate_baseline "$h"
  done
  "$PYTHON" "$REPRO_DIR/collect_t35.py" --selfcheck \
    || fail_exit "baseline four-H self-check <= 1e-6"
  run_group 96 0.6
  "$PYTHON" "$REPRO_DIR/verify_t35.py" --first_window 96 0.6 \
    || fail_exit "first-window history/truth check ETTm1_96"
  "$PYTHON" "$REPRO_DIR/verify_t35.py" --smoke_hash 96 0.6 \
    || fail_exit "smoke-pair hard assertions + SHA-256 evidence ETTm1_96"
  { echo "smoke_ok $(date +%F\ %T)"; echo "verify_json_md5=$(md5sum "$LOGDIR/verify_t35.json" | cut -d' ' -f1)"; } > "$SMOKE_MARK"
  echo "[run_t35_all] MIN DONE: 4 baseline gates + self-check + ETTm1_96 c=0.6 + first-window + smoke-hash OK. Next: bash $0 main"
  exit 0
fi

[ -f "$SMOKE_MARK" ] || fail_exit "run 'bash $0 min' first (need save_t35_align/SMOKE_OK.txt)"

# ---- Stage 1: 4 baseline gates FIRST (c=0.4 = denominator of every Delta) ----
for h in $HORIZONS; do
  gate_baseline "$h"
done

# ---- Stage 2: 32 new-c groups (smoke group c=0.6 skips the run, re-verifies) ----
for h in $HORIZONS; do
  for c in $VARIANT_C; do
    run_group "$h" "$c"
  done
done

echo "[run_t35_all] ALL GROUPS DONE. Collecting..."
"$PYTHON" "$REPRO_DIR/collect_t35.py" || fail_exit "collect_t35"
echo "[run_t35_all] ALL DONE (main): 32 new-c groups + 4 baseline gates + tables/RESULT."
