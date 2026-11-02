#!/bin/bash
# EXP-034 queue: Table-18 interpolation ablation (bicubic/nearest zero-shot;
# bilinear column = EXP-001 reuse, no rerun).
# 用法: bash run_t18_all.sh [min|main]   (默认 main)
#   min : preflight + ETTh1_96 冒烟(双线性基线门禁+探针、双变体 run+verify、
#         首窗历史/变量序 x2)+ SMOKE_OK 标记
#   main: 要求冒烟通过 -> 16 组 --bilinear 全门禁先行(任一失败即停,无补跑路径)
#         -> 其余 30 组 bicubic/nearest run+verify(指纹一致跳过,SKIP 另核对
#         interpolation/L/P 与三份数组) -> collect 汇总
# 失败: 写 logs/FAILED.txt 并终止。组日志唯一写入方是 run_t18_group.sh 自身。
set -u

MODE=${1:-main}
PYTHON=${PYTHON:-python}
GPU_ID=${GPU_ID:-0}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$REPRO_DIR/logs"
mkdir -p "$LOGDIR"
cd "$ROOT"

HORIZONS="96 192 336 720"
DSS="ETTh1 ETTh2 ETTm1 ETTm2"
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
  echo "[run_t18_all] FAILED: $1" | tee "$LOGDIR/FAILED.txt"
  exit 1
}

t8_ctx() {  # Table-8 lookback
  case "$1" in
    ETTh1) echo "2880 24" ;;
    ETTh2) echo "1728 24" ;;
    ETTm1) echo "2304 96" ;;
    ETTm2) echo "4032 96" ;;
  esac
}

fp_ok() {  # fp_ok <DS> <H> <INTERP>: 增强SKIP — SUCCESS 在盘、config 含
           # interpolation/L/P/pred_len、md5+ckpt 行与当前树一致、三份数组齐全
  local ds=$1 h=$2 interp=$3
  local S="$ROOT/save_t18_interp/$interp/${ds}_${h}/SUCCESS.txt"
  [ -f "$S" ] || return 1
  local ctx period
  read -r ctx period <<< "$(t8_ctx "$ds")"
  grep -q "interpolation=$interp" "$S" || return 1
  grep -q "seq_len=$ctx " "$S" || return 1
  grep -q "periodicity=$period " "$S" || return 1
  grep -q "pred_len=$h " "$S" || return 1
  grep -q "norm_const=0.4 " "$S" || return 1
  grep -q "align_const=0.4 " "$S" || return 1
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
  local RD="$ROOT/save_t18_interp/$interp/${ds}_${h}/results/_"
  [ -f "$RD/pred.npy" ] && [ -f "$RD/true.npy" ] && [ -f "$RD/metrics.npy" ]
}

run_variant() {  # run_variant <DS> <H> <INTERP>
  local ds=$1 h=$2 interp=$3
  if fp_ok "$ds" "$h" "$interp"; then
    echo "[run_t18_all] SKIP ${ds}_${h} $interp (done, fingerprint+config+arrays match)"
  else
    echo "[run_t18_all] RUN ${ds}_${h} $interp"
    GPU_ID=$GPU_ID bash "$REPRO_DIR/run_t18_group.sh" "$ds" "$h" "$interp" \
      || fail_exit "${ds}_${h} $interp run (see $LOGDIR/${ds}_${h}_${interp}.log)"
    tail -2 "$LOGDIR/${ds}_${h}_${interp}.log"
  fi
  "$PYTHON" "$REPRO_DIR/verify_t18.py" --group "$ds" "$h" "$interp" \
    || fail_exit "${ds}_${h} $interp verify"
}

gate_bilinear() {  # 只读门禁(含探针), 无补跑路径
  "$PYTHON" "$REPRO_DIR/verify_t18.py" --bilinear "$1" "$2" \
    || fail_exit "bilinear gate $1_$2 (baseline drift/missing -> stop, see hint)"
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
  for ds in $DSS; do
    for h in $HORIZONS; do
      D="$ROOT/save_repro_zeroshot/${ds}_${h}"
      [ -f "$D/SUCCESS.txt" ] || fail_exit "EXP-001 SUCCESS missing: $D"
      for f in pred true metrics; do
        [ -f "$D/results/_/$f.npy" ] || fail_exit "EXP-001 $f.npy missing: $D"
      done
      T="$ROOT/save_fullshot/${ds}_${h}_seed2021/results/_/true.npy"
      [ -f "$T" ] || fail_exit "EXP-003 truth missing: $T"
    done
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
  echo "[run_t18_all] preflight OK (ckpt / 16 EXP-001 groups / EXP-003 truths / exp031 snapshot)"
}

echo "=== EXP-034 run_t18_all mode=$MODE (GPU) ==="
env_snapshot "$LOGDIR/env_snapshot_$MODE.txt"
preflight

SMOKE_MARK="$ROOT/save_t18_interp/SMOKE_OK.txt"

if [ "$MODE" = "min" ]; then
  gate_bilinear ETTh1 96
  for interp in bicubic nearest; do
    run_variant ETTh1 96 "$interp"
    "$PYTHON" "$REPRO_DIR/verify_t18.py" --first_window ETTh1 96 "$interp" \
      || fail_exit "first-window history/order check ETTh1_96 $interp"
  done
  { echo "smoke_ok $(date +%F\ %T)"; echo "verify_json_md5=$(md5sum "$LOGDIR/verify_t18.json" | cut -d' ' -f1)"; } > "$SMOKE_MARK"
  echo "[run_t18_all] MIN DONE: ETTh1_96 bilinear gate + both variants + first-window OK. Next: bash $0 main"
  exit 0
fi

[ -f "$SMOKE_MARK" ] || fail_exit "run 'bash $0 min' first (need save_t18_interp/SMOKE_OK.txt)"

# ---- Stage 1: 16 bilinear gates FIRST (baseline = denominator of every Delta) ----
for ds in $DSS; do
  for h in $HORIZONS; do
    gate_bilinear "$ds" "$h"
  done
done

# ---- Stage 2: 32 variant groups (smoke groups skip the run, re-verify) ----
for ds in $DSS; do
  for h in $HORIZONS; do
    for interp in bicubic nearest; do
      run_variant "$ds" "$h" "$interp"
    done
  done
done

echo "[run_t18_all] ALL GROUPS DONE. Collecting..."
"$PYTHON" "$REPRO_DIR/collect_t18.py" || fail_exit "collect_t18"
echo "[run_t18_all] ALL DONE (main): 32 variant groups + 16 bilinear gates + summary."
