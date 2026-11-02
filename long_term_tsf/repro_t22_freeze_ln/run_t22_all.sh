#!/bin/bash
# EXP-033 queue: Table-22 Freeze inference (GPU) + LN column reuse.
# 用法: bash run_t22_all.sh [min|main]   (默认 main)
#   min : preflight + ETTh1_96 冒烟(1 组推理 + 全门禁 + 首窗历史/变量序检查)
#   main: 要求冒烟通过后跑其余 11 组 + ETTm1 逐组缓存核验(失败只补跑该组,
#         权重/协议不一致则停) + collect 汇总
# 已跑组且本实验 SUCCESS.txt 指纹一致者跳过。失败: 写 logs/FAILED.txt 并终止。
# 注意: 预检不要求 EXP-001 全部 16 组完好(用户修正 3)——ETTm1 缓存逐组核对。
set -u

MODE=${1:-main}
PYTHON=${PYTHON:-python}
GPU_ID=${GPU_ID:-0}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPRO_DIR="$ROOT/repro_t22_freeze_ln"
LOGDIR="$REPRO_DIR/logs"
mkdir -p "$LOGDIR"
cd "$ROOT"

NEW_DS="ETTh1 ETTh2 ETTm2"
HORIZONS="96 192 336 720"

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
  echo "[run_t22_all] FAILED: $1" | tee "$LOGDIR/FAILED.txt"
  exit 1
}

fp_ok() {  # fp_ok <DS> <H>: 本实验 SUCCESS.txt 在盘且 md5/ckpt 行与当前树一致
  local S="$ROOT/save_t22_freeze/$1_$2/SUCCESS.txt"
  [ -f "$S" ] || return 1
  local line f want cur
  while IFS= read -r line; do
    case "$line" in
      md5:*)
        f=${line#md5:}; f=${f%%=*}; want=${line##*=}
        cur=$(md5sum "$ROOT/$f" 2>/dev/null | cut -d' ' -f1)
        [ "$cur" = "$want" ] || return 1 ;;
      ckpt_sha256=*)
        want=${line#ckpt_sha256=}
        cur=$(sha256sum "$ROOT/../ckpt/mae_visualize_vit_base.pth" | cut -d' ' -f1)
        [ "$cur" = "$want" ] || return 1 ;;
    esac
  done < "$S"
  return 0
}

run_group() {  # run_group <DS> <H>: 推理(指纹一致自动跳过) + 验证
  local ds=$1 h=$2
  if fp_ok "$ds" "$h"; then
    echo "[run_t22_all] SKIP ${ds}_${h} (done, fingerprint match)"
  else
    echo "[run_t22_all] RUN ${ds}_${h}"
    # 日志唯一写入方是 run_t22_group.sh 自身(评审收尾 3: 外层不再重定向,
    # 避免内层 > "$LOG" 截断覆盖外层已写的参数头部)
    GPU_ID=$GPU_ID bash "$REPRO_DIR/run_t22_group.sh" "$ds" "$h" save_t22_freeze \
      || fail_exit "${ds}_${h} run (see $LOGDIR/${ds}_${h}.log)"
    tail -2 "$LOGDIR/${ds}_${h}.log"
  fi
  "$PYTHON" "$REPRO_DIR/verify_t22.py" --group "$ds" "$h" || fail_exit "${ds}_${h} verify"
}

cache_group() {  # cache_group <H>: EXP-001 ETTm1 缓存逐组核对 + 按需补跑
  local h=$1 st
  "$PYTHON" "$REPRO_DIR/verify_t22.py" --cache ETTm1 "$h" || fail_exit "ETTm1_${h} cache check"
  st=$("$PYTHON" - "$LOGDIR" "$h" <<'EOF'
import json, sys, os
p = os.path.join(sys.argv[1], "verify_t22.json")
print(json.load(open(p))["cache_ETTm1_" + sys.argv[2]]["status"])
EOF
)
  if [ "$st" != "cache_ok" ]; then
    echo "[run_t22_all] ETTm1_${h} cache $st -> rerun under save_t22_freeze"
    run_group ETTm1 "$h"
  fi
}

echo "=== EXP-033 run_t22_all mode=$MODE (GPU) ==="
env_snapshot "$LOGDIR/env_snapshot_$MODE.txt"

# ---- preflight (不要求 EXP-001 全 16 组; ETTm1 逐组在 cache_group 里核对) ----
CKPT="$ROOT/../ckpt/mae_visualize_vit_base.pth"
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
for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
  for h in $HORIZONS; do
    T="$ROOT/save_fullshot/${ds}_${h}_seed2021/results/_/true.npy"
    [ -f "$T" ] || fail_exit "EXP-003 truth missing: $T"
  done
done
"$PYTHON" - "$REPRO_DIR/aggregate_snapshot.csv" <<'EOF' || fail_exit "aggregate snapshot sha mismatch"
import hashlib, sys
want = "a1622c33b35a54c75e6d4ac5fe1b6649b3797e3234270807c91b87e280f75dcf"
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
assert h.hexdigest() == want, h.hexdigest()
EOF
echo "[run_t22_all] preflight OK (ckpt / EXP-003 truths / snapshot sha)"

SMOKE_MARK="$ROOT/save_t22_freeze/ETTh1_96/SMOKE_OK.txt"

if [ "$MODE" = "min" ]; then
  run_group ETTh1 96
  "$PYTHON" "$REPRO_DIR/verify_t22.py" --first_window ETTh1 96 || fail_exit "first-window history/order check"
  { echo "smoke_ok $(date +%F\ %T)"; echo "verify_json_md5=$(md5sum "$LOGDIR/verify_t22.json" | cut -d' ' -f1)"; } > "$SMOKE_MARK"
  echo "[run_t22_all] MIN DONE: ETTh1_96 smoke OK. Next: bash $0 main"
  exit 0
fi

[ -f "$SMOKE_MARK" ] || fail_exit "run 'bash $0 min' first (need save_t22_freeze/ETTh1_96/SMOKE_OK.txt)"

for ds in $NEW_DS; do
  for h in $HORIZONS; do
    run_group "$ds" "$h"
  done
done

for h in $HORIZONS; do
  cache_group "$h"
done

echo "[run_t22_all] ALL GROUPS DONE. Collecting..."
"$PYTHON" "$REPRO_DIR/collect_t22.py" || fail_exit "collect_t22"
echo "[run_t22_all] ALL DONE (main): 12 new groups (incl. smoke) + ETTm1 cache path + summary."
