#!/bin/bash
# EXP-036 queue: t-SNE of frozen vs encoder-LN-finetuned encoders over the
# 1,900 fixed EXP-002 samples (ImageNet 1000 + ETTm1/Weather/Electricity 300).
# 用法: bash run_t36_all.sh [min|main]   (默认 main)
#   min : preflight + 三份 EXP-003 权重核对(指纹/锚点/首批回放 <=1e-4/拼接证据)
#         + frozen 特征抽取(逐位 vs EXP-002 原件)+ ETTm1-LN 抽取(输入哈希跨条件一致)
#         + ETTm1-LN t-SNE -> SMOKE_OK
#   main: 要求冒烟通过 -> 其余两条件抽取(Electricity-LN / Weather-LN)
#         + 全量特征核验 -> 两个 t-SNE -> 30-NN 表 -> 2x2 出图 -> collect
# 失败: 写 logs/FAILED.txt 并终止(冻结特征或权重回放超差就地停)。
set -u

MODE=${1:-main}
PYTHON=${PYTHON:-python}
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # long_term_tsf/
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"        # repro dir
LOGDIR="$REPRO_DIR/logs"
mkdir -p "$LOGDIR"
rm -f "$LOGDIR/FAILED.txt"
cd "$ROOT"   # EXP-002/EXP-003 数据路径均为 CWD=long_term_tsf 的相对路径

SKLEARN_PIN=1.9.1

env_snapshot() {
  "$PYTHON" - > "$1" <<'EOF'
import platform, sys
import numpy, sklearn, torch
print("Python", platform.python_version())
print("numpy", numpy.__version__)
print("sklearn", sklearn.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("executable", sys.executable)
EOF
  echo "--- pip freeze ---" >> "$1"
  "$PYTHON" -m pip freeze 2>/dev/null | sort >> "$1"
}

fail_exit() {
  echo "[run_t36_all] FAILED: $1" | tee "$LOGDIR/FAILED.txt"
  exit 1
}

preflight() {
  "$PYTHON" - "$REPRO_DIR" <<'EOF' || fail_exit "preflight (see message above)"
import os, sys
sys.path.insert(0, sys.argv[1])
import t36_common as C

def need(cond, msg):
    if not cond:
        print("[preflight] MISSING:", msg)
        sys.exit(1)

h = C.sha256_of(C.CKPT_PATH)
need(h == C.CKPT_SHA256 and os.path.getsize(C.CKPT_PATH) == C.CKPT_BYTES,
     "frozen ckpt %s (sha %s)" % (C.CKPT_PATH, h))
for ds in C.EXP003_DS:
    j = C.exp003_job(ds)
    need(os.path.isfile(os.path.join(j, "args.json")), j + "/args.json")
    need(os.path.isfile(os.path.join(j, "fingerprint.json")), j + "/fingerprint.json")
    need(os.path.isfile(C.exp003_success(ds)), C.exp003_success(ds))
    need(os.path.isfile(os.path.join(j, "checkpoints", "_", "checkpoint.pth")),
         j + "/checkpoints/_/checkpoint.pth")
    need(os.path.isfile(C.exp003_pred(ds)), C.exp003_pred(ds))
need(os.path.isfile(C.ORIG_FEATURES), C.ORIG_FEATURES)
need(os.path.isfile(C.ORIG_COORDS), C.ORIG_COORDS)
need(os.path.isfile(C.ORIG_META), C.ORIG_META)
need(os.path.isfile(C.ORIG_GROUPS), C.ORIG_GROUPS)
need(os.path.isfile(C.ORIG_MASK_REF), C.ORIG_MASK_REF)
for g in C.GROUPS:
    need(os.path.isfile(os.path.join(C.LIST_DIR, g + ".json")), C.LIST_DIR + "/" + g + ".json")
import sklearn
need(sklearn.__version__ == C.SKLEARN_PIN,
     "sklearn %s != pinned %s" % (sklearn.__version__, C.SKLEARN_PIN))
import torch
need(torch.cuda.is_available(), "CUDA GPU")
n_val = len(os.listdir("/dev_data/wlt/data/ImageNet/val"))
need(n_val == 50000, "ImageNet val dir has %d files" % n_val)
print("[preflight] OK (ckpt/3x EXP-003 jobs/EXP-002 artifacts/ImageNet/sklearn/CUDA)")
EOF
  "$PYTHON" "$REPRO_DIR/extract_features_t36.py" --help >/dev/null 2>&1 || fail_exit "extract script import"
}

feats_ok() {  # feats_ok <cond>: 特征+sha sidecar+输入哈希在盘且 sha 复算一致
  local cond=$1
  local f="$ROOT/save_t36_tsne/features_$cond.npy"
  [ -f "$f" ] || return 1
  [ -f "$ROOT/save_t36_tsne/features_${cond}_sha256.txt" ] || return 1
  [ -f "$ROOT/save_t36_tsne/input_hashes_$cond.json" ] || return 1
  [ "$(cat "$ROOT/save_t36_tsne/features_${cond}_sha256.txt")" = "$("$PYTHON" -c "import sys; sys.path.insert(0,'$REPRO_DIR'); import t36_common as C; print(C.sha256_of('$f'))")" ]
}

extract_cond() {  # extract_cond <cond>
  local cond=$1
  if feats_ok "$cond"; then
    echo "[run_t36_all] SKIP extract $cond (features+sha+input hashes on disk)"
  else
    echo "[run_t36_all] RUN extract $cond"
    "$PYTHON" "$REPRO_DIR/extract_features_t36.py" --cond "$cond" \
      || fail_exit "extract $cond"
  fi
}

tsne_done() {  # tsne_done <cond>
  [ -f "$ROOT/save_t36_tsne/coords_$1_1900x2.npy" ]
}

echo "=== EXP-036 run_t36_all mode=$MODE (GPU) ==="
env_snapshot "$LOGDIR/env_snapshot_$MODE.txt"
preflight

SMOKE_MARK="$ROOT/save_t36_tsne/SMOKE_OK.txt"

if [ "$MODE" = "min" ]; then
  "$PYTHON" "$REPRO_DIR/verify_t36_weights.py" || fail_exit "weight verification (3 datasets)"
  extract_cond frozen
  "$PYTHON" "$REPRO_DIR/verify_t36_features.py" --conds frozen \
    || fail_exit "frozen features bitwise vs EXP-002 original"
  extract_cond ettm1_ln
  "$PYTHON" "$REPRO_DIR/verify_t36_features.py" --conds frozen,ettm1_ln \
    || fail_exit "frozen+ettm1_ln feature gates (incl. identical inputs)"
  GPU_ID=$GPU_ID "$PYTHON" "$REPRO_DIR/run_tsne_t36.py" --cond ettm1_ln \
    || fail_exit "t-SNE ettm1_ln (CPU-bound; runs under the driver's env)"
  { echo "smoke_ok $(date +%F\ %T)"; echo "verify_json_md5=$(md5sum "$LOGDIR/verify_t36.json" | cut -d' ' -f1)"; } > "$SMOKE_MARK"
  echo "[run_t36_all] MIN DONE: weights x3 + frozen bitwise + ettm1_ln + first t-SNE. Next: bash $0 main"
  exit 0
fi

[ -f "$SMOKE_MARK" ] || fail_exit "run 'bash $0 min' first (need save_t36_tsne/SMOKE_OK.txt)"

for cond in ecl_ln weather_ln; do
  extract_cond "$cond"
done
"$PYTHON" "$REPRO_DIR/verify_t36_features.py" || fail_exit "full feature gates (4 conds)"

for cond in ecl_ln weather_ln; do
  if tsne_done "$cond"; then
    echo "[run_t36_all] SKIP t-SNE $cond (coords on disk)"
  else
    "$PYTHON" "$REPRO_DIR/run_tsne_t36.py" --cond "$cond" || fail_exit "t-SNE $cond"
  fi
done

"$PYTHON" "$REPRO_DIR/nn_metric_t36.py" || fail_exit "30-NN metric"
"$PYTHON" "$REPRO_DIR/plot_fig36.py" || fail_exit "2x2 figure"
"$PYTHON" "$REPRO_DIR/collect_t36.py" || fail_exit "collect_t36"
echo "[run_t36_all] ALL DONE (main): 4 conditions + 3 t-SNE + NN table + figure + RESULT."
