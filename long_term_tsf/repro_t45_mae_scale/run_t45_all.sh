#!/bin/bash
# EXP-045 master driver: MAE-Base/Large/Huge zero-shot comparison on ETT4,
# accuracy + cost, zero-shot inference on eta.
#   PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python \
#     bash repro_t45_mae_scale/run_t45_all.sh
# Stage ORDER (review correction #1: MG11 consumes timing+collect outputs,
# so it runs LAST, before pack):
#   S0 env -> S1 preflight (3 ckpts sha/bytes + strict load + param count,
#     HARD) -> S1b verify --reuse (MG7: 16 EXP-001 Base groups + pinned
#     Base overall anchor <=1e-6, HARD) -> S2 smoke 6 groups + verify
#     --smoke (HARD; 2 Base replays bitwise vs EXP-001 + Large/Huge x
#     ETTh1/H96, ETTm2/H720) -> S3 main queue 32 new groups (per-group
#     acceptance each; skip-if-done resumes) -> S4 verify --all (MG1-MG10
#     + MG11a coverage, HARD) -> S5 timing (microbenchmark) -> S6 collect
#     -> S7 verify --mg11 (FINAL, HARD) -> S8 pack (<50MB + sha256
#     sidecar on /dev_data/wlt; arrays stay).
# skip-if-done everywhere: rerun after interruption resumes; a SUCCESS.txt
# fingerprint mismatch ABORTs (exit 4) without overwriting.
# Checkpoints: large/huge must already be present under ../ckpt/ (scp'd
# from the CA-verified off-box download per review correction #4 -- this
# driver NEVER downloads).
set -u
PY=${PYTHON_B:?PYTHON_B must point to the eta visionts python}
export PYTHON="$PY"   # run_one_t45.sh reads PYTHON (default "python")
PKG=repro_t45_mae_scale
GPU_ID=${GPU_ID:-0}
export GPU_ID
ROOT="$(pwd)"
[ -f "$PKG/verify_t45.py" ] || { echo "[t45] run from long_term_tsf/"; exit 2; }
mkdir -p "$PKG/logs"

run_group() {  # ARCH_SHORT DS PRED
  local SHORT=$1 DS=$2 PRED=$3 RC
  bash "$PKG/run_one_t45.sh" "$DS" "$PRED" "$SHORT"
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "[t45] RUN FAIL $SHORT $DS pred=$PRED exit=$RC -- stop, preserve scene"
    exit "$RC"
  fi
  $PY "$PKG/verify_t45.py" --group "$SHORT" "$DS" "$PRED"
  if [ $? -ne 0 ]; then
    echo "[t45] GROUP ACCEPT FAIL $SHORT $DS pred=$PRED -- stop, preserve scene"
    exit 20
  fi
}

echo "=== S0 env: $(date +%F\ %T) py=$PY gpu=$GPU_ID host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || exit 1
for f in mae_visualize_vit_base.pth mae_visualize_vit_large.pth \
         mae_visualize_vit_huge.pth; do
  [ -f "../ckpt/$f" ] || { echo "[t45] MISSING ../ckpt/$f -- scp it first (no insecure download in this driver)"; exit 2; }
done

echo "=== S1 preflight (incl. strict load + param count vs paper)"
$PY "$PKG/preflight_t45.py" || { echo "[t45] PREFLIGHT FAIL"; exit 2; }

echo "=== S1b verify --reuse (MG7: Base 16 groups + overall anchor)"
$PY "$PKG/verify_t45.py" --reuse || { echo "[t45] REUSE FAIL"; exit 21; }

echo "=== S2 smoke: 2 Base replays + Large/Huge x ETTh1/H96, ETTm2/H720"
run_group base ETTh1 96
run_group base ETTm1 96
run_group large ETTh1 96
run_group large ETTm2 720
run_group huge ETTh1 96
run_group huge ETTm2 720
$PY "$PKG/verify_t45.py" --smoke || { echo "[t45] VERIFY --smoke FAIL"; exit 3; }
echo "--- smoke budget (wall_time_s per group)"
for f in save_t45_base_replay/ETTh1_96/SUCCESS.txt \
         save_t45_base_replay/ETTm1_96/SUCCESS.txt \
         save_t45_large/ETTh1_96/SUCCESS.txt \
         save_t45_large/ETTm2_720/SUCCESS.txt \
         save_t45_huge/ETTh1_96/SUCCESS.txt \
         save_t45_huge/ETTm2_720/SUCCESS.txt; do
  [ -f "$f" ] && echo "$(grep -h '^wall_time_s=' "$f") $f"
done | tee "$PKG/logs/smoke_budget.txt"

echo "=== S3 main queue (32 new groups; the 4 smoke groups skip-if-done)"
for SHORT in large huge; do
  for DS in ETTh1 ETTh2 ETTm1 ETTm2; do
    for PRED in 96 192 336 720; do
      run_group "$SHORT" "$DS" "$PRED"
    done
  done
done

echo "=== S4 verify --all (MG1-MG10 + MG11a, HARD)"
$PY "$PKG/verify_t45.py" --all || { echo "[t45] VERIFY --all FAIL"; exit 5; }

echo "=== S5 timing (microbenchmark: same batch, 3 archs sequential)"
$PY "$PKG/timing_t45.py" --gpu "$GPU_ID" || { echo "[t45] TIMING FAIL"; exit 6; }

echo "=== S6 collect"
$PY "$PKG/collect_t45.py" || { echo "[t45] COLLECT FAIL"; exit 6; }

echo "=== S7 verify --mg11 (FINAL gate, consumes timing+collect, HARD)"
$PY "$PKG/verify_t45.py" --mg11 || { echo "[t45] MG11 FAIL"; exit 7; }

echo "=== S8 pack (<50MB; pred/true npy and weights NOT included)"
PACK=/dev_data/wlt/temp_t45_pack.tar.gz
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.gpu_poll_*' \
    -C "$ROOT" -czf "$PACK" "$PKG"
ls -l "$PACK"
sha256sum "$PACK" | tee "$PACK.sha256"
echo "[t45] DONE_ALL $(date +%F\ %T)"
