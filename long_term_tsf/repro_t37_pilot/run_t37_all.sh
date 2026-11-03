#!/bin/bash
# EXP-037 driver: two INDEPENDENT pilots.
#   stage a -> Monash FRED-MD + NN5 Daily (official harness, naive, verify) -> A_OK
#   stage b -> ETTh1 All vs Rand-VM formal trainings + verify -> B_OK
#   stage all = a then b (a failures never block b)
#
# Usage (eta, visionts env):
#   PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python GPU_ID=0 \
#     nohup bash repro_t37_pilot/run_t37_all.sh a  > repro_t37_pilot/logs/stage_a.out 2>&1 &
#   PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python GPU_ID=0 \
#     nohup bash repro_t37_pilot/run_t37_all.sh b  > repro_t37_pilot/logs/stage_b.out 2>&1 &
# (cwd-independent: the script locates itself)
#
# Failure policy: any failing step appends a stage-tagged line to
# logs/FAILED.txt; A_OK/B_OK are only written when that stage is fully green.

set -u
PYTHON=${PYTHON:?Set PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python}
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$PKG_DIR")"          # long_term_tsf/
REPO_DIR="$(dirname "$ROOT_DIR")"         # repo root
LOGS="$PKG_DIR/logs"
FAILED="$LOGS/FAILED.txt"
mkdir -p "$LOGS"

log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "FAILED: $*"; echo "[$(date '+%F %T')] $1: $*" >> "$FAILED"; }
clear_stage() { grep -v "] $1: " "$FAILED" > "$FAILED.tmp" 2>/dev/null && mv "$FAILED.tmp" "$FAILED" || true; }

env_snapshot() {
  $PYTHON -c "import sys,torch,numpy,gluonts,sklearn;print('python',sys.version.split()[0]);print('torch',torch.__version__);print('numpy',numpy.__version__);print('gluonts',gluonts.__version__);print('sklearn',sklearn.__version__);print('cuda',torch.cuda.is_available())" \
    > "$LOGS/env_snapshot.txt" 2>&1 || true
  nvidia-smi > "$LOGS/nvidia_smi_start.txt" 2>&1 || true
}

# ---------------------------------------------------------------- preflight A
preflight_a() {
  log "preflight A..."
  $PYTHON - "$REPO_DIR" "$PKG_DIR" <<'EOF' > "$LOGS/preflight_a.log" 2>&1
import json, os, shutil, sys
repo, pkg = sys.argv[1], sys.argv[2]
sys.path.insert(0, pkg)
import t37_common as C  # noqa: E402
out = {"started": C.time_str()}
ckpt = os.path.join(repo, "ckpt", "mae_visualize_vit_base.pth")
assert os.path.isfile(ckpt), "ckpt missing: %s" % ckpt
out["ckpt_sha256"] = C.sha256_of(ckpt)
assert out["ckpt_sha256"] == C.CKPT_SHA256, "ckpt sha mismatch"
assert os.path.getsize(ckpt) == C.CKPT_BYTES, "ckpt byte size mismatch"
os.chdir(C.EVAL_GLUONTS)
import gluonts
out["gluonts"] = gluonts.__version__
import dataset  # official monkeypatch module; raises on incompatible gluonts API
import run      # imports visionts/torch/einops too
out["import_dataset_run"] = "OK"
import torch
assert torch.cuda.is_available(), "CUDA not available"
free, total = torch.cuda.mem_get_info(0)
out["gpu_free_gb"] = round(free / 1024 ** 3, 2)
out["gpu_total_gb"] = round(total / 1024 ** 3, 2)
assert free / total > 0.15, "GPU mostly occupied (free %.1f GB)" % (free / 1024 ** 3)
du = shutil.disk_usage(C.ROOT)
out["disk_free_gb"] = round(du.free / 1024 ** 3, 2)
assert du.free > 10 * 1024 ** 3, "less than 10 GB free on long_term_tsf volume"
out["finished"] = C.time_str()
with open(os.path.join(C.LOG_DIR, "preflight_a.json"), "w") as f:
    json.dump(out, f, indent=1)
print("PREFLIGHT_A_OK gluonts=%s free=%.1fGB" % (out["gluonts"], out["gpu_free_gb"]))
EOF
  rc=$?
  cat "$LOGS/preflight_a.log"
  return $rc
}

# ---------------------------------------------------------------- stage a
stage_a() {
  clear_stage "a"
  rm -f "$PKG_DIR/A_OK"
  log "== stage a (Monash) =="
  env_snapshot
  preflight_a || { fail "a" "preflight"; return 1; }

  for ds in fred_md nn5_daily_without_missing; do
    log "runner $ds..."
    $PYTHON "$PKG_DIR/t37_monash_runner.py" --dataset "$ds" --gpu 0 \
      > "$LOGS/runner_${ds}.log" 2>&1
    rc=$?
    tail -3 "$LOGS/runner_${ds}.log"
    if [ $rc -ne 0 ]; then
      fail "a" "runner $ds rc=$rc (subset paused; the other subset continues)"
    fi
  done

  for ds in fred_md nn5_daily_without_missing; do
    log "naive $ds..."
    P=$($PYTHON -c "import json,sys;print(json.load(open(sys.argv[1]))['autotune']['accepted_p'])" \
      "$LOGS/monash_${ds}.json" 2>/dev/null || echo BAD)
    if [ "$P" = "BAD" ]; then
      fail "a" "naive $ds skipped (no autotune P; vision runner failed)"
      continue
    fi
    $PYTHON "$PKG_DIR/t37_naive_monash.py" --dataset "$ds" --p "$P" \
      > "$LOGS/naive_${ds}.log" 2>&1 || fail "a" "naive $ds"
  done

  $PYTHON "$PKG_DIR/verify_t37_monash.py" > "$LOGS/verify_monash.log" 2>&1
  rc=$?
  cat "$LOGS/verify_monash.log"
  if [ $rc -ne 0 ]; then fail "a" "verify monash"; return 1; fi
  touch "$PKG_DIR/A_OK"
  log "A_OK"
}

# ---------------------------------------------------------------- preflight B
preflight_b() {
  log "preflight B..."
  $PYTHON - "$REPO_DIR" "$PKG_DIR" <<'EOF' > "$LOGS/preflight_b.log" 2>&1
import json, os, shutil, sys
repo, pkg = sys.argv[1], sys.argv[2]
sys.path.insert(0, pkg)
import t37_common as C
out = {"started": C.time_str()}
# code identity vs the audited EXP-003 training tree
fp003 = C.load_json(os.path.join(C.exp003_job(), "fingerprint.json"))
out["code_md5_match"] = {}
mism = []
for rel in C.EXP003_FP_FILES:
    p = os.path.join(repo, rel)
    assert os.path.isfile(p), "code file missing: %s" % p
    got = C.md5_of(p)
    want = fp003["code_md5"][rel]
    out["code_md5_match"][rel] = (got == want)
    if got != want:
        mism.append(rel)
assert not mism, "code md5 differs from EXP-003 fingerprint: %s" % mism
csv_path = os.path.join(repo, C.ETTH1_CSV_REL)
out["etth1_csv_sha256"] = C.sha256_of(csv_path)
want_csv = list(fp003["data_sha256"].values())[0]
assert out["etth1_csv_sha256"] == want_csv, "ETTh1.csv sha != EXP-003 data sha"
ckpt = os.path.join(repo, "ckpt", "mae_visualize_vit_base.pth")
out["ckpt_sha256"] = C.sha256_of(ckpt)
assert out["ckpt_sha256"] == C.CKPT_SHA256
assert os.path.getsize(ckpt) == C.CKPT_BYTES
import torch
assert torch.cuda.is_available()
free, total = torch.cuda.mem_get_info(0)
out["gpu_free_gb"] = round(free / 1024 ** 3, 2)
assert free / total > 0.15, "GPU mostly occupied"
du = shutil.disk_usage(C.ROOT)
assert du.free > 20 * 1024 ** 3, "less than 20 GB free (two full ckpts + results)"
out["finished"] = C.time_str()
with open(os.path.join(C.LOG_DIR, "preflight_b.json"), "w") as f:
    json.dump(out, f, indent=1)
print("PREFLIGHT_B_OK free=%.1fGB" % out["gpu_free_gb"])
EOF
  rc=$?
  cat "$LOGS/preflight_b.log"
  return $rc
}

# ---------------------------------------------------------------- stage b
stage_b() {
  clear_stage "b"
  rm -f "$PKG_DIR/B_OK"
  log "== stage b (ETTh1 all vs rand) =="
  env_snapshot
  preflight_b || { fail "b" "preflight"; return 1; }

  for arm in all rand; do
    log "training arm=$arm..."
    $PYTHON "$PKG_DIR/t37_train_driver.py" --arm "$arm" --gpu 0 \
      > "$LOGS/train_${arm}.log" 2>&1
    rc=$?
    tail -3 "$LOGS/train_${arm}.log"
    if [ $rc -ne 0 ]; then
      fail "b" "training arm=$arm rc=$rc"
      return 1
    fi
  done

  $PYTHON "$PKG_DIR/verify_t37_fullshot.py" > "$LOGS/verify_fullshot.log" 2>&1
  rc=$?
  cat "$LOGS/verify_fullshot.log"
  if [ $rc -ne 0 ]; then fail "b" "verify fullshot"; return 1; fi
  touch "$PKG_DIR/B_OK"
  log "B_OK"
}

case "${1:-}" in
  a) stage_a; A_RC=$?; exit $A_RC ;;
  b) stage_b; exit $? ;;
  all)
    stage_a; A_RC=$?
    log "--- stage a ended (rc=$A_RC; b proceeds regardless) ---"
    stage_b; B_RC=$?
    [ "$A_RC" -eq 0 ] || log "stage a FAILED (see $FAILED)"
    [ "$B_RC" -eq 0 ] || log "stage b FAILED (see $FAILED)"
    [ "$A_RC" -eq 0 ] && [ "$B_RC" -eq 0 ] && exit 0
    exit 1 ;;
  *) echo "usage: $0 {a|b|all}"; exit 2 ;;
esac
