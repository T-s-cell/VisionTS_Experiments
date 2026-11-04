#!/bin/bash
# EXP-038 dispatcher: task A (Monash Table 15, 30 code subsets, AUTHOR-pinned
# env) and task B (4 ETT x 4H x {All, Rand-VM} seed2021, current env) are
# fully independent; EXP-037 artifacts are READ-ONLY.
#
# Usage (eta):
#   PYTHON_A=/dev_data/wlt/conda/envs/visionts_t37author/bin/python \
#   AUTHOR_HOME=/dev_data/wlt/t37_author_home \
#   PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python GPU_ID=0 \
#     nohup bash repro_t38_t15t22/run_t38_all.sh a_all > repro_t38_t15t22/logs/a_all.out 2>&1 &
#   ... run_t38_all.sh b_all > .../b_all.out 2>&1 &
# Stages: preflight_a a_replay a_queue a_verify a_collect pack_a a_all |
#         preflight_b b_pilot b_queue b_verify b_collect pack_b b_all
#
# Failure policy: every A command runs with HOME=$AUTHOR_HOME (honest
# download timing; no default-HOME cache leak). a_replay is a HARD gate
# (exit stops stage A); queue items that crash append to logs/FAILED.txt
# and the queue continues. B pilot failing stops stage B; single failed
# training appends + continues.

set -u
PYTHON_A=${PYTHON_A:?Set PYTHON_A=/dev_data/wlt/conda/envs/visionts_t37author/bin/python}
PYTHON_B=${PYTHON_B:?Set PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python}
AUTHOR_HOME=${AUTHOR_HOME:?Set AUTHOR_HOME=/dev_data/wlt/t37_author_home}
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$PKG_DIR")"
REPO_DIR="$(dirname "$ROOT_DIR")"
LOGS="$PKG_DIR/logs"
FAILED="$LOGS/FAILED.txt"
mkdir -p "$LOGS"

log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "FAILED: $*"; echo "[$(date '+%F %T')] $1: $*" >> "$FAILED"; }
clear_stage() { grep -v "] $1: " "$FAILED" > "$FAILED.tmp" 2>/dev/null \
  && mv "$FAILED.tmp" "$FAILED" || true; }

env_snapshot() {  # $1 = python, $2 = tag
  HOME=$AUTHOR_HOME $1 -c "import sys,torch,numpy,pandas,gluonts;print('python',sys.version.split()[0]);print('torch',torch.__version__);print('numpy',numpy.__version__);print('pandas',pandas.__version__);print('gluonts',gluonts.__version__);print('cuda',torch.cuda.is_available())" \
    > "$LOGS/env_snapshot_$2.txt" 2>&1 || true
  nvidia-smi > "$LOGS/nvidia_smi_$2.txt" 2>&1 || true
}

# ---------------------------------------------------------------- preflight A
preflight_a() {
  log "preflight A (author env)..."
  HOME=$AUTHOR_HOME $PYTHON_A - "$REPO_DIR" "$PKG_DIR" "$AUTHOR_HOME" \
    <<'EOF' > "$LOGS/preflight_a.log" 2>&1
import json, os, shutil, sys
repo, pkg, author_home = sys.argv[1], sys.argv[2], sys.argv[3]
assert os.path.expanduser("~") == author_home, \
    "HOME leak: %s != %s" % (os.path.expanduser("~"), author_home)
sys.path.insert(0, pkg)
import t38_common as C  # noqa: E402
out = {"started": C.time_str(), "home": os.path.expanduser("~")}
ckpt = os.path.join(repo, "ckpt", "mae_visualize_vit_base.pth")
assert os.path.isfile(ckpt), "ckpt missing: %s" % ckpt
out["ckpt_sha256"] = C.sha256_of(ckpt)
assert out["ckpt_sha256"] == C.CKPT_SHA256, "ckpt sha mismatch"
assert os.path.getsize(ckpt) == C.CKPT_BYTES, "ckpt byte size mismatch"
os.chdir(C.EVAL_GLUONTS)
import gluonts
out["gluonts"] = gluonts.__version__
assert gluonts.__version__.startswith("0.14"), \
    "author env must carry gluonts 0.14.x, got %s" % gluonts.__version__
from gluonts.dataset.repository import datasets as gds
# t38_common prepends long_term_tsf/ to sys.path, so the official
# eval_gluonts modules must be front-inserted EXPLICITLY (else `import
# run` resolves to long_term_tsf/run.py, the training entry)
sys.path.insert(0, os.path.join(repo, "eval_gluonts"))
import dataset  # official module; registers the Monash recipes, then check
import run
assert os.path.samefile(run.__file__, os.path.join(repo, "eval_gluonts", "run.py")), \
    "run resolved to %s (not eval_gluonts)" % run.__file__
assert os.path.samefile(dataset.__file__, os.path.join(repo, "eval_gluonts", "dataset.py")), \
    "dataset resolved to %s (not eval_gluonts)" % dataset.__file__
missing = [ds for ds in C.MONASH_ORDER if ds not in gds.dataset_recipes]
assert not missing, "recipes missing in this gluonts: %s" % missing
out["import_dataset_run"] = "OK"
out["recipes_30"] = len([ds for ds in C.MONASH_ORDER
                         if ds in gds.dataset_recipes])
import numpy as np
import pandas as pd
import torch
assert torch.cuda.is_available(), "CUDA not available"
z = torch.zeros(64, 64, device="cuda:0", dtype=torch.float64)
out["cuda_fp64"] = bool(z.isfinite().all().item())
free, total = torch.cuda.mem_get_info(0)
out["gpu_free_gb"] = round(free / 1024 ** 3, 2)
assert free / total > 0.15, "GPU mostly occupied (free %.1f GB)" % (free / 1024 ** 3)
du = shutil.disk_usage(C.ROOT)
out["disk_free_gb"] = round(du.free / 1024 ** 3, 2)
assert du.free > 10 * 1024 ** 3, "less than 10 GB free"
# naive-path constructor self-check (gluonts 0.14.4)
from gluonts.model.forecast import SampleForecast
f0 = SampleForecast(samples=np.zeros((1, 4)),
                    start_date=pd.Period("2020-01-01", freq="D"))
assert f0.samples.shape == (1, 4)
from gluonts.model.evaluation import evaluate_forecasts
import inspect
assert "test_data" in inspect.signature(evaluate_forecasts).parameters
from gluonts.ev.metrics import MAE, MASE, MSE, ND, NRMSE, SMAPE
for m in (MAE, MASE, MSE, ND, NRMSE, SMAPE):
    m()
out["finished"] = C.time_str()
with open(os.path.join(C.LOG_DIR, "preflight_a.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("PREFLIGHT_A_OK gluonts=%s free=%.1fGB" % (out["gluonts"], out["gpu_free_gb"]))
EOF
  rc=$?
  cat "$LOGS/preflight_a.log"
  return $rc
}

# ---------------------------------------------------------------- A helpers
run_one() {  # $1 = ds ; runs runner + naive; returns runner rc
  local ds=$1
  log "runner $ds..."
  HOME=$AUTHOR_HOME $PYTHON_A "$PKG_DIR/t38_monash_runner.py" --dataset "$ds" \
    --gpu 0 > "$LOGS/runner_${ds}.log" 2>&1
  local rc=$?
  tail -3 "$LOGS/runner_${ds}.log"
  if [ $rc -ne 0 ]; then
    fail "a_queue" "runner $ds rc=$rc (subset paused; queue continues)"
    return $rc
  fi
  log "naive $ds..."
  HOME=$AUTHOR_HOME $PYTHON_A "$PKG_DIR/t38_naive_monash.py" --dataset "$ds" \
    > "$LOGS/naive_${ds}.log" 2>&1 || fail "a_queue" "naive $ds"
  return 0
}

# ---------------------------------------------------------------- stage a
a_replay() {
  clear_stage "a"
  rm -f "$PKG_DIR/A_OK"
  log "== stage a: replay gates (fred_md + nn5_daily, HARD) =="
  env_snapshot "$PYTHON_A" a
  preflight_a || { fail "a" "preflight"; return 1; }
  for ds in fred_md nn5_daily_without_missing; do
    run_one "$ds"; rc=$?
    if [ $rc -ne 0 ] || [ -n "$(grep 'REPLAY N1 round-match FAIL' "$LOGS/naive_${ds}.log" 2>/dev/null)" ]; then
      fail "a_replay" "hard gate failed for $ds"; return 1
    fi
  done
  log "a_replay OK (both anchors rel<=1e-3 and N1 round-match)"
}

a_queue() {
  log "== stage a: queue (tourism_monthly first, then 27 official order) =="
  run_one tourism_monthly
  grep "autotune: P =" "$LOGS/runner_tourism_monthly.log" | tail -5 || true
  for ds in m1_monthly monash_m3_monthly monash_m3_other m4_monthly \
            m4_weekly m4_daily m4_hourly tourism_quarterly cif_2016_6 \
            cif_2016_12 australian_electricity_demand bitcoin \
            pedestrian_counts vehicle_trips_without_missing \
            kdd_cup_2018_without_missing weather nn5_weekly \
            car_parts_without_missing traffic_hourly traffic_weekly \
            rideshare_without_missing hospital covid_deaths \
            temperature_rain_without_missing sunspot_without_missing \
            saugeenday us_births; do
    run_one "$ds"
  done
  log "a_queue done"
}

a_verify() {
  HOME=$AUTHOR_HOME $PYTHON_A "$PKG_DIR/verify_t38_monash.py" \
    > "$LOGS/verify_monash.log" 2>&1
  local rc=$?
  cat "$LOGS/verify_monash.log"
  [ $rc -ne 0 ] && fail "a_verify" "verify monash hard gates"
  return $rc
}

a_collect() {
  HOME=$AUTHOR_HOME $PYTHON_A "$PKG_DIR/collect_t38_monash.py" \
    > "$LOGS/collect_monash.log" 2>&1
  local rc=$?
  cat "$LOGS/collect_monash.log"
  [ $rc -ne 0 ] && fail "a_collect" "collect"
  return $rc
}

pack_a() {
  local out="$REPO_DIR/temp/VisionTS_t38_monash_t15_$(date +%Y%m%d).tar.gz"
  local list="$LOGS/pack_a_filelist.txt"
  mkdir -p "$REPO_DIR/temp"
  log "packing A -> $out"
  ( cd "$REPO_DIR" && : > "$list" && \
    for f in t38_common.py t38_monash_runner.py t38_naive_monash.py \
             verify_t38_monash.py collect_t38_monash.py run_t38_all.sh; do
      echo "long_term_tsf/repro_t38_t15t22/$f" >> "$list"
    done
    find long_term_tsf/repro_t38_t15t22/logs -maxdepth 1 \
         \( -name '*.json' -o -name '*.md' -o -name '*.csv' \) >> "$list"
    [ -f long_term_tsf/repro_t38_t15t22/logs/fig5_monash_ratio.png ] && \
      echo "long_term_tsf/repro_t38_t15t22/logs/fig5_monash_ratio.png" >> "$list"
    for f in long_term_tsf/repro_t38_t15t22/logs/runner_*.log \
             long_term_tsf/repro_t38_t15t22/logs/naive_*.log \
             long_term_tsf/repro_t38_t15t22/logs/preflight_a.log; do
      [ -f "$f" ] || continue
      [ -f "${f}.gz" ] || gzip -k "$f"
      echo "${f}.gz" >> "$list"
    done
    ls save_t38_monash/visionts_monash_*.csv \
       save_t38_monash/naive_n1_monash_*.csv \
       save_t38_monash/naive_n2_monash_*.csv >> "$list" 2>/dev/null || true
    sort -u "$list" -o "$list"
    tar czf "$out" -T "$list" )
  local rc=$?
  if [ $rc -ne 0 ]; then fail "pack_a" "tar failed"; return 1; fi
  local size=$(du -b "$out" | cut -f1)
  local sha=$(sha256sum "$out" | cut -d' ' -f1)
  echo "$sha  $out" > "${out}.sha256"
  log "pack_a done: $size bytes sha256 $sha"
  if [ "$size" -gt 52428800 ]; then
    fail "pack_a" "pack exceeds 50MB ($size bytes)"; return 1
  fi
  touch "$PKG_DIR/A_OK"
  log "A_OK"
}

a_all() {
  a_replay || return 1
  a_queue
  a_verify || return 1
  a_collect || return 1
  pack_a || return 1
}

# ---------------------------------------------------------------- preflight B
preflight_b() {
  log "preflight B (current env)..."
  $PYTHON_B - "$REPO_DIR" "$PKG_DIR" <<'EOF' > "$LOGS/preflight_b.log" 2>&1
import json, os, shutil, sys
repo, pkg = sys.argv[1], sys.argv[2]
sys.path.insert(0, pkg)
import t38_common as C  # noqa: E402
out = {"started": C.time_str()}
# code identity vs the audited EXP-003 training tree (12 fingerprint files)
fp003 = C.load_json(os.path.join(C.exp003_job("ETTh1", 96), "fingerprint.json"))
mism = []
out["code_md5_match"] = {}
for rel in C.EXP003_FP_FILES:
    p = os.path.join(repo, rel)
    assert os.path.isfile(p), "code file missing: %s" % p
    got = C.md5_of(p)
    want = fp003["code_md5"][rel]
    out["code_md5_match"][rel] = (got == want)
    if got != want:
        mism.append(rel)
assert not mism, "code md5 differs from EXP-003 fingerprint: %s" % mism
# the four ETT CSVs vs each config's own EXP-003 data sha
out["csv_sha_ok"] = {}
for ds in C.ETT4:
    fp = C.load_json(os.path.join(C.exp003_job(ds, 96), "fingerprint.json"))
    want = list(fp["data_sha256"].values())[0]
    got = C.sha256_of(os.path.join(repo, C.ETT_CSV_REL[ds]))
    out["csv_sha_ok"][ds] = (got == want)
    assert got == want, "%s.csv sha != EXP-003 data sha" % ds
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
assert du.free > 20 * 1024 ** 3, "less than 20 GB free"
out["finished"] = C.time_str()
with open(os.path.join(C.LOG_DIR, "preflight_b.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("PREFLIGHT_B_OK free=%.1fGB" % out["gpu_free_gb"])
EOF
  rc=$?
  cat "$LOGS/preflight_b.log"
  return $rc
}

# ---------------------------------------------------------------- stage b
b_pilot() {
  clear_stage "b"
  rm -f "$PKG_DIR/B_OK"
  log "== stage b: pilot ETTm1_96 both arms (budget probe) =="
  env_snapshot "$PYTHON_B" b
  preflight_b || { fail "b" "preflight"; return 1; }
  for arm in all rand; do
    log "pilot arm=$arm..."
    $PYTHON_B "$PKG_DIR/t38_train_driver.py" --config ETTm1_96 --arm "$arm" \
      --gpu 0 > "$LOGS/train_ETTm1_96_${arm}.log" 2>&1
    rc=$?
    tail -3 "$LOGS/train_ETTm1_96_${arm}.log"
    if [ $rc -ne 0 ]; then fail "b_pilot" "arm=$arm rc=$rc"; return 1; fi
  done
  $PYTHON_B - "$PKG_DIR" <<'EOF'
import json, sys, os
sys.path.insert(0, sys.argv[1])
import t38_common as C
for arm in C.B_ARMS:
    rec = C.load_json(C.train_json("ETTm1", 96, arm))
    ph = rec["phases"]
    tb = rec["sampler_len_check"]["expected_train_batches"]
    print("[pilot] %s: train %.1fs (%.3f s/batch x %d), vali %.1fs, "
          "test %.1fs, peak train %.0fMB vali %.0fMB test %.0fMB"
          % (arm, ph["train_wall"], ph["train_wall"] / tb, tb,
             ph["vali_wall"] or -1, ph["test_wall"] or -1,
             ph["train_peak_mb"] or -1, ph["vali_peak_mb"] or -1,
             ph["test_peak_mb"] or -1))
EOF
  log "b_pilot OK"
}

b_queue() {
  log "== stage b: remaining 14 configs x 2 arms =="
  for cfg in ETTh1_192 ETTh1_336 ETTh1_720 ETTh2_96 ETTh2_192 ETTh2_336 \
             ETTh2_720 ETTm2_96 ETTm2_192 ETTm2_336 ETTm2_720 \
             ETTm1_192 ETTm1_336 ETTm1_720; do
    for arm in all rand; do
      log "training $cfg arm=$arm..."
      $PYTHON_B "$PKG_DIR/t38_train_driver.py" --config "$cfg" --arm "$arm" \
        --gpu 0 > "$LOGS/train_${cfg}_${arm}.log" 2>&1
      rc=$?
      tail -2 "$LOGS/train_${cfg}_${arm}.log"
      [ $rc -ne 0 ] && fail "b_queue" "$cfg $arm rc=$rc (queue continues)"
    done
  done
  log "b_queue done"
}

b_verify() {
  $PYTHON_B "$PKG_DIR/verify_t38_fullshot.py" > "$LOGS/verify_fullshot.log" 2>&1
  local rc=$?
  cat "$LOGS/verify_fullshot.log"
  [ $rc -ne 0 ] && fail "b_verify" "verify fullshot"
  return $rc
}

b_collect() {
  $PYTHON_B "$PKG_DIR/collect_t38_fullshot.py" > "$LOGS/collect_ett.log" 2>&1
  local rc=$?
  cat "$LOGS/collect_ett.log"
  [ $rc -ne 0 ] && fail "b_collect" "collect"
  return $rc
}

pack_b() {
  local out="$REPO_DIR/temp/VisionTS_t38_ett_t22_$(date +%Y%m%d).tar.gz"
  local list="$LOGS/pack_b_filelist.txt"
  mkdir -p "$REPO_DIR/temp"
  log "packing B -> $out"
  ( cd "$REPO_DIR" && : > "$list" && \
    for f in t38_common.py t38_train_driver.py verify_t38_fullshot.py \
             collect_t38_fullshot.py run_t38_all.sh; do
      echo "long_term_tsf/repro_t38_t15t22/$f" >> "$list"
    done
    find long_term_tsf/repro_t38_t15t22/logs -maxdepth 1 -name 'train_*.json' >> "$list"
    for f in verify_t38.json summary_t38_ett.csv RESULT_t38_ett.md \
             preflight_b.json; do
      [ -f "long_term_tsf/repro_t38_t15t22/logs/$f" ] && \
        echo "long_term_tsf/repro_t38_t15t22/logs/$f" >> "$list"
    done
    for f in long_term_tsf/repro_t38_t15t22/logs/train_*.log \
             long_term_tsf/repro_t38_t15t22/logs/preflight_b.log; do
      [ -f "$f" ] || continue
      [ -f "${f}.gz" ] || gzip -k "$f"
      echo "${f}.gz" >> "$list"
    done
    sort -u "$list" -o "$list"
    tar czf "$out" -T "$list" )
  local rc=$?
  if [ $rc -ne 0 ]; then fail "pack_b" "tar failed"; return 1; fi
  local size=$(du -b "$out" | cut -f1)
  local sha=$(sha256sum "$out" | cut -d' ' -f1)
  echo "$sha  $out" > "${out}.sha256"
  log "pack_b done: $size bytes sha256 $sha"
  if [ "$size" -gt 52428800 ]; then
    fail "pack_b" "pack exceeds 50MB ($size bytes)"; return 1
  fi
  touch "$PKG_DIR/B_OK"
  log "B_OK"
}

b_all() {
  b_pilot || return 1
  b_queue
  b_verify || return 1
  b_collect || return 1
  pack_b || return 1
}

case "${1:-}" in
  preflight_a) preflight_a; exit $? ;;
  a_replay) a_replay; exit $? ;;
  a_queue) a_queue; exit $? ;;
  a_verify) a_verify; exit $? ;;
  a_collect) a_collect; exit $? ;;
  pack_a) pack_a; exit $? ;;
  a_all) a_all; exit $? ;;
  preflight_b) preflight_b; exit $? ;;
  b_pilot) b_pilot; exit $? ;;
  b_queue) b_queue; exit $? ;;
  b_verify) b_verify; exit $? ;;
  b_collect) b_collect; exit $? ;;
  pack_b) pack_b; exit $? ;;
  b_all) b_all; exit $? ;;
  *) echo "usage: $0 {preflight_a|a_replay|a_queue|a_verify|a_collect|pack_a|a_all|preflight_b|b_pilot|b_queue|b_verify|b_collect|pack_b|b_all}"; exit 2 ;;
esac
