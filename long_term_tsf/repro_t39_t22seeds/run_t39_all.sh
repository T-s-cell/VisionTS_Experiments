#!/bin/bash
# EXP-039 dispatcher: Table 22 B-line across three seeds. 64 new
# trainings (16 configs x seeds {2022,2023} x {All, Rand-VM}); seed 2021
# is archived from EXP-038 (READ-ONLY, incl. the EXP-037 ETTh1_96 pair).
#
# Usage (eta):
#   PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python GPU_ID=0 \
#     nohup bash repro_t39_t22seeds/run_t39_all.sh all \
#       > repro_t39_t22seeds/logs/all.out 2>&1 &
# Stages: preflight | smoke | queue | verify | collect | pack | all
#
# Failure policy: preflight and smoke-verify are HARD gates (exit stops
# everything); a single failed training appends to logs/FAILED.txt and
# the queue continues. Resume semantics (user correction 2026-11-05):
# an empty save_t39_full is expected only on first launch; a task is
# SKIPPED only when SUCCESS.txt + train json (success + fingerprint
# seed pin) + pred/true npy + final checkpoint are ALL present; a
# partial job dir is moved aside (scene preserved) and re-trained.

set -u
PYTHON_B=${PYTHON_B:?Set PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python}
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
  $1 -c "import sys,torch,numpy;print('python',sys.version.split()[0]);print('torch',torch.__version__);print('numpy',numpy.__version__);print('cuda',torch.cuda.is_available())" \
    > "$LOGS/env_snapshot_$2.txt" 2>&1 || true
  nvidia-smi > "$LOGS/nvidia_smi_$2.txt" 2>&1 || true
}

job_list() {  # $1 = subset name (smoke|all) -> lines "DS_H arm seed"
  $PYTHON_B - "$PKG_DIR" "$1" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import t39_common as X
tasks = X.SMOKE if sys.argv[2] == "smoke" else X.TASKS
for ds, h, arm, s in tasks:
    print("%s_%d %s %d" % (ds, h, arm, s))
EOF
}

# resume completeness: prints SKIP / MOVE:<dir> / RUN for one task
job_decide() {  # $1 = DS_H, $2 = arm, $3 = seed
  $PYTHON_B - "$PKG_DIR" "$1" "$2" "$3" <<'EOF'
import json, os, shutil, sys, time
pkg, cfg, arm, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, pkg)
import t39_common as X   # first: fixes sys.path for the t38 package
import t38_common as C
ds, h = cfg.rsplit("_", 1)
h = int(h)
jd = X.b_job_dir(ds, h, arm, seed)
if not os.path.isdir(jd):
    print("RUN"); sys.exit(0)
succ = os.path.join(jd, "SUCCESS.txt")
def why(msg):
    print("MOVE:%s\t%s" % (jd, msg))
if not os.path.isfile(succ):
    why("no SUCCESS.txt"); sys.exit(0)
try:
    s = C.load_json(succ)
    assert isinstance(s.get("mse"), float) and isinstance(s.get("mae"), float)
    assert s.get("seed") == seed
except Exception as e:
    why("SUCCESS unreadable: %s" % e); sys.exit(0)
p = X.train_json(ds, h, arm, seed)
try:
    rec = C.load_json(p)
    assert rec["success"]["mse"] == s["mse"]
    assert rec["fingerprint"]["seed"] == seed
    assert rec["fingerprint"]["args"]["seed"] == seed
    assert rec.get("requested_seed") == seed
except Exception as e:
    why("train json incomplete: %s" % e); sys.exit(0)
try:
    fpj = C.load_json(os.path.join(jd, "fingerprint.json"))
    assert fpj == rec["fingerprint"], "job-dir fingerprint != train json"
    assert fpj.get("seed") == seed and fpj.get("args", {}).get("seed") == seed
    assert all(isinstance(fpj.get(k), str) and fpj[k]
               for k in ("code_md5", "data_sha256", "ckpt_sha256"))
except Exception as e:
    why("fingerprint incomplete: %s" % e); sys.exit(0)
rd = os.path.join(jd, "results", "_")
for f in ("pred.npy", "true.npy"):
    q = os.path.join(rd, f)
    if not os.path.isfile(q) or os.path.getsize(q) <= 0:
        why("missing/empty %s" % f); sys.exit(0)
    with open(q, "rb") as fh:
        if fh.read(6) != b"\x93NUMPY":
            why("invalid npy header: %s" % f); sys.exit(0)
ck = os.path.join(jd, "checkpoints", "_", "checkpoint.pth")
if not os.path.isfile(ck):
    why("missing final checkpoint"); sys.exit(0)
print("SKIP\tmse=%.6f mae=%.6f json=%s" % (s["mse"], s["mae"], os.path.basename(p)))
EOF
}

job_move_aside() {  # $1 = job dir; preserve the failed scene
  local jd=$1
  local base=$(basename "$jd")
  mkdir -p "$ROOT_DIR/save_t39_full/.failed"
  mv "$jd" "$ROOT_DIR/save_t39_full/.failed/${base}_$(date +%Y%m%d_%H%M%S)"
  log "moved partial job aside: $jd"
}

run_one() {  # $1 = DS_H, $2 = arm, $3 = seed
  local cfg=$1 arm=$2 seed=$3
  local decision tag
  tag="${cfg}_${arm}_s${seed}"
  decision=$(job_decide "$cfg" "$arm" "$seed") || decision="RUN"
  if [[ "$decision" == SKIP* ]]; then
    log "skip-if-done $tag: ${decision#SKIP}"
    return 0
  fi
  if [[ "$decision" == MOVE:* ]]; then
    local mdir=${decision#MOVE:}
    job_move_aside "${mdir%%$'\t'*}"
  fi
  log "training $tag..."
  $PYTHON_B "$PKG_DIR/t39_train_driver.py" --config "$cfg" --arm "$arm" \
    --seed "$seed" --gpu 0 > "$LOGS/train_${tag}.log" 2>&1
  local rc=$?
  tail -2 "$LOGS/train_${tag}.log"
  if [ $rc -ne 0 ]; then
    fail "queue" "$cfg $arm s$seed rc=$rc (queue continues)"
  fi
  return 0
}

# ---------------------------------------------------------------- preflight
preflight() {
  log "preflight (current env)..."
  $PYTHON_B - "$REPO_DIR" "$PKG_DIR" <<'EOF' > "$LOGS/preflight.log" 2>&1
import json, os, shutil, subprocess, sys
repo, pkg = sys.argv[1], sys.argv[2]
sys.path.insert(0, pkg)
import t39_common as X   # first: fixes sys.path for the t38/t22 packages
import t38_common as C
out = {"started": C.time_str()}
assert len(X.TASKS) == 64 and len(set(X.TASKS)) == 64, "TASKS != 64 unique"
assert len(X.SMOKE) == 4 and len(X.QUEUE) == 60
# code identity vs the audited EXP-003 training tree
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
for ds in C.ETT4:
    fp = C.load_json(os.path.join(C.exp003_job(ds, 96), "fingerprint.json"))
    want = list(fp["data_sha256"].values())[0]
    got = C.sha256_of(os.path.join(repo, C.ETT_CSV_REL[ds]))
    assert got == want, "%s.csv sha != EXP-003 data sha" % ds
ckpt = os.path.join(repo, "ckpt", "mae_visualize_vit_base.pth")
assert C.sha256_of(ckpt) == C.CKPT_SHA256
assert os.path.getsize(ckpt) == C.CKPT_BYTES
# EXP-003 LN archives for seeds 2022/2023 (32 dirs; local fallback source)
missing_ln = []
for (ds, h) in X.CONFIGS:
    for s in X.SEEDS:
        if not os.path.isdir(os.path.join(C.ROOT, "save_fullshot",
                                          "%s_%d_seed%d" % (ds, h, s))):
            missing_ln.append("%s_%d_seed%d" % (ds, h, s))
out["ln_seed_dirs_missing_on_eta"] = missing_ln
if missing_ln:
    local = os.path.join(repo, "save_fullshot")
    print("MISSING LN ARCHIVES on eta (scp from %s):" % local)
    for m in missing_ln:
        print("  scp -r %s/%s <eta>:%s/save_fullshot/" %
              (local, m, os.path.join(repo, "long_term_tsf")))
    raise SystemExit("LN archives incomplete (%d missing)" % len(missing_ln))
# LN48 pins vs the archives that ARE present (eta base chosen by marker)
for (ds, h) in X.CONFIGS:
    for s in (2021, 2022, 2023):
        p = X.ln_success_path(ds, h, s)
        assert os.path.isfile(p), "LN SUCCESS missing: %s" % p
        mse, mae = C.parse_success_mse_mae(p)
        pin = X.LN48["%s_%d_seed%d" % (ds, h, s)]
        assert (mse, mae) == pin, "LN pin mismatch %s_%d_s%d" % (ds, h, s)
out["ln48_pins_checked"] = 48
# EXP-038 seed-2021 provenance (30 t38 jsons + 2 t37 reuse)
n38 = 0
for (ds, h) in X.CONFIGS:
    for arm in C.B_ARMS:
        p = X.t38_train_json(ds, h, arm)
        assert os.path.isfile(p), "t38 train json missing: %s" % p
        rec = C.load_json(p)
        assert isinstance(rec["success"]["mse"], float)
        n38 += 1
out["t38_jsons_present"] = n38
assert n38 == 32  # 16 configs x 2 arms (30 t38 + 2 t37 reuse for ETTh1_96)
# save_t39_full state (first launch expects absent/empty; resume counts)
save = X.SAVE_B_BASE
if os.path.isdir(save):
    sub = [d for d in os.listdir(save) if d != ".failed"]
    n_succ = sum(1 for d in sub
                 if os.path.isfile(os.path.join(save, d, "SUCCESS.txt")))
    out["save_t39_full"] = {"dirs": len(sub), "with_success": n_succ}
    print("RESUME STATE: %d job dirs, %d with SUCCESS.txt"
          % (len(sub), n_succ))
else:
    out["save_t39_full"] = "absent (first launch)"
# GPU + disk
import torch
assert torch.cuda.is_available()
q = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.free,memory.total",
     "--format=csv,noheader,nounits"]).decode()
free, total = [float(x) for x in q.strip().splitlines()[0].split(",")]
out["gpu_free_gb"] = round(free / 1024, 2)
assert free / total > 0.15, "GPU mostly occupied"
du = shutil.disk_usage(C.ROOT)
out["disk_free_gb"] = round(du.free / 1024 ** 3, 2)
assert du.free > 20 * 1024 ** 3, "less than 20 GB free"
out["finished"] = C.time_str()
with open(os.path.join(X.LOG_DIR, "preflight.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("PREFLIGHT_OK free=%.1fGB disk=%.0fGB ln48=OK t38=32/32 tasks=64"
      % (out["gpu_free_gb"], out["disk_free_gb"]))
EOF
  rc=$?
  cat "$LOGS/preflight.log"
  return $rc
}

# ---------------------------------------------------------------- smoke
smoke() {
  clear_stage "t39"
  rm -f "$PKG_DIR/T39_OK"
  log "== smoke: ETTh1_96 x seeds{2022,2023} x {all,rand} (4 of 64) =="
  env_snapshot "$PYTHON_B" t39
  preflight || { fail "preflight" "hard gate"; return 1; }
  local n=0
  while read -r cfg arm seed; do
    [ -z "$cfg" ] && continue
    run_one "$cfg" "$arm" "$seed" || true
    n=$((n + 1))
  done < <(job_list smoke)
  [ "$n" -eq 4 ] || { fail "smoke" "job count $n != 4"; return 1; }
  log "smoke verify (seed-effectiveness + numeric gates, HARD)..."
  $PYTHON_B "$PKG_DIR/verify_t39_fullshot.py" --smoke \
    > "$LOGS/verify_smoke.log" 2>&1
  local rc=$?
  cat "$LOGS/verify_smoke.log"
  if [ $rc -ne 0 ]; then
    fail "smoke" "smoke verify FAIL — queue NOT started; preserve scene"
    return 1
  fi
  budget_from_logs
  log "smoke OK (4/4 + verify green) -> auto-continue via stage queue"
}

budget_from_logs() {
  $PYTHON_B - "$PKG_DIR" <<'EOF'
import os, sys
sys.path.insert(0, sys.argv[1])
import t39_common as X   # first: fixes sys.path for the t38 package
import t38_common as C
print("[budget] measured pure-train s/batch (train_wall - vali_wall):")
for ds, h in [("ETTh1", 96)]:
    for arm in C.B_ARMS:
        for s in X.SEEDS:
            p = X.train_json(ds, h, arm, s)
            if not os.path.isfile(p):
                continue
            rec = C.load_json(p)
            ph = rec.get("phases", {})
            w, v = ph.get("train_wall"), ph.get("vali_wall")
            b = rec.get("sampler_len_check", {}).get("expected_train_batches")
            if w and v is not None and b:
                print("  %s_%d s%d %s: %.3f s/batch x %d batches"
                      % (ds, h, s, arm, (w - v) / b, b))
rem = sum(C.expected_train_batches(ds, h) for (ds, h, _, _) in X.QUEUE)
print("[budget] remaining queue batches (approx): %d" % rem)
EOF
}

# ---------------------------------------------------------------- queue
queue() {
  log "== queue: remaining tasks (skip-if-done enforced per task) =="
  local n=0 n_skip=0
  while read -r cfg arm seed; do
    [ -z "$cfg" ] && continue
    decision=$(job_decide "$cfg" "$arm" "$seed") || decision="RUN"
    if [[ "$decision" == SKIP* ]]; then n_skip=$((n_skip + 1)); fi
    run_one "$cfg" "$arm" "$seed"
    n=$((n + 1))
  done < <(job_list all)
  [ "$n" -eq 64 ] || fail "queue" "job count $n != 64 (TASKS assert)"
  log "queue done: $n tasks considered, $n_skip skipped as complete"
}

verify_stage() {
  $PYTHON_B "$PKG_DIR/verify_t39_fullshot.py" > "$LOGS/verify_full.log" 2>&1
  local rc=$?
  cat "$LOGS/verify_full.log"
  [ $rc -ne 0 ] && fail "verify" "verify fullshot"
  return $rc
}

collect_stage() {
  $PYTHON_B "$PKG_DIR/collect_t39_fullshot.py" > "$LOGS/collect.log" 2>&1
  local rc=$?
  cat "$LOGS/collect.log"
  [ $rc -ne 0 ] && fail "collect" "collect"
  return $rc
}

pack() {
  local out="$REPO_DIR/temp/VisionTS_t39_t22seeds_$(date +%Y%m%d).tar.gz"
  local list="$LOGS/pack_filelist.txt"
  mkdir -p "$REPO_DIR/temp"
  log "packing -> $out"
  ( cd "$REPO_DIR" && : > "$list" && \
    for f in t39_common.py t39_train_driver.py verify_t39_fullshot.py \
             collect_t39_fullshot.py run_t39_all.sh README_t39.md; do
      echo "long_term_tsf/repro_t39_t22seeds/$f" >> "$list"
    done
    # import-time deps (t39_common -> t38_common -> t22_common); packed
    # at their canonical relative paths so the tree is importable
    # standalone (sys.path fixes are __file__-relative)
    for f in long_term_tsf/repro_t38_t15t22/t38_common.py \
             long_term_tsf/repro_t22_freeze_ln/t22_common.py; do
      [ -f "$f" ] || { fail "pack" "dep missing: $f"; return 1; }
      echo "$f" >> "$list"
    done
    find long_term_tsf/repro_t39_t22seeds/logs -maxdepth 1 \
         -name 'train_*_seed20*.json' >> "$list"
    ntr=$(grep -c 'train_.*_seed20.*\.json$' "$list")
    [ "$ntr" -eq 64 ] || { fail "pack" "train jsons in pack = $ntr (want 64: smoke 4 + queue 60)"; return 1; }
    for f in verify_t39.json verify_t39_smoke.json recompute_f64.json \
             recompute_f64_smoke.json summary_t39_ett.csv RESULT_t39_ett.md \
             fig_t39_seed_effect.png preflight.json; do
      [ -f "long_term_tsf/repro_t39_t22seeds/logs/$f" ] && \
        echo "long_term_tsf/repro_t39_t22seeds/logs/$f" >> "$list"
    done
    # LN evidence: 48 x {SUCCESS.txt, fingerprint.json} (~300KB)
    nl=0
    for s in 2021 2022 2023; do
      for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
        for h in 96 192 336 720; do
          d="long_term_tsf/save_fullshot/${ds}_${h}_seed${s}"
          for f in SUCCESS.txt fingerprint.json; do
            [ -f "$d/$f" ] || { fail "pack" "LN evidence missing: $d/$f"; exit 1; }
            echo "$d/$f" >> "$list"; nl=$((nl + 1))
          done
        done
      done
    done
    [ "$nl" -eq 96 ] || { fail "pack" "LN evidence files = $nl (want 96)"; exit 1; }
    # aggregate snapshot + freeze summary (pinned / audited baselines)
    for f in long_term_tsf/repro_t22_freeze_ln/aggregate_snapshot.csv \
             long_term_tsf/repro_t22_freeze_ln/summary_t22.csv; do
      [ -f "$f" ] || { fail "pack" "baseline missing: $f"; exit 1; }
      echo "$f" >> "$list"
    done
    # EXP-038 seed-2021 provenance: 30 t38 train jsons + 4 t37 files
    n38=0
    for f in long_term_tsf/repro_t38_t15t22/logs/train_*.json; do
      [ -f "$f" ] || continue
      echo "$f" >> "$list"; n38=$((n38 + 1))
    done
    [ "$n38" -eq 30 ] || { fail "pack" "t38 train jsons = $n38 (want 30)"; exit 1; }
    for f in long_term_tsf/repro_t37_pilot/logs/train_all.json \
             long_term_tsf/repro_t37_pilot/logs/train_rand.json \
             long_term_tsf/save_t37_full/ETTh1_96_seed2021_all/SUCCESS.txt \
             long_term_tsf/save_t37_full/ETTh1_96_seed2021_rand/SUCCESS.txt; do
      [ -f "$f" ] || { fail "pack" "reuse evidence missing: $f"; exit 1; }
      echo "$f" >> "$list"
    done
    for f in long_term_tsf/repro_t39_t22seeds/logs/train_*.log \
             long_term_tsf/repro_t39_t22seeds/logs/preflight.log; do
      [ -f "$f" ] || continue
      [ -f "${f}.gz" ] || gzip -k "$f"
      echo "${f}.gz" >> "$list"
    done
    sort -u "$list" -o "$list"
    tar czf "$out" -T "$list" )
  local rc=$?
  if [ $rc -ne 0 ]; then fail "pack" "tar failed"; return 1; fi
  local size=$(du -b "$out" | cut -f1)
  local sha=$(sha256sum "$out" | cut -d' ' -f1)
  echo "$sha  $out" > "${out}.sha256"
  log "pack done: $size bytes sha256 $sha"
  if [ "$size" -gt 52428800 ]; then
    fail "pack" "pack exceeds 50MB ($size bytes)"; return 1
  fi
  touch "$PKG_DIR/T39_OK"
  log "T39_OK"
}

all() {
  smoke || return 1
  queue
  verify_stage || return 1
  collect_stage || return 1
  pack || return 1
}

case "${1:-}" in
  preflight) preflight; exit $? ;;
  smoke) smoke; exit $? ;;
  queue) queue; exit $? ;;
  verify) verify_stage; exit $? ;;
  collect) collect_stage; exit $? ;;
  pack) pack; exit $? ;;
  all) all; exit $? ;;
  *) echo "usage: $0 {preflight|smoke|queue|verify|collect|pack|all}"; exit 2 ;;
esac
