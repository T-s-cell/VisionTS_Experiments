#!/bin/bash
# EXP-042 dispatcher: paper Table 22 remaining finetune types (Bias /
# MLP / Attention) on the ORIGINAL MAE-base pretrained weights.  144
# new trainings = 16 configs x 3 seeds {2021,2022,2023} x 3 ft types.
# LN (EXP-003), All/Rand-VM (EXP-038 seed-2021 + EXP-039) and Freeze
# (EXP-033) are reused from READ-ONLY archives.
#
# Usage (eta):
#   PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python GPU_ID=0 \
#     nohup bash repro_t42_fttypes/run_t42_all.sh all \
#       > repro_t42_fttypes/logs/all.out 2>&1 &
# Stages: preflight | smoke | queue | verify | collect | pack | all
#
# Failure policy: preflight, smoke-verify and decide anomalies are HARD
# gates (exit stops everything, scene preserved); a single failed
# training appends to logs/FAILED.txt and the queue continues.  A task
# is SKIPPED only when SUCCESS.txt + train json (seed pins + init/
# frozen verdicts) + pred/true npy + final checkpoint (file sha ==
# delivered) are ALL present; a partial job dir (no SUCCESS.txt) is
# moved aside (scene preserved) and re-trained.  ANY anomaly under an
# existing SUCCESS.txt (fingerprint/pin mismatch etc., decide verdict
# ABORT:* or decide rc!=0) stops the pipeline immediately — the runner
# NEVER falls back to RUN, mirroring run_train's native ABORT (rc=3,
# dir kept).

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

job_list() {  # $1 = subset name (smoke|all) -> lines "DS_H ft seed"
  $PYTHON_B - "$PKG_DIR" "$1" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import t42_common as X
tasks = X.SMOKE if sys.argv[2] == "smoke" else X.TASKS
for ds, h, s, ft in tasks:
    print("%s_%d %s %d" % (ds, h, ft, s))
EOF
}

job_decide() {  # $1 = DS_H, $2 = ft, $3 = seed
  $PYTHON_B "$PKG_DIR/verify_t42_fullshot.py" --decide "$1" "$2" "$3"
}

job_move_aside() {  # $1 = job dir; preserve the failed scene
  local jd=$1
  local base=$(basename "$jd")
  mkdir -p "$ROOT_DIR/save_t42_fttypes/.failed"
  mv "$jd" "$ROOT_DIR/save_t42_fttypes/.failed/${base}_$(date +%Y%m%d_%H%M%S)"
  log "moved partial job aside: $jd"
}

run_one() {  # $1 = DS_H, $2 = ft, $3 = seed
  local cfg=$1 ft=$2 seed=$3
  local decision rc tag
  tag="${cfg}_${ft}_s${seed}"
  decision=$(job_decide "$cfg" "$ft" "$seed")
  rc=$?
  if [ $rc -ne 0 ]; then
    # decide itself failed (non-zero rc incl. ABORT rc=3): hard stop,
    # scene preserved — never fall back to RUN
    fail "decide" "$tag decide rc=$rc: $decision — ABORT, scene preserved, pipeline stopped"
    exit 1
  fi
  if [[ "$decision" == SKIP* ]]; then
    log "skip-if-done $tag: ${decision#SKIP}"
    return 0
  fi
  if [[ "$decision" == ABORT:* ]]; then
    fail "decide" "$tag $decision — scene preserved, pipeline stopped"
    exit 1
  fi
  if [[ "$decision" == MOVE:* ]]; then
    local mdir=${decision#MOVE:}
    job_move_aside "${mdir%%$'\t'*}"
  elif [[ "$decision" != "RUN" ]]; then
    fail "decide" "$tag unexpected decide verdict '$decision' — pipeline stopped"
    exit 1
  fi
  log "training $tag..."
  $PYTHON_B "$PKG_DIR/t42_train_driver.py" --config "$cfg" --ft "$ft" \
    --seed "$seed" --gpu 0 > "$LOGS/train_${tag}.log" 2>&1
  rc=$?
  tail -2 "$LOGS/train_${tag}.log"
  if [ $rc -ne 0 ]; then
    fail "t42" "$cfg $ft s$seed rc=$rc (queue continues)"
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
import t42_common as X   # first: fixes sys.path for the t39/t38 packages
import t38_common as C
out = {"started": C.time_str()}
assert len(X.TASKS) == 144 and len(set(X.TASKS)) == 144, "TASKS != 144 unique"
assert len(X.SMOKE) == 9 and len(X.QUEUE) == 135
assert len(X.TRIPLES) == 48 and sorted(X.TRIPLES) == \
    sorted(set((ds, h, s) for ds, h, s, _ in X.TASKS))
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
# EXP-003 archives: ALL 48 (ds,H,seed) dirs + LN48 pins vs archives
for (ds, h) in X.CONFIGS:
    assert os.path.isdir(C.exp003_job(ds, h)), \
        "EXP-003 seed-2021 archive missing: %s_%d" % (ds, h)
n_ln = 0
for (ds, h) in X.CONFIGS:
    for s in X.LN_KEYS_SEEDS:
        p = X.ln_success_path(ds, h, s)
        assert os.path.isfile(p), "LN SUCCESS missing: %s" % p
        mse, mae = C.parse_success_mse_mae(p)
        pin = X.LN48["%s_%d_seed%d" % (ds, h, s)]
        assert (mse, mae) == pin, "LN pin mismatch %s_%d_s%d" % (ds, h, s)
        n_ln += 1
out["exp003_archives_checked"] = 48
out["ln48_pins_checked"] = n_ln
# reference evidence: EXP-039 64 jsons + EXP-038 30 (+2 t37 reuse)
n39 = n38 = 0
for (ds, h) in X.CONFIGS:
    for arm in C.B_ARMS:
        for s in (2022, 2023):
            assert os.path.isfile(X.t39_train_json(ds, h, arm, s)), \
                "t39 json missing: %s_%d_%s_s%d" % (ds, h, arm, s)
            n39 += 1
        p = X.t38_train_json(ds, h, arm)
        assert os.path.isfile(p), "t38 json missing: %s_%d_%s" % (ds, h, arm)
        assert isinstance(C.load_json(p)["success"]["mse"], float)
        n38 += 1
out["t39_jsons"] = n39
out["t38_jsons"] = n38
assert n39 == 64 and n38 == 32
assert len(C.load_freeze16()) == 16, "Freeze16 incomplete"
# finetune-scope audit (writes scope_audit_t42.json + frozen_digest.json)
subprocess.check_call([sys.executable,
                       os.path.join(pkg, "t42_scope_audit.py")])
assert os.path.isfile(X.SCOPE_JSON)
assert os.path.isfile(X.FROZEN_DIGEST_JSON)
scope = C.load_json(X.SCOPE_JSON)
assert sorted(scope["per_ft"].keys()) == sorted(X.FT_TYPES)
# save_t42_fttypes state (first launch expects absent/empty; resume counts)
save = X.SAVE_BASE
if os.path.isdir(save):
    sub = [d for d in os.listdir(save) if d != ".failed"]
    n_succ = sum(1 for d in sub
                 if os.path.isfile(os.path.join(save, d, "SUCCESS.txt")))
    out["save_t42_fttypes"] = {"dirs": len(sub), "with_success": n_succ}
    print("RESUME STATE: %d job dirs, %d with SUCCESS.txt"
          % (len(sub), n_succ))
else:
    out["save_t42_fttypes"] = "absent (first launch)"
# GPU + disk (hard: >=100GB free; plus per-remaining-task estimate)
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
n_done = 0
if os.path.isdir(save):
    for (ds, h, s, ft) in X.TASKS:
        if os.path.isfile(os.path.join(X.job_dir(ds, h, s, ft),
                                       "SUCCESS.txt")):
            n_done += 1
n_remaining = 144 - n_done
need_gb = n_remaining * 0.6 + 20.0   # ckpt 0.42 + pred/true ~0.13 + margin
out["remaining_tasks"] = n_remaining
out["disk_need_gb"] = round(need_gb, 1)
assert du.free / 1024 ** 3 >= 100.0, \
    "less than 100 GB free (gate)"
assert du.free / 1024 ** 3 >= need_gb, \
    "disk %d GiB < estimated need %.1f GiB (%d remaining tasks)" % (
        du.free / 1024 ** 3, need_gb, n_remaining)
out["finished"] = C.time_str()
with open(os.path.join(X.LOG_DIR, "preflight.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("PREFLIGHT_OK free=%.1fGB disk=%.0fGB (need %.1f, remaining %d) "
      "exp003=48 ln48=48 t39=64 t38=32 tasks=144 scope=OK"
      % (out["gpu_free_gb"], out["disk_free_gb"], need_gb, n_remaining))
EOF
  rc=$?
  cat "$LOGS/preflight.log"
  return $rc
}

# ---------------------------------------------------------------- smoke
smoke() {
  clear_stage "t42"
  rm -f "$PKG_DIR/T42_OK"
  log "== smoke: ETTh1_96 x 3 seeds x {bias,mlp,attn} (9 of 144) =="
  env_snapshot "$PYTHON_B" t42
  preflight || { fail "preflight" "hard gate"; return 1; }
  local n=0
  while read -r cfg ft seed; do
    [ -z "$cfg" ] && continue
    run_one "$cfg" "$ft" "$seed" || true
    n=$((n + 1))
  done < <(job_list smoke)
  [ "$n" -eq 9 ] || { fail "smoke" "job count $n != 9"; return 1; }
  log "smoke verify (scope/init/frozen/truth/f64/re-entry gates, HARD)..."
  $PYTHON_B "$PKG_DIR/verify_t42_fullshot.py" --smoke \
    > "$LOGS/verify_smoke.log" 2>&1
  local rc=$?
  cat "$LOGS/verify_smoke.log"
  if [ $rc -ne 0 ]; then
    fail "smoke" "smoke verify FAIL — queue NOT started; preserve scene"
    return 1
  fi
  budget_from_logs
  log "smoke OK (9/9 + verify green) -> auto-continue via stage queue"
}

budget_from_logs() {
  $PYTHON_B - "$PKG_DIR" <<'EOF'
import os, sys
sys.path.insert(0, sys.argv[1])
import t42_common as X   # first: fixes sys.path for the t38 package
import t38_common as C
print("[budget] measured pure-train s/batch (train_wall - vali_wall):")
for ds, h in [("ETTh1", 96)]:
    for ft in X.FT_TYPES:
        for s in X.SEEDS:
            p = X.train_json(ds, h, s, ft)
            if not os.path.isfile(p):
                continue
            rec = C.load_json(p)
            ph = rec.get("phases", {})
            w, v = ph.get("train_wall"), ph.get("vali_wall")
            b = rec.get("sampler_len_check", {}).get("expected_train_batches")
            if w and v is not None and b:
                print("  %s_%d s%d %s: %.3f s/batch x %d batches"
                      % (ds, h, s, ft, (w - v) / b, b))
rem = sum(C.expected_train_batches(ds, h)
          for (ds, h, _, _) in X.QUEUE)
print("[budget] remaining queue train batches (approx): %d" % rem)
EOF
}

# ---------------------------------------------------------------- queue
queue() {
  log "== queue: remaining tasks (skip-if-done enforced per task) =="
  local n=0 n_skip=0 decision
  while read -r cfg ft seed; do
    [ -z "$cfg" ] && continue
    decision=$(job_decide "$cfg" "$ft" "$seed") || decision=""
    if [[ "$decision" == SKIP* ]]; then n_skip=$((n_skip + 1)); fi
    run_one "$cfg" "$ft" "$seed"   # hard-stops the pipeline on decide failure/ABORT
    n=$((n + 1))
  done < <(job_list all)
  [ "$n" -eq 144 ] || fail "queue" "job count $n != 144 (TASKS assert)"
  log "queue done: $n tasks considered, $n_skip skipped as complete"
}

verify_stage() {
  $PYTHON_B "$PKG_DIR/verify_t42_fullshot.py" > "$LOGS/verify_full.log" 2>&1
  local rc=$?
  cat "$LOGS/verify_full.log"
  [ $rc -ne 0 ] && fail "verify" "verify fullshot"
  return $rc
}

collect_stage() {
  $PYTHON_B "$PKG_DIR/collect_t42_fttypes.py" > "$LOGS/collect.log" 2>&1
  local rc=$?
  cat "$LOGS/collect.log"
  [ $rc -ne 0 ] && fail "collect" "collect"
  return $rc
}

pack() {
  local out="$REPO_DIR/temp/VisionTS_t42_fttypes_$(date +%Y%m%d).tar.gz"
  local list="$LOGS/pack_filelist.txt"
  mkdir -p "$REPO_DIR/temp"
  log "packing -> $out"
  ( cd "$REPO_DIR" && : > "$list" && \
    for f in t42_common.py t42_scope_audit.py t42_train_driver.py \
             verify_t42_fullshot.py collect_t42_fttypes.py run_t42_all.sh \
             README_t42.md; do
      echo "long_term_tsf/repro_t42_fttypes/$f" >> "$list"
    done
    # import-time deps (t42_common -> t39_common -> t38_common ->
    # t22_common) packed at their canonical relative paths
    for f in long_term_tsf/repro_t39_t22seeds/t39_common.py \
             long_term_tsf/repro_t38_t15t22/t38_common.py \
             long_term_tsf/repro_t22_freeze_ln/t22_common.py; do
      [ -f "$f" ] || { fail "pack" "dep missing: $f"; return 1; }
      echo "$f" >> "$list"
    done
    find long_term_tsf/repro_t42_fttypes/logs -maxdepth 1 \
         -name 'train_*_seed20*.json' >> "$list"
    ntr=$(grep -c 'train_.*_seed20.*\.json$' "$list")
    [ "$ntr" -eq 144 ] || { fail "pack" "train jsons in pack = $ntr (want 144)"; return 1; }
    for f in verify_t42.json verify_t42_smoke.json recompute_f64.json \
             recompute_f64_smoke.json reentry_check_t42.json \
             reentry_check_t42_smoke.json summary_t42.csv \
             RESULT_t42_fttypes.md fig_t42_fttypes.png preflight.json \
             scope_audit_t42.json frozen_digest.json; do
      [ -f "long_term_tsf/repro_t42_fttypes/logs/$f" ] && \
        echo "long_term_tsf/repro_t42_fttypes/logs/$f" >> "$list"
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
    # All/Rand reference evidence: t39 64 train jsons + t38 30 train jsons
    # + the EXP-037 reuse pair (ETTh1/H96/seed2021 via t38_train_json)
    n39=0
    for f in long_term_tsf/repro_t39_t22seeds/logs/train_*.json; do
      [ -f "$f" ] || continue
      echo "$f" >> "$list"; n39=$((n39 + 1))
    done
    [ "${n39:-0}" -eq 64 ] || { fail "pack" "t39 jsons = ${n39:-0} (want 64)"; exit 1; }
    n38=0
    for f in long_term_tsf/repro_t38_t15t22/logs/train_*.json; do
      [ -f "$f" ] || continue
      echo "$f" >> "$list"; n38=$((n38 + 1))
    done
    [ "$n38" -eq 30 ] || { fail "pack" "t38 train jsons = $n38 (want 30)"; exit 1; }
    for f in long_term_tsf/repro_t37_pilot/logs/train_all.json \
             long_term_tsf/repro_t37_pilot/logs/train_rand.json; do
      [ -f "$f" ] || { fail "pack" "t37 reuse evidence missing: $f"; exit 1; }
      echo "$f" >> "$list"
    done
    for f in long_term_tsf/repro_t22_freeze_ln/aggregate_snapshot.csv \
             long_term_tsf/repro_t22_freeze_ln/summary_t22.csv; do
      [ -f "$f" ] || { fail "pack" "baseline missing: $f"; exit 1; }
      echo "$f" >> "$list"
    done
    for f in long_term_tsf/repro_t42_fttypes/logs/train_*.log \
             long_term_tsf/repro_t42_fttypes/logs/preflight.log; do
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
  touch "$PKG_DIR/T42_OK"
  log "T42_OK"
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
