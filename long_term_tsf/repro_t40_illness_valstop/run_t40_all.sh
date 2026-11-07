#!/usr/bin/env bash
# EXP-040 runner: preflight -> audit -> smoke (two-round control) ->
# queue (11) -> verify -> collect -> pack.
#
# Usage: GPU_ID=0 PYTHON_B=/path/to/python bash run_t40_all.sh <stage>
# Stages:
#   preflight  GPU/disk/env/frozen-materials checks (no training)
#   audit      t40_audit.py (AG1-AG9) over the EXP-003 archives
#   smoke      ctrl round (A-only) + main round (A/B/C) on (24, 2021),
#              then verify --smoke as the HARD gate (TG2 trajectory
#              equality + RNG audit + 3 locks + lr assertions)
#   queue      remaining 11 tasks, skip-if-done, sequential
#   verify     full TG1-TG10 over 12 main tasks + ctrl
#   collect    summary_t40.csv + RESULT md + figure
#   pack       code + logs (NO ckpt/npy) -> temp-ready tar.gz (<50MB)
set -u
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_B="${PYTHON_B:-python}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

log() { echo "[$(date +%H:%M:%S)] $*"; }

task_name() { echo "illness_$1_seed$2"; }

job_done() {  # h seed -> 0 if the MAIN round is complete
    local jd
    jd="$(dirname "$PKG_DIR")/save_t40_valstop/$(task_name "$1" "$2")"
    [ -f "$jd/SUCCESS.txt" ] && [ -f "$jd/curves.npz" ] \
        && [ -f "$PKG_DIR/logs/train_$(task_name "$1" "$2")_main.json" ]
}

stage_preflight() {
    log "preflight: python $("$PYTHON_B" -V 2>&1)"
    (
        cd "$PKG_DIR" || exit 1
        "$PYTHON_B" - <<'EOF'
import os, sys
import torch, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
import t40_common as X, t38_common as C
csv_path = os.path.join(X.ROOT, X.CSV_REL)
ck_path = os.path.abspath(os.path.join(X.ROOT, "..", X.CKPT_REL))
assert os.path.isfile(csv_path), "missing illness csv: " + csv_path
assert C.sha256_of(csv_path) == X.CSV_SHA256, "csv sha drift"
assert os.path.isfile(ck_path), "missing frozen ckpt: " + ck_path
assert C.sha256_of(ck_path) == X.CKPT_SHA256, "ckpt sha drift"
assert len(X.TASKS) == 12 and X.QUEUE and X.SMOKE == (24, 2021)
for h, s in X.TASKS:
    assert os.path.isdir(X.exp003_job(h, s)), "missing archive %d %d" % (h, s)
    assert X.theta_log(h, s), "missing theta log %d %d" % (h, s)
print("preflight-checks OK: csv/ckpt sha pinned, 12 archives, 12 theta logs")
EOF
    )
    df -h "$(dirname "$PKG_DIR")" | tail -1
    local avail_gb
    avail_gb=$(df -BG --output=avail "$(dirname "$PKG_DIR")" | tail -1 | tr -dc '0-9')
    log "disk avail ${avail_gb}G (need >= 20G for lock copies)"
    [ "$avail_gb" -ge 20 ] || { log "PREFLIGHT FAIL: insufficient disk"; exit 1; }
    log "preflight OK"
}

stage_audit() {
    "$PYTHON_B" "$PKG_DIR/t40_audit.py" --gpu 0 || exit 1
}

stage_smoke() {
    log "smoke round 1/2: ctrl (A-only) on (24, 2021)"
    "$PYTHON_B" "$PKG_DIR/t40_train_driver.py" --h 24 --seed 2021 \
        --round ctrl --gpu 0 || exit 1
    log "smoke round 2/2: main (A/B,C) on (24, 2021)"
    "$PYTHON_B" "$PKG_DIR/t40_train_driver.py" --h 24 --seed 2021 \
        --round main --gpu 0 || exit 1
    log "smoke HARD gate: verify --smoke (TG1/TG2/TG6-TG10)"
    "$PYTHON_B" "$PKG_DIR/verify_t40_fullshot.py" --smoke --gpu 0 || exit 1
    log "smoke PASS: two-round trajectory equality + 3 locks verified"
}

stage_queue() {
    local n=0
    for h in 24 36 48 60; do
        for s in 2021 2022 2023; do
            { [ "$h" == "24" ] && [ "$s" == "2021" ]; } && continue
            n=$((n+1))
            if job_done "$h" "$s"; then
                log "queue %2d/11 SKIP (done): %s" "$n" "$(task_name "$h" "$s")"
                continue
            fi
            log "queue %2d/11 RUN: %s" "$n" "$(task_name "$h" "$s")"
            "$PYTHON_B" "$PKG_DIR/t40_train_driver.py" --h "$h" --seed "$s" \
                --round main --gpu 0 || exit 1
        done
    done
    log "queue complete"
}

stage_verify() {
    "$PYTHON_B" "$PKG_DIR/verify_t40_fullshot.py" --gpu 0 || exit 1
}

stage_collect() {
    "$PYTHON_B" "$PKG_DIR/collect_t40.py" || exit 1
}

stage_pack() {
    local OUT="temp_t40_pack"
    rm -rf "$OUT" && mkdir -p "$OUT/repro_t40_illness_valstop/logs"
    cp "$PKG_DIR"/*.py "$PKG_DIR/run_t40_all.sh" "$PKG_DIR"/README_t40.md \
       "$OUT/repro_t40_illness_valstop/"
    cp "$PKG_DIR"/logs/*.json "$PKG_DIR"/logs/*.csv \
       "$PKG_DIR"/logs/*.md "$PKG_DIR"/logs/*.png \
       "$OUT/repro_t40_illness_valstop/logs/" 2>/dev/null
    # curves.npz + small per-job jsons (NO ckpt, NO npy arrays);
    # the smoke ctrl round (A-only) job_records are included too — they
    # are the TG2 same-environment trajectory-equivalence evidence
    mkdir -p "$OUT/repro_t40_illness_valstop/job_records"
    local jd
    for h in 24 36 48 60; do
        for s in 2021 2022 2023; do
            jd="$(dirname "$PKG_DIR")/save_t40_valstop/$(task_name "$h" "$s")"
            [ -d "$jd" ] || continue
            mkdir -p "$OUT/repro_t40_illness_valstop/job_records/$(task_name "$h" "$s")"
            cp "$jd"/curves.npz "$jd"/trajectory.json "$jd"/phases.json \
               "$jd"/lock_events.json "$jd"/rng_audit.json "$jd"/args.json \
               "$jd"/fingerprint.json "$jd"/SUCCESS.txt \
               "$OUT/repro_t40_illness_valstop/job_records/$(task_name "$h" "$s")/" 2>/dev/null
        done
    done
    local cd_="$(dirname "$PKG_DIR")/save_t40_valstop_ctrl/illness_24_seed2021"
    if [ -d "$cd_" ]; then
        mkdir -p "$OUT/repro_t40_illness_valstop/job_records_ctrl/illness_24_seed2021"
        cp "$cd_"/curves.npz "$cd_"/trajectory.json "$cd_"/phases.json \
           "$cd_"/lock_events.json "$cd_"/rng_audit.json "$cd_"/args.json \
           "$cd_"/fingerprint.json "$cd_"/SUCCESS.txt \
           "$OUT/repro_t40_illness_valstop/job_records_ctrl/illness_24_seed2021/" 2>/dev/null
    fi
    local tarball
    tarball="${OUT}.tar.gz"
    tar -czf "$tarball" "$OUT"
    local sz
    sz=$(stat -c%s "$tarball")
    log "pack: $tarball ($sz bytes)"
    [ "$sz" -lt $((50*1024*1024)) ] || { log "PACK FAIL: >50MB"; exit 1; }
    sha256sum "$tarball" > "$tarball.sha256"
    log "pack OK + sidecar"
}

case "${1:-}" in
    preflight) stage_preflight ;;
    audit)     stage_audit ;;
    smoke)     stage_smoke ;;
    queue)     stage_queue ;;
    verify)    stage_verify ;;
    collect)   stage_collect ;;
    pack)      stage_pack ;;
    *) echo "usage: $0 {preflight|audit|smoke|queue|verify|collect|pack}"; exit 2 ;;
esac
