#!/usr/bin/env bash
# EXP-041 runner: preflight -> smoke (end-to-end mini run, HARD gate) ->
# queue (full 300 + 3 fits) -> verify (FG1-FG7) -> collect -> pack.
#
# Usage: GPU_ID=0 \
#        PYTHON_B=/dev_data/wlt/conda/envs/visionts/bin/python \
#        PYTHON_A=/dev_data/wlt/conda/envs/visionts_t37author/bin/python \
#        bash run_t41_all.sh <stage>
# sample_monash.py runs in the AUTHOR env (gluonts cache provenance,
# HOME=/dev_data/wlt/t37_author_home); everything else in the visionts env.
set -u
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_B="${PYTHON_B:-python}"
PYTHON_A="${PYTHON_A:-$PYTHON_B}"
GPU_ID="${GPU_ID:-0}"
AUTHOR_HOME="${AUTHOR_HOME:-/dev_data/wlt/t37_author_home}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

log() { echo "[$(date +%H:%M:%S)] $*"; }

stage_preflight() {
    log "preflight: visionts env $("$PYTHON_B" -V 2>&1), author env $("$PYTHON_A" -V 2>&1)"
    "$PYTHON_B" - <<'EOF' || exit 1
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("t41_common.py")), "repro_t41_fig7_monash") if False else ".")
import t41_common as X, t38_common as T38
import torch, numpy, sklearn
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
assert sklearn.__version__.startswith(X.SKLEARN_PIN_PREFIX), \
    "sklearn %s != %s* (t-SNE determinism pin)" % (sklearn.__version__, X.SKLEARN_PIN_PREFIX)
fp = X.frozen_features_path()
assert os.path.isfile(fp), "missing frozen features: " + fp
sha = X.sha256_of(fp)
assert sha == X.FROZEN_FEATURES_SHA256, "frozen features sha drift: " + sha
print("frozen features sha OK")
for grp in ("ImageNet", "ETTm1", "Weather", "Electricity"):
    assert os.path.isfile(X.sampling_list_path(grp)), "missing list " + grp
assert os.path.isfile(X.mask_ref_path()), "missing mask_ref.json"
import json
ref = X.load_json(X.mask_ref_path())
ids = ref["ids_keep_order"]
assert len(ids) == 70 and all(i % 14 < 5 for i in ids)
for ds in ("fred_md", "nn5_daily_without_missing", "m1_monthly"):
    assert os.path.isfile(X.monash_json(ds)), "missing t38 archive " + ds
print("preflight-checks OK (env/sklearn pin/frozen sha/lists/mask_ref/t38 archives)")
EOF
    "$PYTHON_B" - "$PKG_DIR" <<'EOF' || exit 1
import json, os, sys
sys.path.insert(0, sys.argv[1])
import t41_common as X
rec = {"sampling_list_sha256": {g: X.sha256_of(X.sampling_list_path(g))
                                for g in ("ImageNet", "ETTm1", "Weather",
                                          "Electricity")},
       "recorded": X.time_str()}
X.dump_json(X.SMOKE_JSON, rec)
print("preflight sampling-list shas recorded")
EOF
    df -h "$(dirname "$PKG_DIR")" | tail -1
    log "preflight OK"
}

stage_smoke() {
    # 3 rows x <=2 samples, end-to-end: sample -> extract -> 1939-row fit
    # (rs2021, no stability) -> NN sanity.  HARD gate via verify --smoke.
    log "smoke 1/4: sample (3 rows x 2)"
    HOME="$AUTHOR_HOME" "$PYTHON_A" "$PKG_DIR/sample_monash.py" --smoke 2 \
        || exit 1
    log "smoke 2/4: extract"
    "$PYTHON_B" "$PKG_DIR/extract_features_monash.py" || exit 1
    log "smoke 3/4: joint fit (rs2021 only, recheck)"
    "$PYTHON_B" "$PKG_DIR/run_tsne_joint.py" --rs 2021
    log "smoke 4/4: nn + verify --smoke"
    "$PYTHON_B" "$PKG_DIR/nn_metric_t41.py" || exit 1
    "$PYTHON_B" "$PKG_DIR/verify_t41.py" --smoke || exit 1
    log "smoke PASS"
}

stage_queue() {
    log "queue 1/4: full sampling (300)"
    rm -f "$PKG_DIR/../../save_t41_monash/tsne_coords_2200x2_rs2022.npy" \
          "$PKG_DIR/../../save_t41_monash/tsne_coords_2200x2_rs2023.npy"
    HOME="$AUTHOR_HOME" "$PYTHON_A" "$PKG_DIR/sample_monash.py" || exit 1
    log "queue 2/4: full extraction (300)"
    "$PYTHON_B" "$PKG_DIR/extract_features_monash.py" || exit 1
    log "queue 3/4: joint t-SNE (rs2021 + recheck + rs2022 + rs2023)"
    "$PYTHON_B" "$PKG_DIR/run_tsne_joint.py" || exit 1
    log "queue 4/4: 30-NN"
    "$PYTHON_B" "$PKG_DIR/nn_metric_t41.py" || exit 1
    log "queue complete"
}

stage_verify() {
    "$PYTHON_B" "$PKG_DIR/verify_t41.py" || exit 1
}

stage_collect() {
    "$PYTHON_B" "$PKG_DIR/fig7_plots.py" || exit 1
    "$PYTHON_B" "$PKG_DIR/collect_t41.py" || exit 1
}

stage_pack() {
    local OUT="temp_t41_pack"
    rm -rf "$OUT" && mkdir -p "$OUT/repro_t41_fig7_monash/logs"
    cp "$PKG_DIR"/*.py "$PKG_DIR/run_t41_all.sh" "$PKG_DIR/README_t41.md" \
       "$OUT/repro_t41_fig7_monash/" 2>/dev/null
    cp "$PKG_DIR"/logs/*.json "$PKG_DIR"/logs/*.csv \
       "$PKG_DIR"/logs/*.md "$PKG_DIR"/logs/*.png \
       "$OUT/repro_t41_fig7_monash/logs/" 2>/dev/null
    mkdir -p "$OUT/repro_t41_fig7_monash/job_records"
    local SB="$(dirname "$PKG_DIR")/save_t41_monash"
    cp "$SB"/monash_manifest.json "$SB"/extract_env.json \
       "$SB"/tsne_params.json "$SB"/nn_summary.json "$SB"/nn_table.csv \
       "$SB"/input_hashes_monash.json "$SB"/groups_monash.npy \
       "$OUT/repro_t41_fig7_monash/job_records/" 2>/dev/null
    # coords npy are small (~35 KB each) and ARE included, together with
    # nn_neighbors.npz (archived neighbour indices -> the 30-NN table can
    # be recomputed independently from the pack)
    cp "$SB"/tsne_coords_2200x2*.npy "$SB"/nn_neighbors.npz \
       "$OUT/repro_t41_fig7_monash/job_records/" 2>/dev/null
    # v3: review follow-up probe artifacts (json is small; nn_neighbors_b1
    # allows independent recomputation of the B=1 30-NN table)
    cp "$SB"/probe_batch_diff.json "$SB"/nn_table_b1.csv \
       "$SB"/nn_neighbors_b1.npz \
       "$OUT/repro_t41_fig7_monash/job_records/" 2>/dev/null
    # features npy / windows npz stay on eta (NOT packed)
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
    smoke)     stage_smoke ;;
    queue)     stage_queue ;;
    verify)    stage_verify ;;
    collect)   stage_collect ;;
    pack)      stage_pack ;;
    *) echo "usage: $0 {preflight|smoke|queue|verify|collect|pack}"; exit 2 ;;
esac
