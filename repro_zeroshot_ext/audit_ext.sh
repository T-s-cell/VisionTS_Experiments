#!/usr/bin/env bash
# Audit package + isolation verification for repro_zeroshot_ext.
#  1) verify the 12 fingerprint-locked files still match pre_snapshot.txt
#  2) verify git status only grew by repro_zeroshot_ext entries
#  3) tar code+logs+results+splits+selection (+cache) into /home/wlt/MMTS/temp/
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
SNAP="$HERE/results/pre_snapshot.txt"
cd "$REPO"
FAIL=0

echo "== 1) locked files md5 vs snapshot =="
sed -n '/^## locked files md5/,$p' "$SNAP" | grep -E '^[0-9a-f]{32} ' > /tmp/ext_locked_now.txt || true
md5sum repro_fullshot/run_train.py repro_fullshot/exp_univariate.py \
       long_term_tsf/exp/exp_long_term_forecasting.py long_term_tsf/exp/exp_basic.py \
       long_term_tsf/models/VisionTS.py long_term_tsf/data_provider/data_factory.py \
       long_term_tsf/data_provider/data_loader.py long_term_tsf/utils/tools.py \
       long_term_tsf/utils/metrics.py visionts/model.py visionts/models_mae.py \
       visionts/util.py > /tmp/ext_locked_check.txt
if diff -u /tmp/ext_locked_now.txt /tmp/ext_locked_check.txt; then
  echo "locked md5: OK (12 files unchanged)"
else
  echo "locked md5: MISMATCH"; FAIL=1
fi

echo "== 2) git status isolation =="
echo "-- tracked modifications (must be empty):"
TRACKED=$(git status --porcelain | grep -vE '^\?\?' || true)
echo "$TRACKED"
[ -z "$TRACKED" ] || { echo "git status: tracked files modified"; FAIL=1; }
echo "-- repro_fullshot/ still untracked & present:"
git status --porcelain | grep -q '^?? repro_fullshot/' || { echo "repro_fullshot/ entry changed"; FAIL=1; }
echo "-- untracked entries outside repro_zeroshot_ext/repro_fullshot (must be empty):"
EXTRA=$(git status --porcelain -uall | grep '^??' | grep -vE '^\?\? (repro_zeroshot_ext/|repro_fullshot/)' || true)
echo "$EXTRA"
[ -z "$EXTRA" ] || { echo "git status: unexpected untracked additions"; FAIL=1; }

echo "== 3) package =="
OUT_DIR="/home/wlt/MMTS/temp"
mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$OUT_DIR/VisionTS_zeroshot_ext_$TS.tar.gz"
tar -czf "$OUT" \
  --exclude='__pycache__' --exclude='repro_zeroshot_ext/dataset' \
  -C "$REPO" \
  repro_zeroshot_ext/*.py repro_zeroshot_ext/*.sh repro_zeroshot_ext/README.md \
  repro_zeroshot_ext/splits.json \
  repro_zeroshot_ext/P_selection.json repro_zeroshot_ext/P_selection_run1.json \
  repro_zeroshot_ext/results repro_zeroshot_ext/results_run1_preserved \
  repro_zeroshot_ext/logs \
  repro_zeroshot_ext/cache repro_zeroshot_ext/cache_run1_preserved \
  repro_zeroshot_ext/cache_cpu_check \
  repro_zeroshot_ext/featvis/*.py repro_zeroshot_ext/featvis/*.sh \
  repro_zeroshot_ext/featvis/README.md \
  repro_zeroshot_ext/featvis/selection repro_zeroshot_ext/featvis/lists \
  repro_zeroshot_ext/featvis/features repro_zeroshot_ext/featvis/tsne \
  repro_zeroshot_ext/featvis/figures repro_zeroshot_ext/featvis/highdim \
  repro_zeroshot_ext/featvis/logs \
  repro_zeroshot_ext/featvis_ltfs_style/*.py repro_zeroshot_ext/featvis_ltfs_style/*.sh \
  repro_zeroshot_ext/featvis_ltfs_style/README.md \
  repro_zeroshot_ext/featvis_ltfs_style/REPORT.md \
  repro_zeroshot_ext/featvis_ltfs_style/lists \
  repro_zeroshot_ext/featvis_ltfs_style/features \
  repro_zeroshot_ext/featvis_ltfs_style/tsne \
  repro_zeroshot_ext/featvis_ltfs_style/figures \
  repro_zeroshot_ext/featvis_ltfs_style/logs
ls -l "$OUT"
echo "audit package -> $OUT"
[ "$FAIL" -eq 0 ] && echo "AUDIT OK" || { echo "AUDIT FAILED"; exit 1; }
