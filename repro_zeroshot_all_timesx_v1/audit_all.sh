#!/usr/bin/env bash
# Audit packer. Usage: bash audit_all.sh --preexec | --final
# Packs to /home/wlt/MMTS/temp/VisionTS_zeroshot_all_timesx_v1_{preexec|final}_audit_<ts>.tar.gz
# and asserts size < 50 MB. Includes full code, frozen protocol + complete input
# arrays (amendment 2026-10-10), provenance and results. Excludes weights/zip.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
TEMP_DIR=/home/wlt/MMTS/temp
MODE="$1"
TS=$(date +%Y%m%d_%H%M%S)
STAGE="$([ "$MODE" = "--preexec" ] && echo preexec || echo final)"
OUT="$TEMP_DIR/VisionTS_zeroshot_all_timesx_v1_${STAGE}_audit_${TS}.tar.gz"
mkdir -p "$TEMP_DIR"

MEMBERS=(README.md RUN_COMMANDS.md protocol.py common.py freeze_full.py verify_cache.py
  preflight.py infer_fill.py assemble.py evaluate.py recompute_check.py verify_acceptance.py
  stage_guard.py make_report.py run_all.sh audit_all.sh reuse_plan.json preflight_sample.json
  preflight_report.json reuse_final.json frozen results logs)

if [ "$MODE" != "--preexec" ] && [ "$MODE" != "--final" ]; then
  echo "usage: bash audit_all.sh --preexec|--final" >&2; exit 2
fi

in_members() {
  local f
  for f in "${MEMBERS[@]}"; do
    [ "$f" = "$1" ] && return 0
  done
  return 1
}

# final pack additionally carries cache_all (new inference npz + manifest)
[ "$MODE" = "--final" ] && MEMBERS+=(cache_all)
# final pack MUST contain the preflight verdicts and the new cache
if [ "$MODE" = "--final" ]; then
  for f in preflight_report.json reuse_final.json cache_all; do
    in_members "$f" || { echo "$f not in MEMBERS" >&2; exit 1; }
  done
fi
# preexec must not require results/; final requires acceptance
if [ "$MODE" = "--preexec" ]; then
  for f in frozen/inventory.json frozen/vars.json frozen/frozen_arrays.npz reuse_plan.json preflight_sample.json; do
    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
  done
else
  for f in results/preds.jsonl.gz results/acceptance.json results/recompute_check.json \
           preflight_report.json reuse_final.json cache_all/manifest.json; do
    [ -f "$f" ] || { echo "missing $f — run the pipeline first" >&2; exit 1; }
  done
fi

rm -rf /tmp/vz_pack && mkdir -p /tmp/vz_pack
for m in "${MEMBERS[@]}"; do
  if [ -e "$m" ]; then
    cp -r "$m" /tmp/vz_pack/
  elif [ "$MODE" = "--final" ] && { [ "$m" = "preflight_report.json" ] || [ "$m" = "reuse_final.json" ] || [ "$m" = "cache_all" ]; }; then
    echo "missing required final-pack member: $m — run the pipeline first" >&2; exit 1
  fi
done
# drop bulky or irrelevant logs from packs (keep .gz/.json/.txt evidence)
find /tmp/vz_pack/logs -type f ! -name '*.txt' ! -name '*.json' ! -name '*.jsonl.gz' ! -name '*.log' -delete 2>/dev/null || true

( cd /tmp/vz_pack && find . -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 sha256sum > SHA256SUMS )
PACK_INFO=$(mktemp)
{
  echo "mode: $STAGE"
  echo "created: $(date -Is)"
  echo "host: $(uname -a)"
  echo "git: $(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo 'no-git') ($(git -C "$HERE" status --porcelain | wc -l) dirty files)"
  echo "protocol_version: TimesX-all-native-P1-v1"
} > "$PACK_INFO"
cp "$PACK_INFO" /tmp/vz_pack/PACK_INFO.txt

tar -C /tmp/vz_pack -czf "$OUT" .
SIZE=$(stat -c %s "$OUT")
if [ "$SIZE" -ge $((50 * 1024 * 1024)) ]; then
  echo "ERROR: $OUT is $SIZE bytes (>=50MB)" >&2; exit 1
fi
rm -rf /tmp/vz_pack "$PACK_INFO"
echo "pack: $OUT ($((SIZE / 1024)) KB)"
sha256sum "$OUT"
