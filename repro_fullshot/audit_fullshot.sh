#!/usr/bin/env bash
# Build the small audit package for the LN full-shot reproduction.
#
# Includes: repro_fullshot code + logs + check reports + outputs/,
#           save_fullshot/ small artifacts (args.json, fingerprint.json,
#           SUCCESS.txt, valid_loss.json, metrics.npy).
# Excludes: checkpoints/ (checkpoint.pth), pred.npy, true.npy, __pycache__.
# Run on the LOCAL machine after artifacts are rsynced back.
set -eu

BASE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$BASE/.." && pwd)
OUT_DIR=${1:-/home/wlt/MMTS/temp}
OUT="$OUT_DIR/VisionTS_fullshot_$(date +%Y%m%d_%H%M%S).tar.gz"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

rsync -a \
  --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'pred.npy' --exclude 'true.npy' \
  --exclude 'checkpoint.pth' \
  "$ROOT/repro_fullshot" "$STAGE/"

if [ -d "$ROOT/save_fullshot" ]; then
  rsync -a \
    --exclude '__pycache__/' \
    --exclude 'pred.npy' --exclude 'true.npy' \
    --exclude 'checkpoint.pth' \
    "$ROOT/save_fullshot" "$STAGE/"
else
  echo "WARN: $ROOT/save_fullshot not found locally; packaging code/logs only" >&2
fi

# Supplement (2026-10-01 audit request): per-server source snapshots, launch
# scripts, env records, fingerprint verification, paper.
if [ -d "$ROOT/audit_supplement" ]; then
  rsync -a \
    --exclude '__pycache__/' --exclude '*.pyc' \
    "$ROOT/audit_supplement" "$STAGE/"
fi

echo "--- staged contents ---"
du -sh "$STAGE"/*
find "$STAGE" -type f | wc -l | xargs echo 'files:'

mkdir -p "$OUT_DIR"
tar -C "$STAGE" -czf "$OUT" .
echo "audit package -> $OUT ($(du -sh "$OUT" | cut -f1))"
