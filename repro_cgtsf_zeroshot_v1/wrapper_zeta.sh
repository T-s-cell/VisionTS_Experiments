#!/usr/bin/env bash
# zeta wrapper for the CGTSF zero-shot round (wlt account).
# Usage: bash wrapper_zeta.sh <script.py> [args...]
# Env pinned before any CUDA init; single-modality round needs no LLM key.
set -euo pipefail
SUBDIR="$(cd "$(dirname "$0")" && pwd)"
PY=/dev_data/wlt/conda/envs/visionts/bin/python

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUBLAS_WORKSPACE_CONFIG=':4096:8'
export PYTHONHASHSEED=0

cd "$SUBDIR"
exec "$PY" "$@"
