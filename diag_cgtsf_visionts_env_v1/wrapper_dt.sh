#!/usr/bin/env bash
# dt (WSL2, RTX 5060 Ti) wrapper for the VisionTS environment-drift
# diagnostic. Usage: bash wrapper_dt.sh <script.py> [args...]
# Env pinned before any CUDA init; uses the CLONED diag env (original
# torch_50_series env is preserved untouched).
set -euo pipefail
SUBDIR="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/miniconda3/envs/diag_vts_env/bin/python"

# WSL2: non-interactive PATH lacks the Windows-side nvidia-smi
export PATH="/usr/lib/wsl/lib:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=':4096:8'
export PYTHONHASHSEED=0

cd "$SUBDIR"
exec "$PY" "$@"
