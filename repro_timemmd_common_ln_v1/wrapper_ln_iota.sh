#!/usr/bin/env bash
# iota wrapper: deterministic + offline env, GPU0 only, conda env
# visionts_mmd (clone of aurora + timm; aurora itself is never modified).
# Usage: bash repro_timemmd_common_ln_v1/wrapper_ln_iota.sh <cmd...>
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source /dev_data/anaconda3/etc/profile.d/conda.sh
conda activate visionts_mmd
cd "$(dirname "$0")/.."
exec "$@"
