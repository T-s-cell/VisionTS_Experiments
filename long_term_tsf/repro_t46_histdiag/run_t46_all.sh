#!/bin/bash
# EXP-046 全流程(eta)。用法:
#   HF_HOME=/dev_data/wlt/hf_cache WORKERS=8 TIMEOUT=300 bash run_t46_all.sh <repo_root> [save_dir]
# 顺序:select → preflight → sfdx selftest(超时真杀验证)→ infer_vts(全量核验)
#   → 冒烟(moirai/sfdx 各 16 样本 + 指标汇总)→ 门禁过后自动续全量 1792
#   → features → metrics → analyze → figures → verify(六门禁)。
set -euo pipefail
REPO=${1:?usage: run_t46_all.sh <repo_root> [save_dir]}
SAVE=${2:-save_t46_histdiag}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
: "${HF_HOME:?HF_HOME must point to the HF cache containing moirai-1.0-R-base}"
WORKERS=${WORKERS:-8}
TIMEOUT=${TIMEOUT:-300}

echo "=== [1/12] select(冻结清单)==="
python select_t46.py .
echo "=== [2/12] preflight(版本/签名/权重/数据)==="
python preflight_t46.py "$REPO" "$SAVE"
echo "=== [3/12] sfdx selftest(超时真杀 + 管线存活)==="
python infer_sfdx_t46.py "$REPO" "$SAVE" --selftest
echo "=== [4/12] infer_vts(切片复用 + 全 1792 真值核验)==="
python infer_vts_t46.py "$REPO" "$SAVE"
echo "=== [5/12] smoke: moirai(ETTh1/ETTm1 各 8)==="
python infer_moirai_t46.py "$REPO" "$SAVE" --smoke
echo "=== [6/12] smoke: sfdx AutoETS/AutoARIMA ==="
python infer_sfdx_t46.py "$REPO" "$SAVE" --smoke --workers "$WORKERS" --timeout "$TIMEOUT"
echo "=== [7/12] smoke 指标汇总(对齐/真值/有限性)==="
python collect_metrics_t46.py "$REPO" "$SAVE" --smoke
echo "=== 冒烟门禁通过,自动续全量 1792 ==="
echo "=== [8/12] full: moirai ==="
python infer_moirai_t46.py "$REPO" "$SAVE"
echo "=== [9/12] full: sfdx AutoETS/AutoARIMA ==="
python infer_sfdx_t46.py "$REPO" "$SAVE" --workers "$WORKERS" --timeout "$TIMEOUT"
echo "=== [10/12] features + 全量指标 ==="
python build_features_t46.py "$REPO"
python collect_metrics_t46.py "$REPO" "$SAVE"
echo "=== [11/12] analyze + figures ==="
python analyze_t46.py "$REPO" "$SAVE"
python figures_t46.py "$REPO" "$SAVE"
echo "=== [12/12] verify 六门禁 ==="
python verify_t46.py "$REPO" "$SAVE"
echo "=== EXP-046 pipeline done ==="
