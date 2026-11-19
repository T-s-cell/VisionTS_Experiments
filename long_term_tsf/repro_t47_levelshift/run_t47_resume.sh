#!/bin/bash
# EXP-047 续跑(S1-b 容差修订 1e-3 后):重跑 VTS 门禁 + 补 moirai/sfdx/finalize 门禁,
# 复用全部冒烟推理 npz(不重推),然后全量推理 → 分析 → verify。
# 用法: bash run_t47_resume.sh <repo_root>
set -euo pipefail

REPO_ROOT=$1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_DIR="$SCRIPT_DIR/save_t47_levelshift"
VTS_PY=${VTS_PY:-/dev_data/wlt/conda/envs/visionts/bin/python}
T46_PY=${T46_PY:-/dev_data/wlt/conda/envs/t46diag/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
export HF_HOME=${HF_HOME:-/dev_data/wlt/hf_cache}
cd "$SCRIPT_DIR"

echo "== [R0] 冒烟 npz 完整性(复用,不重推)"
ls -la "$SAVE_DIR"/pred_vts_t47_smoke.npz "$SAVE_DIR"/pred_moirai_t47_smoke.npz \
       "$SAVE_DIR"/pred_autoets_t47_smoke.npz "$SAVE_DIR"/pred_autoarima_t47_smoke.npz \
       "$SAVE_DIR"/ledger_t47_smoke.csv

echo "== [R1] 冒烟门禁(vts 容差 1e-3 + moirai + sfdx + finalize)"
$VTS_PY smoke_t47.py "$REPO_ROOT" "$SAVE_DIR" --part vts
$T46_PY smoke_t47.py "$REPO_ROOT" "$SAVE_DIR" --part moirai
$T46_PY smoke_t47.py "$REPO_ROOT" "$SAVE_DIR" --part sfdx
$T46_PY smoke_t47.py "$REPO_ROOT" "$SAVE_DIR" --part finalize

echo "== [R2] 全量推理"
$VTS_PY infer_vts_t47.py "$REPO_ROOT" "$SAVE_DIR"
$T46_PY infer_moirai_t47.py "$REPO_ROOT" "$SAVE_DIR"
$T46_PY infer_sfdx_t47.py "$REPO_ROOT" "$SAVE_DIR" --workers 32 --timeout 1800

echo "== [R3] 指标 + 分析 + 图 + 报告数字"
$T46_PY collect_metrics_t47.py "$REPO_ROOT" "$SAVE_DIR"
$T46_PY analyze_t47.py "$REPO_ROOT" "$SAVE_DIR"
$T46_PY figures_t47.py "$REPO_ROOT" "$SAVE_DIR"
$T46_PY gen_report_numbers_t47.py "$REPO_ROOT" "$SAVE_DIR"

echo "== [R4] verify 门禁 G1–G10"
$T46_PY verify_t47.py "$REPO_ROOT" "$SAVE_DIR"
echo "EXP-047 RESUME ALL DONE"
