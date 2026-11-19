#!/bin/bash
# EXP-047 全流程(eta):预检 → 冻结链(select/build)→ 冒烟门禁 → 全量推理 → 分析 → 门禁。
# 冒烟任何 FAIL → set -e 中止,不续全量。
# 用法: bash run_t47_all.sh <repo_root>          # repo_root = <VisionTS_Experiments>/long_term_tsf
# 环境变量: VTS_PY / T46_PY / GPU_ID
set -euo pipefail

REPO_ROOT=$1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_DIR="$SCRIPT_DIR/save_t47_levelshift"
VTS_PY=${VTS_PY:-/dev_data/wlt/conda/envs/visionts/bin/python}
T46_PY=${T46_PY:-/dev_data/wlt/conda/envs/t46diag/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
export HF_HOME=${HF_HOME:-/dev_data/wlt/hf_cache}   # moirai-1.0-R-base 缓存(同 t46 约定)
cd "$SCRIPT_DIR"

echo "== [0/7] 冻结链(select + build;sha 必须与本地冻结值一致)"
$T46_PY select_t47.py
$T46_PY build_variants_t47.py "$REPO_ROOT" "$SAVE_DIR"
python3 - "$SAVE_DIR" <<'EOF'
import hashlib, sys
save = sys.argv[1]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
exp = {
    'baseline_list_t47.csv':
        '5c750c645af06fbb3b7912f10ed88bd7f3082c60b5996e1c5d820362d6a7814d',
    'manifest_t47.csv':
        '94aec758b2e490baa31d4fed96828a576e113be4dc3bac6069c928c5231a99ed',
    'baseline_ETTh1.npz':
        'd19478024b5404bd783344fb52f0b882753e59ab120db76dbdf4918e3dd277eb',
    'baseline_ETTh2.npz':
        '081b00981106f51221785145de58a2f5d355abf17101fa8d530e090f123bfbd6',
    'baseline_ETTm1.npz':
        '420d622a171d58c69c331dbc708758b77a39c682383a27a3da34e4f54daa21d1',
    'baseline_ETTm2.npz':
        '5aaacd20b404c96b1faddd57dbc2eed751e961fe094d62a3b900fe2e93b4c20f',
}
got = {
    'baseline_list_t47.csv': sha('baseline_list_t47.csv'),
    'manifest_t47.csv': sha('manifest_t47.csv'),
}
for ds in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
    got['baseline_%s.npz' % ds] = sha(save + '/baseline_%s.npz' % ds)
for k, v in exp.items():
    assert got[k] == v, (k, got[k][:16], v[:16])
print('frozen chain shas OK (6/6)')
EOF

echo "== [1/7] 预检(vts / t46diag 两 env)"
$VTS_PY preflight_t47.py "$REPO_ROOT" "$SAVE_DIR" --env vts
$T46_PY preflight_t47.py "$REPO_ROOT" "$SAVE_DIR" --env t46diag

echo "== [2/7] sfdx selftest(超时真杀 + 池存活)"
$T46_PY infer_sfdx_t47.py "$REPO_ROOT" "$SAVE_DIR" --selftest

echo "== [3/7] 冒烟推理(smoke npz:三方法)"
$VTS_PY infer_vts_t47.py "$REPO_ROOT" "$SAVE_DIR" --smoke
$T46_PY infer_moirai_t47.py "$REPO_ROOT" "$SAVE_DIR" --smoke
$T46_PY infer_sfdx_t47.py "$REPO_ROOT" "$SAVE_DIR" --smoke --workers 8 --timeout 1800

echo "== [4/7] 冒烟门禁(S1 双层复现 / S2 δ=0 实推 / S3 恒等 / S4 对齐)"
$VTS_PY smoke_t47.py "$REPO_ROOT" "$SAVE_DIR" --part vts
$T46_PY smoke_t47.py "$REPO_ROOT" "$SAVE_DIR" --part moirai
$T46_PY smoke_t47.py "$REPO_ROOT" "$SAVE_DIR" --part sfdx
$T46_PY smoke_t47.py "$REPO_ROOT" "$SAVE_DIR" --part finalize   # FAIL → 非零退出

echo "== [5/7] 全量推理"
$VTS_PY infer_vts_t47.py "$REPO_ROOT" "$SAVE_DIR"
$T46_PY infer_moirai_t47.py "$REPO_ROOT" "$SAVE_DIR"
$T46_PY infer_sfdx_t47.py "$REPO_ROOT" "$SAVE_DIR" --workers 32 --timeout 1800

echo "== [6/7] 指标 + 分析 + 图 + 报告数字"
$T46_PY collect_metrics_t47.py "$REPO_ROOT" "$SAVE_DIR"
$T46_PY analyze_t47.py "$REPO_ROOT" "$SAVE_DIR"
$T46_PY figures_t47.py "$REPO_ROOT" "$SAVE_DIR"
$T46_PY gen_report_numbers_t47.py "$REPO_ROOT" "$SAVE_DIR"

echo "== [7/7] verify 门禁 G1–G10"
$T46_PY verify_t47.py "$REPO_ROOT" "$SAVE_DIR"
echo "EXP-047 ALL DONE"
