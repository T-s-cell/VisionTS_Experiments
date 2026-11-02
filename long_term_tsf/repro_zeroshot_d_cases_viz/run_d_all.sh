#!/usr/bin/env bash
# EXP-032 driver: env snapshot -> precheck -> select_windows -> replay_figures -> report
# Usage (eta): PYTHON=/dev_data/wlt/conda/envs/visionts/bin/python bash repro_zeroshot_d_cases_viz/run_d_all.sh
set -uo pipefail
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."   # long_term_tsf/
PKG=repro_zeroshot_d_cases_viz
mkdir -p "$PKG/logs"

fail() { echo "[run_d_all] FAIL: $*" | tee "$PKG/logs/FAILED.txt"; exit 1; }

{
  echo "== EXP-032 env snapshot $(date -Is) =="
  $PYTHON - <<'PYEOF'
import sys
import numpy, pandas, matplotlib
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
except Exception as e:
    print("torch import failed:", e)
print("numpy", numpy.__version__, "pandas", pandas.__version__,
      "matplotlib", matplotlib.__version__)
PYEOF
  (nvidia-smi | head -12) 2>&1 || true
} > "$PKG/logs/env_snapshot.txt" 2>&1
cat "$PKG/logs/env_snapshot.txt"

$PYTHON - <<'PYEOF' || fail "precheck"
import os, sys
sys.path.insert(0, "repro_zeroshot_d_cases_viz")
import d_common as D
ck = os.path.join(D.ROOT, "..", "ckpt", "mae_visualize_vit_base.pth")
assert os.path.isfile(ck), ck
assert os.path.getsize(ck) == 447725789, os.path.getsize(ck)
for m, p in D.win_npz_paths().items():
    assert os.path.isfile(p), p
for m, p in D.frozen_arrays_paths().items():
    for f in ("pred.npy", "true.npy"):
        assert os.path.isfile(os.path.join(p, f)), (m, f)
print("[precheck] ckpt sha256", D.sha256_of(ck))
print("[precheck] OK: ckpt + 3 win npz + 3 method pred/true arrays on disk")
PYEOF

$PYTHON "$PKG/select_windows_d.py"  || fail "select_windows_d"
$PYTHON "$PKG/replay_figures_d.py"  || fail "replay_figures_d"
$PYTHON "$PKG/report_d.py"          || fail "report_d"

echo "[run_d_all] ALL DONE (selection gated on frozen table; gates a<=1e-6 b<=1e-4)"
