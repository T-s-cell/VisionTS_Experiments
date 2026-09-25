#!/bin/bash
# 打 Figure 7 复现审计包（本地运行, 在 rsync 回传之后）。
# 只含代码/清单/日志/图表/报告等小文件, 不含 features npy 与 ImageNet 数据。
set -eu
REPO=/home/wlt/MMTS/VisionTS_Experiments
FIG7=$REPO/repro_fig7
OUT=/home/wlt/MMTS/temp
STAMP=$(date +%Y%m%d)
mkdir -p "$OUT"

cd "$REPO"
TAR="$OUT/VisionTS_fig7_${STAMP}.tar.gz"
tar -czf "$TAR" \
  --exclude='repro_fig7/outputs/features_*.npy' \
  --exclude='repro_fig7/__pycache__' \
  --exclude='repro_fig7/logs/.gpu_poll_*' \
  repro_fig7
echo "audit package -> $TAR"
ls -lh "$TAR"
tar -tzf "$TAR" | wc -l
