#!/bin/bash
# EXP-043 acceptance round (2026-11-12): true cross-path check + f64 reuse
# recompute + acceptance-counts sidecar.  Read-only wrt frozen assets and
# the original ledger (selection.json mtimes/sha untouched).
set -u
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf
PY=/dev_data/wlt/conda/envs/visionts/bin/python
PKG=repro_t43_period_val
LOGD=$PKG/logs

echo "=== P0 preserve delivery-round evidence ==="
cp -n $LOGD/verify_t43.json $LOGD/verify_t43_delivery_20261112_v1.json
sha256sum $LOGD/selection.json | tee $LOGD/xcheck_driver_selection_sha_before.txt

echo "=== P1 true cross-path check, 6 smoke configs ==="
for dsP in "ETTh1 24" "ETTh1 168" "ETTh1 1" "ETTm1 96" "ETTm1 672" "ETTm1 1"; do
  set -- $dsP
  echo "--- xcheck $1 P$2 ---"
  $PY $PKG/crosspath_true.py --ds $1 --P $2 || { echo XCHECK_RUN_FAIL; exit 10; }
done

echo "=== P2 verify --xcheck ==="
$PY $PKG/verify_t43.py --xcheck || { echo VERIFY_XCHECK_FAIL; exit 11; }

echo "=== P3 verify --all ==="
$PY $PKG/verify_t43.py --all || { echo VERIFY_ALL_FAIL; exit 12; }

echo "=== P4 collect (regenerate RESULT md with corrected wording + counts) ==="
$PY $PKG/collect_t43.py || { echo COLLECT_FAIL; exit 13; }

echo "=== P5 acceptance_counts sidecar (fills ledger summary:null; ledger untouched) ==="
$PY - <<'EOF'
import json, os, sys
sys.path.insert(0, "repro_t43_period_val")
import t43_common as X

def load(p):
    with open(p) as f:
        return json.load(f)

vf = load(X.VALFULL_JSON)
v1 = load(os.path.join(X.LOG_DIR, "verify_t43_delivery_20261112_v1.json"))
va = load(X.VERIFY_JSON)
xs = [load(os.path.join(X.LOG_DIR, "xcheck_%s_P%d_H96.json" % (ds, P)))
      for ds, P in (("ETTh1", 24), ("ETTh1", 168), ("ETTh1", 1),
                    ("ETTm1", 96), ("ETTm1", 672), ("ETTm1", 1))]
rec = {
 "purpose": "fills the acceptance-count null in the original ledger "
            "(selection.json accept_before_select.summary was null at "
            "generation time); the ledger file itself is preserved "
            "byte-identical -- counts live here",
 "valfull_at_ledger_time": {k: vf[k] for k in
    ("modes", "n_checks", "n_fail", "all_pass", "wall_s")},
 "verify_delivery_round": {k: v1[k] for k in
    ("modes", "n_checks", "n_fail", "all_pass", "wall_s")},
 "verify_acceptance_round": {k: va[k] for k in
    ("modes", "n_checks", "n_fail", "all_pass", "wall_s")},
 "xcheck_configs": [{k: x[k] for k in
    ("ds", "P", "pred_bitwise", "true_bitwise", "pred_max_abs_diff",
     "true_max_abs_diff", "sha256_pred_official", "sha256_pred_valpass",
     "official_mse", "abs_diff_mse_official_vs_valjson")} for x in xs],
 "xcheck_all_bitwise": all(x["pred_bitwise"] and x["true_bitwise"]
                           for x in xs),
 "f64_recompute_note": "verify --all now recomputes reused-arm metrics "
    "with float64 casts (gates reuse_exp*:*:recompute_f64); per-group "
    "diffs recorded inside verify_t43.json checks",
}
out = os.path.join(X.LOG_DIR, "acceptance_counts.json")
with open(out, "w") as f:
    json.dump(rec, f, indent=1)
    f.write("\n")
print("[acceptance_counts] wrote", out)
EOF
[ $? -eq 0 ] || { echo ACCEPTANCE_COUNTS_FAIL; exit 14; }

sha256sum $LOGD/selection.json | tee $LOGD/xcheck_driver_selection_sha_after.txt
echo DONE_ALL
