"""EXP-043 selection ledger — runs ONLY after the 48 validation configs
are accepted (verify_t43.py --valfull all_pass), BEFORE any new test run
(user correction #3: accept-then-select; selection uses unrounded f64
values; SSE accumulated in float64 upstream).

Outputs (logs/):
  val_table.csv   48 rows (ds, P, H, n, mse f64 repr, mae f64 repr)
  selection.json  per-ds equal-weight mean val MSE per candidate P
                  (unrounded), chosen P, tie-break record, rule statement
  LEDGER_OK       sha256 of the above + the 12 val jsons + acceptance ref

Selection rule (frozen, a SUPPLEMENTARY setting of this round — the paper
specifies selection on the validation set but not the aggregation):
per dataset, choose P minimizing the equal-weight mean of the 4 per-H
validation MSEs (unrounded f64); ties -> prefer the paper P, then the
smaller P.  NO test result may influence this step; the ledger sha is
embedded in every subsequent test fingerprint.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t43_common as X

RULE = ("per dataset: argmin over candidate P of the equal-weight mean of "
        "the 4 per-H validation MSEs (float64, unrounded); ties -> prefer "
        "the paper P (ETTh 24 / ETTm 96), then the smaller P.  Supplementary "
        "setting of EXP-043, fixed before any test evaluation; the paper "
        "specifies validation selection but not this aggregation.")


def main():
    # ---- hard precondition: the 48 val configs are accepted ----
    if not os.path.isfile(X.VALFULL_JSON):
        raise SystemExit("[select] %s missing -> run verify_t43.py --valfull "
                         "first (accept-before-select is frozen)" % X.VALFULL_JSON)
    vf = X.load_json(X.VALFULL_JSON)
    if not vf.get("all_pass"):
        raise SystemExit("[select] valfull verify not all_pass (%s) -> ABORT; "
                         "ledger generation is gated on 48-config acceptance"
                         % vf.get("summary"))

    rows, per_ds = [], {}
    for ds in X.ETT4:
        means = {}
        for P in X.CANDIDATE_P[ds]:
            ms, mas = [], []
            for h in X.HORIZONS:
                j = X.load_json(X.vjson_path(ds, P, h))
                assert j["ds"] == ds and j["P"] == P and j["H"] == h
                assert j["n_windows"] == X.expected_n_windows(ds, h)
                assert j["truth_bitwise_vs_truth_from_std"] and j["finite"] \
                    and j["geometry_ok"]
                rows.append((ds, P, h, j["n_windows"],
                             repr(j["mse_from_sse"]), repr(j["mae_from_sae"]),
                             repr(j["sse_f64"])))
                ms.append(j["mse_from_sse"])
                mas.append(j["mae_from_sae"])
            means[P] = {"mean_mse": sum(ms) / len(ms),
                        "mean_mae": sum(mas) / len(mas),
                        "per_H_mse": ms}
        ranked = sorted(means.items(), key=lambda kv: kv[1]["mean_mse"])
        best = ranked[0][1]["mean_mse"]
        tied = [k for k, v in means.items() if v["mean_mse"] == best]
        if len(tied) == 1:
            chosen, tie = tied[0], "no tie (unique float64 minimum)"
        else:
            paper = X.PAPER_P[ds]
            if paper in tied:
                chosen, tie = paper, "tie among %s -> paper P %d" % (tied, paper)
            else:
                chosen, tie = min(tied), "tie among %s -> smaller P %d" \
                    % (tied, min(tied))
        per_ds[ds] = {
            "candidates": {str(p): means[p] for p in X.CANDIDATE_P[ds]},
            "chosen_P": chosen,
            "paper_P": X.PAPER_P[ds],
            "agrees_with_paper": chosen == X.PAPER_P[ds],
            "tie_break": tie,
            "margin_second_minus_best_mse":
                ranked[1][1]["mean_mse"] - ranked[0][1]["mean_mse"]
                if len(ranked) > 1 else 0.0,
        }

    os.makedirs(X.LOG_DIR, exist_ok=True)
    with open(X.LEDGER_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ds", "P", "H", "n_windows", "mse_f64_repr",
                    "mae_f64_repr", "sse_f64_repr"])
        for r in rows:
            w.writerow(r)

    sel = {
        "experiment": "EXP-043 periodic-P validation selection",
        "rule": RULE,
        "accept_before_select": {
            "gate": "verify_t43.py --valfull (48 configs)",
            "json": X.VALFULL_JSON,
            "all_pass": vf.get("all_pass"),
            "summary": vf.get("summary"),
        },
        "per_ds": per_ds,
        "chosen_P_summary": {ds: per_ds[ds]["chosen_P"] for ds in X.ETT4},
        "note": "ledger precedes every new test run; test fingerprints embed "
                "ledger_sha256; re-selection after seeing test results is "
                "forbidden by protocol.",
    }
    X.dump_json(X.LEDGER_JSON, sel)

    entries = {
        "val_table.csv": X.sha256_of(X.LEDGER_CSV),
        "selection.json": X.sha256_of(X.LEDGER_JSON),
    }
    for ds in X.ETT4:
        for P in X.CANDIDATE_P[ds]:
            for h in X.HORIZONS:
                entries["val_%s_P%d_H%d.json" % (ds, P, h)] = \
                    X.sha256_of(X.vjson_path(ds, P, h))
    lines = ["# EXP-043 ledger: selection is frozen; tests may now run.",
             "accept_gate=verify_t43_valfull all_pass=%s n_checks=%s"
             % (vf.get("all_pass"), vf.get("summary", {}).get("n_checks")),
             "rule=%s" % RULE.replace("\n", " "),
             "chosen=" + " ".join("%s:%d" % (ds, per_ds[ds]["chosen_P"])
                                  for ds in X.ETT4)]
    for k, v in entries.items():
        lines.append("sha256:%s=%s" % (k, v))
    X.write_text(X.LEDGER_OK, lines)

    # the ledger file's own sha is consumed ONLY by test fingerprints (and
    # recorded in delivery); it is not embedded back into selection.json
    # (self-reference impossible).
    ledger_sha = X.sha256_of(X.LEDGER_OK)
    print("[select] chosen P: %s" % sel["chosen_P_summary"])
    print("[select] ledger_sha256=%s" % ledger_sha)


if __name__ == "__main__":
    main()
