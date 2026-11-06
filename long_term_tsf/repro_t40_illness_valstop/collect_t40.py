"""EXP-040 collector: summary_t40.csv + RESULT_t40_illness_valstop.md +
figures.  Reads the t40 train jsons (12 main tasks) + audit json + verify
json; aggregates per-H 3-seed mean±std (ddof=1) per rule; paired diffs
A-B / A-C / B-C per (H, seed); paper Table 21 comparison; theta-curve
reference paragraph; budget table from phases.json.

Reporting discipline (frozen):
  - observed effects are reported; causal attribution is NOT claimed;
  - approaching paper values is NOT an acceptance criterion;
  - controlled diagnostic: does NOT recover undisclosed author details;
  - theta archived curves are cross-environment REFERENCE ONLY.
"""
import csv
import os
import sys

import numpy as np

_T40_DIR = os.path.dirname(os.path.abspath(__file__))
if _T40_DIR not in sys.path:
    sys.path.insert(0, _T40_DIR)
import t40_common as X              # noqa: E402
import t38_common as C              # noqa: E402


def ms(vals):
    v = np.asarray(vals, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1))


def fmt(m, s):
    return "%.4f±%.4f" % (m, s)


def main():
    os.chdir(X.ROOT)
    os.makedirs(X.LOG_DIR, exist_ok=True)
    audit = C.load_json(X.AUDIT_JSON) if os.path.isfile(X.AUDIT_JSON) else {}
    verify = C.load_json(X.VERIFY_JSON) if os.path.isfile(X.VERIFY_JSON) \
        else {}

    rows = []
    for h, s in X.TASKS:
        jd = X.job_dir(h, s, "main")
        succ = C.load_json(os.path.join(jd, "SUCCESS.txt"))
        rec = C.load_json(X.train_json(h, s, "main"))
        lock = C.load_json(X.lock_events_path(h, s, "main"))
        fin = {e["rule"]: e for e in lock["final"]}
        for r in X.RULES:
            t = succ["test"][r]
            rows.append({
                "h": h, "seed": s, "rule": r,
                "mse": t["mse"], "mae": t["mae"],
                "best_valid_epoch": t["best_valid_epoch"],
                "locked": t["locked"],
                "trigger_epoch": fin[r]["trigger_epoch"],
                "n_epochs": rec["finished_epochs"],
                "train_s": round(rec["train_wall_s"], 1),
                "ckpt_sha256": fin[r]["ckpt_sha256"],
            })
    csv_path = X.SUMMARY_CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[t40-collect] wrote %s (%d rows)" % (csv_path, len(rows)))

    # ---- aggregates --------------------------------------------------------
    def get(h, seed, rule, key):
        for r in rows:
            if (r["h"], r["seed"], r["rule"]) == (h, seed, rule):
                return r[key]
        raise KeyError((h, seed, rule, key))

    L = ["# EXP-040: Illness validation-rule / early-stop diagnostic",
         "",
         "Frozen EXP-003 recipe (LN, seq_len=104, Adam 1e-4, bs=256,",
         "patience=3, lradj=type1, max 100 epochs; seeds 2021/2022/2023),",
         "three validation rules evaluated on the SAME shared trajectory:",
         "",
         "- **A** full fixed val set, element-weighted MSE",
         "  (EXP-003-consistent, mirrors `Exp_Univariate.vali`);",
         "- **B** same full fixed set, equal-weight per-batch criterion",
         "  means (official base-vali semantics; the partial last batch",
         "  gets 1/n_batches weight — distortion probe);",
         "- **C** official counterfactual: per-epoch permutation +",
         "  drop_last truncation to floor(N/256) full batches +",
         "  equal-weight batch means (dedicated RandomState; simulates",
         "  the official selection rule, not its exact RNG interleaving).",
         "",
         "**Controlled diagnostic, NOT a reproduction of undisclosed",
         "author details; approaching paper values is NOT an acceptance",
         "criterion.** Observed effects and causal attribution are kept",
         "separate: nothing below claims to identify WHY the paper",
         "differs.",
         ""]
    vg = verify.get("gates", {})
    if vg:
        L.append("Gates: **%s** (TG1-TG10, %d checks, %d fail) -> %s"
                 % ("ALL PASS" if verify.get("all_pass") else "FAIL",
                    verify.get("summary", {}).get("n_checks", 0),
                    verify.get("summary", {}).get("n_fail", 0),
                    "logs/verify_t40.json"))
        L.append("")

    L.append("## T1 per-H test performance (3-seed mean±std, ddof=1)")
    L.append("")
    L.append("| H | A MSE | B MSE | C MSE | A MAE | B MAE | C MAE | "
             "paper T21 MSE | paper MAE | EXP-003 arch MSE (ref) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    per_h = {}
    for h in X.HS:
        cells = {}
        for r in X.RULES:
            m_mse, s_mse = ms([get(h, s, r, "mse") for s in X.SEEDS])
            m_mae, s_mae = ms([get(h, s, r, "mae") for s in X.SEEDS])
            cells[r] = (m_mse, s_mse, m_mae, s_mae)
        per_h[h] = cells
        arch_m = np.mean([audit.get("archived_success", {}).get(
            "illness_%d_seed%d" % (h, s), {}).get("mse", np.nan)
            for s in X.SEEDS])
        pm, pa = X.PAPER_T21[h]
        L.append("| %d | %s | %s | %s | %s | %s | %s | %.3f | %.3f | %.4f |"
                 % (h,
                    fmt(cells["A"][0], cells["A"][1]),
                    fmt(cells["B"][0], cells["B"][1]),
                    fmt(cells["C"][0], cells["C"][1]),
                    fmt(cells["A"][2], cells["A"][3]),
                    fmt(cells["B"][2], cells["B"][3]),
                    fmt(cells["C"][2], cells["C"][3]),
                    pm, pa, arch_m))

    L.append("")
    L.append("## T2 overall (4-H equal-weight, 3-seed mean±std)")
    L.append("")
    L.append("| rule | MSE | MAE | paper avg MSE | paper avg MAE |")
    L.append("|---|---|---|---|---|")
    overall = {}
    for r in X.RULES:
        m_mse, s_mse = ms([np.mean([get(h, s, r, "mse")
                                    for h in X.HS]) for s in X.SEEDS])
        m_mae, s_mae = ms([np.mean([get(h, s, r, "mae")
                                    for h in X.HS]) for s in X.SEEDS])
        overall[r] = (m_mse, s_mse, m_mae, s_mae)
        L.append("| %s | %s | %s | %.3f | %.3f |"
                 % (r, fmt(m_mse, s_mse), fmt(m_mae, s_mae),
                    X.PAPER_T21["avg"][0], X.PAPER_T21["avg"][1]))
    arch_all = np.mean([v["mse"] for v in
                        audit.get("archived_success", {}).values()])
    L.append("| EXP-003 archive (A-rule selection, seed=2021 only "
             "reference) | %.4f | — | %.3f | %.3f |"
             % (arch_all, X.PAPER_T21["avg"][0], X.PAPER_T21["avg"][1]))

    L.append("")
    L.append("## T3 paired rule differences (test MSE; per (H, seed), "
             "then overall)")
    L.append("")
    L.append("| pair | per-H per-seed values (2021 / 2022 / 2023) | "
             "overall per-seed | direction-consistent |")
    L.append("|---|---|---|---|")
    for pair in (("A", "B"), ("A", "C"), ("B", "C")):
        ra, rb = pair
        per_seed_overall = []
        for s in X.SEEDS:
            d = np.mean([get(h, s, ra, "mse") - get(h, s, rb, "mse")
                         for h in X.HS])
            per_seed_overall.append(d)
        parts = []
        for h in X.HS:
            per_seed = ["%+.4f" % (get(h, s, ra, "mse")
                                   - get(h, s, rb, "mse"))
                        for s in X.SEEDS]
            parts.append("H%d: %s" % (h, "/".join(per_seed)))
        consist = all((d < 0) == (per_seed_overall[0] < 0)
                      for d in per_seed_overall)
        L.append("| %s-%s | %s | %s | %s |"
                 % (ra, rb, "; ".join(parts),
                    "/".join("%+.4f" % d for d in per_seed_overall),
                    "yes" if consist else "NO"))

    L.append("")
    L.append("## T4 selection behaviour (best epoch / stop epoch / locked)")
    L.append("")
    L.append("| H | rule | best_valid_epoch (3 seeds) | locked (3 seeds) | "
             "n_epochs run (3 seeds) |")
    L.append("|---|---|---|---|---|")
    for h in X.HS:
        for r in X.RULES:
            be = [get(h, s, r, "best_valid_epoch") for s in X.SEEDS]
            lk = [get(h, s, r, "locked") for s in X.SEEDS]
            ne = [get(h, s, r, "n_epochs") for s in X.SEEDS]
            L.append("| %d | %s | %s | %s | %s |"
                     % (h, r,
                        "/".join(str(int(b)) for b in be),
                        "/".join("Y" if x else "n" for x in lk),
                        "/".join(str(n) for n in ne)))

    L.append("")
    L.append("## T5 archived theta curves / early-stop replay (reference "
             "only, cross-env)")
    L.append("")
    ag9 = audit.get("gates", {}).get("AG9", {})
    if ag9:
        n_checked = sum(1 for v in ag9.values()
                        if v.get("status") == "checked")
        n_match = sum(1 for v in ag9.values() if v.get("match"))
        L.append("Audit AG9 replayed the REAL EarlyStopping tie semantics "
                 "(ties improve) over %d parseable theta logs; archived "
                 "best_valid_epoch reproduced in %d/%d checked. The theta "
                 "curves themselves are compared side-by-side (no "
                 "threshold) in `logs/verify_t40.json` TG3 — the t40 "
                 "training stream intentionally diverges from epoch 2 "
                 "(isolated validation generators instead of the "
                 "official global-RNG consumption), so curve agreement "
                 "is expected only qualitatively on the early epochs."
                 % (n_checked, n_match, n_checked))
    else:
        L.append("audit json missing — run t40_audit.py first.")

    L.append("")
    L.append("## Budget (measured, from phases.json)")
    L.append("")
    tot = 0.0
    per_h_s = {}
    for h, s in X.TASKS:
        rec = C.load_json(X.train_json(h, s, "main"))
        tot += rec["train_wall_s"]
        per_h_s.setdefault(h, []).append(rec["train_wall_s"])
    for h in X.HS:
        L.append("- H%d: mean wall %.1f s/task (%s)"
                 % (h, float(np.mean(per_h_s[h])),
                    ", ".join("%.0f" % v for v in per_h_s[h])))
    L.append("- total measured train wall (12 tasks): %.1f min"
             % (tot / 60.0))

    L.append("")
    L.append("## Reading guide (frozen wording)")
    L.append("")
    L.append("1. Differences between rules measure the SELECTION "
             "criterion's effect on the frozen recipe — they do not "
             "identify the cause of the gap to the paper's numbers.")
    L.append("2. Rule C is the documented official counterfactual with a "
             "dedicated permutation stream, not an RNG-exact replay of "
             "the original runs.")
    L.append("3. Any agreement or disagreement with paper Table 21 is "
             "reported as an observation; no threshold against the paper "
             "is part of acceptance.")

    with open(X.RESULT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("[t40-collect] wrote %s" % X.RESULT_MD)

    # ---- figure ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        x = np.arange(len(X.HS))
        wdt = 0.25
        for i, r in enumerate(X.RULES):
            means = [per_h[h][r][0] for h in X.HS]
            stds = [per_h[h][r][1] for h in X.HS]
            ax.bar(x + (i - 1) * wdt, means, wdt, yerr=stds,
                   capsize=3, label="rule %s" % r)
        pm = [X.PAPER_T21[h][0] for h in X.HS]
        ax.plot(x, pm, "k^--", label="paper T21")
        ax.set_xticks(x, [str(h) for h in X.HS])
        ax.set_ylabel("test MSE (3-seed mean±std)")
        ax.set_title("EXP-040 Illness: validation rules")
        ax.legend(fontsize=8)
        ax = axes[1]
        for i, pair in enumerate((("A", "B"), ("A", "C"))):
            ra, rb = pair
            ds = [[get(h, s, ra, "mse") - get(h, s, rb, "mse")
                   for h in X.HS] for s in X.SEEDS]
            x = np.arange(len(X.HS))
            for s_i, s in enumerate(X.SEEDS):
                ax.plot(x, ds[s_i], "o-", alpha=0.6,
                        label="%s-%s seed%d" % (ra, rb, s))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [str(h) for h in X.HS])
        ax.set_ylabel("paired ΔMSE")
        ax.set_title("paired rule differences")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(X.LOG_DIR, "fig_t40_valstop.png"), dpi=150)
        print("[t40-collect] wrote fig_t40_valstop.png")
    except Exception as e:   # matplotlib optional on headless boxes
        print("[t40-collect] figure skipped: %s" % e)

    # ---- self-check ---------------------------------------------------------
    assert len(rows) == 36, "expected 36 rule-unit rows, got %d" % len(rows)
    for h, s in X.TASKS:
        for r in X.RULES:
            m = get(h, s, r, "mse")
            assert np.isfinite(m) and m > 0, (h, s, r, m)
    print("[t40-collect] self-check ok (36/36 units, finite, positive)")


if __name__ == "__main__":
    main()
