#!/usr/bin/env python3
"""Stage 00: recompute the composite ranking and freeze the 18 selected vars.

Rule (frozen before looking at any figure): over all 190 variables, rank
std_mse_P1 and std_mae_P1 ascending (float tie -> var_key), composite = mean of
the two ranks, final tie -> var_key. Groups: best = ranks 1-6, middle = 93-98,
worst = 185-190. Post-hoc diagnosis only; never used to retune hyperparameters.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DIRS, ensure_dirs

CSV_PATH = os.path.join(os.path.dirname(HERE), "results", "test_variable_level.csv")

FROZEN = [
    (1, "traffic__tour_de_france_96_12_4_10events", "best"),
    (2, "electronic_technology__drones_96_12_4_10events", "best"),
    (3, "shopping__air_conditioner_96_12_4_10events", "best"),
    (4, "science__earthquake_96_12_4_10events", "best"),
    (5, "shopping__back_to_school_96_12_4_10events", "best"),
    (6, "public_health__climate_change_96_12_4_10events", "best"),
    (93, "public_policy__national_debt_96_12_4_10events", "middle"),
    (94, "climate__endangered_species_96_12_4_10events", "middle"),
    (95, "CropsAndStaples__rice_usd_cwt_96_12_12_10events", "middle"),
    (96, "pets__biodiversity_96_12_4_10events", "middle"),
    (97, "arts__food_wine_festivals_96_12_4_10events", "middle"),
    (98, "public_health__hiv_aids_96_12_4_10events", "middle"),
    (185, "finance__financial_regulation_96_12_4_10events", "worst"),
    (186, "economy__cost_of_living_96_12_4_10events", "worst"),
    (187, "public_health__obesity_96_12_4_10events", "worst"),
    (188, "pets__animal_welfare_96_12_4_10events", "worst"),
    (189, "science__nobel_prize_96_12_4_10events", "worst"),
    (190, "climate__sustainable_fashion_96_12_4_10events", "worst"),
]


def main():
    rows = [r for r in csv.DictReader(open(CSV_PATH)) if r["P"] == "1"]
    assert len(rows) == 190, f"expected 190 P=1 rows, got {len(rows)}"

    def ranks(key):
        order = sorted(rows, key=lambda r: (float(r[key]), r["var_key"]))
        return {r["var_key"]: i + 1 for i, r in enumerate(order)}

    rm, ra = ranks("std_mse"), ranks("std_mae")
    for r in rows:
        r["rank_mse"] = rm[r["var_key"]]
        r["rank_mae"] = ra[r["var_key"]]
        r["rank_composite"] = (rm[r["var_key"]] + ra[r["var_key"]]) / 2.0
    order = sorted(rows, key=lambda r: (r["rank_composite"], r["var_key"]))

    frozen_by_rank = {rank: (vk, grp) for rank, vk, grp in FROZEN}
    picked = []
    for pos in (1, 2, 3, 4, 5, 6, 93, 94, 95, 96, 97, 98,
                185, 186, 187, 188, 189, 190):
        r = order[pos - 1]
        exp_vk, grp = frozen_by_rank[pos]
        assert r["var_key"] == exp_vk, f"rank {pos}: got {r['var_key']}, expected {exp_vk}"
        picked.append((pos, r, grp))

    ensure_dirs()
    out = os.path.join(DIRS["selection"], "selected_vars.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank_composite", "rank_mse", "rank_mae", "var_key", "domain",
                    "sel_group", "frequency", "p", "std_mse_P1", "std_mae_P1", "n_test"])
        for pos, r, grp in picked:
            freq = "daily" if r["group"] in ("CommodityPrice", "Currency") else "weekly"
            w.writerow([pos, r["rank_mse"], r["rank_mae"], r["var_key"], r["domain"],
                        grp, freq, 1, r["std_mse"], r["std_mae"], r["n_test"]])

    n_test = sum(int(r["n_test"]) for _, r, _ in picked)
    assert n_test == 117, f"expected 117 test windows, got {n_test}"
    print(json.dumps({"selected": 18, "test_windows": n_test, "out": out}))
    print("ranking recomputation matches the frozen list; selection frozen")


if __name__ == "__main__":
    main()
