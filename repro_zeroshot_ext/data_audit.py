#!/usr/bin/env python3
"""S0 data audit for TimesX_Datasets.zip. Read-only, stdlib only.

Produces results/audit_report.json with: zip hashes, per-variable facts
(sample counts, observation interval vs sample-start interval, timestamp
monotonicity), zero-variance-history census, SearchTrend >100 stats,
and a check for any bundled scoring protocol.
"""
import argparse
import hashlib
import json
import os
import statistics
import zipfile
from collections import Counter
from datetime import datetime

EPS_STD = 1e-8
EXPECTED_LEN = {"past": 96, "future": 12}
PROTOCOL_KEYWORDS = ["mse", "mae", "mase", "split", "train set", "test set",
                     "validation", "metric", "evaluation protocol", "benchmark score"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ts(s):
    return datetime.fromisoformat(s)


def days(td):
    return td.total_seconds() / 86400.0


def domain_of(rel_parts):
    """rel_parts: path parts after the wrapper dir, e.g. ['CommodityPrice','CropsAndStaples','x.json']."""
    if len(rel_parts) == 2:
        return rel_parts[0], rel_parts[0]
    return rel_parts[0], rel_parts[1]


def audit_variable(group, domain, path, raw):
    info = json.loads(raw)
    dinfo = info.get("dataset_info", {})
    samples = info["samples"]
    v = {
        "group": group,
        "domain": domain,
        "file": path,
        "var": path.rsplit("/", 1)[-1][:-5],
        "declared_variable_name": dinfo.get("variable_name"),
        "declared_total_samples": dinfo.get("total_samples"),
        "n_samples": len(samples),
        "freq": sorted({s.get("freq") for s in samples}),
        "len_ok": True,
        "date_equals_future_start": True,
        "past_monotone": True,
        "future_monotone": True,
        "sample_order_monotone": True,
        "obs_interval_days": [],
        "start_interval_days": [],
    }
    zero_var = []
    over100 = 0
    vmin, vmax = None, None
    prev_start = None
    for s in samples:
        pt, ft = s["past_time"], s["future_time"]
        if not (len(pt["timestamp"]) == EXPECTED_LEN["past"] == len(pt["value"])
                and len(ft["timestamp"]) == EXPECTED_LEN["future"] == len(ft["value"])):
            v["len_ok"] = False
        if s.get("date") != ft["timestamp"][0][:10]:
            v["date_equals_future_start"] = False
        pts = [parse_ts(t) for t in pt["timestamp"]]
        fts = [parse_ts(t) for t in ft["timestamp"]]
        if any(pts[i] >= pts[i + 1] for i in range(len(pts) - 1)):
            v["past_monotone"] = False
        if any(fts[i] >= fts[i + 1] for i in range(len(fts) - 1)):
            v["future_monotone"] = False
        v["obs_interval_days"].append(round(statistics.median(
            [days(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]), 6))
        cur_start = fts[0]
        if prev_start is not None:
            if cur_start < prev_start:
                v["sample_order_monotone"] = False
            v["start_interval_days"].append(round(days(cur_start - prev_start), 6))
        prev_start = cur_start

        pv = [float(x) for x in pt["value"]]
        fv = [float(x) for x in ft["value"]]
        mean = sum(pv) / len(pv)
        pstd = (sum((x - mean) ** 2 for x in pv) / len(pv)) ** 0.5  # ddof=0
        if pstd < EPS_STD:
            zero_var.append({
                "var": v["var"], "domain": domain, "sample_idx": s.get("idx"),
                "date": s.get("date"),
                "past_constant_value": pv[0],
                "future_min": min(fv), "future_max": max(fv),
                "future_varies": (max(fv) - min(fv)) > EPS_STD,
            })
        lo, hi = min(pv + fv), max(pv + fv)
        vmin = lo if vmin is None else min(vmin, lo)
        vmax = hi if vmax is None else max(vmax, hi)
        if group == "SearchTrend":
            over100 += sum(1 for x in pv + fv if x > 100)

    v["obs_interval_days"] = round(statistics.median(v["obs_interval_days"]), 6)
    v["start_interval_days"] = (round(statistics.median(v["start_interval_days"]), 6)
                                if v["start_interval_days"] else None)
    v["first_future_start"] = samples[0]["future_time"]["timestamp"][0]
    v["last_future_end"] = samples[-1]["future_time"]["timestamp"][-1]
    v["value_min"] = vmin
    v["value_max"] = vmax
    v["n_over100"] = over100
    v["declared_matches"] = v["declared_total_samples"] in (None, len(samples))
    extras = {"zero_variance_samples": zero_var}
    return v, extras


def scan_protocol(text):
    hits = []
    low = text.lower()
    for kw in PROTOCOL_KEYWORDS:
        if kw in low:
            i = low.find(kw)
            hits.append({"keyword": kw, "snippet": text[max(0, i - 60):i + 120]})
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="/home/wlt/MMTS/TimesX_Datasets.zip")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results", "audit_report.json")

    report = {"zip": {"path": args.zip, "size_bytes": os.path.getsize(args.zip),
                      "sha256": sha256_file(args.zip)},
              "groups": Counter(), "domains": {}, "variables": [],
              "zero_variance_census": [], "protocol_hits": [], "bundled_docs": []}

    with zipfile.ZipFile(args.zip) as z:
        jsons = sorted(n for n in z.namelist() if n.endswith(".json"))
        for n in z.namelist():
            if n.lower().endswith((".md", ".txt", ".rst")):
                txt = z.read(n).decode("utf-8", "replace")
                report["bundled_docs"].append({"path": n, "bytes": len(txt),
                                               "protocol_hits": scan_protocol(txt)})
        wrapper = jsons[0].split("/")[0] if jsons else ""
        for n in jsons:
            rel = n[len(wrapper) + 1:] if wrapper and n.startswith(wrapper + "/") else n
            parts = rel.split("/")
            group, domain = domain_of(parts)
            v, extras = audit_variable(group, domain, n, z.read(n))
            report["variables"].append(v)
            report["zero_variance_census"].extend(extras["zero_variance_samples"])
            report["groups"][group] += 1
            d = report["domains"].setdefault(domain, {"group": group, "n_vars": 0, "n_samples": 0})
            d["n_vars"] += 1
            d["n_samples"] += v["n_samples"]

    report["groups"] = dict(report["groups"])
    totals = {"n_vars": len(report["variables"]),
              "n_samples": sum(v["n_samples"] for v in report["variables"]),
              "n_domains": len(report["domains"])}
    report["totals"] = totals
    report["totals_check"] = {
        "expected_vars": 190, "expected_samples": 8106, "expected_domains": 19,
        "vars_ok": totals["n_vars"] == 190,
        "samples_ok": totals["n_samples"] == 8106,
        "domains_ok": totals["n_domains"] == 19,
        "all_len_ok": all(v["len_ok"] for v in report["variables"]),
        "all_monotone": all(v["past_monotone"] and v["future_monotone"]
                            and v["sample_order_monotone"] for v in report["variables"]),
        "all_date_matches_future_start": all(v["date_equals_future_start"] for v in report["variables"]),
        "declared_all_match": all(v["declared_matches"] for v in report["variables"]),
        "zero_var_expected": 4, "zero_var_found": len(report["zero_variance_census"]),
        "zero_var_with_future_change": sum(1 for z_ in report["zero_variance_census"] if z_["future_varies"]),
    }

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    print(json.dumps(report["totals_check"], indent=2))
    print("groups:", report["groups"])
    for d, info in sorted(report["domains"].items()):
        print(f"  {d:35s} vars={info['n_vars']:3d} samples={info['n_samples']}")
    freq_by_group = {}
    for v in report["variables"]:
        freq_by_group.setdefault(v["group"], Counter()).update({v["freq"][0]: 1})
    print("freq by group:", {k: dict(c) for k, c in freq_by_group.items()})
    ivals = [(v["var"], v["group"], v["obs_interval_days"], v["start_interval_days"])
             for v in report["variables"]]
    for g in ("CommodityPrice", "Currency", "SearchTrend"):
        rows = [r for r in ivals if r[1] == g]
        print(f"{g}: obs_interval min/med/max = "
              f"{min(r[2] for r in rows)}/{statistics.median(r[2] for r in rows)}/{max(r[2] for r in rows)} d, "
              f"start_interval min/med/max = "
              f"{min(r[3] for r in rows)}/{statistics.median(r[3] for r in rows)}/{max(r[3] for r in rows)} d")
    print("zero-variance census:")
    for z_ in report["zero_variance_census"]:
        print("  ", z_)
    print("docs:", [(d["path"], d["protocol_hits"]) for d in report["bundled_docs"]])
    print("report ->", out)


if __name__ == "__main__":
    main()
