#!/usr/bin/env python3
"""TimesX data access: iterate variables/samples straight from the zip (read-only).

Stdlib only, so splits can be frozen on machines without numpy/torch.
"""
import json
import os
import zipfile
from datetime import datetime

WRAPPER = "TimesX_Datasets"
GROUP_CANDIDATES = {1: "daily", 2: "weekly"}  # path depth -> group key (unused placeholder)
DAILY_GROUPS = {"CommodityPrice", "Currency"}
P_CANDIDATES = {"daily": [1, 5], "weekly": [1, 52]}
EPS_STD = 1e-8


def find_zip(explicit=None):
    cands = [
        explicit,
        os.environ.get("TIMESX_ZIP"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "TimesX_Datasets.zip"),
        "/home/wlt/MMTS/TimesX_Datasets.zip",
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError("TimesX_Datasets.zip not found (set TIMESX_ZIP or pass --zip)")


def parse_ts(s):
    return datetime.fromisoformat(s)


def domain_of(rel_parts):
    if len(rel_parts) == 1:
        return rel_parts[0], rel_parts[0]
    return rel_parts[0], rel_parts[1]


def make_sample_id(domain, var, future_start, idx, seen):
    sid = f"{domain}__{var}__{future_start.strftime('%Y%m%dT%H%M%S')}"
    if sid in seen:
        sid = f"{sid}__i{idx}"
    seen.add(sid)
    return sid


class Sample:
    __slots__ = ("idx", "date", "freq", "sample_id", "future_start", "future_end",
                 "past_ts", "past_val", "future_ts", "future_val")

    def __init__(self, idx, date, freq, sample_id, future_start, future_end,
                 past_ts, past_val, future_ts, future_val):
        self.idx = idx
        self.date = date
        self.freq = freq
        self.sample_id = sample_id
        self.future_start = future_start
        self.future_end = future_end
        self.past_ts = past_ts
        self.past_val = past_val
        self.future_ts = future_ts
        self.future_val = future_val


class Variable:
    def __init__(self, group, domain, var, path):
        self.group = group
        self.domain = domain
        self.var = var
        self.path = path
        self.samples = []

    @property
    def var_key(self):
        return f"{self.domain}__{self.var}"

    @property
    def p_candidates(self):
        return P_CANDIDATES["daily" if self.group in DAILY_GROUPS else "weekly"]


def iter_variables(zip_path, load_values=True):
    """Yield Variable objects, samples sorted by (future_start, idx)."""
    with zipfile.ZipFile(zip_path) as z:
        jsons = sorted(n for n in z.namelist() if n.endswith(".json"))
        wrapper = jsons[0].split("/")[0] if jsons else ""
        for n in jsons:
            rel = n[len(wrapper) + 1:] if wrapper and n.startswith(wrapper + "/") else n
            parts = rel.split("/")
            group, domain = domain_of(parts[:-1] if parts[-1] else parts)
            var = parts[-1][:-5]
            info = json.loads(z.read(n))
            v = Variable(group, domain, var, n)
            seen = set()
            for s in info["samples"]:
                pt, ft = s["past_time"], s["future_time"]
                fts = [parse_ts(t) for t in ft["timestamp"]]
                v.samples.append(Sample(
                    idx=s.get("idx"), date=s.get("date"), freq=s.get("freq"),
                    sample_id=make_sample_id(domain, var, fts[0], s.get("idx"), seen),
                    future_start=fts[0], future_end=fts[-1],
                    past_ts=pt["timestamp"], past_val=pt["value"] if load_values else None,
                    future_ts=ft["timestamp"], future_val=ft["value"] if load_values else None,
                ))
            v.samples.sort(key=lambda s: (s.future_start, s.idx if s.idx is not None else 0))
            yield v


def pstdev(values):
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
