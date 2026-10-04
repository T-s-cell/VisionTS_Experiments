#!/usr/bin/env python3
"""Stage 04 — audit + REPORT.md (EXP-009).

Checks (all recorded in audit/audit_report.json; any failure -> exit 1):
  1 completeness   Z0__timesx == all 2,474 windows / 190 vars / 19 domains in
                   canonical split_manifest order; per-domain F0 subsets match
                   the Z0 slice; slices reassemble the full set
  2 checkpoints    19 ln.pt file SHA256 == selection.json selected_ckpt_sha256
                   (L1 re-verified here; L2 names/shapes/values were verified
                   at load time and recorded per domain)
  3 features       every npz shape/finite; ids/order records present
  4 hook validity  20 models (Z0 + 19 F0) each show pred-bitwise-equal and
                   captured==direct forward_encoder in logs/extract.jsonl
  5 immutability   baseline snapshot re-diff: zero changes to any input
  6 provenance     _cache_index.json covers every artifact; sha256 + cache key
                   both match current recomputation
  7 metrics        per-domain std MSE/MAE recomputed window -> var -> var-equal
                   (stored d); Z0 rows must equal EXP-004 domain_summary.csv
                   Z0/single; F0 s2021 from its own npz; never 3-seed means
Also verifies every coords file (row count, finite, fit_key) and writes
REPORT.md (per-domain one-line table + conclusion boundaries + provenance).

--postaudit: re-attestation of the FINAL files (default full mode). Loads
audit/postaudit_changelog.json (hash-anchored record of the post-campaign
figure-presentation edits), verifies: declared changes old/new/diff hashes,
config diff confined to figures keys, generation-time records untouched
(baseline / cache index / original report), and all 40 feature cache keys
bound to the generation-time fingerprints. Writes
audit/audit_report_final.json; the original audit_report.json is preserved.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "exp009_common", os.path.join(HERE, "common.py"))
E9 = importlib.util.module_from_spec(_spec)
sys.modules["exp009_common"] = E9
_spec.loader.exec_module(E9)

AUDIT_JSON = os.path.join(E9.AUDIT, "audit_report.json")
AUDIT_JSON_FINAL = os.path.join(E9.AUDIT, "audit_report_final.json")
CHANGELOG_PATH = os.path.join(E9.AUDIT, "postaudit_changelog.json")
REPORT_MD = os.path.join(E9.HERE, "REPORT.md")
EXTRACT_LOG = os.path.join(E9.LOGS, "extract.jsonl")
METRIC_ATOL = 1e-6
TSNE_CFG = {k: E9.CFG["tsne"][k] for k in (
    "n_components", "perplexity", "learning_rate", "init",
    "early_exaggeration", "max_iter", "random_state", "metric",
    "method", "angle")}


class Checks:
    def __init__(self):
        self.results = {}

    def add(self, name, ok, **details):
        self.results[name] = {"status": "OK" if ok else "FAIL", **details}
        print(f"[audit] {'OK  ' if ok else 'FAIL'} {name}: "
              f"{json.dumps(details, ensure_ascii=False)[:400]}", flush=True)
        return ok

    @property
    def all_ok(self):
        return all(r["status"] == "OK" for r in self.results.values())


def expected_order():
    """Canonical [(domain, var_key, sample_id)] from EXP-004 manifests."""
    rows = E9.test_rows_all_domains()
    return ([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])


def check_completeness(ck, domains_subset):
    exp_dom, exp_vk, exp_sid = expected_order()
    z = np.load(os.path.join(E9.FEATURES, "Z0__timesx.npz"), allow_pickle=False)
    z_sid = [str(s) for s in z["sample_ids"]]
    z_vk = [str(s) for s in z["var_keys"]]
    z_dom = [str(s) for s in z["domains"]]
    ck.add("completeness.z0_full",
           z_sid == exp_sid and z_vk == exp_vk and z_dom == exp_dom,
           n_windows=len(z_sid), n_vars=len(set(z_vk)),
           n_domains=len(set(z_dom)))
    ck.add("completeness.counts",
           len(z_sid) == 2474 and len(set(z_vk)) == 190 and len(set(z_dom)) == 19,
           n_windows=len(z_sid), n_vars=len(set(z_vk)), n_domains=len(set(z_dom)))

    mism, slice_rows = [], []
    for d in domains_subset:
        f = np.load(os.path.join(E9.FEATURES, f"F0__{d}__s{E9.F0_SEED}"
                                 f"__timesx.npz"), allow_pickle=False)
        mask = [i for i, dd in enumerate(z_dom) if dd == d]
        exp = [z_sid[i] for i in mask]
        got = [str(s) for s in f["sample_ids"]]
        if got != exp:
            mism.append({"domain": d, "n_expected": len(exp),
                         "n_got": len(got)})
        else:
            slice_rows.append((d, got))
    ck.add("completeness.domain_slices", not mism, mismatches=mism)

    union = [sid for _d, sids in sorted(slice_rows) for sid in sids]
    full_mode = len(domains_subset) == len(E9.DOMAIN_NAMES)
    if full_mode:
        ck.add("completeness.slices_reassemble_full", union == z_sid,
               mode="full")
    else:
        ck.add("completeness.slices_reassemble_full", True,
               mode="subset", note="deferred to full mode")


def check_checkpoints(ck, domains_subset):
    sel = E9.f0_selection()
    with open(E9.CKPT_TABLE) as f:
        table = json.load(f)
    by_dom = {r["domain"]: r for r in table}
    bad = []
    for d in domains_subset:
        r = by_dom.get(d)
        if r is None:
            bad.append({"domain": d, "problem": "missing from ckpt_table"})
            continue
        sha = E9.sha256_file(E9.f0_ckpt_path(d))
        if sha != sel[d]["selected_ckpt_sha256"]:
            bad.append({"domain": d, "problem": "file sha drift"})
        if not r.get("names_shapes_values_verified") or r["n_tensors"] != 84:
            bad.append({"domain": d, "problem": "L2 record missing"})
    ck.add("checkpoints.binding", not bad, problems=bad, n=len(domains_subset))


def check_features(ck, domains_subset):
    idx = json.load(open(E9.CACHE_INDEX)) if os.path.isfile(E9.CACHE_INDEX) \
        else {}
    names = ["Z0__imagenet.npz", "Z0__timesx.npz"]
    names += [f"F0__{d}__s{E9.F0_SEED}__{k}.npz"
              for d in domains_subset for k in ("imagenet", "timesx")]
    bad = []
    for n in names:
        p = os.path.join(E9.FEATURES, n)
        if n not in idx:
            bad.append({"artifact": n, "problem": "not in cache index"})
            continue
        if idx[n]["sha256"] != E9.sha256_file(p):
            bad.append({"artifact": n, "problem": "sha256 drift"})
        z = np.load(p, allow_pickle=False)
        if not np.isfinite(z["feats"]).all() or \
                z["feats"].shape[1] != E9.FLAT_DIM:
            bad.append({"artifact": n, "problem": "shape/finite"})
    ck.add("features.shape_finite_provenance", not bad, problems=bad,
           n_artifacts=len(names))


def check_hook_equivalence(ck, domains_subset):
    """Only events from the CURRENT campaign count: everything after the last
    baseline snapshot (older events belong to superseded extractions)."""
    with open(EXTRACT_LOG) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    last_base = max((i for i, r in enumerate(lines)
                     if r.get("event") == "baseline"), default=-1)
    events = [r for r in lines[last_base + 1:]
              if r.get("event") == "equivalence_check"]
    full_mode = len(domains_subset) == len(E9.DOMAIN_NAMES)
    need = 1 + len(domains_subset)
    ok = (len(events) >= need
          and all(e.get("pred_bitwise_equal") and e.get("hook_equals_direct")
                  for e in events))
    if not full_mode and len(events) < 1 + len(E9.DOMAIN_NAMES):
        note = "subset campaign; full run must cover all 20 models"
    else:
        note = "captured vision_model.norm == direct forward_encoder " \
               "(bitwise); instrumentation never changed predictions"
    ck.add("hook_equivalence", ok, n_models=len(events), need=need, note=note)


def check_immutability(ck, domains_subset, changelog=None):
    full_mode = len(domains_subset) == len(E9.DOMAIN_NAMES)
    with open(E9.BASELINE_JSON) as f:
        base = json.load(f)["entries"]
    # recompute the same entry set as 01_extract_features.build_baseline
    entries = {}
    for name, p in E9.REUSED_SOURCES:
        entries[f"reused/{name}"] = E9.sha256_file(p)
    vt = os.path.join(E9.REPO_ROOT, "visionts")
    for fn in sorted(os.listdir(vt)):
        if fn.endswith(".py"):
            entries[f"visionts/{fn}"] = E9.sha256_file(os.path.join(vt, fn))
    entries["ckpt/mae_visualize_vit_base.pth"] = E9.sha256_file(
        os.path.join(E9.REPO_ROOT, "ckpt", "mae_visualize_vit_base.pth"))
    for p in (E9.SPLIT_MANIFEST,
              os.path.join(E9.LN_DIR, "manifests", "protocol.json"),
              os.path.join(E9.LN_DIR, "manifests", "split_counts.csv"),
              E9.DATA_CACHE, E9.SELECTION_JSON, E9.DOMAIN_SUMMARY_CSV,
              os.path.join(E9.LN_DIR, "results", "overall.csv"),
              E9.FIG7_LIST_JSON):
        entries[f"ln_inputs/{os.path.basename(p)}"] = E9.sha256_file(p)
    for fn in sorted(os.listdir(E9.LN_PREDICTIONS)):
        if fn.endswith(".npz"):
            entries[f"ln_predictions/{fn}"] = E9.sha256_file(
                os.path.join(E9.LN_PREDICTIONS, fn))
    for root, _dirs, files in os.walk(E9.HERE):
        rel_root = os.path.relpath(root, E9.HERE)
        if rel_root.split(os.sep)[0] in ("features", "coords", "figures",
                                         "manifests", "audit", "logs",
                                         "__pycache__"):
            continue
        for fn in sorted(files):
            if fn.endswith((".py", ".sh", ".yaml")):
                rel = os.path.normpath(os.path.join(rel_root, fn))
                entries[f"exp009_code/{rel}"] = E9.sha256_file(
                    os.path.join(root, fn))

    def diff(prefix):
        sel_b = {k: v for k, v in base.items() if k.startswith(prefix)}
        sel_e = {k: v for k, v in entries.items() if k.startswith(prefix)}
        return (sorted(set(sel_e) - set(sel_b)),
                sorted(set(sel_b) - set(sel_e)),
                sorted(k for k in set(sel_e) & set(sel_b)
                       if sel_e[k] != sel_b[k]))

    a, r, ch = diff("exp009_code/")
    if changelog is None:
        ck.add("immutability.exp009_code",
               (not a and not r and not ch) or (not full_mode),
               n_entries=len(entries), added=a, removed=r, changed=ch,
               note=None if full_mode else
               "informational in subset mode: code iterates between campaigns; "
               "the full run rebuilds the baseline at extract start")
    else:
        declared = {c["file"]: c for c in changelog["changes"]}
        ch_undecl = [f for f in ch
                     if f[len("exp009_code/"):] not in declared]
        gen_ok = all(base.get(f"exp009_code/{f}") == c["old_sha256"]
                     for f, c in declared.items())
        ck.add("immutability.exp009_code_generation_inputs",
               full_mode and not a and not r and not ch_undecl and gen_ok,
               declared=sorted(declared), added=a, removed=r,
               changed_undeclared=ch_undecl,
               baseline_matches_changelog_old_sha=gen_ok,
               note="every generation-time input byte-identical to "
                    "baseline.json except the changelog-declared files "
                    "(whose baseline hashes themselves match the declared "
                    "old_sha256)")
    for prefix in ("reused/", "visionts/", "ckpt/", "ln_inputs/",
                   "ln_predictions/"):
        a, r, ch = diff(prefix)
        ck.add(f"immutability.{prefix.rstrip('/')}",
               not a and not r and not ch,
               added=a, removed=r, changed=ch)


def check_declared_changes(ck, changelog):
    """Every changelog-declared change: baseline hash == declared old, current
    file == declared new, archived diff byte-exact, scope declared; the config
    diff must stay confined to the figures-section keys."""
    base = json.load(open(E9.BASELINE_JSON))["entries"]
    allowed_scopes = {"figure_presentation_only", "audit_tooling_only"}
    cfg_keys_allowed = {"imagenet_style", "domain_rank_header"}
    probs = []
    for c in changelog["changes"]:
        rel = c["file"]
        if base.get(f"exp009_code/{rel}") != c["old_sha256"]:
            probs.append({"file": rel, "problem": "old_sha256 != baseline"})
        p = os.path.join(E9.HERE, rel)
        if not os.path.isfile(p) or \
                E9.sha256_file(p) != c["new_sha256"]:
            probs.append({"file": rel, "problem": "new_sha256 != current"})
        dp = os.path.join(E9.AUDIT, c["diff"])
        if not os.path.isfile(dp) or E9.sha256_file(dp) != c["diff_sha256"]:
            probs.append({"file": rel, "problem": "diff missing/ drifted"})
        if c.get("scope") not in allowed_scopes:
            probs.append({"file": rel, "problem": "undeclared scope"})
        if rel.endswith(".yaml"):
            with open(dp) as f:
                body = [l for l in f if l[:1] in "+-"
                        and not l.startswith(("+++", "---"))]
            keys = {m.group(1) for l in body
                    if (m := re.search(r'"([a-z_0-9]+)":', l))}
            if not keys <= cfg_keys_allowed:
                probs.append({"file": rel, "problem": "config keys beyond "
                              "figures section", "keys": sorted(keys)})
    ck.add("postaudit.declared_changes_verified", not probs,
           n_declared=len(changelog["changes"]), problems=probs,
           note="old/new/diff hashes all verified; config diff confined to "
                "figures-section keys (presentation-only)")


def check_preserved_records(ck, changelog):
    """Generation-time records must survive byte-identical (never overwritten):
    audit/baseline.json, features/_cache_index.json, original audit_report.json."""
    probs = []
    for rel, want in changelog["preserved_records"].items():
        p = os.path.join(E9.HERE, rel)
        got = E9.sha256_file(p) if os.path.isfile(p) else None
        if got != want:
            probs.append({"file": rel, "got": got, "want": want})
    ck.add("postaudit.preserved_records_untouched", not probs,
           n=len(changelog["preserved_records"]), problems=probs,
           note="generation-time baseline / cache index / original audit "
                "report are byte-identical to their pre-restyle values")


def check_cache_binding(ck, changelog):
    """Close the generation-version loop for all 40 feature artifacts:
    (a) each cache key is reproducible from its recorded key_components;
    (b) the components' code/config fingerprints equal the changelog's
    frozen GENERATION-time fingerprints (changed files pinned by old_md5,
    unchanged files must equal the current file)."""
    gen = changelog["generation_fingerprints"]
    idx = json.load(open(E9.CACHE_INDEX))
    probs = []
    for name, ent in idx.items():
        comps = ent["key_components"]
        key = hashlib.sha256(json.dumps(
            comps, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
        if key != ent["key"]:
            probs.append({"artifact": name,
                          "problem": "key not reproducible from components"})
        if comps["exp009_config"] != gen["exp009_config_md5"]:
            probs.append({"artifact": name,
                          "problem": "config md5 != generation fingerprint"})
        for fn, md5v in comps["exp009_code"].items():
            want = gen["exp009_code_md5"].get(fn)
            if want is None:
                want = E9.md5_file(os.path.join(E9.HERE, fn))
            if md5v != want:
                probs.append({"artifact": name, "file": fn,
                              "problem": "code md5 != generation fingerprint"})
    ck.add("postaudit.cache_key_generation_binding", not probs,
           n_artifacts=len(idx), problems=probs[:20], n_problems=len(probs),
           note="all cache keys reproducible from recorded components and "
                "bound to the generation-time code/config fingerprints")


def coords_fit_key(domain, state):
    """Independently recompute stage-02's resume key from current inputs
    (auditor recomputes; never trusts the recorded value alone)."""
    tag = state if state == E9.STATE_BEFORE else \
        f"F0__{domain}__s{E9.F0_SEED}"
    shas = {n: E9.sha256_file(os.path.join(E9.FEATURES, n))
            for n in (f"{tag}__imagenet.npz", f"{tag}__timesx.npz")}
    comps = {"domain": domain, "state": state, "tsne": TSNE_CFG,
             "flat_dim": E9.FLAT_DIM,
             "imagenet_sha": shas[f"{tag}__imagenet.npz"],
             "timesx_sha": shas[f"{tag}__timesx.npz"]}
    return hashlib.sha256(json.dumps(
        comps, sort_keys=True).encode()).hexdigest()


def check_coords(ck, domains_subset):
    bad = []
    for d in domains_subset:
        for state in (E9.STATE_BEFORE, E9.STATE_AFTER):
            p = os.path.join(E9.COORDS, f"{d}__{state}.csv")
            pp = os.path.join(E9.COORDS, f"{d}__{state}.params.json")
            if not (os.path.isfile(p) and os.path.isfile(pp)):
                bad.append({"fit": f"{d}/{state}", "problem": "missing"})
                continue
            par = json.load(open(pp))
            with open(p) as f:
                rows = list(csv.DictReader(f))
            tsx_n = sum(1 for r in rows if r["source"] == "timesx")
            if len(rows) != E9.IMAGENET_N + tsx_n or tsx_n != par["n_timesx"]:
                bad.append({"fit": f"{d}/{state}", "problem": "row counts"})
            if not all(np.isfinite(float(r["x"])) and np.isfinite(float(r["y"]))
                       for r in rows):
                bad.append({"fit": f"{d}/{state}", "problem": "non-finite"})
            if par.get("fit_key") != coords_fit_key(d, state):
                bad.append({"fit": f"{d}/{state}", "problem": "stale fit_key"})
            if par["tsne"] != TSNE_CFG:
                bad.append({"fit": f"{d}/{state}", "problem": "tsne params"})
    ck.add("coords.rows_finite_fresh", not bad, problems=bad)


def read_domain_summary_z0():
    out = {}
    with open(E9.DOMAIN_SUMMARY_CSV) as f:
        for r in csv.DictReader(f):
            if r["method"] == "Z0" and r["seed_stat"] == "single":
                out[r["domain"]] = (float(r["MSE"]), float(r["MAE"]))
    assert len(out) == 19
    return out


def check_metrics(ck, domains_subset):
    z0_ref = read_domain_summary_z0()
    rows, bad = [], []
    for d in domains_subset:
        m0 = E9.metrics_from_npz(E9.Z0_PRED_NPZ, domain=d)
        mf = E9.metrics_from_npz(os.path.join(
            E9.LN_PREDICTIONS,
            f"F0__{d}__s{E9.F0_SEED}__test.npz"))
        ref = z0_ref[d]
        if abs(m0["mse"] - ref[0]) > METRIC_ATOL or \
                abs(m0["mae"] - ref[1]) > METRIC_ATOL:
            bad.append({"domain": d, "z0_recomputed": m0["mse"],
                        "z0_exp004": ref[0]})
        zf = np.load(os.path.join(E9.LN_PREDICTIONS,
                                  f"F0__{d}__s{E9.F0_SEED}__test.npz"),
                     allow_pickle=False)
        if str(zf["method"]) != "F0" or int(zf["seed"]) != E9.F0_SEED:
            bad.append({"domain": d, "problem": "npz method/seed mismatch"})
        ep = E9.f0_selection()[d]["best_epoch"]
        if int(zf["epoch"]) != ep:
            bad.append({"domain": d, "problem": "npz epoch != best_epoch"})
        dmse = 100.0 * (mf["mse"] - m0["mse"]) / m0["mse"]
        rows.append({"domain": d, "n_vars": m0["n_vars"],
                     "n_windows": m0["n_windows"],
                     "z0_mse": m0["mse"], "z0_mae": m0["mae"],
                     "f0_mse": mf["mse"], "f0_mae": mf["mae"],
                     "delta_mse_pct": dmse, "best_epoch": ep})
    ck.add("metrics.z0_matches_exp004", not bad, problems=bad,
           rule="window(d-normalized) -> var mean -> var-equal in domain")
    overall_z0 = float(np.mean([r["z0_mse"] for r in rows]))
    overall_f0 = float(np.mean([r["f0_mse"] for r in rows]))
    return rows, overall_z0, overall_f0


def write_report(rows, overall_z0, overall_f0, mode, postaudit=False):
    n_imp = sum(1 for r in rows if r["f0_mse"] < r["z0_mse"])
    lines = [
        "# EXP-009 — TimesX 分域 LN 微调前后编码器特征 t-SNE（REPORT）",
        "",
        f"模式：**{mode}** | 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"协议：exp009_tsne_v1 | 基线代码：dcd2425",
        "",
        "## 方法（一句话）",
        "对每个领域，把 1,000 张冻结 ImageNet-1k val 参考图与该领域全部原生测试窗"
        "（仅 past 96 步）的编码器特征（vision_model.norm 全 57 token 展平 43,776 维，"
        "无池化/PCA）做一次联合 t-SNE；微调前 = Z0（原始 ImageNet 预训练 MAE-base），"
        "微调后 = F0 seed2021 每域 best_ft（EXP-004 selection.json，非 fallback）。",
        "",
        "## 逐域结果（单种子 s2021；std MSE/MAE：窗口 d 归一 → 变量平均 → 域内变量等权）",
        "",
        "| domain | n_vars | n_win | Z0 MSE | Z0 MAE | F0_s2021 MSE | F0_s2021 MAE "
        "| ΔMSE% | best_epoch | 图 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['domain']} | {r['n_vars']} | {r['n_windows']} "
            f"| {r['z0_mse']:.6f} | {r['z0_mae']:.6f} "
            f"| {r['f0_mse']:.6f} | {r['f0_mae']:.6f} "
            f"| {r['delta_mse_pct']:+.2f} | {r['best_epoch']} "
            f"| figures/{r['domain']}/ |")
    lines += [
        f"| **overall（19 域等权）** | 190 | 2474 | {overall_z0:.6f} "
        f"| — | {overall_f0:.6f} | — "
        f"| {100.0 * (overall_f0 - overall_z0) / overall_z0:+.2f} | — | — |",
        "",
        f"单种子 s2021 下 F0 优于 Z0 的领域数：**{n_imp}/{len(rows)}**"
        f"（ΔMSE% 为负 = 微调更优）。",
        "",
        "## 结论边界（务必一并阅读）",
        "- 事后探索性可视化：单种子（s2021）、非盲测（与 EXP-004 共享数据与划分）。",
        "- 仅观测编码器 vision_model.norm 输出；不能将预测误差变化归因于"
        "“编码器模态距离”变化的机制。",
        "- t-SNE 距离仅定性；前后独立拟合，绝对坐标不可跨图直接比较。",
        "- 指标为单种子复算（Z0 已与 EXP-004 domain_summary 逐域核对一致），"
        "不代表三种子均值。",
        "",
        "## 产物",
        "- 主图 38 张：`figures/<domain>/{Z0,F0_s2021}.png`（300 dpi）＋"
        " 19 张并排图 `figures/_side_by_side/`＋`index.html`。",
        "- 坐标 38 份：`coords/<domain>__{Z0,F0_s2021}.csv`（含 sample_id/来源/"
        "域/变量/状态/坐标）与 `.params.json`（KL、n_iter、冻结参数、输入指纹）。",
        "- 特征缓存（大文件，留在服务器）：`features/`（清单位 manifests 内、"
        "SHA256 与缓存键见 `features/_cache_index.json`）。",
        "- 审计：`audit/audit_report.json`（完整性/检查点/特征/hook/不可变性/"
        "溯源/指标 七类检查）。",
    ]
    if postaudit:
        lines += [
            "",
            "## 绘图阶段后复核（postaudit re-attestation）",
            "- 特征提取与 t-SNE 之后，应用户要求做了两轮纯绘图改动（ImageNet 参考图"
            "点样式；每域性能排名题头），并为此扩展了审计脚本本身。",
            "- 三处代码/配置变更均经哈希锚定：生成时文件内容先重建并双哈希验证"
            "（SHA256 == audit/baseline.json 且 MD5 == _cache_index.json 生成指纹），"
            "再以 unified diff 存档于 `audit/code_diff/`，见"
            " `audit/postaudit_changelog.json`。",
            "- `audit/baseline.json`、`features/_cache_index.json` 与原始"
            " `audit/audit_report.json` 原样保留（本报告的"
            " postaudit.preserved_records_untouched 检查项逐字节验证）。",
            "- 本复核报告：`audit/audit_report_final.json`（覆盖最终交付文件；"
            "特征/坐标/指标检查全部按现状重算通过）。",
        ]
    tmp = REPORT_MD + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, REPORT_MD)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", default=None,
                    help="subset mode for the small trial (default: all 19, "
                         "full mode)")
    ap.add_argument("--postaudit", action="store_true",
                    help="re-attest the FINAL files against the generation-"
                         "time records via audit/postaudit_changelog.json; "
                         "writes audit_report_final.json and never touches "
                         "the original audit_report.json")
    args = ap.parse_args()
    changelog = None
    if args.postaudit:
        assert not args.domains, "--postaudit runs full mode only"
        with open(CHANGELOG_PATH) as f:
            changelog = json.load(f)
    domains_subset = (args.domains.split(",") if args.domains
                      else E9.DOMAIN_NAMES)
    if changelog is not None:
        mode = "postaudit re-attestation (full 19 domains, FINAL files)"
    else:
        mode = "full (19 domains)" if not args.domains else \
            f"PARTIAL (domains: {','.join(domains_subset)})"

    ck = Checks()
    check_completeness(ck, domains_subset)
    check_checkpoints(ck, domains_subset)
    check_features(ck, domains_subset)
    check_hook_equivalence(ck, domains_subset)
    check_immutability(ck, domains_subset, changelog)
    if changelog is not None:
        check_declared_changes(ck, changelog)
        check_preserved_records(ck, changelog)
        check_cache_binding(ck, changelog)
    check_coords(ck, domains_subset)
    rows, overall_z0, overall_f0 = check_metrics(ck, domains_subset)

    if not args.domains:
        ck.add("full_coverage", len(rows) == 19, n=len(rows))

    out_json = AUDIT_JSON_FINAL if changelog is not None else AUDIT_JSON
    E9.atomic_write_json(
        {"protocol_version": E9.CFG["protocol_version"],
         "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": mode,
         "changelog": "audit/postaudit_changelog.json"
         if changelog is not None else None,
         "all_ok": ck.all_ok, "checks": ck.results},
        out_json)
    write_report(rows, overall_z0, overall_f0, mode,
                 postaudit=changelog is not None)
    print(f"[audit] ALL {'OK' if ck.all_ok else 'PROBLEMS'} — "
          f"{out_json}", flush=True)
    sys.exit(0 if ck.all_ok else 1)


if __name__ == "__main__":
    main()
