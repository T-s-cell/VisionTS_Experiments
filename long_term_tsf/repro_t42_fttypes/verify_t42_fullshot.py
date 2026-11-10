"""EXP-042 gates: 144 tasks = 16 configs x 3 seeds x 3 ft types
(bias/mlp/attn), anchored against the EXP-003 audited archives (code
md5 / data sha / truth) and the frozen MAE-base ckpt.

Default: verifies ALL 144 tasks + VG7 re-entry over all 144 + report
aggregation; appends a 'fullshot' section to logs/verify_t42.json and
writes logs/recompute_f64.json (144 records) + logs/reentry_check_t42.json.
--smoke: verifies only ETTh1_96 x 3 seeds x 3 ft (9 units) -> standalone
logs/verify_t42_smoke.json + recompute_f64_smoke.json + reentry over the
9 smoke tasks; same gate structure (VG8/VG9 skipped).
--decide DS_H FT SEED: re-entry decide mode for run_t42_all.sh (prints
SKIP / MOVE:<dir> / RUN; exit 0).

Gates (hard):
  VG1 provenance    per task: train json present; fingerprint
                    code_md5/data_sha256/ckpt_sha256 == the config's own
                    EXP-003 fingerprint; args deltas ONLY the t42
                    whitelist {model_id, save_dir, ft_type, gpu, seed}
                    with {model_id, save_dir, ft_type, seed} present and
                    vm_pretrained == 1 (original pretrained weights);
                    seed pinned 4 ways (requested/fingerprint/
                    fingerprint.args/success)
  VG2 scope 3-way   scope_audit actual == scope predicate == verify's
                    independent predicate re-derivation from the frozen
                    digest key names, per ft; driver init trainable
                    names/count identical across ALL tasks of the ft
                    (and == scope); sets pairwise distinct across fts;
                    bias includes patch_embed.proj.bias; encoder+
                    decoder coverage statement recorded
  VG3 init==frozen  per task: vs_ckpt verdict ALL_KEYS_EQUAL_TO_CKPT
                    with n_equal == n_keys == frozen digest n_keys
  VG4 frozen kept   per task: final ckpt file sha == delivered train
                    json final_ckpt_sha256; reloaded from disk, EVERY
                    frozen key bitwise == frozen digest (hard); >=1
                    trainable key differs (recorded); driver's own
                    post_frozen_check verdict cross-checked
  VG5 truth         per task: true.npy shape == formula and bitwise ==
                    EXP-003 (seed-independent, sequential test loader);
                    144 checks (48 triples x 3 ft)
  VG6 metrics       per task: pred/true shapes == EXP-003 archived,
                    finite; float64 MSE/MAE recompute vs SUCCESS
                    <= 1e-6 (144 records)
  VG7 re-entry      per task completeness (skip-if-done semantics:
                    SUCCESS + train json pins + job fingerprint +
                    EXP-003 anchors + FULL config pin incl. expected
                    args rebuild + arrays actually loaded + checkpoint
                    file sha == delivered); all 144 reported to
                    reentry_check_t42.json
  VG8 references    LN48 pins == EXP-003 per-seed archives (48);
                    EXP-039 All/Rand train jsons (64) + EXP-038 seed-2021
                    jsons (30+2 reuse) readable; Freeze16 = 16 rows
  VG9 queue state   after queue: every one of the 144 re-entry decisions
                    is SKIP (no incomplete/overwritable job); native
                    run_train ABORT semantics (fingerprint mismatch ->
                    rc=3, dir kept) pinned by the EXP-003 code md5 in VG1
  VG10 artifacts    per task: train json fields (argv/phases/sampler+
                    len check ok/first_batch/init/args_delta_check/
                    final_ckpt_sha256/timing) + job dir files (args.json,
                    fingerprint.json, SUCCESS.txt, checkpoint.pth,
                    pred.npy, true.npy)

Aggregation (report, full only): per (ft, ds, seed) 4H equal mean ->
per-ft per-seed dataset / per-seed overall -> cross-seed mean +/- std
(ddof=1); paired diffs vs LN (same seed, same ds).  Paper Table 22 is
compared ONLY at the dataset-4H-mean and overall layers.
"""
import argparse
import os
import sys

import numpy as np

import t42_common as X   # first: fixes sys.path for the t39/t38/t22 pkgs
import t38_common as C
import t22_common as T

SMOKE_TASKS = X.SMOKE

F64_REC = []
_CACHE = {}


def gate(rows, key, ok, detail):
    rows.append({"gate": key, "verdict": "PASS" if ok else "FAIL",
                 "detail": detail})
    return ok


CKPT_PATH = os.path.join(C.ROOT, "..", "ckpt", "mae_visualize_vit_base.pth")


def frozen_sd():
    import torch
    if "frozen" not in _CACHE:
        ck = torch.load(CKPT_PATH, map_location="cpu")
        _CACHE["frozen"] = ck["model"]
    return _CACHE["frozen"]


def prefix_sd_sha(sd, prefix="vm.vision_model."):
    return {k[len(prefix):]: C.sha256_bytes(
        sd[k].detach().cpu().contiguous().numpy().tobytes())
        for k in sd if k.startswith(prefix)}


# --------------------------------------------------------------------------
def check_vg1_vg3_vg6(ds, h, s, ft, rows, summary, dig):
    tag = "%s_%d_s%d_%s" % (ds, h, s, ft)
    p = X.train_json(ds, h, s, ft)
    if not os.path.isfile(p):
        gate(rows, "VG1_presence[%s]" % tag, False,
             "train json missing: %s" % p)
        return
    rec = C.load_json(p)
    gate(rows, "VG1_presence[%s]" % tag, True, "train json present")

    fp003 = C.load_json(os.path.join(C.exp003_job(ds, h), "fingerprint.json"))
    fp = rec["fingerprint"]
    gate(rows, "VG1_code_md5[%s]" % tag, fp["code_md5"] == fp003["code_md5"],
         "12 code files md5 == EXP-003(%s)" %
         os.path.dirname(C.exp003_job(ds, h)))
    gate(rows, "VG1_data_sha[%s]" % tag,
         fp["data_sha256"] == fp003["data_sha256"],
         "%s sha %s" % (ds, list(fp["data_sha256"].values())[0][:16]))
    got_ckpt = list(fp["ckpt_sha256"].values())[0]
    gate(rows, "VG1_ckpt_sha[%s]" % tag, got_ckpt == C.CKPT_SHA256,
         "fingerprint ckpt %s... pinned %s..." %
         (got_ckpt[:16], C.CKPT_SHA256[:16]))
    adv = rec.get("args_delta_check", {})
    deltas = sorted(adv.get("deltas", {}).keys())
    ok_delta = adv.get("verdict") == "ONLY_ALLOWED_DELTAS" and \
        not adv.get("unexpected") and \
        set(deltas) == set(X.FT_ALLOWED_ARG_DELTA)
    gate(rows, "VG1_args_delta[%s]" % tag, ok_delta,
         "deltas %s (whitelist exactly observed)" % deltas)
    gate(rows, "VG1_vm_pretrained[%s]" % tag,
         fp.get("args", {}).get("vm_pretrained") == 1,
         "vm_pretrained == 1 (original MAE-base pretrained weights)")
    gate(rows, "VG1_seed_pinned[%s]" % tag,
         rec.get("requested_seed") == s and fp.get("seed") == s
         and fp.get("args", {}).get("seed") == s
         and rec.get("success", {}).get("seed") == s,
         "requested/fingerprint/fingerprint.args/success seed all == %d" % s)

    # VG3: init == frozen ckpt
    ia = rec["init"]
    vc = ia["vs_ckpt"]
    gate(rows, "VG3_init_eq_ckpt[%s]" % tag,
         vc.get("verdict") == "ALL_KEYS_EQUAL_TO_CKPT"
         and vc["n_equal"] == vc["n_keys"] == dig["n_keys"],
         "%d/%d keys == ckpt (digest n_keys %d)" %
         (vc["n_equal"], vc["n_keys"], dig["n_keys"]))

    # VG6: shapes/finiteness/f64 recompute
    n = C.expected_n_windows(ds, h)
    res_dir = os.path.join(X.job_dir(ds, h, s, ft), "results", "_")
    pred = np.load(os.path.join(res_dir, "pred.npy"))
    true = np.load(os.path.join(res_dir, "true.npy"))
    ref_pred = np.load(os.path.join(C.exp003_job(ds, h), "results", "_",
                                    "pred.npy"), mmap_mode="r")
    metric = T.load_metric()
    pr = pred[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
    tr = true[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
    finite = bool(np.isfinite(pr).all() and np.isfinite(tr).all())
    m = T.metrics_from_arrays(metric, pr.astype(np.float64),
                              tr.astype(np.float64))
    succ = rec["success"]
    dmse = abs(float(m[1]) - succ["mse"])
    dmae = abs(float(m[0]) - succ["mae"])
    gate(rows, "VG6_metrics[%s]" % tag,
         pred.shape == ref_pred.shape and finite
         and dmse <= C.METRIC_TOL and dmae <= C.METRIC_TOL
         and np.isfinite(float(m[1])) and np.isfinite(float(m[0])),
         "shape %s==EXP-003 finite=%s f64 %.10f/%.10f vs SUCCESS "
         "%.10f/%.10f (d %.2e/%.2e)" %
         (pred.shape, finite, m[1], m[0], succ["mse"], succ["mae"],
          dmse, dmae))
    summary[(ds, h, s, ft)] = {"mse": succ["mse"], "mae": succ["mae"]}
    F64_REC.append({"ds": ds, "h": h, "seed": s, "ft": ft,
                    "mse_f64": float(m[1]), "mae_f64": float(m[0]),
                    "mse_success": succ["mse"], "mae_success": succ["mae"],
                    "d_mse": dmse, "d_mae": dmae})
    return rec


def check_vg4(ds, h, s, ft, rows):
    tag = "%s_%d_s%d_%s" % (ds, h, s, ft)
    rec = C.load_json(X.train_json(ds, h, s, ft))
    ck = os.path.join(X.job_dir(ds, h, s, ft), "checkpoints", "_",
                      "checkpoint.pth")
    if not os.path.isfile(ck):
        gate(rows, "VG4_ckpt_present[%s]" % tag, False,
             "final ckpt missing: %s" % ck)
        return
    fsha = C.sha256_of(ck)
    gate(rows, "VG4_file_sha[%s]" % tag,
         fsha == rec.get("final_ckpt_sha256"),
         "ckpt file sha == delivered final_ckpt_sha256 (%s...)" % fsha[:16])
    import torch
    dig_shas = _CACHE["frozen_shas"]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    got = prefix_sd_sha(sd)
    trainable = set(rec["init"]["trainable_names"])
    frozen_changed = [k for k in dig_shas
                      if got.get(k) != dig_shas[k]]
    trainable_changed = [k for k in frozen_changed if k in trainable]
    pf = rec.get("post_frozen_check", {})
    gate(rows, "VG4_frozen_bitwise[%s]" % tag,
         len(frozen_changed) == 0
         and pf.get("verdict") == "FROZEN_KEYS_BITWISE_UNCHANGED",
         "frozen keys changed %d/%d; driver post_frozen_check verdict %s"
         % (len(frozen_changed), len(dig_shas), pf.get("verdict", "?")))
    gate(rows, "VG4_trainable_moved[%s]" % tag,
         len(trainable_changed) >= 1,
         "%d/%d trainable keys differ from frozen (recorded)"
         % (len(trainable_changed), len(trainable)))


def check_vg5(ds, h, s, ft, rows):
    tag = "%s_%d_s%d_%s" % (ds, h, s, ft)
    n = C.expected_n_windows(ds, h)
    true003 = np.load(os.path.join(C.exp003_job(ds, h), "results", "_",
                                   "true.npy"))
    true = np.load(os.path.join(X.job_dir(ds, h, s, ft), "results", "_",
                                "true.npy"))
    gate(rows, "VG5_truth_bitwise[%s]" % tag,
         true.shape == true003.shape and np.array_equal(true, true003),
         "true %s bitwise==EXP-003 (formula n=%d)" % (true.shape, n))


def check_vg2(rows, tasks, dig):
    """Scope 3-way: scope audit == verify predicate re-derivation ==
    driver init records (identical across all tasks of the ft)."""
    scope = C.load_json(X.SCOPE_JSON)

    def predicate(ft, n):
        if "ln" == ft:
            return "norm" in n
        if "bias" == ft:
            return "bias" in n
        if "none" == ft:
            return False
        if "mlp" in ft:
            return ".mlp." in n
        if "attn" in ft:
            return ".attn." in n
        raise ValueError(ft)

    key_names = set(dig["keys"].keys())
    ok_all = True
    for ft in X.FT_TYPES:
        ent = scope["per_ft"].get(ft)
        ok_all &= gate(rows, "VG2_scope_record[%s]" % ft, ent is not None,
                       "scope audit entry present" if ent else "missing")
        if ent is None:
            continue
        preds = sorted(n for n in key_names if predicate(ft, n))
        ok_all &= gate(rows, "VG2_3way[%s]" % ft,
                       ent["trainable_names"] == ent["predicate_set"]
                       == preds and ent["predicate_match"],
                       "audit actual == audit predicate == verify "
                       "re-derivation (%d names)" % len(preds))
        ok_all &= gate(rows, "VG2_bias_conv[%s]" % ft,
                       (ft != "bias")
                       or ("patch_embed.proj.bias" in ent["trainable_names"]
                           and ent.get("patch_embed_proj_bias_included")),
                       "patch_embed.proj.bias (Conv2d) explicitly in the "
                       "bias trainable set")
        ok_all &= gate(rows, "VG2_coverage[%s]" % ft,
                       ent["encoder_blocks_trainable"] > 0
                       and ent["decoder_blocks_trainable"] > 0
                       and "AND" in ent["coverage_statement"],
                       "%s" % ent["coverage_statement"])
        # driver init identical across ALL tasks of this ft
        names_ref, cnt_ref = None, None
        n_dr = 0
        for (ds, h, s, f) in tasks:
            if f != ft:
                continue
            rec = C.load_json(X.train_json(ds, h, s, ft))
            ia = rec["init"]
            if names_ref is None:
                names_ref, cnt_ref = ia["trainable_names"], ia["trainable_count"]
            same = ia["trainable_names"] == names_ref and \
                ia["trainable_count"] == cnt_ref and \
                ia["trainable_names"] == ent["trainable_names"]
            if not same:
                ok_all &= gate(rows, "VG2_driver_names[%s_%s_%d_s%d]"
                               % (ft, ds, h, s), False,
                               "driver trainable record != scope audit")
                continue
            n_dr += 1
        ok_all &= gate(rows, "VG2_driver_names[%s]" % ft,
                       n_dr == sum(1 for t in tasks if t[3] == ft),
                       "%d/%d tasks: init trainable list == scope audit "
                       "(%d names, %d params)"
                       % (n_dr, sum(1 for t in tasks if t[3] == ft),
                          len(names_ref), cnt_ref))
    sets = {ft: set(scope["per_ft"][ft]["trainable_names"])
            for ft in X.FT_TYPES if ft in scope.get("per_ft", {})}
    ok_all &= gate(rows, "VG2_ft_sets_distinct",
                   len({frozenset(v) for v in sets.values()})
                   == len(X.FT_TYPES) == len(sets),
                   "3 trainable sets pairwise distinct")
    return ok_all


# --------------------------------------------------------------------------
# re-entry (VG7 / VG9 / runner decide mode) — one implementation
# --------------------------------------------------------------------------
def reentry_expected_args(ds, h, s, ft, jd):
    job = X.job_name(ds, h, s, ft)
    exp003_args = C.load_json(os.path.join(C.exp003_job(ds, h), "args.json"))
    expected = dict(exp003_args)
    expected.update({
        "model_id": job,
        "save_dir": os.path.abspath(jd),
        "ft_type": ft,
        "vm_pretrained": 1,
        "gpu": 0,
        "seed": s,
    })
    return expected, exp003_args


def reentry_check_one(ds, h, s, ft, mode):
    """mode 'decide': SKIP / MOVE:<dir>\\t<reason> / RUN.
    mode 'report': PASS / FAIL[<job>]: <reason> / RUN."""
    jd = X.job_dir(ds, h, s, ft)
    job = X.job_name(ds, h, s, ft)

    def _why(msg):
        if mode == "decide":
            return "MOVE:%s\t%s" % (jd, msg)
        return "FAIL[%s]: %s" % (job, msg)

    if not os.path.isdir(jd):
        return "RUN", "no job dir"
    succ = os.path.join(jd, "SUCCESS.txt")
    if not os.path.isfile(succ):
        return _why("no SUCCESS.txt"), ""
    try:
        sv = C.load_json(succ)
        assert isinstance(sv.get("mse"), float) and \
            isinstance(sv.get("mae"), float)
        assert sv.get("seed") == s
    except Exception as e:
        return _why("SUCCESS unreadable: %s" % e), ""
    p = X.train_json(ds, h, s, ft)
    try:
        rec = C.load_json(p)
        assert rec["success"]["mse"] == sv["mse"]
        assert rec["fingerprint"]["seed"] == s
        assert rec["fingerprint"]["args"]["seed"] == s
        assert rec.get("requested_seed") == s
        assert rec["init"]["vs_ckpt"]["verdict"] == \
            "ALL_KEYS_EQUAL_TO_CKPT"
        assert rec["post_frozen_check"]["verdict"] == \
            "FROZEN_KEYS_BITWISE_UNCHANGED"
    except Exception as e:
        return _why("train json incomplete: %s" % e), ""
    try:
        fpj = C.load_json(os.path.join(jd, "fingerprint.json"))
        assert fpj == rec["fingerprint"], "job fingerprint != train json"
        assert fpj.get("seed") == s and fpj.get("args", {}).get("seed") == s
        assert all(fpj.get(k)
                   for k in ("code_md5", "data_sha256", "ckpt_sha256"))
    except Exception as e:
        return _why("fingerprint incomplete: %s" % e), ""
    try:
        fp3 = C.load_json(os.path.join(C.exp003_job(ds, h),
                                       "fingerprint.json"))
        assert fpj["code_md5"] == fp3.get("code_md5")
        assert fpj["data_sha256"] == fp3.get("data_sha256")
        assert fpj["ckpt_sha256"] == fp3.get("ckpt_sha256")
    except Exception as e:
        return _why("independent anchors: %s" % e), ""
    try:
        expected, _ = reentry_expected_args(ds, h, s, ft, jd)
        got_args = C.load_json(os.path.join(jd, "args.json"))
        assert got_args == expected, "job args.json != expected config"
        assert rec["fingerprint"]["args"] == expected
    except Exception as e:
        return _why("config pin: %s" % e), ""
    try:
        rd = os.path.join(jd, "results", "_")
        ref_shapes = {}
        for name in ("pred.npy", "true.npy"):
            q = os.path.join(rd, name)
            if not os.path.isfile(q) or os.path.getsize(q) <= 0:
                raise AssertionError("missing/empty %s" % name)
            with open(q, "rb") as fh:
                if fh.read(6) != b"\x93NUMPY":
                    raise AssertionError("invalid npy header: %s" % name)
            ref = np.load(os.path.join(C.exp003_job(ds, h), "results", "_",
                                       name), mmap_mode="r")
            arr = np.load(q, allow_pickle=False)
            assert arr.shape == ref.shape, \
                "%s shape %s != EXP-003 %s" % (name, arr.shape, ref.shape)
            assert np.isfinite(arr).all(), "non-finite in %s" % name
            ref_shapes[name] = list(arr.shape)
    except Exception as e:
        return _why("array validity: %s" % e), ""
    try:
        ck = os.path.join(jd, "checkpoints", "_", "checkpoint.pth")
        if not os.path.isfile(ck):
            raise AssertionError("missing final checkpoint")
        fsha = C.sha256_of(ck)
        assert fsha == rec.get("final_ckpt_sha256"), \
            "ckpt file sha != delivered final_ckpt_sha256"
    except Exception as e:
        return _why("checkpoint validity: %s" % e), ""
    detail = "mse=%.6f mae=%.6f shape=%s" % (sv["mse"], sv["mae"],
                                             ref_shapes["pred.npy"])
    if mode == "decide":
        return "SKIP\t%s (json=%s)" % (detail, os.path.basename(p)), ""
    return "PASS", detail


def run_reentry(tasks, mode, rows=None):
    records, n_skip, n_move = [], 0, 0
    for (ds, h, s, ft) in tasks:
        verdict, detail = reentry_check_one(ds, h, s, ft, mode)
        if verdict.startswith("SKIP"):
            n_skip += 1
        elif verdict.startswith("MOVE"):
            n_move += 1
        if rows is not None and mode == "report":
            ok = verdict == "PASS"
            gate(rows, "VG7_reentry[%s_%d_s%d_%s]" % (ds, h, s, ft), ok,
                 detail or verdict)
        records.append({"ds": ds, "h": h, "seed": s, "ft": ft,
                        "verdict": verdict, "detail": detail})
    return records, n_skip, n_move


# --------------------------------------------------------------------------
def check_vg8(rows):
    """References loadable: LN48 pins vs archives; t39/t38 jsons; Freeze16."""
    ok_all = True
    for (ds, h) in X.CONFIGS:
        for s in X.LN_KEYS_SEEDS:
            p = X.ln_success_path(ds, h, s)
            if not os.path.isfile(p):
                ok_all &= gate(rows, "VG8_ln_archive[%s_%d_s%d]" %
                               (ds, h, s), False,
                               "EXP-003 SUCCESS missing: %s" % p)
                continue
            mse, mae = C.parse_success_mse_mae(p)
            pin = X.LN48["%s_%d_seed%d" % (ds, h, s)]
            ok_all &= gate(rows, "VG8_ln_pin[%s_%d_s%d]" % (ds, h, s),
                           (mse, mae) == pin,
                           "archive == pin %.16f/%.16f" % (mse, mae))
    n39 = 0
    for (ds, h) in X.CONFIGS:
        for arm in C.B_ARMS:
            for s in (2022, 2023):
                p = X.t39_train_json(ds, h, arm, s)
                ok = os.path.isfile(p)
                if ok:
                    rec = C.load_json(p)
                    ok = isinstance(rec.get("success", {}).get("mse"), float)
                n39 += int(ok)
                if not ok:
                    ok_all &= gate(rows, "VG8_t39[%s_%d_%s_s%d]"
                                   % (ds, h, arm, s), False,
                                   "missing/incomplete: %s" % p)
    ok_all &= gate(rows, "VG8_t39_coverage", n39 == 64,
                   "%d/64 EXP-039 All/Rand train jsons (seeds 22/23)" % n39)
    n38 = 0
    for (ds, h) in X.CONFIGS:
        for arm in C.B_ARMS:
            p = X.t38_train_json(ds, h, arm)
            ok = os.path.isfile(p)
            if ok:
                rec = C.load_json(p)
                ok = isinstance(rec.get("success", {}).get("mse"), float) \
                    and rec["success"].get("seed") == 2021
            n38 += int(ok)
            if not ok:
                ok_all &= gate(rows, "VG8_t38[%s_%d_%s]" % (ds, h, arm),
                               False, "missing/incomplete: %s" % p)
    ok_all &= gate(rows, "VG8_t38_coverage", n38 == 32,
                   "%d/32 EXP-038 seed-2021 jsons (30 t38 + 2 t37 reuse)"
                   % n38)
    freeze16 = C.load_freeze16()
    ok_all &= gate(rows, "VG8_freeze16_coverage", len(freeze16) == 16,
                   "16 freeze rows from %s; %s" % (C.t22_summary_csv(),
                                                   C.FREEZE16_SOURCE_NOTE))
    return ok_all


def check_vg9(rows, tasks):
    records, n_skip, n_move = run_reentry(tasks, "decide")
    gate(rows, "VG9_all_complete_after_queue",
         n_skip == len(tasks) and n_move == 0,
         "%d/%d re-entry decisions SKIP, %d MOVE (native run_train "
         "fingerprint-mismatch ABORT rc=3 semantics pinned by the "
         "EXP-003 code md5 in VG1)" % (n_skip, len(tasks), n_move))
    return records


def check_vg10(ds, h, s, ft, rows):
    tag = "%s_%d_s%d_%s" % (ds, h, s, ft)
    rec = C.load_json(X.train_json(ds, h, s, ft))
    jd = X.job_dir(ds, h, s, ft)
    need_json = ["argv", "fingerprint", "phases", "sampler",
                 "sampler_len_check", "first_batch", "init",
                 "args_delta_check", "final_ckpt_sha256", "timing",
                 "success", "post_frozen_check"]
    missing = [k for k in need_json if k not in rec]
    slc = rec.get("sampler_len_check", {})
    missing.append("" if slc.get("ok") else "sampler_len_check.ok")
    missing.append("" if rec.get("first_batch", {}).get("done")
                   else "first_batch.done")
    missing.append("" if rec["init"]["vs_ckpt"].get("verdict")
                   == "ALL_KEYS_EQUAL_TO_CKPT" else "init.vs_ckpt.verdict")
    need_files = ["args.json", "fingerprint.json", "SUCCESS.txt",
                  os.path.join("checkpoints", "_", "checkpoint.pth"),
                  os.path.join("results", "_", "pred.npy"),
                  os.path.join("results", "_", "true.npy")]
    missing_f = [f for f in need_files
                 if not os.path.isfile(os.path.join(jd, f))]
    gate(rows, "VG10_artifacts[%s]" % tag,
         not any(missing) and not missing_f,
         "json fields %d/%d, files %d/%d present"
         % (len(need_json) - len([m for m in missing if m]),
            len(need_json), len(need_files) - len(missing_f),
            len(need_files)))


# --------------------------------------------------------------------------
def mean_std(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def collect_refs():
    """-> refs[k][ds][seed] = (mse4h, mae4h); k in ln/all/rand; freeze
    handled separately (seed-independent single values)."""
    refs = {"ln": {}, "all": {}, "rand": {}}
    for ds in C.ETT4:
        for k in refs:
            refs[k][ds] = {}
        for s in X.LN_KEYS_SEEDS:
            refs["ln"][ds][s] = (
                sum(X.LN48["%s_%d_seed%d" % (ds, h, s)][0]
                    for h in C.HORIZONS) / 4.0,
                sum(X.LN48["%s_%d_seed%d" % (ds, h, s)][1]
                    for h in C.HORIZONS) / 4.0)
        for arm, k in (("all", "all"), ("rand", "rand")):
            for s in X.LN_KEYS_SEEDS:
                vals_m, vals_a = [], []
                for h in C.HORIZONS:
                    p = X.t38_train_json(ds, h, arm) if s == 2021 else \
                        X.t39_train_json(ds, h, arm, s)
                    rec = C.load_json(p)
                    vals_m.append(rec["success"]["mse"])
                    vals_a.append(rec["success"]["mae"])
                refs[k][ds][s] = (sum(vals_m) / 4.0, sum(vals_a) / 4.0)
    return refs


def aggregate(summary, rows):
    """Per (ft, ds, seed) 4H means -> per-seed overall -> cross-seed
    mean/std (ddof=1); LN paired diffs same-seed same-ds."""
    refs = collect_refs()
    per_seed_ds = {ft: {s: {} for s in X.LN_KEYS_SEEDS}
                   for ft in X.FT_TYPES}
    for ft in X.FT_TYPES:
        for s in X.LN_KEYS_SEEDS:
            for ds in C.ETT4:
                cells = [summary.get((ds, h, s, ft)) for h in C.HORIZONS]
                if any(c is None for c in cells):
                    gate(rows, "AGG_coverage[%s_%s_s%d]" % (ft, ds, s),
                         False, "missing horizons")
                    continue
                per_seed_ds[ft][s][ds] = {
                    "mse": sum(c["mse"] for c in cells) / 4.0,
                    "mae": sum(c["mae"] for c in cells) / 4.0}
    per_seed_ov = {ft: {} for ft in X.FT_TYPES}
    for ft in X.FT_TYPES:
        for s in X.LN_KEYS_SEEDS:
            dsm = per_seed_ds[ft][s]
            if len(dsm) == 4:
                per_seed_ov[ft][s] = {
                    "mse": sum(dsm[ds]["mse"] for ds in C.ETT4) / 4.0,
                    "mae": sum(dsm[ds]["mae"] for ds in C.ETT4) / 4.0}
    agg_ds = {}
    for ft in X.FT_TYPES:
        agg_ds[ft] = {}
        for ds in C.ETT4:
            mses = [per_seed_ds[ft][s][ds]["mse"] for s in X.LN_KEYS_SEEDS]
            maes = [per_seed_ds[ft][s][ds]["mae"] for s in X.LN_KEYS_SEEDS]
            mm, msd = mean_std(mses)
            am, asd = mean_std(maes)
            agg_ds[ft][ds] = {"mse_mean": mm, "mse_std": msd,
                              "mae_mean": am, "mae_std": asd,
                              "per_seed_mse": mses, "per_seed_mae": maes}
    agg_ov = {}
    for ft in X.FT_TYPES:
        mses = [per_seed_ov[ft][s]["mse"] for s in X.LN_KEYS_SEEDS]
        maes = [per_seed_ov[ft][s]["mae"] for s in X.LN_KEYS_SEEDS]
        mm, msd = mean_std(mses)
        am, asd = mean_std(maes)
        agg_ov[ft] = {"mse_mean": mm, "mse_std": msd, "mae_mean": am,
                      "mae_std": asd, "per_seed_mse": mses,
                      "per_seed_mae": maes}
    # paired diffs vs LN (same seed, same ds, 4H-mean layer)
    paired = {}
    for ft in X.FT_TYPES:
        paired[ft] = {}
        for ds in C.ETT4:
            per_seed = {}
            for s in X.LN_KEYS_SEEDS:
                per_seed[s] = {
                    "mse": per_seed_ds[ft][s][ds]["mse"]
                    - refs["ln"][ds][s][0],
                    "mae": per_seed_ds[ft][s][ds]["mae"]
                    - refs["ln"][ds][s][1]}
            pm, psd = mean_std([v["mse"] for v in per_seed.values()])
            pam, pasd = mean_std([v["mae"] for v in per_seed.values()])
            paired[ft][ds] = {"per_seed": per_seed, "mse_mean": pm,
                              "mse_std": psd, "mae_mean": pam,
                              "mae_std": pasd}
    return per_seed_ds, per_seed_ov, agg_ds, agg_ov, paired, refs


def build_comparison(agg_ds, agg_ov, refs):
    cmp_rows = []
    for ft in X.FT_TYPES:
        for ds, m in agg_ds[ft].items():
            pmse, pmae = X.PAPER_T42[ds][ft]
            cmp_rows.append({"ds": ds, "col": ft,
                             "mse_mean": m["mse_mean"],
                             "mse_std": m["mse_std"],
                             "mae_mean": m["mae_mean"],
                             "mae_std": m["mae_std"],
                             "mse_paper": pmse, "mae_paper": pmae})
        pmse, pmae = X.PAPER_T42_OVERALL[ft]
        cmp_rows.append({"ds": "Overall", "col": ft,
                         "mse_mean": agg_ov[ft]["mse_mean"],
                         "mse_std": agg_ov[ft]["mse_std"],
                         "mae_mean": agg_ov[ft]["mae_mean"],
                         "mae_std": agg_ov[ft]["mae_std"],
                         "mse_paper": pmse, "mae_paper": pmae})
    return cmp_rows


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="verify only the 9 ETTh1_96 units, standalone out")
    ap.add_argument("--decide", nargs=3, metavar=("DS_H", "FT", "SEED"),
                    help="re-entry decide mode for the runner")
    args = ap.parse_args()

    if args.decide:
        ds, hh = args.decide[0].rsplit("_", 1)
        verdict, _ = reentry_check_one(ds, int(hh), int(args.decide[2]),
                                       args.decide[1], "decide")
        print(verdict)
        return

    tasks = SMOKE_TASKS if args.smoke else X.TASKS
    dig = C.load_json(X.FROZEN_DIGEST_JSON)
    ck = frozen_sd()
    _CACHE["frozen_shas"] = {
        k: C.sha256_bytes(v.detach().cpu().contiguous().numpy().tobytes())
        for k, v in ck.items()}

    rows, summary = [], {}
    for (ds, h, s, ft) in tasks:
        p = X.train_json(ds, h, s, ft)
        if not os.path.isfile(p):
            gate(rows, "VG1_presence[%s_%d_s%d_%s]" % (ds, h, s, ft), False,
                 "train json missing: %s" % p)
            continue
        rec = check_vg1_vg3_vg6(ds, h, s, ft, rows, summary, dig)
        if rec is not None:
            check_vg4(ds, h, s, ft, rows)
            check_vg5(ds, h, s, ft, rows)
            check_vg10(ds, h, s, ft, rows)

    check_vg2(rows, tasks, dig)

    # VG7 re-entry
    if args.smoke:
        recs7, _, _ = run_reentry(tasks, "decide")
        gate(rows, "VG7_smoke_complete", all(
            r["verdict"].startswith("SKIP") for r in recs7),
            "%d/9 smoke units complete (SKIP)" %
            sum(1 for r in recs7 if r["verdict"].startswith("SKIP")))
        C.dump_json(X.REENTRY_JSON.replace("reentry_check_t42.json",
                                           "reentry_check_t42_smoke.json"),
                    {"summary": {"n_tasks": len(recs7),
                                 "n_skip": sum(1 for r in recs7
                                               if r["verdict"].startswith(
                                                   "SKIP"))},
                     "records": recs7})
    else:
        recs7, _, _ = run_reentry(tasks, "report", rows)
        n_pass7 = sum(1 for r in recs7 if r["verdict"] == "PASS")
        gate(rows, "VG7_coverage", len(recs7) == 144,
             "%d/144 re-entry reports (%d PASS)" % (len(recs7), n_pass7))
        C.dump_json(X.REENTRY_JSON, {"summary": {
            "n_tasks": len(recs7), "n_pass": n_pass7,
            "n_fail": len(recs7) - n_pass7}, "records": recs7})

    if args.smoke:
        agg = {}
        cmp_rows = []
    else:
        check_vg8(rows)
        check_vg9(rows, tasks)
        per_seed_ds, per_seed_ov, agg_ds, agg_ov, paired, refs = \
            aggregate(summary, rows)
        cmp_rows = build_comparison(agg_ds, agg_ov, refs)
        agg = {"per_seed_dataset_4h_mean":
               {"%s_s%d" % (ft, s): d for ft in X.FT_TYPES
                for s, d in per_seed_ds[ft].items()},
               "per_seed_overall": {ft: {str(s): d for s, d in v.items()}
                                    for ft, v in per_seed_ov.items()},
               "agg_dataset_4h_mean": agg_ds, "agg_overall": agg_ov,
               "paired_vs_ln": paired}

    n_fail = sum(1 for g in rows if g["verdict"] == "FAIL")
    expected_f64 = 9 if args.smoke else 144

    def skey(k):
        return "%s_%d_s%d_%s" % (k[0], k[1], k[2], k[3])

    out = {
        "mode": "smoke" if args.smoke else "full",
        "all_pass": n_fail == 0,
        "n_gates": len(rows),
        "n_pass": sum(1 for g in rows if g["verdict"] == "PASS"),
        "gates": rows,
        "summary": {skey(k): v for k, v in summary.items()},
        "timing": {}, "gpu_peak": {},
    }
    if not args.smoke:
        out["aggregation"] = agg
        out["table22_comparison"] = cmp_rows
    for (ds, h, s, ft) in tasks:
        p = X.train_json(ds, h, s, ft)
        if not os.path.isfile(p):
            continue
        rec = C.load_json(p)
        key = "%s_%d_%s_s%d" % (ds, h, ft, s)
        out["timing"][key] = rec["timing"]
        ph = rec.get("phases", {})
        peaks = {"train_peak_mb": ph.get("train_peak_mb"),
                 "vali_peak_mb": ph.get("vali_peak_mb"),
                 "test_peak_mb": ph.get("test_peak_mb")}
        vals = [v for v in peaks.values() if v is not None]
        peaks["overall_max_mb"] = max(vals) if vals else None
        out["gpu_peak"][key] = peaks
    out["recompute_f64"] = {
        "n_records": len(F64_REC),
        "coverage_ok": len(F64_REC) == expected_f64,
        "tol": C.METRIC_TOL, "records": F64_REC}
    if args.smoke:
        out_path = X.VERIFY_JSON.replace("verify_t42.json",
                                         "verify_t42_smoke.json")
        C.dump_json(out_path, out)
        C.dump_json(os.path.join(X.LOG_DIR, "recompute_f64_smoke.json"),
                    out["recompute_f64"])
    else:
        out_path = X.VERIFY_JSON
        C.merge_json(out_path, "fullshot", out)
        C.dump_json(X.RECOMPUTE_JSON, out["recompute_f64"])
    for g in rows:
        if g["verdict"] == "FAIL":
            print("[t42][FAIL] %s %s" % (g["gate"], g["detail"]))
    print("[t42]%s %d/%d gates pass (%d FAIL) -> %s" %
          ("[smoke]" if args.smoke else "", out["n_pass"], out["n_gates"],
           n_fail, out_path))
    if not args.smoke and agg:
        for ft in X.FT_TYPES:
            v = out["aggregation"]["agg_overall"][ft]
            print("[t42] overall %s: %.4f+-%.4f / %.4f+-%.4f "
                  "(paper %.3f/%.3f)" %
                  (ft, v["mse_mean"], v["mse_std"], v["mae_mean"],
                   v["mae_std"], X.PAPER_T42_OVERALL[ft][0],
                   X.PAPER_T42_OVERALL[ft][1]))
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
