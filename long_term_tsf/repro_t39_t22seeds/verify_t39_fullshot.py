"""EXP-039 gates: 16 configs x 2 seeds (2022/2023) x 2 arms (64 new
trainings), anchored against the frozen EXP-038 seed-2021 archive
(30 t38 train jsons + EXP-037 ETTh1_96 reuse pair), the EXP-003 LN
per-seed archives (48 pins) and EXP-033 Freeze16.

Default: verifies ALL 64 arms and appends a 'fullshot' section to
logs/verify_t39.json + writes logs/recompute_f64.json (64 records).
--smoke: verifies only ETTh1_96 (4 arms) -> logs/verify_t39_smoke.json
+ logs/recompute_f64_smoke.json; same gate structure.

Gates per config (hard):
  BG0 presence      train json for every (arm, seed) of this round
  BG1 provenance    fingerprint code_md5/data_sha256/ckpt_sha256 == the
                    config's own EXP-003 fingerprint; args deltas only
                    {model_id, save_dir, ft_type, vm_pretrained, gpu,
                    seed}; BG1_seed_pinned: fingerprint.seed ==
                    fingerprint.args.seed == success.seed == requested
  BG2 truth         true.npy shape == formula and bitwise == EXP-003
                    (seed-independent, sequential test loader)
  BG3 metrics       float64 recompute of MSE/MAE from pred/true with
                    finiteness checks vs SUCCESS.txt <= 1e-6 (64 records)
  BG4 init          per seed: All init == ckpt per-key, Rand differs,
                    init shas differ across arms; trainable name lists
                    equal across arms AND across all 3 seeds (2021 pair
                    from the t38/t37 archives)
  BG5 RNG           same seed: rng_after_init + rng_at_train_entry equal
                    across arms; cross-seed: all four RNG sub-states
                    pairwise distinct across the 3 seeds
  BG6 sample order  exactly 1 RandomSampler iter per arm; epoch index
                    count == formula; same-seed order sha equal across
                    arms; cross-seed order sha distinct
  BG7 first batch   same-seed first train-batch hashes equal across
                    arms; cross-seed distinct
  BG8 final ckpts   per config 6 final checkpoints (4 new + 2 seed-2021
                    archived): TENSOR-CONTENT digests pairwise distinct
                    (15/15 pairs) and every ckpt has >=1 vision_model
                    key differing from the frozen ImageNet weights
  BG9 LN anchors    per seed: EXP-003 archive SUCCESS == LN48 pin
                    (seed2021 additionally == LN16 via t39_common
                    import-time assert); aggregate_snapshot.csv (sha
                    pinned) 3-seed mean/std(ddof=1) cross-check <= 1e-6;
                    LN overall self-check == user anchor; FREEZE16
                    coverage
  BG10 provenance   EXP-038 seed-2021 evidence: 30 t38 train jsons +
                    2 t37 reuse jsons readable with complete success

Aggregation (report): per (ds, seed) the 4H equal-weight mean of
All/Rand (measured this round), LN (that seed's pin), Freeze (shared);
per-seed overall; cross-seed mean +/- std (ddof=1). Paper Table 22 is
compared only at the dataset-4H-mean and overall layers.
"""
import argparse
import csv
import os
import sys

import numpy as np

import t39_common as X   # first: fixes sys.path for the t38/t22 packages
import t38_common as C
import t22_common as T

SMOKE_CONFIGS = [("ETTh1", 96)]

F64_REC = []
_CACHE = {}

CKPT_PATH = os.path.join(C.ROOT, "..", "ckpt", "mae_visualize_vit_base.pth")


def gate(rows, key, ok, detail):
    rows.append({"gate": key, "verdict": "PASS" if ok else "FAIL",
                 "detail": detail})
    return ok


def canonical_sd_sha(sd):
    import torch
    h = __import__("hashlib").sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(("|%s|%s|" % (t.dtype, tuple(t.shape))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def check_arm_seed(ds, h, arm, s, rows, summary):
    tag = "%s_%d_s%d_%s" % (ds, h, s, arm)
    p = X.train_json(ds, h, arm, s)
    if not os.path.isfile(p):
        return gate(rows, "BG0_presence[%s]" % tag, False,
                    "train json missing: %s" % p)
    rec = C.load_json(p)
    gate(rows, "BG0_presence[%s]" % tag, True, "train json present")

    fp003 = C.load_json(os.path.join(C.exp003_job(ds, h), "fingerprint.json"))
    fp = rec["fingerprint"]
    gate(rows, "BG1_code_md5[%s]" % tag, fp["code_md5"] == fp003["code_md5"],
         "12 code files md5 == EXP-003(%s)" %
         os.path.dirname(C.exp003_job(ds, h)))
    gate(rows, "BG1_data_sha[%s]" % tag,
         fp["data_sha256"] == fp003["data_sha256"],
         "%s sha %s" % (ds, list(fp["data_sha256"].values())[0][:16]))
    got_ckpt = list(fp["ckpt_sha256"].values())[0]
    gate(rows, "BG1_ckpt_sha[%s]" % tag, got_ckpt == C.CKPT_SHA256,
         "fingerprint ckpt %s... pinned %s..." %
         (got_ckpt[:16], C.CKPT_SHA256[:16]))
    adv = rec.get("args_delta_check", {})
    gate(rows, "BG1_args_delta[%s]" % tag,
         adv.get("verdict") == "ONLY_ALLOWED_DELTAS"
         and not adv.get("unexpected"),
         "deltas %s" % sorted(adv.get("deltas", {})))
    gate(rows, "BG1_seed_pinned[%s]" % tag,
         rec.get("requested_seed") == s and fp.get("seed") == s
         and fp.get("args", {}).get("seed") == s
         and rec.get("success", {}).get("seed") == s,
         "requested/fingerprint/fingerprint.args/success seed all == %d" % s)

    n = C.expected_n_windows(ds, h)
    true003 = np.load(os.path.join(C.exp003_job(ds, h), "results", "_",
                                   "true.npy"))
    res_dir = os.path.join(X.b_job_dir(ds, h, arm, s), "results", "_")
    pred = np.load(os.path.join(res_dir, "pred.npy"))
    true = np.load(os.path.join(res_dir, "true.npy"))
    gate(rows, "BG2_truth_bitwise[%s]" % tag,
         true.shape == true003.shape and np.array_equal(true, true003),
         "true %s bitwise==EXP-003 (formula n=%d)" % (true.shape, n))
    metric = T.load_metric()
    pr = pred[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
    tr = true[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
    finite = bool(np.isfinite(pr).all() and np.isfinite(tr).all())
    m = T.metrics_from_arrays(metric, pr.astype(np.float64),
                              tr.astype(np.float64))
    succ = rec["success"]
    dmse = abs(float(m[1]) - succ["mse"])
    dmae = abs(float(m[0]) - succ["mae"])
    gate(rows, "BG3_metrics[%s]" % tag,
         finite and dmse <= C.METRIC_TOL and dmae <= C.METRIC_TOL
         and np.isfinite(float(m[1])) and np.isfinite(float(m[0])),
         "finite=%s f64 recomp %.10f/%.10f vs SUCCESS %.10f/%.10f "
         "(d %.2e/%.2e)" % (finite, m[1], m[0], succ["mse"], succ["mae"],
                            dmse, dmae))
    summary[(ds, h)][(arm, s)] = {"mse": succ["mse"], "mae": succ["mae"]}
    F64_REC.append({"ds": ds, "h": h, "arm": arm, "seed": s,
                    "source": "this run",
                    "mse_f64": float(m[1]), "mae_f64": float(m[0]),
                    "mse_success": succ["mse"], "mae_success": succ["mae"],
                    "d_mse": dmse, "d_mae": dmae})
    return True


def check_seed_pair(ds, h, s, rows, recs21):
    """Cross-arm gates at one seed, plus cross-seed inequality gates
    against the archived seed-2021 evidence (recs21[arm] or None for
    configs ETTh1_96 uses t37; others use t38 logs)."""
    tag = "%s_%d_s%d" % (ds, h, s)
    recs = {}
    for arm in C.B_ARMS:
        p = X.train_json(ds, h, arm, s)
        if os.path.isfile(p):
            recs[arm] = C.load_json(p)
    if len(recs) != 2:
        return False

    ia, ir = recs["all"]["init"], recs["rand"]["init"]
    gate(rows, "BG4_all_eq_ckpt[%s]" % tag,
         ia["vs_ckpt"]["verdict"] == "ALL_KEYS_EQUAL_TO_CKPT",
         "%d/%d keys == ckpt" % (ia["vs_ckpt"]["n_equal"],
                                 ia["vs_ckpt"]["n_keys"]))
    gate(rows, "BG4_rand_differs[%s]" % tag,
         ir["vs_ckpt"]["verdict"] == "RANDOM_INIT_RECORDED",
         "%d/%d keys differ from ckpt" % (ir["vs_ckpt"]["n_differ"],
                                          ir["vs_ckpt"]["n_keys"]))
    gate(rows, "BG4_init_shas_differ[%s]" % tag,
         ia["canonical_sha256"] != ir["canonical_sha256"],
         "all %s... rand %s..." % (ia["canonical_sha256"][:16],
                                   ir["canonical_sha256"][:16]))

    # trainable lists equal across arms AND across all three seeds
    lists = {s: (ia["trainable_names"], ia["trainable_count"])}
    s21_a = recs21.get("all") if recs21 else None
    s21_r = recs21.get("rand") if recs21 else None
    ok21 = bool(s21_a and s21_r)
    if ok21:
        lists[2021] = (s21_a["init"]["trainable_names"],
                       s21_a["init"]["trainable_count"])
    vals = set(v for v, _ in lists.values())
    gate(rows, "BG4_trainable_lists_consistent[%s]" % tag,
         ia["trainable_names"] == ir["trainable_names"]
         and len(vals) == 1 and ok21,
         "%d trainable params; identical lists across arms%s" %
         (ia["trainable_count"],
          " and seeds 2021/2022/2023" if ok21 else " (2021 evidence miss)"))

    for key in ("rng_after_init", "rng_at_train_entry"):
        a = ia[key] if key == "rng_after_init" else recs["all"][key]
        r = ir[key] if key == "rng_after_init" else recs["rand"][key]
        gate(rows, "BG5_%s_same_seed[%s]" % (key, tag), a == r,
             "identical across arms at seed %d" % s)
        if ok21:
            b = (s21_a["init"][key] if key == "rng_after_init"
                 else s21_a[key])
            c = (s21_r["init"][key] if key == "rng_after_init"
                 else s21_r[key])
            sub_ok = all(len({a[k], r[k], b[k], c[k]}) == 3
                         for k in a) and set(a) == set(b)
            gate(rows, "BG5_%s_cross_seed_differs[%s]" % (key, tag),
                 sub_ok,
                 "all %d sub-states pairwise distinct across seeds "
                 "2021/2022/2023" % len(a))

    sa, sr = recs["all"]["sampler"], recs["rand"]["sampler"]
    exp_epoch = C.expected_epoch_len(ds, h)
    gate(rows, "BG6_single_epoch_iter[%s]" % tag,
         sa["n_randomsampler_iters"] == 1 and
         sr["n_randomsampler_iters"] == 1,
         "iters all=%d rand=%d" % (sa["n_randomsampler_iters"],
                                   sr["n_randomsampler_iters"]))
    ea, er = sa["iters"][0], sr["iters"][0]
    gate(rows, "BG6_epoch_len[%s]" % tag,
         ea["n"] == er["n"] == exp_epoch,
         "n_idx %d/%d expect %d" % (ea["n"], er["n"], exp_epoch))
    gate(rows, "BG6_order_same_seed[%s]" % tag, ea["sha256"] == er["sha256"],
         "epoch index sha %s... equal across arms" % ea["sha256"][:16])
    if ok21:
        e21_a = s21_a["sampler"]["iters"][0]
        e21_r = s21_r["sampler"]["iters"][0]
        gate(rows, "BG6_order_cross_seed_differs[%s]" % tag,
             len({ea["sha256"], er["sha256"], e21_a["sha256"],
                  e21_r["sha256"]}) == 2
             and ea["sha256"] != e21_a["sha256"],
             "2022/2023 orders differ from 2021 (arms-in-seed equal)")

    fa, fr = recs["all"]["first_batch"], recs["rand"]["first_batch"]
    ok = fa.get("done") and fr.get("done") and \
        [t["sha256"] for t in fa["tensors"]] == \
        [t["sha256"] for t in fr["tensors"]]
    gate(rows, "BG7_first_batch_same_seed[%s]" % tag, ok,
         "tensors %s" % [t["shape"] for t in fa.get("tensors", [])])
    if ok21:
        f21a = s21_a["first_batch"]
        f21r = s21_r["first_batch"]
        gate(rows, "BG7_first_batch_cross_seed_differs[%s]" % tag,
             [t["sha256"] for t in fa["tensors"]] !=
             [t["sha256"] for t in f21a["tensors"]]
             and [t["sha256"] for t in f21r["tensors"]] !=
             [t["sha256"] for t in f21a["tensors"]],
             "first-batch hashes differ from seed 2021")

    for arm in C.B_ARMS:
        sc = recs[arm]["sampler_len_check"]
        gate(rows, "BG6_formula_check[%s_%s]" % (tag, arm), sc["ok"],
             "sampler %d vs formula %d" % (sc["got"],
                                           sc["expected_epoch_len"]))
    return True


def final_ckpt_path(ds, h, arm, s):
    if s == 2021:
        return os.path.join(X.t38_job_dir(ds, h, arm), "checkpoints", "_",
                            "checkpoint.pth")
    return os.path.join(X.b_job_dir(ds, h, arm, s), "checkpoints", "_",
                        "checkpoint.pth")


def check_bg8(ds, h, rows, cache):
    """6 final ckpts per config (4 new + 2 seed-2021 archived):
    tensor-content digests pairwise distinct; >=1 vision_model key
    differs from the frozen ImageNet weights in every ckpt."""
    tag = "%s_%d" % (ds, h)
    if "frozen" not in cache:
        import torch
        ck = torch.load(CKPT_PATH, map_location="cpu")
        if isinstance(ck, dict) and "model" in ck:
            ck = ck["model"]
        cache["frozen"] = {k: v for k, v in ck.items()}
    frozen = cache["frozen"]
    frozen_shas = {k: C.sha256_bytes(v.numpy().tobytes())
                   for k, v in frozen.items()}
    digests, changed = [], {}
    import torch
    vm_prefix = "vm.vision_model."
    for arm in C.B_ARMS:
        for s in (2021, 2022, 2023):
            p = final_ckpt_path(ds, h, arm, s)
            if not os.path.isfile(p):
                gate(rows, "BG8_presence[%s_s%d_%s]" % (tag, s, arm),
                     False, "final ckpt missing: %s" % p)
                return
            sd = torch.load(p, map_location="cpu")
            digests.append(canonical_sd_sha(sd))
            mapped = {kk[len(vm_prefix):]: kk for kk in sd
                      if kk.startswith(vm_prefix)}
            changed["s%d_%s" % (s, arm)] = sum(
                1 for k, fk in frozen_shas.items()
                if k in mapped
                and C.sha256_bytes(sd[mapped[k]].detach().cpu().contiguous()
                                   .numpy().tobytes()) != fk)
    n_pairs = len(digests) * (len(digests) - 1) // 2
    n_distinct = len(set(digests))
    gate(rows, "BG8_tensor_digests_pairwise_distinct[%s]" % tag,
         n_distinct == len(digests) == 6,
         "%d/%d pairwise-distinct tensor-content digests (%d pairs)" %
         (n_distinct, len(digests), n_pairs))
    gate(rows, "BG8_all_differ_from_frozen[%s]" % tag,
         all(v >= 1 for v in changed.values()),
         "changed vision_model keys per ckpt: %s" %
         {k: v for k, v in sorted(changed.items())})


def check_bg9(rows):
    """LN48 triple anchor + Freeze16 coverage."""
    ok_all = True
    for (ds, h) in X.CONFIGS:
        for s in X.LN_KEYS_SEEDS:
            p = X.ln_success_path(ds, h, s)
            if not os.path.isfile(p):
                ok_all &= gate(rows, "BG9_ln_archive[%s_%d_s%d]" %
                               (ds, h, s), False,
                               "EXP-003 SUCCESS missing: %s" % p)
                continue
            mse, mae = C.parse_success_mse_mae(p)
            pin = X.LN48["%s_%d_seed%d" % (ds, h, s)]
            got_seed = C.load_json(p).get("seed")
            ok_all &= gate(rows, "BG9_ln_pin[%s_%d_s%d]" % (ds, h, s),
                           (mse, mae) == pin and got_seed == s,
                           "archive %.16f/%.16f seed=%s == pin, seed %d" %
                           (mse, mae, str((pin[0], pin[1]))[:40], s))
    # aggregate_snapshot.csv cross-check: 3-seed mean and std(ddof=1)
    agg = T.load_aggregate()  # validates pinned sha256, returns means
    snap = {}
    with open(T.AGG_SNAPSHOT) as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], int(r["pred"]))
            if key[0] in C.ETT4:
                snap[key] = (float(r["mse_mean"]), float(r["mse_std"]),
                             float(r["mae_mean"]), float(r["mae_std"]))
    ok = len(snap) == 16
    for (ds, h), vals in snap.items():
        mses = [X.LN48["%s_%d_seed%d" % (ds, h, s)][0]
                for s in X.LN_KEYS_SEEDS]
        maes = [X.LN48["%s_%d_seed%d" % (ds, h, s)][1]
                for s in X.LN_KEYS_SEEDS]
        ok &= (abs(sum(mses) / 3.0 - vals[0]) <= C.METRIC_TOL
               and abs(float(np.std(mses, ddof=1)) - vals[1]) <= C.METRIC_TOL
               and abs(sum(maes) / 3.0 - vals[2]) <= C.METRIC_TOL
               and abs(float(np.std(maes, ddof=1)) - vals[3]) <= C.METRIC_TOL)
    ok_all &= gate(rows, "BG9_aggregate_snapshot",
                   ok, "48 pins vs aggregate_snapshot.csv mean/std "
                       "(ddof=1) all <= 1e-6; load_aggregate sha-gated")
    ov = X.ln_overall()
    ok_all &= gate(rows, "BG9_ln_overall_anchor",
                   abs(ov[0] - X.LN_OVERALL_ANCHOR[0]) <= C.METRIC_TOL
                   and abs(ov[1] - X.LN_OVERALL_ANCHOR[1]) <= C.METRIC_TOL,
                   "LN 3-seed overall %.16f/%.16f == user anchor" % ov)
    freeze16 = C.load_freeze16()
    ok_all &= gate(rows, "BG9_freeze16_coverage",
                   len(freeze16) == 16,
                   "16 freeze rows from %s; %s" % (C.t22_summary_csv(),
                                                   C.FREEZE16_SOURCE_NOTE))
    return ok_all


def check_bg10(rows):
    """EXP-038 seed-2021 provenance: 30 t38 train jsons + 2 t37 reuse."""
    ok_all = True
    n_ok = 0
    for (ds, h) in X.CONFIGS:
        for arm in C.B_ARMS:
            p = X.t38_train_json(ds, h, arm)
            if not os.path.isfile(p):
                ok_all &= gate(rows, "BG10_t38[%s_%d_%s]" % (ds, h, arm),
                               False, "t38 train json missing: %s" % p)
                continue
            rec = C.load_json(p)
            ok = isinstance(rec.get("success", {}).get("mse"), float) and \
                isinstance(rec.get("success", {}).get("mae"), float) and \
                rec["success"].get("seed") == 2021
            ok_all &= gate(rows, "BG10_t38[%s_%d_%s]" % (ds, h, arm), ok,
                           "t38 seed2021 evidence complete (mse/mae/seed)")
            n_ok += int(ok)
    gate(rows, "BG10_t38_coverage", n_ok == 30,
         "%d/30 t38 train jsons usable" % n_ok)
    return ok_all


def mean_std(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def aggregate(summary, rows):
    """Per (ds, seed) 4H equal means -> per-seed overall -> cross-seed
    mean +/- std (ddof=1). Paper anchors only at dataset/overall layer."""
    freeze16 = C.load_freeze16()
    per_seed_ds = {s: {} for s in X.LN_KEYS_SEEDS}   # s -> ds -> col -> m
    for s in X.LN_KEYS_SEEDS:
        for ds in C.ETT4:
            cells = [summary.get((ds, h), {}) for h in C.HORIZONS]
            have = [c for c in cells
                    if ("all", s) in c and ("rand", s) in c]
            if len(have) != 4:
                gate(rows, "AGG_coverage[%s_s%d]" % (ds, s), False,
                     "only %d/4 horizons present" % len(have))
                continue
            out = {}
            for k in ("all", "rand"):
                out[k] = {"mse": sum(c[(k, s)]["mse"] for c in have) / 4.0,
                          "mae": sum(c[(k, s)]["mae"] for c in have) / 4.0}
            out["ln"] = {"mse": sum(X.LN48["%s_%d_seed%d" % (ds, h, s)][0]
                                    for h in C.HORIZONS) / 4.0,
                         "mae": sum(X.LN48["%s_%d_seed%d" % (ds, h, s)][1]
                                    for h in C.HORIZONS) / 4.0}
            out["freeze"] = {
                "mse": sum(freeze16[(ds, h)][0] for h in C.HORIZONS) / 4.0,
                "mae": sum(freeze16[(ds, h)][1] for h in C.HORIZONS) / 4.0}
            out["n_horizons"] = 4
            per_seed_ds[s][ds] = out
        gate(rows, "AGG_coverage[%s_s%d]" % ("all_ds", s),
             len(per_seed_ds[s]) == 4,
             "seed %d: %d/4 datasets aggregated" % (s, len(per_seed_ds[s])))
    # per-seed overall
    per_seed_ov = {s: {} for s in X.LN_KEYS_SEEDS}
    for s, ds_map in per_seed_ds.items():
        if len(ds_map) != 4:
            continue
        for k in ("all", "rand", "ln", "freeze"):
            per_seed_ov[s][k] = {
                "mse": sum(ds_map[ds][k]["mse"] for ds in C.ETT4) / 4.0,
                "mae": sum(ds_map[ds][k]["mae"] for ds in C.ETT4) / 4.0}
    # cross-seed mean/std
    agg_ds = {}
    for ds in C.ETT4:
        if all(ds in per_seed_ds[s] for s in X.LN_KEYS_SEEDS):
            agg_ds[ds] = {}
            for k in ("all", "rand", "ln", "freeze"):
                mses = [per_seed_ds[s][ds][k]["mse"]
                        for s in X.LN_KEYS_SEEDS]
                maes = [per_seed_ds[s][ds][k]["mae"]
                        for s in X.LN_KEYS_SEEDS]
                mm, msd = mean_std(mses)
                am, asd = mean_std(maes)
                agg_ds[ds][k] = {"mse_mean": mm, "mse_std": msd,
                                 "mae_mean": am, "mae_std": asd,
                                 "per_seed_mse": mses, "per_seed_mae": maes}
    agg_ov = {}
    if all(per_seed_ov[s] for s in X.LN_KEYS_SEEDS):
        for k in ("all", "rand", "ln", "freeze"):
            mses = [per_seed_ov[s][k]["mse"] for s in X.LN_KEYS_SEEDS]
            maes = [per_seed_ov[s][k]["mae"] for s in X.LN_KEYS_SEEDS]
            mm, msd = mean_std(mses)
            am, asd = mean_std(maes)
            agg_ov[k] = {"mse_mean": mm, "mse_std": msd,
                         "mae_mean": am, "mae_std": asd,
                         "per_seed_mse": mses, "per_seed_mae": maes}
    cmp_rows = []
    for ds, m in agg_ds.items():
        for k in ("all", "rand", "ln", "freeze"):
            pmse, pmae = C.PAPER_T22[ds][k]
            cmp_rows.append({"ds": ds, "col": k,
                             "mse_mean": m[k]["mse_mean"],
                             "mse_std": m[k]["mse_std"],
                             "mae_mean": m[k]["mae_mean"],
                             "mae_std": m[k]["mae_std"],
                             "mse_paper": pmse, "mae_paper": pmae})
    for k in ("all", "rand", "ln", "freeze"):
        if k in agg_ov:
            pmse, pmae = C.PAPER_T22_OVERALL[k]
            cmp_rows.append({"ds": "Overall", "col": k,
                             "mse_mean": agg_ov[k]["mse_mean"],
                             "mse_std": agg_ov[k]["mse_std"],
                             "mae_mean": agg_ov[k]["mae_mean"],
                             "mae_std": agg_ov[k]["mae_std"],
                             "mse_paper": pmse, "mae_paper": pmae})
    return per_seed_ds, per_seed_ov, agg_ds, agg_ov, cmp_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="verify only ETTh1_96 (4 arms), standalone output")
    args = ap.parse_args()
    configs = SMOKE_CONFIGS if args.smoke else X.CONFIGS

    rows, summary = [], {}
    for ds, h in configs:
        summary[(ds, h)] = {}
        # archived seed-2021 evidence for cross-seed gates
        recs21 = {}
        for arm in C.B_ARMS:
            p21 = X.t38_train_json(ds, h, arm)
            if os.path.isfile(p21):
                recs21[arm] = C.load_json(p21)
        for s in X.SEEDS:
            for arm in C.B_ARMS:
                check_arm_seed(ds, h, arm, s, rows, summary)
            check_seed_pair(ds, h, s, rows, recs21)
        check_bg8(ds, h, rows, _CACHE)
    check_bg9(rows)
    check_bg10(rows)
    per_seed_ds, per_seed_ov, agg_ds, agg_ov, cmp_rows = \
        aggregate(summary, rows)

    n_fail = sum(1 for g in rows if g["verdict"] == "FAIL")
    expected_f64 = 4 if args.smoke else 64
    out = {
        "mode": "smoke" if args.smoke else "full",
        "all_pass": n_fail == 0,
        "n_gates": len(rows), "n_pass": sum(1 for g in rows
                                            if g["verdict"] == "PASS"),
        "gates": rows,
        "summary": {"%s_%d" % k: {"%s_s%d" % (a, s): v for (a, s), v
                                  in cell.items()}
                    for k, cell in summary.items()},
        "per_seed_dataset_4h_mean": {str(s): d for s, d
                                     in per_seed_ds.items()},
        "per_seed_overall": {str(s): d for s, d in per_seed_ov.items()},
        "agg_dataset_4h_mean": agg_ds,
        "agg_overall": agg_ov,
        "table22_comparison": cmp_rows,
        "timing": {},
        "gpu_peak": {},
    }
    for (ds, h) in configs:
        for arm in C.B_ARMS:
            for s in X.SEEDS:
                p = X.train_json(ds, h, arm, s)
                if not os.path.isfile(p):
                    continue
                rec = C.load_json(p)
                key = "%s_%d_%s_s%d" % (ds, h, arm, s)
                out["timing"][key] = rec["timing"]
                ph = rec.get("phases", {})
                peaks = {"train_peak_mb": ph.get("train_peak_mb"),
                         "vali_peak_mb": ph.get("vali_peak_mb"),
                         "test_peak_mb": ph.get("test_peak_mb")}
                vals = [v for v in peaks.values() if v is not None]
                peaks["overall_max_mb"] = max(vals) if vals else None
                out["gpu_peak"][key] = peaks
    out["recompute_f64"] = {
        "n_records": len(F64_REC), "coverage_ok": len(F64_REC) == expected_f64,
        "tol": C.METRIC_TOL, "records": F64_REC}
    if args.smoke:
        out_path = X.VERIFY_JSON.replace("verify_t39.json",
                                         "verify_t39_smoke.json")
        C.dump_json(out_path, out)
        C.dump_json(os.path.join(X.LOG_DIR, "recompute_f64_smoke.json"),
                    out["recompute_f64"])
    else:
        out_path = X.VERIFY_JSON
        C.merge_json(out_path, "fullshot", out)
        C.dump_json(X.RECOMPUTE_JSON, out["recompute_f64"])
    for g in rows:
        if g["verdict"] == "FAIL":
            print("[t39][FAIL] %s %s" % (g["gate"], g["detail"]))
    print("[t39]%s %d/%d gates pass (%d FAIL) -> %s" %
          ("[smoke]" if args.smoke else "", out["n_pass"], out["n_gates"],
           n_fail, out_path))
    if not args.smoke and agg_ov:
        for k in ("all", "rand", "ln", "freeze"):
            v = agg_ov[k]
            print("[t39] overall %s: %.4f+-%.4f / %.4f+-%.4f "
                  "(paper %.3f/%.3f)" %
                  (k, v["mse_mean"], v["mse_std"], v["mae_mean"],
                   v["mae_std"], C.PAPER_T22_OVERALL[k][0],
                   C.PAPER_T22_OVERALL[k][1]))
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
