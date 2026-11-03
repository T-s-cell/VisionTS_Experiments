"""EXP-037 task B gates (ETTh1 H=96 seed=2021, All vs Rand-VM full finetune).
Reads logs/train_{all,rand}.json + the two formal-training job dirs under
save_t37_full/ and the EXP-003 audit records; appends a 'fullshot' section
to logs/verify_t37.json. Exit 0 only if all gates pass.

Gates:
  BG1 provenance: both arms' fingerprint code_md5 == EXP-003 eta
      fingerprint code_md5; data_sha256 equal to EXP-003's; ckpt_sha256 ==
      pinned frozen ckpt; args deltas vs EXP-003 args.json only the allowed
      set {model_id, save_dir, ft_type, vm_pretrained, gpu}
  BG2 test truth bitwise: both arms' results/_/true.npy == EXP-003 seed2021
      original, element for element ((2785*7, 96, 1) float32)
  BG3 metrics: standardization-space MSE/MAE recomputed in float64 from
      pred/true (restored to (2785, 96, 7)) vs each arm SUCCESS.txt <= 1e-6
  BG4 init evidence (user correction #2): all arm initial vision_model ==
      ckpt tensors bitwise (0 differing keys); rand arm differs (>=1 key,
      count recorded); canonical init shas differ across arms; trainable
      name lists identical across arms
  BG5 RNG: torch/np/python state hashes identical across arms both after
      VisionTS.__init__ and at train entry
  BG6 sample order: exactly 1 RandomSampler epoch per arm, 51,751 indices,
      sha256 of the index sequence identical across arms (first/last 20
      also equal)
  BG7 first train batch tensor hashes identical across arms (auxiliary)
  BG8 final checkpoints: sha256 recorded, both differ from the frozen ckpt
      and from each other
  BG9 anchors: EXP-003 LN SUCCESS == pinned LN anchor; EXP-033 freeze
      SUCCESS (eta, if present) == pinned freeze anchor
"""
import os
import sys

import numpy as np

import t37_common as C
import t22_common as T


def gate(gates, tag, ok, detail):
    gates.append((tag, "PASS" if ok else "FAIL", detail))


def main():
    gates = []
    recs = {arm: C.load_json(C.train_json(arm)) for arm in C.B_ARMS}

    fp003_path = os.path.join(C.exp003_job(), "fingerprint.json")
    fp003 = C.load_json(fp003_path)
    for arm in C.B_ARMS:
        fp = recs[arm]["fingerprint"]
        gate(gates, "BG1_code_md5[%s]" % arm,
             fp["code_md5"] == fp003["code_md5"],
             "12 code files md5 %s EXP-003 eta fingerprint"
             % ("==" if fp["code_md5"] == fp003["code_md5"] else "!="))
        gate(gates, "BG1_data_sha[%s]" % arm,
             fp["data_sha256"] == fp003["data_sha256"],
             "ETTh1.csv sha %s" % list(fp["data_sha256"].values())[0][:16])
        got_ckpt = list(fp["ckpt_sha256"].values())[0]
        gate(gates, "BG1_ckpt_sha[%s]" % arm, got_ckpt == C.CKPT_SHA256,
             "fingerprint ckpt sha %s... pinned %s..." % (got_ckpt[:16], C.CKPT_SHA256[:16]))
        adv = recs[arm]["args_delta_check"]
        gate(gates, "BG1_args_delta[%s]" % arm,
             adv["verdict"] == "ONLY_ALLOWED_DELTAS" and not adv["unexpected"],
             "deltas %s" % sorted(adv["deltas"]))

    # BG2 truth bitwise
    n_flat, h = C.ETTH1["flat_rows"], C.ETTH1["pred"]
    n = T.expected_n_windows("ETTh1", 96)
    assert (n, n * 7) == (C.ETTH1["n_windows"], n_flat)
    exp003_true_path = os.path.join(C.exp003_job(), "results", "_", "true.npy")
    true003 = np.load(exp003_true_path)
    gate(gates, "BG2_exp003_truth_shape", true003.shape == (n * 7, h, 1),
         "EXP-003 true %s" % (true003.shape,))
    metric = T.load_metric()
    summary = {}
    for arm in C.B_ARMS:
        res_dir = os.path.join(C.b_job_dir(arm), "results", "_")
        pred = np.load(os.path.join(res_dir, "pred.npy"))
        true = np.load(os.path.join(res_dir, "true.npy"))
        gate(gates, "BG2_truth_bitwise[%s]" % arm,
             true.shape == true003.shape and np.array_equal(true, true003),
             "true %s bitwise==EXP-003: %s" %
             (true.shape, bool(true.shape == true003.shape and np.array_equal(true, true003))))
        pr = pred[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
        tr = true[..., 0].reshape(n, 7, h).transpose(0, 2, 1)
        m = T.metrics_from_arrays(metric, pr, tr)  # [mae, mse, rmse, mape, mspe] f64
        succ = recs[arm]["success"]
        dmse = abs(float(m[1]) - succ["mse"])
        dmae = abs(float(m[0]) - succ["mae"])
        gate(gates, "BG3_metrics[%s]" % arm,
             dmse <= C.METRIC_TOL and dmae <= C.METRIC_TOL,
             "recomp %.10f/%.10f vs SUCCESS %.10f/%.10f (d %.2e/%.2e tol %.0e)" %
             (m[1], m[0], succ["mse"], succ["mae"], dmse, dmae, C.METRIC_TOL))
        summary[arm] = {"mse_recomp": float(m[1]), "mae_recomp": float(m[0]),
                        "mse_success": succ["mse"], "mae_success": succ["mae"]}

    # BG4 init evidence
    ia, ir = recs["all"]["init"], recs["rand"]["init"]
    gate(gates, "BG4_all_eq_ckpt", ia["vs_ckpt"]["verdict"] == "ALL_KEYS_EQUAL_TO_CKPT",
         "%d/%d keys == ckpt, %d differ" % (ia["vs_ckpt"]["n_equal"],
                                            ia["vs_ckpt"]["n_keys"], ia["vs_ckpt"]["n_differ"]))
    gate(gates, "BG4_rand_differs", ir["vs_ckpt"]["verdict"] == "RANDOM_INIT_RECORDED",
         "%d/%d keys differ from ckpt (%d equal: default-init LN/bias coincidences)"
         % (ir["vs_ckpt"]["n_differ"], ir["vs_ckpt"]["n_keys"], ir["vs_ckpt"]["n_equal"]))
    gate(gates, "BG4_init_shas_differ",
         ia["canonical_sha256"] != ir["canonical_sha256"],
         "all %s... rand %s..." % (ia["canonical_sha256"][:16], ir["canonical_sha256"][:16]))
    gate(gates, "BG4_trainable_lists_equal",
         ia["trainable_names"] == ir["trainable_names"]
         and ia["trainable_count"] == ir["trainable_count"],
         "%d trainable params, identical name lists" % ia["trainable_count"])
    gate(gates, "BG4_init_args",
         ia["init_args"]["finetune_type"] == "full" == ir["init_args"]["finetune_type"]
         and ia["init_args"]["load_ckpt"] is True and ir["init_args"]["load_ckpt"] is False
         and ia["init_args"]["arch"] == ir["init_args"]["arch"],
         "arch %s | ft full/full | load_ckpt True(all)/False(rand)" % ia["init_args"]["arch"])
    gate(gates, "BG4_init_called_once",
         ia["called"] == 1 and ir["called"] == 1,
         "VisionTS constructed once per arm")

    # BG5 RNG equality (rng_after_init lives under rec["init"])
    for key, path in (("rng_after_init", lambda r: r["init"][key]),
                      ("rng_at_train_entry", lambda r: r[key])):
        a, r = path(recs["all"]), path(recs["rand"])
        gate(gates, "BG5_%s" % key, a == r,
             "identical across arms (%s)" % ",".join(sorted(a)))

    # BG6 sampler order
    sa, sr = recs["all"]["sampler"], recs["rand"]["sampler"]
    ok = sa["n_randomsampler_iters"] == 1 and sr["n_randomsampler_iters"] == 1
    gate(gates, "BG6_single_epoch_iter", ok,
         "RandomSampler iterations all=%d rand=%d" %
         (sa["n_randomsampler_iters"], sr["n_randomsampler_iters"]))
    ea, er = sa["iters"][0], sr["iters"][0]
    gate(gates, "BG6_epoch_len", ea["n"] == er["n"] == 7 * 7393,
         "n_idx all=%d rand=%d expect %d (7393 windows x 7 vars)" %
         (ea["n"], er["n"], 7 * 7393))
    gate(gates, "BG6_order_identical",
         ea["sha256"] == er["sha256"] and ea["first20"] == er["first20"]
         and ea["last20"] == er["last20"],
         "epoch index sha %s... first20 %s..%s last20 %s..%s" %
         (ea["sha256"][:16], ea["first20"][0], ea["first20"][-1],
          er["last20"][0], er["last20"][-1]))

    # BG7 first batch
    fa, fr = recs["all"]["first_batch"], recs["rand"]["first_batch"]
    ok = fa.get("done") and fr.get("done") and \
        [t["sha256"] for t in fa["tensors"]] == [t["sha256"] for t in fr["tensors"]]
    gate(gates, "BG7_first_batch_identical", ok,
         "batch tensors %s | first sha %s..." %
         ([t["shape"] for t in fa.get("tensors", [])],
          fa["tensors"][0]["sha256"][:16] if fa.get("tensors") else "?"))

    # BG8 final checkpoints
    sha_all = recs["all"]["final_ckpt_sha256"]
    sha_rand = recs["rand"]["final_ckpt_sha256"]
    gate(gates, "BG8_final_ckpts",
         sha_all != sha_rand and sha_all != C.CKPT_SHA256 and sha_rand != C.CKPT_SHA256,
         "all %s... rand %s... (frozen %s...)" %
         (sha_all[:16], sha_rand[:16], C.CKPT_SHA256[:16]))

    # BG9 anchors
    ln_succ = os.path.join(C.exp003_job(), "SUCCESS.txt")
    mse, mae = C.parse_success_mse_mae(ln_succ)
    gate(gates, "BG9_ln_anchor",
         abs(mse - C.LN_ANCHOR["mse"]) <= 1e-9 and abs(mae - C.LN_ANCHOR["mae"]) <= 1e-9,
         "EXP-003 SUCCESS %.10f/%.10f vs pinned %.10f/%.10f" %
         (mse, mae, C.LN_ANCHOR["mse"], C.LN_ANCHOR["mae"]))
    freeze_job = C.first_existing(C._B_FREEZE_CANDIDATES, "SUCCESS.txt")
    freeze_path = os.path.join(freeze_job, "SUCCESS.txt")
    if os.path.isfile(freeze_path):
        mse, mae = C.parse_success_mse_mae(freeze_path)
        gate(gates, "BG9_freeze_anchor",
             abs(mse - C.FREEZE_ANCHOR["mse"]) <= 1e-9 and
             abs(mae - C.FREEZE_ANCHOR["mae"]) <= 1e-9,
             "EXP-033 SUCCESS %.10f/%.10f vs pinned %.10f/%.10f (%s)" %
             (mse, mae, C.FREEZE_ANCHOR["mse"], C.FREEZE_ANCHOR["mae"], freeze_job))
    else:
        gates.append(("BG9_freeze_anchor", "WARN",
                      "freeze SUCCESS not on this machine; pinned value used (%s)" % freeze_job))

    all_pass = all(g[1] in ("PASS", "WARN") for g in gates)
    out = {
        "all_pass": all_pass,
        "n_gates": len(gates), "n_pass": sum(1 for g in gates if g[1] == "PASS"),
        "gates": [{"gate": g[0], "verdict": g[1], "detail": g[2]} for g in gates],
        "summary": summary,
        "timing": {arm: recs[arm]["timing"] for arm in C.B_ARMS},
        "gpu_peak": {arm: recs[arm]["gpu_peak"] for arm in C.B_ARMS},
        "final_ckpt_sha256": {arm: recs[arm]["final_ckpt_sha256"] for arm in C.B_ARMS},
    }
    C.merge_json(C.VERIFY_JSON, "fullshot", out)
    for g in gates:
        print("[t37][B] %-30s %-5s %s" % (g[0], g[1], g[2]))
    print("[t37][B] ALL %s (%d/%d gates pass)"
          % ("PASS" if all_pass else "FAIL", out["n_pass"], out["n_gates"]))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
