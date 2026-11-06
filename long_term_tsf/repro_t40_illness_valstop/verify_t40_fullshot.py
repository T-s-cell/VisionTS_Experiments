"""EXP-040 verifier: gates TG1-TG10 over the delivered main round
(12 tasks x rules A,B,C = 36 units) plus the same-environment control
round (smoke task, rule A only).

  TG1  RNG audit complete: every train json rng_violations == [] and
       rng_audit continuity (post_hooks[e] == pre_epoch[e+1]) 100%
  TG2  same-environment trajectory equivalence (HARD): ctrl (A-only)
       vs main (A/B/C) on (24, 2021): identical init digest + per common
       epoch identical train sampler-order hash AND identical post-epoch
       state_dict digest; common prefix >= 2 epochs (stop epochs may
       differ — each round stops under its own rule set)
  TG3  archived theta curves vs this round's A curves — side-by-side
       REFERENCE ONLY (cross-env; no threshold)
  TG4  test truth alignment: results/<rule>/true.npy bitwise ==
       EXP-003 archived true.npy (36 units)
  TG5  float64 metric recompute: pred/true -> mse/mae vs SUCCESS
       <= 1e-6, finite everywhere (36 units)
  TG6  best epoch <-> lock correspondence: (a) state-machine replay
       from recorded val scores reproduces improved flags / best epoch /
       trigger epoch EXACTLY (real EarlyStopping semantics; never
       first-argmin); (b) locked ckpt reload recomputes the recorded
       best val score <= 1e-5 — rule C uses the epoch's SAVED sample
       indices (curves.npz c_idx_e<best>), never re-randomized
  TG7  post-lock selection invariance: locked rules have
       len(events) == trigger_epoch (no further state-machine calls),
       counter array constant after the trigger, ckpt file sha ==
       recorded final sha, test-reload state digest == recomputed
  TG8  coverage completeness: 12 x 3 = 36 units with {lock ckpt,
       valid_loss.json, curve arrays, test metrics} all present;
       12 main + 1 ctrl train jsons
  TG9  lr dual record: train_lr_used / lr_after_update match the type1
       cadence (train_lr_used[1]=1e-4, [e>=2]=1e-4*0.5^(e-2);
       lr_after_update[e]=1e-4*0.5^(e-1)); the all-locked break epoch
       has no update (NaN allowed there only)
  TG10 re-entry completeness (read-only): SUCCESS(seed/rules) +
       fingerprint(seed pins) + curves/trajectory/phases/rng_audit/
       lock_events + per-rule artifacts for 12 main + 1 ctrl

Run: python verify_t40_fullshot.py [--gpu 0] [--smoke]
     --smoke restricts the task loops to the smoke task (24, 2021) and
     keeps the TG2 ctrl-vs-main hard gate — used by the runner's smoke
     stage before the queue starts.
Writes logs/verify_t40.json; exit 0 iff every gate passes.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_T40_DIR = os.path.dirname(os.path.abspath(__file__))
if _T40_DIR not in sys.path:
    sys.path.insert(0, _T40_DIR)
import t40_common as X              # noqa: E402
import t38_common as C              # noqa: E402

FP_CODE_FILES = [
    "repro_fullshot/run_train.py", "repro_fullshot/exp_univariate.py",
    "long_term_tsf/exp/exp_long_term_forecasting.py",
    "long_term_tsf/exp/exp_basic.py", "long_term_tsf/models/VisionTS.py",
    "long_term_tsf/data_provider/data_factory.py",
    "long_term_tsf/data_provider/data_loader.py",
    "long_term_tsf/utils/tools.py", "long_term_tsf/utils/metrics.py",
    "visionts/model.py", "visionts/models_mae.py", "visionts/util.py",
]


def tensor_digest(sd):
    import torch
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        t = sd[k]
        if torch.is_tensor(t):
            t = t.detach().cpu().contiguous()
            h.update(str(tuple(t.shape)).encode())
            h.update(b"\x00")
            h.update(t.numpy().tobytes())
        else:
            h.update(repr(t).encode())
        h.update(b"\x01")
    return h.hexdigest()


def replay_state_machine(scores):
    """Mirror of utils.tools.EarlyStopping (patience=3, delta=0): ties
    count as improvement (re-save, counter reset); stop when counter
    reaches patience.  Returns (improved_flags, best_epoch, trigger)."""
    best, best_ep, counter, trig = None, None, 0, None
    flags = []
    for i, v in enumerate(scores, start=1):
        s = -v
        if best is None:
            best, best_ep = s, i
            flags.append(True)
        elif s < best:
            counter += 1
            flags.append(False)
            if trig is None and counter >= 3:
                trig = i
        else:
            best, best_ep, counter = s, i, 0
            flags.append(True)
    return flags, best_ep, trig


class Gate:
    def __init__(self, name):
        self.name = name
        self.checks = 0
        self.fails = []
        self.info = {}

    def check(self, cond, msg):
        self.checks += 1
        if not cond:
            self.fails.append(msg)
        return bool(cond)

    def done(self, rec):
        rec[self.name] = {"n_checks": self.checks,
                          "n_fail": len(self.fails),
                          "fails": self.fails[:20], "info": self.info}
        print("[t40-verify] %s: %d/%d %s"
              % (self.name, self.checks - len(self.fails), self.checks,
                 "PASS" if not self.fails else "FAIL(%d) %s"
                 % (len(self.fails), self.fails[0])))
        return not self.fails


def load_job_records(X, C, h, seed, round_tag):
    jd = X.job_dir(h, seed, round_tag)
    rec = C.load_json(X.train_json(h, seed, round_tag))
    succ = C.load_json(os.path.join(jd, "SUCCESS.txt"))
    fp = C.load_json(os.path.join(jd, "fingerprint.json"))
    rng = C.load_json(X.rng_audit_path(h, seed, round_tag))
    lock = C.load_json(X.lock_events_path(h, seed, round_tag))
    traj = C.load_json(os.path.join(jd, "trajectory.json"))
    phases = C.load_json(os.path.join(jd, "phases.json"))
    npz = np.load(X.curves_path(h, seed, round_tag), allow_pickle=False)
    return dict(jd=jd, rec=rec, succ=succ, fp=fp, rng=rng, lock=lock,
                traj=traj, phases=phases, npz=npz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="verify only the smoke task (hard TG2 gate)")
    a = ap.parse_args()

    os.chdir(X.ROOT)
    os.makedirs(X.LOG_DIR, exist_ok=True)
    out = {"started": C.time_str(), "gates": {}, "smoke_mode": a.smoke}
    allok = True

    tasks = [X.SMOKE] if a.smoke else list(X.TASKS)
    rounds = [("main", h, s) for (h, s) in tasks]
    ctrl = load_job_records(X, C, *X.SMOKE, "ctrl") if os.path.isfile(
        X.train_json(*X.SMOKE, "ctrl")) else None

    # ---- TG1: RNG audit ---------------------------------------------------
    g = Gate("TG1")
    for round_tag, h, s in rounds:
        R = load_job_records(X, C, h, s, round_tag)
        g.check(R["rec"]["rng_violations"] == [],
                "%s main: rng_violations non-empty" % R["rec"]["job"])
        per = R["rng"]["per_epoch"]
        for k in range(len(per) - 1):
            okk = all(per[k]["post_hooks"][st] == per[k + 1]["pre_epoch"][st]
                      for st in ("python", "numpy", "torch_cpu", "torch_cuda"))
            g.check(okk, "%s main: rng continuity broken at epoch %d"
                    % (R["rec"]["job"], per[k]["epoch"]))
        g.check(len(per) == R["rec"]["finished_epochs"],
                "%s main: rng_audit epochs != finished_epochs"
                % R["rec"]["job"])
    if ctrl is not None:
        g.check(ctrl["rec"]["rng_violations"] == [],
                "ctrl: rng_violations non-empty")
    allok &= g.done(out["gates"])

    # ---- TG2: ctrl-vs-main trajectory equivalence (HARD) ------------------
    g = Gate("TG2")
    h0, s0 = X.SMOKE
    if ctrl is None:
        g.check(False, "ctrl round missing")
    else:
        mainr = load_job_records(X, C, h0, s0, "main")
        g.check(ctrl["traj"]["init_state_sha256"]
                == mainr["traj"]["init_state_sha256"],
                "init state digest differs ctrl vs main")
        n_common = min(len(ctrl["traj"]["epochs"]),
                       len(mainr["traj"]["epochs"]))
        g.check(n_common >= 2,
                "common prefix %d epochs < 2" % n_common)
        for k in range(n_common):
            ec, em = ctrl["traj"]["epochs"][k], mainr["traj"]["epochs"][k]
            g.check(ec["epoch"] == em["epoch"], "epoch numbering drift")
            g.check(ec["order"]["sha256"] == em["order"]["sha256"]
                    and ec["order"]["n_indices"] == em["order"]["n_indices"],
                    "sampler order differs at epoch %d" % ec["epoch"])
            g.check(ec["state_sha256"] == em["state_sha256"],
                    "parameter trajectory differs at epoch %d"
                    % ec["epoch"])
        g.check(ctrl["rec"]["rules"] == ["A"],
                "ctrl rules != [A]")
        g.check(mainr["rec"]["rules"] == ["A", "B", "C"],
                "main rules != [A,B,C]")
        # same seed/env provenance
        g.check(ctrl["rec"]["seed"] == s0 == mainr["rec"]["seed"],
                "seed mismatch between rounds")
        g.info = {"n_common_epochs": n_common,
                  "ctrl_epochs": len(ctrl["traj"]["epochs"]),
                  "main_epochs": len(mainr["traj"]["epochs"])}
    allok &= g.done(out["gates"])

    # ---- TG3: theta-curve reference (SOFT, never fails the run) -----------
    g = Gate("TG3")
    audit_path = X.AUDIT_JSON
    if not os.path.isfile(audit_path):
        g.check(False, "audit json missing (run t40_audit first)")
    else:
        aud = C.load_json(audit_path)
        curves = aud.get("gates", {}).get("AG8_curves", {})
        for h, s in tasks:
            key = "illness_%d_seed%d" % (h, s)
            R = load_job_records(X, C, h, s, "main")
            a_vals = R["npz"]["A_val"].tolist()
            th = curves.get(key, {})
            g.info[key] = {"theta_n": th.get("n_parsed", 0),
                           "theta_loss": th.get("vali_loss", []),
                           "t40_A_val": a_vals,
                           "note": "reference-only (cross-env)"}
            g.check(True, "")   # presence check only; no threshold
    allok &= g.done(out["gates"])   # TG3 never hard-fails beyond audit file

    # ---- TG4/TG5: test truth + f64 recompute ------------------------------
    from utils.metrics import metric
    g4, g5 = Gate("TG4"), Gate("TG5")
    recompute = []
    for h, s in tasks:
        ref_true = np.load(os.path.join(X.exp003_job(h, s),
                                        "results", "_", "true.npy"))
        R = load_job_records(X, C, h, s, "main")
        for r in X.RULES:
            rdir = X.rule_results_dir(h, s, r, "main")
            true = np.load(os.path.join(rdir, "true.npy"))
            pred = np.load(os.path.join(rdir, "pred.npy"))
            g4.check(np.array_equal(true, ref_true),
                     "illness_%d_seed%d rule %s: true.npy != EXP-003"
                     % (h, s, r))
            g5.check(np.isfinite(pred).all() and np.isfinite(true).all(),
                     "illness_%d_seed%d rule %s: non-finite arrays"
                     % (h, s, r))
            p64 = pred.astype(np.float64)
            t64 = true.astype(np.float64)
            mae64, mse64, _, _, _ = metric(p64, t64)
            mse_rec = R["succ"]["test"][r]["mse"]
            mae_rec = R["succ"]["test"][r]["mae"]
            d_mse = abs(mse64 - mse_rec)
            d_mae = abs(mae64 - mae_rec)
            g5.check(d_mse <= 1e-6 and d_mae <= 1e-6,
                     "illness_%d_seed%d rule %s: f64 recompute d_mse=%.3g "
                     "d_mae=%.3g" % (h, s, r, d_mse, d_mae))
            recompute.append({"task": "illness_%d_seed%d" % (h, s),
                              "rule": r, "mse_f64": mse64, "mae_f64": mae64,
                              "d_mse": d_mse, "d_mae": d_mae})
    allok &= g4.done(out["gates"])
    allok &= g5.done(out["gates"])
    with open(X.RECOMPUTE_JSON, "w") as f:
        json.dump({"n_units": len(recompute), "records": recompute}, f,
                  indent=1)

    # ---- TG6/TG7: state machine + locks ------------------------------------
    g6, g7 = Gate("TG6"), Gate("TG7")
    for h, s in tasks:
        R = load_job_records(X, C, h, s, "main")
        for r in X.RULES:
            scores = R["npz"]["%s_val" % r].tolist()
            improved_rec = R["npz"]["%s_improved" % r].tolist()
            counter_rec = R["npz"]["%s_counter" % r].tolist()
            flags, best_ep, trig = replay_state_machine(scores)
            g6.check(flags == improved_rec,
                     "illness_%d_seed%d %s: state-machine replay improved "
                     "flags != recorded" % (h, s, r))
            fin = next(e for e in R["lock"]["final"] if e["rule"] == r)
            vl = C.load_json(os.path.join(
                X.locks_dir(h, s, r, "main"), "valid_loss.json"))
            g6.check(best_ep == fin["best_valid_epoch"] == vl[
                         "best_valid_epoch"],
                     "illness_%d_seed%d %s: best epoch replay %d != final "
                     "%s / valid_loss %s" % (h, s, r, best_ep,
                                             fin["best_valid_epoch"],
                                             vl["best_valid_epoch"]))
            if fin["locked"]:
                g6.check(trig == fin["trigger_epoch"],
                         "illness_%d_seed%d %s: trigger replay %s != %s"
                         % (h, s, r, trig, fin["trigger_epoch"]))
                g7.check(len(R["lock"]["events"][r]) == fin["trigger_epoch"],
                         "illness_%d_seed%d %s: events after lock (%d > %d)"
                         % (h, s, r, len(R["lock"]["events"][r]),
                            fin["trigger_epoch"]))
                t = fin["trigger_epoch"]
                tail = counter_rec[t - 1:]
                g7.check(all(c == tail[0] for c in tail),
                         "illness_%d_seed%d %s: counter changed after lock"
                         % (h, s, r))
            else:
                g6.check(trig is None,
                         "illness_%d_seed%d %s: replay found a trigger on "
                         "a no-trigger rule" % (h, s, r))
                g7.check(len(R["lock"]["events"][r]) == len(scores),
                         "illness_%d_seed%d %s: no-trigger events %d != "
                         "epochs %d" % (h, s, r,
                                        len(R["lock"]["events"][r]),
                                        len(scores)))
            ck = X.rule_ckpt(h, s, r, "main")
            g7.check(C.sha256_of(ck) == fin["ckpt_sha256"],
                     "illness_%d_seed%d %s: ckpt file sha != recorded"
                     % (h, s, r))
            g7.check(R["rec"]["test"][r]["state_sha256"]
                     == fin["state_sha256"],
                     "illness_%d_seed%d %s: test-reload digest != final"
                     % (h, s, r))
    allok &= g7.done(out["gates"])

    # TG6(b): locked-ckpt val recompute (needs GPU) — after TG7 file checks
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, default_collate
    from data_provider.data_factory import data_provider
    for h, s in tasks:
        R = load_job_records(X, C, h, s, "main")
        args0 = C.load_json(os.path.join(R["jd"], "args.json"))
        args0["use_gpu"] = True
        args0["gpu"] = 0   # eta remap (archived H48/H60 carried theta gpu=1)
        from exp_univariate import Exp_Univariate
        exp = Exp_Univariate(argparse.Namespace(**args0))
        val_data, _ = data_provider(exp.args, "val")
        N_val = len(val_data)
        bs = exp.args.batch_size
        n_c = N_val // bs
        criterion = nn.MSELoss()

        def recompute_val(rule, idx_used=None):
            exp.model.eval()
            sse, cnt, b_losses = 0.0, 0, []
            with torch.no_grad():
                batches = []
                if idx_used is None:
                    ggen = torch.Generator()
                    ggen.manual_seed(0)
                    loader = DataLoader(val_data, batch_size=bs,
                                        shuffle=False, num_workers=0,
                                        drop_last=False, generator=ggen)
                    for bx, by, bxm, bym in loader:
                        batches.append((bx, by, bxm, bym))
                else:
                    for st in range(0, len(idx_used), bs):
                        idx = idx_used[st:st + bs]
                        batches.append(default_collate(
                            [val_data[int(i)] for i in idx]))
                for bx, by, bxm, bym in batches:
                    bx = bx.float().to(exp.device)
                    by = by.float()
                    bxm = bxm.float().to(exp.device)
                    bym = bym.float().to(exp.device)
                    dec_inp = torch.zeros_like(
                        by[:, -exp.args.pred_len:, :]).float()
                    dec_inp = torch.cat(
                        [by[:, :exp.args.label_len, :], dec_inp],
                        dim=1).float().to(exp.device)
                    out = exp.model(bx, bxm, dec_inp, bym)
                    f_dim = -1 if exp.args.features == "MS" else 0
                    out = out[:, -exp.args.pred_len:, f_dim:]
                    by = by[:, -exp.args.pred_len:, f_dim:].to(exp.device)
                    diff = out - by
                    sse += diff.pow(2).sum().item()
                    cnt += diff.numel()
                    b_losses.append(criterion(out.detach().cpu(),
                                              by.detach().cpu()).item())
            exp.model.train()
            return sse / cnt, float(np.average(b_losses))

        fin = {e["rule"]: e for e in R["lock"]["final"]}
        for r in X.RULES:
            sd_t = torch.load(X.rule_ckpt(h, s, r, "main"),
                              map_location="cpu", weights_only=False)
            exp.model.load_state_dict(sd_t)
            best_loss = fin[r]["best_valid_loss"]
            a_v, b_v = recompute_val(r)
            got = {"A": a_v, "B": b_v}
            if r == "C":
                key = "c_idx_e%d" % fin[r]["best_valid_epoch"]
                idx = R["npz"][key]
                _, c_v = recompute_val("C", idx)
                got["C"] = c_v
            g6.check(abs(got[r] - best_loss) <= 1e-5,
                     "illness_%d_seed%d %s: locked-ckpt val recompute "
                     "%.7f != recorded %.7f" % (h, s, r, got[r], best_loss))
        del exp
        torch.cuda.empty_cache()
    allok &= g6.done(out["gates"])

    # ---- TG8: coverage completeness ----------------------------------------
    g = Gate("TG8")
    for h, s in tasks:
        npz = np.load(X.curves_path(h, s, "main"), allow_pickle=False)
        for r in X.RULES:
            g.check(os.path.isfile(X.rule_ckpt(h, s, r, "main")),
                    "missing lock ckpt illness_%d_seed%d %s" % (h, s, r))
            g.check(os.path.isfile(os.path.join(
                X.locks_dir(h, s, r, "main"), "valid_loss.json")),
                "missing valid_loss illness_%d_seed%d %s" % (h, s, r))
            rdir = X.rule_results_dir(h, s, r, "main")
            for fn in ("metrics.npy", "pred.npy", "true.npy"):
                g.check(os.path.isfile(os.path.join(rdir, fn)),
                        "missing %s illness_%d_seed%d %s"
                        % (fn, h, s, r))
            for key in ("%s_val" % r, "%s_improved" % r, "%s_counter" % r):
                g.check(key in npz.files,
                        "missing curve %s illness_%d_seed%d" % (key, h, s))
            if r == "C":
                g.check("c_idx_e1" in npz.files,
                        "missing c_idx_e1 illness_%d_seed%d" % (h, s))
    allok &= g.done(out["gates"])

    # ---- TG9: lr dual record -----------------------------------------------
    g = Gate("TG9")
    for h, s in tasks:
        for round_tag in (["main"]):
            R = load_job_records(X, C, h, s, round_tag)
            used = R["npz"]["train_lr_used"]
            after = R["npz"]["lr_after_update"]
            n = len(used)
            g.check(n == R["rec"]["finished_epochs"],
                    "illness_%d_seed%d: lr history != finished_epochs"
                    % (h, s))
            for k in range(n):
                e = k + 1
                if abs(used[k] - X.expected_train_lr(e)) > 1e-12:
                    g.check(False, "illness_%d_seed%d: train_lr_used[%d]=%r "
                            "!= %r" % (h, s, e, used[k],
                                       X.expected_train_lr(e)))
                    break
            for k in range(n):
                e = k + 1
                if np.isnan(after[k]):
                    g.check(k == n - 1,
                            "illness_%d_seed%d: NaN lr_after_update at "
                            "non-final epoch %d" % (h, s, e))
                    break
                if abs(after[k] - X.expected_lr_after_update(e)) > 1e-12:
                    g.check(False, "illness_%d_seed%d: "
                            "lr_after_update[%d]=%r != %r"
                            % (h, s, e, after[k],
                               X.expected_lr_after_update(e)))
                    break
    if ctrl is not None:
        used = ctrl["npz"]["train_lr_used"]
        for k in range(len(used)):
            if abs(used[k] - X.expected_train_lr(k + 1)) > 1e-12:
                g.check(False, "ctrl train_lr_used[%d] drifted" % (k + 1))
                break
    allok &= g.done(out["gates"])

    # ---- TG10: re-entry completeness (read-only) ----------------------------
    g = Gate("TG10")
    tree_md5 = {rel: C.md5_of(os.path.join(X.REPO_ROOT, rel))
                for rel in FP_CODE_FILES}
    for round_tag, h, s in rounds:
        R = load_job_records(X, C, h, s, round_tag)
        g.check(R["succ"].get("seed") == s
                and R["succ"].get("rules") == list(X.RULES),
                "SUCCESS pins illness_%d_seed%d" % (h, s))
        g.check(R["fp"].get("seed") == s
                and R["fp"].get("args", {}).get("seed") == s,
                "fingerprint seed pins illness_%d_seed%d" % (h, s))
        bad = [k for k in FP_CODE_FILES
               if R["fp"].get("code_md5", {}).get(k) != tree_md5[k]]
        g.check(not bad, "code_md5 drift illness_%d_seed%d: %s"
                % (h, s, bad[:3]))
        g.check(R["fp"]["data_sha256"].get(
            os.path.join("dataset", "illness", "national_illness.csv"))
            == X.CSV_SHA256, "csv sha pin illness_%d_seed%d" % (h, s))
        g.check(R["rec"]["requested_seed"] == s,
                "requested_seed pin illness_%d_seed%d" % (h, s))
        for fn in ("args.json", "curves.npz", "trajectory.json",
                   "phases.json", "lock_events.json", "SUCCESS.txt"):
            g.check(os.path.isfile(os.path.join(R["jd"], fn)),
                    "missing %s illness_%d_seed%d" % (fn, h, s))
    if ctrl is not None:
        g.check(ctrl["succ"].get("rules") == ["A"], "ctrl SUCCESS rules")
        g.check(ctrl["fp"].get("seed") == X.SMOKE[1], "ctrl seed pin")
    allok &= g.done(out["gates"])

    out["finished"] = C.time_str()
    out["all_pass"] = bool(allok)
    n_checks = sum(v["n_checks"] for v in out["gates"].values())
    n_fail = sum(v["n_fail"] for v in out["gates"].values())
    out["summary"] = {"n_checks": n_checks, "n_fail": n_fail,
                      "all_pass": bool(allok)}
    with open(X.VERIFY_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print("[t40-verify] TOTAL %d/%d checks pass -> %s"
          % (n_checks - n_fail, n_checks, X.VERIFY_JSON))
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
