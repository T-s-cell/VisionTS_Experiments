"""EXP-040 driver: one Illness (H, seed) trained ONCE under the frozen
EXP-003 recipe, with three validation rules evaluated on the shared
trajectory.  Adding the extra validation hooks must NOT change the
training RNG stream or per-epoch sample order — enforced structurally
(A/B loader carries an explicit dedicated torch.Generator; rule C uses
a dedicated np.random.RandomState and manual collation, zero DataLoader
RNG surface) and audited at runtime (four-state RNG hashes around every
hook, hard-asserted unchanged).

Run (from anywhere; the driver chdirs to long_term_tsf — EXP-003 CWD
contract):
  python t40_train_driver.py --h 24 --seed 2021 --round main  [--gpu 0]
      [--max-epochs 100]
  --round main -> rules A,B,C into save_t40_valstop/<H>_seed<s>/
  --round ctrl -> rule A only into save_t40_valstop_ctrl/<H>_seed<s>/
                  (same-environment control for the TG2 trajectory gate)

Mirrored zero-change from the audited code paths:
  - args replayed from the config's OWN EXP-003 archived args.json
    (save_fullshot/illness_<H>_seed<s>/args.json); allowed delta keys
    are EXACTLY {model_id, save_dir} (both point at this round's job).
  - seeding identical to repro_fullshot/run_train.py: random.seed ->
    torch.manual_seed -> np.random.seed BEFORE any model construction.
  - model construction / datasets / train loader via Exp_Univariate and
    data_provider unchanged (train loader shuffle=True, drop_last=True,
    num_workers=10, generator=None — identical global-RNG draw pattern
    to EXP-003).
  - lr cadence via the official adjust_learning_rate(model_optim,
    epoch+1, args) AFTER each epoch's training (skipped only when the
    loop breaks because ALL enabled rules locked — mirrors the base
    train break semantics).  Per-epoch BOTH the lr actually used for
    training and the lr after the end-of-epoch update are recorded and
    asserted (TG9).
  - early stopping via the REAL tools.EarlyStopping (patience=3,
    delta=0; ties count as improvement/re-save) with one clone per rule
    saving into locks/<rule>/checkpoint.pth + valid_loss.json; a locked
    rule receives no further calls (selection frozen; TG7).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

_T40_DIR = os.path.dirname(os.path.abspath(__file__))
if _T40_DIR not in sys.path:
    sys.path.insert(0, _T40_DIR)
import t40_common as X              # noqa: E402  (pulls in t38_common)
import t38_common as C              # noqa: E402


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ---- four-state RNG snapshots ---------------------------------------------
def rng_snapshot():
    import torch
    snap = {}
    snap["torch_cpu"] = _sha_bytes(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        h = hashlib.sha256()
        for st in torch.cuda.get_rng_state_all():
            h.update(st.numpy().tobytes())
        snap["torch_cuda"] = h.hexdigest()
    else:
        snap["torch_cuda"] = "cuda-unavailable"
    nps = np.random.get_state()
    nh = hashlib.sha256()
    nh.update(nps[0].encode())
    nh.update(np.ascontiguousarray(nps[1], dtype=np.uint32).tobytes())
    nh.update(np.asarray(nps[2:], dtype=np.int64).tobytes())
    snap["numpy"] = nh.hexdigest()
    py = random.getstate()
    ph = hashlib.sha256()
    ph.update(str(py[0]).encode())
    ph.update(np.asarray(py[1], dtype=np.uint64).tobytes())
    ph.update(str(py[2]).encode())
    snap["python"] = ph.hexdigest()
    return snap


def rng_assert_unchanged(pre, post, where, rec):
    bad = [k for k in pre if pre[k] != post.get(k)]
    if bad:
        rec["rng_violations"].append({"where": where, "keys": bad})
        raise AssertionError("RNG leak at %s: %s" % (where, bad))


def canonical_state_sha(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(("|%s|%s|" % (t.dtype, tuple(t.shape))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class OrderRecorder:
    """Wraps loader.batch_sampler: full delegation, records the exact
    per-iteration index batches (the sampler order) without changing any
    RNG consumption.  DataLoader._index_sampler returns
    loader.batch_sampler, so iter(loader) iterates THIS object; the
    inner BatchSampler still drives the original RandomSampler (whose
    global-stream base_seed draw happens exactly as without the proxy)."""

    def __init__(self, inner):
        self._inner = inner
        self.orders = []

    def __iter__(self):
        idxs = []
        for batch in self._inner:
            idxs.append([int(i) for i in batch])
            yield batch
        self.orders.append(idxs)

    def __len__(self):
        return len(self._inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def order_hashes(self):
        out = []
        for idxs in self.orders:
            flat = [i for b in idxs for i in b]
            out.append({"n_batches": len(idxs), "n_indices": len(flat),
                        "sha256": _sha_bytes(
                            np.asarray(flat, dtype=np.int64).tobytes())})
        return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True, choices=X.HS)
    ap.add_argument("--seed", type=int, required=True, choices=X.SEEDS)
    ap.add_argument("--round", dest="round_tag", default="main",
                    choices=["main", "ctrl"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=100,
                    help="cap on train_epochs (100 = frozen recipe)")
    return ap


def main():
    args_cli = build_parser().parse_args()
    h, seed = args_cli.h, args_cli.seed
    round_tag = args_cli.round_tag
    rules = X.allowed_rules(round_tag)
    job = X.job_name(h, seed, round_tag)
    jd = os.path.abspath(X.job_dir(h, seed, round_tag))

    os.makedirs(X.LOG_DIR, exist_ok=True)
    os.makedirs(X.save_base(round_tag), exist_ok=True)
    if os.path.isfile(os.path.join(jd, "SUCCESS.txt")):
        raise SystemExit("[t40] refusing: SUCCESS.txt already exists in %s"
                         % jd)

    exp003_path = X.exp003_job(h, seed)
    exp003_args_path = os.path.join(exp003_path, "args.json")
    assert os.path.isfile(exp003_args_path), \
        "EXP-003 args.json missing: %s" % exp003_args_path
    exp003_args = C.load_json(exp003_args_path)
    assert exp003_args.get("seed") == seed, \
        "replay base seed mismatch: archive %r != requested %d" % (
            exp003_args.get("seed"), seed)
    pins = ("ft_type", "vm_pretrained", "patience", "lradj",
            "train_epochs", "batch_size", "seq_len")
    assert (exp003_args.get("ft_type") == "ln"
            and exp003_args.get("vm_pretrained") == 1
            and exp003_args.get("patience") == 3
            and exp003_args.get("lradj") == "type1"
            and exp003_args.get("train_epochs") == 100
            and exp003_args.get("batch_size") == 256
            and exp003_args.get("seq_len") == 104), \
        "EXP-003 recipe pins drifted: %s" % {k: exp003_args.get(k)
                                             for k in pins}

    csv_path = os.path.join(X.ROOT, X.CSV_REL)
    ckpt_path = os.path.abspath(os.path.join(X.ROOT, "..", X.CKPT_REL))
    assert os.path.isfile(csv_path), "illness csv missing: %s" % csv_path
    assert C.sha256_of(csv_path) == X.CSV_SHA256, "illness csv sha drifted"
    assert os.path.isfile(ckpt_path), "frozen ckpt missing: %s" % ckpt_path
    assert C.sha256_of(ckpt_path) == X.CKPT_SHA256, "frozen ckpt sha drifted"

    os.chdir(X.ROOT)   # long_term_tsf — EXP-003 CWD contract

    # ---- replay args: delta keys EXACTLY {model_id, save_dir, gpu} -----
    # (gpu is an environment fact: EXP-003 theta ran H48/H60 on cuda:1;
    #  eta maps everything to cuda:0 via CUDA_VISIBLE_DEVICES)
    args = argparse.Namespace(**json.loads(json.dumps(exp003_args)))
    args.model_id = job
    args.save_dir = jd
    args.use_gpu = True
    args.gpu = 0

    log("%s rules=%s job_dir=%s" % (job, "+".join(rules), jd))

    # ---- seed exactly like run_train.py, BEFORE model construction -----
    # Deterministic-kernel mode: the TG2 ctrl-vs-main hard gate compares
    # parameter trajectories BITWISE; stock CUDA kernels (atomics) are
    # run-to-run nondeterministic, so both rounds pin deterministic
    # kernel selection (recorded in the fingerprint; training math is
    # unchanged).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    import torch.nn as nn
    from torch.utils.data import DataLoader, default_collate
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    rec = {"job": job, "h": h, "seed": seed, "round": round_tag,
           "rules": list(rules), "requested_seed": seed,
           "started": C.time_str(), "max_epochs": args_cli.max_epochs,
           "rng_violations": [],
           "gpu": torch.cuda.get_device_name(args_cli.gpu)
           if torch.cuda.is_available() else "cpu",
           "env": {"python": sys.version.split()[0],
                   "torch": torch.__version__, "numpy": np.__version__,
                   "deterministic": {
                       "cublas_workspace_config":
                           os.environ["CUBLAS_WORKSPACE_CONFIG"],
                       "cudnn_deterministic": True,
                       "cudnn_benchmark": False,
                       "use_deterministic_algorithms": True,
                       "why": "TG2 ctrl-vs-main bitwise trajectory gate"}}}

    # fingerprint BEFORE training (run_train order), relative-path form
    repo_root = X.REPO_ROOT
    csv_rel = os.path.normpath(os.path.join(args.root_path, args.data_path))
    ckpt_key = os.path.join(args.vm_ckpt, "mae_visualize_vit_base.pth")
    code_md5 = {}
    for rel in ["repro_fullshot/run_train.py",
                "repro_fullshot/exp_univariate.py",
                "long_term_tsf/exp/exp_long_term_forecasting.py",
                "long_term_tsf/exp/exp_basic.py",
                "long_term_tsf/models/VisionTS.py",
                "long_term_tsf/data_provider/data_factory.py",
                "long_term_tsf/data_provider/data_loader.py",
                "long_term_tsf/utils/tools.py",
                "long_term_tsf/utils/metrics.py",
                "visionts/model.py", "visionts/models_mae.py",
                "visionts/util.py"]:
        p = os.path.join(repo_root, rel)
        code_md5[rel] = C.md5_of(p) if os.path.isfile(p) else "MISSING:" + p
    fingerprint = {"args": {k: v for k, v in vars(args).items()},
                   "seed": seed,
                   "data_sha256": {csv_rel: X.CSV_SHA256},
                   "ckpt_sha256": {ckpt_key: X.CKPT_SHA256},
                   "code_md5": code_md5, "env": rec["env"]}
    os.makedirs(jd, exist_ok=True)
    with open(os.path.join(jd, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    with open(os.path.join(jd, "fingerprint.json"), "w") as f:
        json.dump(fingerprint, f, indent=1)

    from exp_univariate import Exp_Univariate
    from utils.tools import EarlyStopping, adjust_learning_rate
    from utils.metrics import metric
    from data_provider.data_factory import data_provider

    class T40Exp(Exp_Univariate):
        """Only test_rule() is new; everything else inherited zero-change
        (vali() disabled — the A/B pass is an exact inline mirror of
        Exp_Univariate.vali plus the B records, one shared forward pass)."""

        def vali(self, vali_data, vali_loader, criterion):
            raise RuntimeError("base vali path must not run in EXP-040")

    exp = T40Exp(args)
    vm = exp.model.vm.vision_model   # wrapper.Model -> vm: VisionTS -> MAE ViT
    sd = vm.state_dict()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    eq = sum(1 for k in ck if k in sd and sd[k].shape == ck[k].shape
             and torch.equal(sd[k].cpu(), ck[k].cpu()))
    assert eq == len(ck), "vision_model init != frozen ckpt (%d/%d)" % (
        eq, len(ck))
    trainable = sorted(n for n, p in vm.named_parameters()
                       if p.requires_grad)
    rec["init"] = {"vision_model_vs_ckpt": {"n_keys": len(ck),
                                            "n_equal": eq,
                                            "n_differ": len(ck) - eq},
                   "trainable_count": len(trainable),
                   "state_sha256": canonical_state_sha(exp.model.state_dict())}
    log("model init ok: %d trainable LN tensors, vision_model == ckpt"
        % len(trainable))

    # ---- datasets / loaders --------------------------------------------
    _, train_loader = exp._get_data(flag="train")            # official params
    val_data, _ = data_provider(args, "val")                 # audited dataset
    N_val = len(val_data)
    assert N_val == X.VAL_N_PIN[h], \
        "val N %d != pin %d" % (N_val, X.VAL_N_PIN[h])
    n_batches_c = N_val // args.batch_size
    assert n_batches_c == X.C_BATCHES_PIN[h]
    train_steps = len(train_loader)
    rec["coverage"] = {"val_n": N_val,
                       "val_dropped": N_val - 256 * n_batches_c,
                       "c_batches": n_batches_c, "train_steps": train_steps}
    log("coverage: val N=%d, C batches=%d (drop %d), train_steps=%d"
        % (N_val, n_batches_c, N_val - 256 * n_batches_c, train_steps))

    # A/B loader: same dataset, sequential complete, ISOLATED generator
    gen_ab = torch.Generator()
    gen_ab.manual_seed(X.gen_seed(h, seed, "AB", round_tag))
    ab_loader = DataLoader(val_data, batch_size=args.batch_size,
                           shuffle=False, num_workers=0, drop_last=False,
                           generator=gen_ab)

    # train-loader order recorder (delegating proxy, no RNG change)
    # (torch 2.12 guards DataLoader.__setattr__ post-init; torch itself
    # stores these attrs via object.__setattr__, so we swap the same way)
    recorder = OrderRecorder(train_loader.batch_sampler)
    object.__setattr__(train_loader, "batch_sampler", recorder)

    model_optim = exp._select_optimizer()
    criterion = nn.MSELoss()

    c_rs = np.random.RandomState(X.c_stream_seed(h, seed, round_tag))

    def ab_pass():
        total_sse, total_cnt, b_losses = 0.0, 0, []
        exp.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in ab_loader:
                batch_x = batch_x.float().to(exp.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(exp.device)
                batch_y_mark = batch_y_mark.float().to(exp.device)
                dec_inp = torch.zeros_like(
                    batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :args.label_len, :], dec_inp],
                    dim=1).float().to(exp.device)
                outputs = exp.model(batch_x, batch_x_mark, dec_inp,
                                    batch_y_mark)
                f_dim = -1 if args.features == "MS" else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(exp.device)
                diff = outputs - batch_y
                total_sse += diff.pow(2).sum().item()
                total_cnt += diff.numel()
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                b_losses.append(criterion(pred, true).item())
        exp.model.train()
        return total_sse / total_cnt, b_losses, total_cnt

    def c_pass(idx_used):
        losses, elem = [], []
        exp.model.eval()
        with torch.no_grad():
            for start in range(0, len(idx_used), args.batch_size):
                idx = idx_used[start:start + args.batch_size]
                assert len(idx) == args.batch_size, \
                    "rule C must only see full batches"
                samples = [val_data[int(i)] for i in idx]
                batch_x, batch_y, batch_x_mark, batch_y_mark = \
                    default_collate(samples)
                batch_x = batch_x.float().to(exp.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(exp.device)
                batch_y_mark = batch_y_mark.float().to(exp.device)
                dec_inp = torch.zeros_like(
                    batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :args.label_len, :], dec_inp],
                    dim=1).float().to(exp.device)
                outputs = exp.model(batch_x, batch_x_mark, dec_inp,
                                    batch_y_mark)
                f_dim = -1 if args.features == "MS" else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(exp.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                losses.append(criterion(pred, true).item())
                elem.append(int(true.numel()))
        exp.model.train()
        return losses, elem

    # ---- per-rule early stopping (the REAL state machine) ---------------
    es = {}
    for r in rules:
        lockd = os.path.join(jd, "locks", r)
        os.makedirs(lockd, exist_ok=True)
        es[r] = {"sm": EarlyStopping(patience=args.patience, verbose=True),
                 "dir": lockd, "locked": False, "trigger_epoch": None,
                 "events": []}

    max_epochs = args_cli.max_epochs
    curve = {r: {"val": [], "improved": [], "counter": [], "n_val_elem": []}
             for r in rules}
    c_idx_by_epoch = []
    lr_used_hist, lr_after_hist, train_loss_hist = [], [], []
    rng_audit = {"per_epoch": []}
    trajectory = []            # per-epoch: state sha + train sampler order
    phases = []                # per-epoch timing/hooks/locks records

    t_train0 = time.time()
    for epoch in range(max_epochs):
        e1 = epoch + 1
        lr_used = model_optim.param_groups[0]["lr"]
        rng_pre_epoch = rng_snapshot()
        t0 = time.time()
        exp.model.train()
        train_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in \
                enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat(
                [batch_y[:, :args.label_len, :], dec_inp],
                dim=1).float().to(exp.device)
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(exp.device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            loss.backward()
            model_optim.step()
        t_train = time.time() - t0

        rng_post_train = rng_snapshot()
        state_sha = canonical_state_sha(exp.model.state_dict())
        order_rec = recorder.order_hashes()[-1]
        trajectory.append({"epoch": e1, "state_sha256": state_sha,
                           "order": order_rec})
        train_loss_mean = float(np.average(train_loss)) if train_loss \
            else float("nan")
        train_loss_hist.append(train_loss_mean)

        epoch_rec = {"epoch": e1, "train_s": t_train,
                     "train_loss_mean": train_loss_mean,
                     "lr_used": lr_used, "hooks": []}

        # ---- validation hooks (isolated; four-state audited) ----------
        vals = {}
        if "A" in rules or "B" in rules:
            pre = rng_snapshot()
            t1 = time.time()
            a_val, b_losses, n_elem = ab_pass()
            dt = time.time() - t1
            post = rng_snapshot()
            rng_assert_unchanged(pre, post, "AB-hook epoch %d" % e1, rec)
            epoch_rec["hooks"].append({"hook": "AB", "s": dt,
                                       "n_batches": len(b_losses),
                                       "n_elem": n_elem})
            vals["A"] = a_val
            curve["A"]["val"].append(a_val)
            curve["A"]["n_val_elem"].append(n_elem)
            if "B" in rules:
                vals["B"] = float(np.average(b_losses))
                curve["B"]["val"].append(vals["B"])
                curve["B"]["n_val_elem"].append(n_elem)
        if "C" in rules:
            pre = rng_snapshot()
            t1 = time.time()
            perm = c_rs.permutation(N_val)
            idx_used = perm[:n_batches_c * args.batch_size].astype(np.int64)
            c_losses, c_elem = c_pass(idx_used)
            dt = time.time() - t1
            post = rng_snapshot()
            rng_assert_unchanged(pre, post, "C-hook epoch %d" % e1, rec)
            epoch_rec["hooks"].append({"hook": "C", "s": dt,
                                       "n_batches": len(c_losses),
                                       "n_elem_total": int(sum(c_elem))})
            vals["C"] = float(np.average(c_losses))
            curve["C"]["val"].append(vals["C"])
            curve["C"]["n_val_elem"].append(int(sum(c_elem)))
            c_idx_by_epoch.append(idx_used)

        # ---- per-rule early stopping (locked rules skipped) -----------
        lock_rec_epoch = {}
        for r in rules:
            st = es[r]
            if st["locked"]:
                lock_rec_epoch[r] = "locked-skip"
                continue
            sm = st["sm"]
            sm(vals[r], exp.model, st["dir"])
            improved = (sm.val_loss_min == vals[r])
            st["events"].append({"epoch": e1, "val": vals[r],
                                 "improved": bool(improved),
                                 "counter": int(sm.counter)})
            curve[r]["improved"].append(bool(improved))
            curve[r]["counter"].append(int(sm.counter))
            if sm.early_stop:
                st["locked"] = True
                st["trigger_epoch"] = e1
                lock_rec_epoch[r] = "LOCKED"
                log("rule %s LOCKED at epoch %d (val %.7f)"
                    % (r, e1, vals[r]))
            else:
                lock_rec_epoch[r] = ("improved" if improved
                                     else "counter=%d" % sm.counter)
        epoch_rec["locks"] = lock_rec_epoch

        # ---- lr update (official cadence; skipped when breaking) ------
        all_locked = all(es[r]["locked"] for r in rules)
        if not all_locked:
            adjust_learning_rate(model_optim, e1, args)
            lr_after = model_optim.param_groups[0]["lr"]
        else:
            lr_after = None
        lr_used_hist.append(lr_used)
        lr_after_hist.append(lr_after)
        epoch_rec["lr_after_update"] = lr_after

        # ---- TG9 dual-lr assertions (every completed epoch) -----------
        assert abs(lr_used - X.expected_train_lr(e1)) <= 1e-12, \
            "TG9 train_lr_used epoch %d: %r != %r" % (
                e1, lr_used, X.expected_train_lr(e1))
        if lr_after is not None:
            assert abs(lr_after - X.expected_lr_after_update(e1)) <= 1e-12, \
                "TG9 lr_after_update epoch %d: %r != %r" % (
                    e1, lr_after, X.expected_lr_after_update(e1))

        rng_audit["per_epoch"].append({
            "epoch": e1,
            "pre_epoch": rng_pre_epoch, "post_train": rng_post_train,
            "post_hooks": rng_snapshot(),
        })
        phases.append(epoch_rec)
        log("E%03d train %.1fs loss %.7f | %s | lr_after=%s"
            % (e1, t_train, train_loss_mean,
               " ".join("%s=%.7f" % (r, vals[r]) for r in rules),
               lr_after if lr_after is not None else "none(all-locked)"))
        if all_locked:
            log("all rules locked at epoch %d -> break (no lr update, "
                "mirrors base train break)" % e1)
            break

    train_wall = time.time() - t_train0

    # ---- finalize rule selections ---------------------------------------
    lock_events = []
    for r in rules:
        st = es[r]
        ck = X.rule_ckpt(h, seed, r, round_tag)
        assert os.path.isfile(ck), "rule %s checkpoint missing: %s" % (r, ck)
        vl = C.load_json(os.path.join(st["dir"], "valid_loss.json"))
        lock_events.append({
            "rule": r, "locked": st["locked"],
            "trigger_epoch": st["trigger_epoch"],
            "best_valid_loss": vl["best_valid_loss"],
            "best_valid_epoch": vl["best_valid_epoch"],
            "n_improvements": sum(1 for e in st["events"] if e["improved"]),
            "ckpt_sha256": C.sha256_of(ck),
            "state_sha256": None,   # filled at test time after reload
        })
        if not st["locked"]:
            log("rule %s: NO-TRIGGER after %d epochs; best=epoch %d "
                "(checkpoint.pth = state-machine best-so-far)"
                % (r, len(st["events"]), vl["best_valid_epoch"]))

    # ---- per-rule test (strict reload -> official test path) ------------
    test_data, test_loader = data_provider(args, "test")
    test_rec = {}
    for r in rules:
        ck = X.rule_ckpt(h, seed, r, round_tag)
        sd_t = torch.load(ck, map_location="cpu", weights_only=False)
        assert isinstance(sd_t, dict) and len(sd_t) > 0
        exp.model.load_state_dict(sd_t)   # strict (default)
        for ent in lock_events:
            if ent["rule"] == r:
                ent["state_sha256"] = canonical_state_sha(sd_t)
        preds, trues = [], []
        exp.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(exp.device)
                batch_y = batch_y.float().to(exp.device)
                batch_x_mark = batch_x_mark.float().to(exp.device)
                batch_y_mark = batch_y_mark.float().to(exp.device)
                dec_inp = torch.zeros_like(
                    batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :args.label_len, :], dec_inp],
                    dim=1).float().to(exp.device)
                outputs = exp.model(batch_x, batch_x_mark, dec_inp,
                                    batch_y_mark)
                f_dim = -1 if args.features == "MS" else 0
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        rdir = X.rule_results_dir(h, seed, r, round_tag)
        os.makedirs(rdir, exist_ok=True)
        np.save(os.path.join(rdir, "metrics.npy"),
                np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(rdir, "pred.npy"), preds)
        np.save(os.path.join(rdir, "true.npy"), trues)
        ent = next(e for e in lock_events if e["rule"] == r)
        test_rec[r] = {"mse": float(mse), "mae": float(mae),
                       "rmse": float(rmse), "mape": float(mape),
                       "mspe": float(mspe),
                       "best_valid_loss": ent["best_valid_loss"],
                       "best_valid_epoch": ent["best_valid_epoch"],
                       "locked": ent["locked"],
                       "trigger_epoch": ent["trigger_epoch"],
                       "ckpt_sha256": ent["ckpt_sha256"],
                       "state_sha256": ent["state_sha256"],
                       "n_test_samples": int(preds.shape[0])}
        log("rule %s TEST mse=%.7f mae=%.7f (best epoch %d, locked=%s)"
            % (r, mse, mae, ent["best_valid_epoch"], ent["locked"]))

    # ---- curves.npz + trajectory/phases/rng records ----------------------
    npz = {"epochs": np.arange(1, len(lr_used_hist) + 1, dtype=np.int64),
           "train_loss_mean": np.asarray(train_loss_hist, dtype=np.float64),
           "train_lr_used": np.asarray(lr_used_hist, dtype=np.float64),
           "lr_after_update": np.asarray(
               [np.nan if v is None else v for v in lr_after_hist],
               dtype=np.float64)}
    for r in rules:
        npz["%s_val" % r] = np.asarray(curve[r]["val"], dtype=np.float64)
        npz["%s_improved" % r] = np.asarray(curve[r]["improved"], dtype=bool)
        npz["%s_counter" % r] = np.asarray(curve[r]["counter"], dtype=np.int64)
        npz["%s_n_val_elem" % r] = np.asarray(curve[r]["n_val_elem"],
                                              dtype=np.int64)
    for k, idx in enumerate(c_idx_by_epoch, start=1):
        npz["c_idx_e%d" % k] = idx
    np.savez_compressed(os.path.join(jd, "curves.npz"), **npz)

    with open(os.path.join(jd, "trajectory.json"), "w") as f:
        json.dump({"init_state_sha256": rec["init"]["state_sha256"],
                   "epochs": trajectory}, f, indent=1)
    with open(os.path.join(jd, "phases.json"), "w") as f:
        json.dump(phases, f, indent=1)
    with open(X.rng_audit_path(h, seed, round_tag), "w") as f:
        json.dump(rng_audit, f, indent=1)
    with open(X.lock_events_path(h, seed, round_tag), "w") as f:
        json.dump({"rules": list(rules),
                   "events": {r: es[r]["events"] for r in rules},
                   "final": lock_events}, f, indent=1)

    # ---- SUCCESS + train json -------------------------------------------
    rec["finished_epochs"] = len(lr_used_hist)
    rec["train_wall_s"] = train_wall
    rec["test"] = test_rec
    rec["finished"] = C.time_str()
    with open(os.path.join(jd, "SUCCESS.txt"), "w") as f:
        json.dump({"job": job, "finished": rec["finished"], "seed": seed,
                   "rules": list(rules),
                   "test": {r: {"mse": test_rec[r]["mse"],
                                "mae": test_rec[r]["mae"],
                                "best_valid_epoch":
                                    test_rec[r]["best_valid_epoch"],
                                "locked": test_rec[r]["locked"]}
                            for r in rules},
                   "val_n": N_val}, f, indent=1)
    with open(X.train_json(h, seed, round_tag), "w") as f:
        json.dump(rec, f, indent=1)
    log("%s DONE in %.1fs: %s"
        % (job, train_wall, {r: (test_rec[r]["mse"], test_rec[r]["mae"])
                             for r in rules}))


if __name__ == "__main__":
    main()
