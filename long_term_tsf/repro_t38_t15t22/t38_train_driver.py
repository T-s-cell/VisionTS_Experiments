"""EXP-038 task B driver: ONE config/horizon/arm formal training executed
ZERO-CHANGE through repro_fullshot/run_train.py main(), with
observation-only hooks (H1-H4, identical to EXP-037) plus per-phase
timing/peak-memory wrappers (train/vali/test; vali runs once inside
train at epochs=1, so train_pure = train_wall - vali_wall).

Run: python t38_train_driver.py --config ETTm1_96 --arm all [--gpu 0]
(chdirs to long_term_tsf; same CWD contract as EXP-003)

Generalization from EXP-037's t37_train_driver.py:
  - argv rebuilt from the config's OWN EXP-003 audited args.json
    (save_fullshot/<DS>_<H>_seed2021/args.json); deltas only
    {model_id, save_dir, ft_type, vm_pretrained, gpu};
  - job name <DS>_<H>_seed2021_<arm>; a job dir that already has
    SUCCESS.txt is REFUSED (run_train SKIP must never silently reuse a
    previous run);
  - hooks: H1 VisionTS.__init__ (canonical sha, trainable list, RNG,
    per-key bitwise vs frozen ckpt; all arm MUST equal, rand arm MUST
    differ); H2 Exp_Univariate.__init__ RNG at train entry; H3
    RandomSampler full-epoch index sequence (count/sha asserted against
    the formula 7*(12*month_rows - L - h + 1)); H4 first train-batch
    tensor hashes;
  - record: logs/train_<DS>_<H>_<arm>.json.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

import t38_common as C


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def rng_hashes():
    import torch
    parts = {}
    parts["torch_cpu"] = C.sha256_bytes(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        parts["torch_cuda"] = C.sha256_bytes(
            torch.cuda.get_rng_state().numpy().tobytes())
    nps = np.random.get_state()
    _nh = __import__("hashlib").sha256()
    _nh.update(nps[0].encode())
    _nh.update(np.ascontiguousarray(nps[1], dtype=np.uint32).tobytes())
    _nh.update(np.asarray(nps[2:], dtype=np.int64).tobytes())
    parts["numpy"] = _nh.hexdigest()
    py = random.getstate()
    parts["python"] = C.sha256_bytes(np.asarray(py[1], dtype=np.uint64).tobytes())
    return parts


def canonical_state_sha(state_dict):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(("|%s|%s|" % (t.dtype, tuple(t.shape))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e.g. ETTm1_96")
    ap.add_argument("--arm", required=True, choices=C.B_ARMS)
    ap.add_argument("--gpu", type=int, default=0)
    dargs = ap.parse_args()
    ds, h = dargs.config.rsplit("_", 1)
    ds, h = ds, int(h)
    assert ds in C.ETT4 and h in C.HORIZONS, "unknown config %s" % dargs.config
    arm = dargs.arm
    job = C.config_job(ds, h, arm)
    save_dir = os.path.abspath(C.b_job_dir(ds, h, arm))

    os.makedirs(C.LOG_DIR, exist_ok=True)
    os.makedirs(C.SAVE_B_BASE, exist_ok=True)
    if os.path.isfile(os.path.join(save_dir, "SUCCESS.txt")):
        raise SystemExit("[t38] refusing: SUCCESS.txt already exists in %s "
                         "(move the dir away before re-running)" % save_dir)
    os.chdir(C.ROOT)  # long_term_tsf — EXP-003 CWD contract

    exp003_args_path = os.path.join(C.exp003_job(ds, h), "args.json")
    assert os.path.isfile(exp003_args_path), \
        "EXP-003 args.json missing: %s" % exp003_args_path
    exp003_args = C.load_json(exp003_args_path)
    ckpt_path = os.path.abspath(os.path.join(C.ROOT, "..", "ckpt",
                                             "mae_visualize_vit_base.pth"))
    assert os.path.isfile(ckpt_path), "frozen ckpt missing: %s" % ckpt_path
    ckpt_sha = C.sha256_of(ckpt_path)
    assert ckpt_sha == C.CKPT_SHA256, "ckpt sha mismatch: %s" % ckpt_sha

    # ---- build argv from the audited EXP-003 recipe -------------------
    BOOL_SKIP = {k for k, v in exp003_args.items() if isinstance(v, bool)}
    SKIP = BOOL_SKIP | {"save_dir", "model_id", "ft_type", "vm_pretrained",
                        "archive_dir"}
    argv = ["run_train.py"]
    for k, v in exp003_args.items():
        if k in SKIP:
            continue
        if isinstance(v, (list, tuple)):  # nargs='+' args (p_hidden_dims)
            argv += ["--%s" % k] + [str(x) for x in v]
        else:
            argv += ["--%s" % k, str(v)]
    argv += ["--model_id", job,
             "--save_dir", save_dir,
             "--ft_type", "full",
             "--vm_pretrained", "1" if arm == "all" else "0",
             "--gpu", str(dargs.gpu)]
    log("%s argv: %s" % (job, " ".join(argv)))

    # ---- install observation-only hooks --------------------------------
    import torch
    import torch.utils.data as tud

    rec = {"config": dargs.config, "ds": ds, "h": h, "arm": arm, "job": job,
           "started": C.time_str(), "argv": argv, "ckpt_path": ckpt_path,
           "ckpt_sha256": ckpt_sha,
           "env": {"python": sys.version.split()[0],
                   "torch": torch.__version__, "numpy": np.__version__,
                   "gpu": torch.cuda.get_device_name(dargs.gpu)
                   if torch.cuda.is_available() else "cpu"}}

    init_rec = {"called": 0}
    import visionts.model as vmodel
    assert os.path.samefile(vmodel.__file__,
                            os.path.join(C.REPO_ROOT, "visionts", "model.py")), \
        vmodel.__file__
    _vinit = vmodel.VisionTS.__init__

    def vinit_wrapper(self, arch="mae_base", finetune_type="ln",
                      ckpt_dir="./ckpt/", load_ckpt=True):
        _vinit(self, arch, finetune_type, ckpt_dir, load_ckpt)
        init_rec["called"] += 1
        init_rec["init_args"] = {"arch": arch, "finetune_type": finetune_type,
                                 "ckpt_dir": ckpt_dir, "load_ckpt": load_ckpt}
        sd = self.vision_model.state_dict()
        init_rec["canonical_sha256"] = canonical_state_sha(sd)
        trainable = [n for n, p in self.vision_model.named_parameters()
                     if p.requires_grad]
        init_rec["trainable_names"] = sorted(trainable)
        init_rec["trainable_count"] = len(trainable)
        init_rec["trainable_numel"] = int(sum(p.numel() for p in
                                              self.vision_model.parameters()
                                              if p.requires_grad))
        init_rec["n_params_total"] = int(sum(p.numel() for p in
                                             self.vision_model.parameters()))
        init_rec["rng_after_init"] = rng_hashes()
        ck = torch.load(ckpt_path, map_location="cpu")["model"]
        eq, diff = [], []
        with torch.no_grad():
            for k in sorted(ck.keys()):
                same = k in sd and sd[k].shape == ck[k].shape and \
                    torch.equal(sd[k].cpu(), ck[k].cpu())
                (eq if same else diff).append(k)
        init_rec["vs_ckpt"] = {"n_keys": len(ck), "n_equal": len(eq),
                               "n_differ": len(diff), "differ_keys": diff[:400]}
        if arm == "all":
            assert len(diff) == 0, \
                "all arm init != ckpt tensors: %s..." % diff[:5]
            init_rec["vs_ckpt"]["verdict"] = "ALL_KEYS_EQUAL_TO_CKPT"
        else:
            assert len(diff) > 0, \
                "rand arm init unexpectedly identical to ckpt"
            init_rec["vs_ckpt"]["verdict"] = "RANDOM_INIT_RECORDED"

    vmodel.VisionTS.__init__ = vinit_wrapper

    import exp_univariate as eu
    assert os.path.samefile(eu.__file__,
                            os.path.join(C.REPO_ROOT, "repro_fullshot",
                                         "exp_univariate.py")), eu.__file__
    _einit = eu.Exp_Univariate.__init__
    phase = {"train_wall": None, "train_peak_mb": None, "vali_wall": None,
             "vali_peak_mb": None, "test_wall": None, "test_peak_mb": None,
             "train_end_peak_mb": None, "n_vali_calls": 0}
    gpu = dargs.gpu

    def peak_now():
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(gpu) / 1024 ** 2
        return None

    def einit_wrapper(self, *a, **kw):
        _einit(self, *a, **kw)
        rec["rng_at_train_entry"] = rng_hashes()
        _tr, _va, _te = self.train, self.vali, self.test
        vali_state = {"first": True}

        def train_w(setting):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(gpu)
            t = time.time()
            try:
                return _tr(setting)
            finally:
                phase["train_wall"] = time.time() - t
                phase["train_end_peak_mb"] = peak_now()

        def vali_w(*a, **kw):
            if vali_state["first"]:
                vali_state["first"] = False
                phase["train_peak_mb"] = peak_now()  # train batches only
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(gpu)
            phase["n_vali_calls"] += 1
            t = time.time()
            try:
                return _va(*a, **kw)
            finally:
                phase["vali_wall"] = (phase["vali_wall"] or 0.0) + \
                    (time.time() - t)
                phase["vali_peak_mb"] = peak_now()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(gpu)

        def test_w(*a, **kw):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(gpu)
            t = time.time()
            try:
                return _te(*a, **kw)
            finally:
                phase["test_wall"] = time.time() - t
                phase["test_peak_mb"] = peak_now()

        self.train = train_w
        self.vali = vali_w
        self.test = test_w

    eu.Exp_Univariate.__init__ = einit_wrapper

    _rs_iter = tud.RandomSampler.__iter__
    rs_seqs = []

    def rs_iter_wrapper(self):
        it = _rs_iter(self)
        seq = []
        handle = {"n": None, "first20": None, "last20": None, "seq": seq}
        rs_seqs.append(handle)

        def gen():
            try:
                for i in it:
                    seq.append(int(i))
                    yield i
            finally:
                handle["n"] = len(seq)
        return gen()

    tud.RandomSampler.__iter__ = rs_iter_wrapper

    first_batch = {"done": False}
    _dl_iter = tud.DataLoader.__iter__

    def dl_iter_wrapper(self):
        it = _dl_iter(self)
        if not isinstance(getattr(self, "sampler", None), tud.RandomSampler):
            return it

        def gen():
            for b in it:
                if not first_batch["done"]:
                    first_batch["done"] = True
                    first_batch["tensors"] = [
                        {"shape": list(t.shape),
                         "sha256": C.sha256_bytes(
                             np.ascontiguousarray(
                                 t.detach().cpu().numpy()).tobytes())}
                        for t in list(b)[:4]]
                yield b
        return gen()

    tud.DataLoader.__iter__ = dl_iter_wrapper

    # ---- zero-change formal training ------------------------------------
    torch.zeros(1, device="cuda:%d" % gpu)  # lazy-init before peak reset
    torch.cuda.reset_peak_memory_stats(gpu)
    t0 = time.time()
    sys.argv = argv
    import run_train
    assert os.path.samefile(run_train.__file__,
                            os.path.join(C.REPO_ROOT, "repro_fullshot",
                                         "run_train.py")), run_train.__file__
    assert init_rec["called"] == 0, \
        "VisionTS constructed before hooks were armed"
    rc = run_train.main()
    wall = time.time() - t0
    assert rc == 0, "run_train.main() returned %s" % rc

    # ---- post-training evidence -----------------------------------------
    for hnd in rs_seqs:
        arr = np.asarray(hnd["seq"], dtype=np.int64)
        hnd["sha256"] = C.sha256_bytes(arr.tobytes())
        hnd["first20"] = arr[:20].tolist()
        hnd["last20"] = arr[-20:].tolist()
        hnd.pop("seq")
    rec["sampler"] = {"n_randomsampler_iters": len(rs_seqs), "iters": rs_seqs}
    exp_epoch = C.expected_epoch_len(ds, h)
    got_len = rs_seqs[0]["n"] if rs_seqs else -1
    rec["sampler_len_check"] = {
        "expected_epoch_len": exp_epoch,
        "expected_train_batches": C.expected_train_batches(ds, h),
        "expected_eval_batches": C.expected_eval_batches(ds, h),
        "got": got_len, "ok": got_len == exp_epoch}
    rec["first_batch"] = first_batch
    rec["init"] = init_rec
    rec["phases"] = phase
    rec["args_delta_check"] = compare_args(exp003_args, save_dir)
    rec["fingerprint"] = C.load_json(os.path.join(save_dir, "fingerprint.json"))
    rec["success"] = C.load_json(os.path.join(save_dir, "SUCCESS.txt"))
    final_ckpt = os.path.join(save_dir, "checkpoints", "_", "checkpoint.pth")
    rec["final_ckpt_sha256"] = C.sha256_of(final_ckpt)
    rec["timing"] = {"wall_sec": wall}
    if torch.cuda.is_available():
        rec["gpu_peak"] = {
            "allocated_mb": torch.cuda.max_memory_allocated(gpu) / 1024 ** 2,
            "reserved_mb": torch.cuda.max_memory_reserved(gpu) / 1024 ** 2}
    rec["finished"] = C.time_str()
    C.dump_json(C.train_json(ds, h, arm), rec)
    log("%s: mse %.8f mae %.8f, sampler %d (expect %d), wall %.1f s "
        "(train %.1f vali %.1f test %.1f)"
        % (job, rec["success"]["mse"], rec["success"]["mae"], got_len,
           exp_epoch, wall, phase["train_wall"] or -1,
           phase["vali_wall"] or -1, phase["test_wall"] or -1))
    if not rec["sampler_len_check"]["ok"]:
        raise SystemExit("[t38] sampler epoch length mismatch")
    log("DONE")


def compare_args(exp003_args, save_dir):
    got = C.load_json(os.path.join(save_dir, "args.json"))
    deltas = {k: {"exp003": exp003_args.get(k, "<missing>"),
                  "this": got.get(k, "<missing>")}
              for k in set(exp003_args) | set(got)
              if exp003_args.get(k, "<missing>") != got.get(k, "<missing>")}
    unexpected = sorted(set(deltas) - C.FP_ALLOWED_ARG_DELTA)
    assert not unexpected, \
        "unexpected args deltas vs EXP-003: %s (%s)" % (unexpected, deltas)
    return {"deltas": deltas, "unexpected": unexpected,
            "verdict": "ONLY_ALLOWED_DELTAS"}


if __name__ == "__main__":
    main()
