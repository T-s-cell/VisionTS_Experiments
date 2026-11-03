"""EXP-037 task B driver: one formal training arm ("all" or "rand") executed
ZERO-CHANGE through repro_fullshot/run_train.py main(), with observation-only
hooks installed in this process to satisfy the EXP-037 hash-evidence gates
(user correction #2: evidence must come from the two formal trainings, not a
side probe).

Run: python t37_train_driver.py --arm all [--gpu 0]
(chdirs to long_term_tsf itself; same CWD contract as EXP-003)

Hooks (installed before run_train.main(), never alter control flow or RNG):
  H1 visionts.model.VisionTS.__init__ post-hook -> canonical state_dict
     sha256 (sorted keys + dtype/shape/bytes), trainable-name list + count,
     RNG-state hashes (torch/np/python), and per-key bitwise comparison of
     the initialised vision_model against the frozen ckpt tensors
     (all arm: must equal; rand arm: recorded, expected to differ);
  H2 exp_univariate.Exp_Univariate.__init__ post-hook -> RNG-state hashes
     at train entry (nothing consumes RNG between init and train);
  H3 torch RandomSampler.__iter__ wrapper -> full index sequence of the
     train epoch (n, sha256 of int64 bytes, first/last 20);
  H4 torch DataLoader.__iter__ wrapper (flag=='train' only) -> hashes of
     the first batch's 4 tensors (auxiliary evidence).

argv is rebuilt from the EXP-003 ETTh1_96_seed2021 args.json so every
non-delta hyperparameter is pinned to the audited recipe; the only deltas
are model_id/save_dir/ft_type=full/vm_pretrained(1|0)/gpu. After main()
returns 0, the record train_<arm>.json carries: passed argv, fingerprint
(copy), args-delta check, init evidence, sampler evidence, first-batch
hashes, wall time, in-process CUDA peak, SUCCESS mse/mae, final ckpt sha.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

import t37_common as C


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
    _nh = hashlib.sha256()
    _nh.update(nps[0].encode())
    _nh.update(np.ascontiguousarray(nps[1], dtype=np.uint32).tobytes())
    _nh.update(np.asarray(nps[2:], dtype=np.int64).tobytes())
    parts["numpy"] = _nh.hexdigest()
    py = random.getstate()
    parts["python"] = C.sha256_bytes(np.asarray(py[1], dtype=np.uint64).tobytes())
    return parts


def canonical_state_sha(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(("|%s|%s|" % (t.dtype, tuple(t.shape))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=C.B_ARMS)
    ap.add_argument("--gpu", type=int, default=0)
    dargs = ap.parse_args()
    arm = dargs.arm
    job = C.B_JOB[arm]
    save_dir = os.path.abspath(C.b_job_dir(arm))

    os.makedirs(C.LOG_DIR, exist_ok=True)
    os.makedirs(C.SAVE_B_BASE, exist_ok=True)
    os.chdir(C.ROOT)  # long_term_tsf — EXP-003 CWD contract

    exp003_args_path = os.path.join(C.exp003_job(), "args.json")
    assert os.path.isfile(exp003_args_path), "EXP-003 args.json missing: %s" % exp003_args_path
    exp003_args = C.load_json(exp003_args_path)
    ckpt_path = os.path.abspath(os.path.join(C.ROOT, "..", "ckpt",
                                             "mae_visualize_vit_base.pth"))
    assert os.path.isfile(ckpt_path), "frozen ckpt missing: %s" % ckpt_path
    ckpt_sha = C.sha256_of(ckpt_path)
    assert ckpt_sha == C.CKPT_SHA256, "ckpt sha mismatch: %s" % ckpt_sha

    # ---- build argv from the audited EXP-003 recipe -------------------
    # bools are all default in args.json and one (--distil) is an
    # action='store_false' trap; deltas are passed explicitly below.
    BOOL_SKIP = {k for k, v in exp003_args.items() if isinstance(v, bool)}
    SKIP = BOOL_SKIP | {"save_dir", "model_id", "ft_type", "vm_pretrained",
                        "archive_dir"}
    argv = ["run_train.py"]
    for k, v in exp003_args.items():
        if k in SKIP:
            continue
        if isinstance(v, (list, tuple)):  # nargs='+' args (e.g. p_hidden_dims)
            argv += ["--%s" % k] + [str(x) for x in v]
        else:
            argv += ["--%s" % k, str(v)]
    argv += ["--model_id", job,
             "--save_dir", save_dir,
             "--ft_type", "full",
             "--vm_pretrained", "1" if arm == "all" else "0",
             "--gpu", str(dargs.gpu)]
    log("%s argv: %s" % (arm, " ".join(argv)))

    # ---- install observation-only hooks --------------------------------
    import torch
    import torch.utils.data as tud

    rec = {"arm": arm, "job": job, "started": C.time_str(),
           "argv": argv, "ckpt_path": ckpt_path, "ckpt_sha256": ckpt_sha,
           "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                   "numpy": np.__version__,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}}

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
        trainable = [n for n, p in self.vision_model.named_parameters() if p.requires_grad]
        init_rec["trainable_names"] = sorted(trainable)
        init_rec["trainable_count"] = len(trainable)
        init_rec["trainable_numel"] = int(sum(p.numel() for p in self.vision_model.parameters()
                                              if p.requires_grad))
        init_rec["n_params_total"] = int(sum(p.numel() for p in self.vision_model.parameters()))
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
            assert len(diff) == 0, "all arm init != ckpt tensors: %s..." % diff[:5]
            init_rec["vs_ckpt"]["verdict"] = "ALL_KEYS_EQUAL_TO_CKPT"
        else:
            assert len(diff) > 0, "rand arm init unexpectedly identical to ckpt"
            init_rec["vs_ckpt"]["verdict"] = "RANDOM_INIT_RECORDED"

    vmodel.VisionTS.__init__ = vinit_wrapper

    import exp_univariate as eu
    assert os.path.samefile(eu.__file__,
                            os.path.join(C.REPO_ROOT, "repro_fullshot",
                                         "exp_univariate.py")), eu.__file__
    _einit = eu.Exp_Univariate.__init__

    def einit_wrapper(self, *a, **kw):
        _einit(self, *a, **kw)
        rec["rng_at_train_entry"] = rng_hashes()

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
        if getattr(self.dataset, "flag", None) != "train":
            return it

        def gen():
            for b in it:
                if not first_batch["done"]:
                    first_batch["done"] = True
                    first_batch["tensors"] = [
                        {"shape": list(t.shape),
                         "sha256": C.sha256_bytes(
                             np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())}
                        for t in list(b)[:4]]
                yield b
        return gen()

    tud.DataLoader.__iter__ = dl_iter_wrapper

    # ---- zero-change formal training ------------------------------------
    # run_train is imported AFTER the hooks are installed; importing it
    # never triggers training (main() is called explicitly below).
    torch.zeros(1, device="cuda:%d" % dargs.gpu)  # lazy-init before peak reset
    torch.cuda.reset_peak_memory_stats(dargs.gpu)
    t0 = time.time()
    sys.argv = argv
    import run_train
    assert os.path.samefile(run_train.__file__,
                            os.path.join(C.REPO_ROOT, "repro_fullshot",
                                         "run_train.py")), run_train.__file__
    assert init_rec["called"] == 0, "VisionTS constructed before hooks were armed"
    rc = run_train.main()
    wall = time.time() - t0
    assert rc == 0, "run_train.main() returned %s" % rc

    # ---- post-training evidence -----------------------------------------
    for h in rs_seqs:
        arr = np.asarray(h["seq"], dtype=np.int64)
        h["sha256"] = C.sha256_bytes(arr.tobytes())
        h["first20"] = arr[:20].tolist()
        h["last20"] = arr[-20:].tolist()
        h.pop("seq")
    rec["sampler"] = {"n_randomsampler_iters": len(rs_seqs), "iters": rs_seqs}
    rec["first_batch"] = first_batch
    rec["init"] = init_rec
    rec["args_delta_check"] = compare_args(exp003_args, save_dir)
    rec["fingerprint"] = C.load_json(os.path.join(save_dir, "fingerprint.json"))
    rec["success"] = C.load_json(os.path.join(save_dir, "SUCCESS.txt"))
    final_ckpt = os.path.join(save_dir, "checkpoints", "_", "checkpoint.pth")
    rec["final_ckpt_sha256"] = C.sha256_of(final_ckpt)
    rec["timing"] = {"wall_sec": wall}
    if torch.cuda.is_available():
        rec["gpu_peak"] = {"allocated_mb": torch.cuda.max_memory_allocated(dargs.gpu) / 1024 ** 2,
                           "reserved_mb": torch.cuda.max_memory_reserved(dargs.gpu) / 1024 ** 2}
    rec["finished"] = C.time_str()
    C.dump_json(C.train_json(arm), rec)
    log("%s: mse %.8f mae %.8f, %d sampler iters, wall %.1f s"
        % (arm, rec["success"]["mse"], rec["success"]["mae"],
           len(rs_seqs), wall))
    log("DONE")


def compare_args(exp003_args, save_dir):
    got = C.load_json(os.path.join(save_dir, "args.json"))
    deltas = {k: {"exp003": exp003_args.get(k, "<missing>"), "this": got.get(k, "<missing>")}
              for k in set(exp003_args) | set(got)
              if exp003_args.get(k, "<missing>") != got.get(k, "<missing>")}
    unexpected = sorted(set(deltas) - C.FP_ALLOWED_ARG_DELTA - {"gpu"})
    assert not unexpected, "unexpected args deltas vs EXP-003: %s (%s)" % (unexpected, deltas)
    return {"deltas": deltas, "unexpected": unexpected,
            "verdict": "ONLY_ALLOWED_DELTAS"}


if __name__ == "__main__":
    main()
