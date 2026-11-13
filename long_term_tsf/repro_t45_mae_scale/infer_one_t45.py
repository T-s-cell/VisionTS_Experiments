"""EXP-045 launcher wrapper: runs EXP-001's run.py UNCHANGED with
--vm_arch <mae_base|mae_large|mae_huge> (injected by run_one_t45.sh) while
recording evidence from the real inference path:
  1. the VisionTS constructor call: arch AS PASSED (mae_base/large/huge),
     finetune_type, ckpt_dir
  2. every update_config call: patch size, patch grid (num_patch), the
     visible column count num_patch_input (npi) + pre-clamp npi_raw,
     mask ratio, pad, scale_x, resize size tuples PLUS norm_const and
     align_const AS STORED ON THE MODEL (parameter effectivity evidence)
  3. every pre-interpolation 2D segmentation rearrange
     pattern 'b n (p f) -> (b n) 1 f p'
All framework files are untouched; we only observe via a module-level
einops proxy and class-level __init__/update_config wrappers in
visionts.model (same mechanism as t44's infer_one_t44.py / t43).

Usage:
  python infer_one_t45.py --shape_log <path> -- <run.py args...>
(the leading run.py argv is passed through verbatim)
"""
import argparse
import inspect
import json
import os
import runpy
import sys
import time

AP = argparse.ArgumentParser(add_help=False)
AP.add_argument("--shape_log", required=True)
AP.add_argument("--run_py", default="run.py")
FLAGS, REST = AP.parse_known_args()
while REST and REST[0] == "--":  # argparse keeps the separator in REST
    REST.pop(0)
sys.argv = ["run.py"] + REST

ROOT = os.path.dirname(os.path.abspath(FLAGS.run_py))
sys.path.insert(0, ROOT)
REPO = os.path.dirname(ROOT)  # where the visionts/ package lives
if REPO not in sys.path:
    sys.path.append(REPO)

import importlib

vm = importlib.import_module("visionts.model")

RECORDS = []


class _EinopsProxy:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def rearrange(self, tensor, pattern, **kw):
        out = self._real.rearrange(tensor, pattern, **kw)
        if pattern == "b n (p f) -> (b n) 1 f p":
            RECORDS.append({"type": "segmentation", "f": kw.get("f"),
                            "out_shape": [int(x) for x in out.shape],
                            "t": time.time()})
        return out


_orig_init = vm.VisionTS.__init__
_INIT_SIG = inspect.signature(_orig_init)


def _patched_init(self, *a, **k):
    bound = _INIT_SIG.bind(self, *a, **k)
    bound.apply_defaults()
    _orig_init(self, *a, **k)
    RECORDS.append({
        "type": "init",
        "arch": str(bound.arguments["arch"]),
        "finetune_type": str(bound.arguments["finetune_type"]),
        "ckpt_dir": str(bound.arguments["ckpt_dir"]),
        "load_ckpt": bool(bound.arguments["load_ckpt"]),
        "t": time.time()})


_orig_uc = vm.VisionTS.update_config
_UC_SIG = inspect.signature(_orig_uc)


def _patched_update_config(self, *a, **k):
    # record the values PASSED to update_config (run.py's injection);
    # align_const is a local in update_config and is NOT stored on self,
    # so self-attribute access would crash (t44 smoke lesson)
    bound = _UC_SIG.bind(self, *a, **k)
    bound.apply_defaults()
    passed_norm = float(bound.arguments["norm_const"])
    passed_align = float(bound.arguments["align_const"])
    passed_interp = str(bound.arguments["interpolation"])
    _orig_uc(self, *a, **k)
    RECORDS.append({
        "type": "update_config",
        "periodicity": int(self.periodicity),
        "context_len": int(self.context_len),
        "pred_len": int(self.pred_len),
        "patch_size": int(self.patch_size),
        "num_patch": int(self.num_patch),
        "image_size": int(self.image_size),
        "pad_left": int(self.pad_left), "pad_right": int(self.pad_right),
        "num_patch_input": int(self.num_patch_input),
        "norm_const": passed_norm,
        "align_const": passed_align,
        "interpolation": passed_interp,
        "scale_x": float(self.scale_x),
        "input_resize_size": str(getattr(self.input_resize, "size", None)),
        "output_resize_size": str(getattr(self.output_resize, "size", None)),
        "mask_ratio": float(self.mask_ratio),
        "t": time.time()})


vm.einops = _EinopsProxy(vm.einops)
vm.VisionTS.__init__ = _patched_init
vm.VisionTS.update_config = _patched_update_config


def _flush():
    d = os.path.dirname(os.path.abspath(FLAGS.shape_log))
    os.makedirs(d, exist_ok=True)
    with open(FLAGS.shape_log, "w") as f:
        for r in RECORDS:
            f.write(json.dumps(r) + "\n")


try:
    runpy.run_path(FLAGS.run_py, run_name="__main__")
finally:
    _flush()
