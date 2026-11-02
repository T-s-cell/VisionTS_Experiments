"""Launch EXP-001's run.py UNCHANGED with --periodicity 1 while recording
evidence from the real inference path:
  1. every update_config call (pad/num_patch_input/scale_x geometry)
  2. every pre-interpolation 2D segmentation rearrange
     pattern 'b n (p f) -> (b n) 1 f p'  -> P=1 gives (B*nvars, 1, 1, ctx)
All framework files are untouched; we only observe via a module-level
einops proxy and a class-level update_config wrapper in visionts.model.
Usage:
  python infer_one_p1.py --shape_log <path> -- <run.py args...>
(the leading run.py argv is passed through verbatim)
"""
import argparse
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


_orig_uc = vm.VisionTS.update_config


def _patched_update_config(self, *a, **k):
    _orig_uc(self, *a, **k)
    RECORDS.append({
        "type": "update_config",
        "periodicity": int(self.periodicity),
        "context_len": int(self.context_len),
        "pred_len": int(self.pred_len),
        "pad_left": int(self.pad_left), "pad_right": int(self.pad_right),
        "num_patch_input": int(self.num_patch_input),
        "scale_x": float(self.scale_x),
        "input_resize_size": str(getattr(self.input_resize, "size", None)),
        "output_resize_size": str(getattr(self.output_resize, "size", None)),
        "mask_ratio": float(self.mask_ratio),
        "t": time.time()})


vm.einops = _EinopsProxy(vm.einops)
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
