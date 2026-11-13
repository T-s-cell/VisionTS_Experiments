"""Shared constants/helpers for EXP-045: VisionTS MAE-Base / Large / Huge
zero-shot comparison on the ETT4 datasets — paper Table 2 / Table 17
(Appendix B.7, paper_full.md L448-450: "MAE (Base) 112M / (Large) 330M /
(Huge) 657M").  Same frozen protocol as EXP-001 (r=c=0.4, L=table 8,
P=table 6, bilinear, original orientation, features=M, bs=32, seed=2021,
train_epochs=0, no inverse); ONLY --vm_arch changes.  Base 16 groups are
EXP-001 archives (read-only reuse); 32 new groups (Large 16 + Huge 16).

Review corrections (2026-11-13, user):
  #1 acceptance ORDER: reuse gate -> smoke -> main queue (per-group
     acceptance) -> timing -> collect -> MG11 -> pack (MG11 consumes the
     collect means json and the timing json, so it runs LAST);
  #2 single-purpose dirs: save_t45_base_replay/ (2 default replays only),
     save_t45_large/, save_t45_huge/ -- there is NO save_t45_base/ (the
     16 real Base groups are EXP-001 reuse); per-group log names embed
     ds/H/arch;
  #3 geometry checks the VISIBLE WIDTH (= patch_size * npi, element 2 of
     the resize size tuple; base/large 16*npi, huge 14*npi), the patch
     grid (num_patch x num_patch) and the visible column count -- NOT a
     fixed-height-224 divisibility claim;
  #4 checkpoint provenance: downloads use NORMAL HTTPS with a real CA
     chain (downloaded off-box, then scp'd); no insecure fallback; every
     file gated by sha256+bytes here and by strict state-dict load +
     parameter count in preflight;
  #5 NO six-dataset paper overall numbers are quoted anywhere (bottom-row
     OCR risk: 0.34617 vs Table 2 Huge overall ~0.315 inconsistent);
     only the verified ETT cells of Table 17 are cited; the ETT-only
     overall is self-computed;
  #6 timing protocol: torch.inference_mode() + model.eval(), the three
     archs run SEQUENTIALLY with the model freed (del + empty_cache)
     between archs -- no grad tracking, no cross-model memory residue.

Model source, framework files and EXP-001 archives are READ-ONLY.
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # long_term_tsf/
REPO = os.path.dirname(ROOT)
PKG = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PKG, "logs")

SAVE_REPLAY = os.path.join(ROOT, "save_t45_base_replay")  # 2 default replays ONLY
SAVE_LARGE = os.path.join(ROOT, "save_t45_large")         # 16 new Large groups
SAVE_HUGE = os.path.join(ROOT, "save_t45_huge")           # 16 new Huge groups

EXP001_BASE = os.path.join(ROOT, "save_repro_zeroshot")   # Base 16, reuse read-only
EXP030_BASE = os.path.join(ROOT, "save_a2p1_zeroshot")    # lineage md5 reference

CKPT_DIR_REL = "../ckpt/"
MAE_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"

ARCHS = ["mae_base", "mae_large", "mae_huge"]
ARCH_SHORT = {"mae_base": "base", "mae_large": "large", "mae_huge": "huge"}
ARCH_SHORT_INV = {v: k for k, v in ARCH_SHORT.items()}

# builder/ckpt file/patch size/patch grid/paper parameter count
# (Table 17 header, paper_full.md L448-450)
ARCH_FACTS = {
    "mae_base": dict(builder="mae_vit_base_patch16", ckpt="mae_visualize_vit_base.pth",
                     patch_size=16, num_patch=14, paper_params_m=112),
    "mae_large": dict(builder="mae_vit_large_patch16", ckpt="mae_visualize_vit_large.pth",
                      patch_size=16, num_patch=14, paper_params_m=330),
    "mae_huge": dict(builder="mae_vit_huge_patch14", ckpt="mae_visualize_vit_huge.pth",
                     patch_size=14, num_patch=16, paper_params_m=657),
}

# provenance: sha256+bytes measured off-box on the downloaded files, then
# scp'd to eta and gated here at preflight (correction #4: sha/bytes bind
# the scp'd copy to the CA-verified download; strict load + parameter
# count bind the CONTENT to the official MaskedAutoencoderViT structure)
CKPTS = {
    "mae_base": dict(sha256="c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e",
                     bytes=447725789),
    "mae_large": dict(sha256="d0072183ec21f3b3e50874a957254ea4d03c2d7255d0ad47aebb85a868a6bdb6",
                      bytes=1318315181),
    "mae_huge": dict(sha256="49ca3f48bfde7cd51feb4f6e63fd1b859e5bc11824730426a124bcda301f037e",
                     bytes=2628481101),
}
CKPT_FILE = {a: os.path.join(REPO, "ckpt", ARCH_FACTS[a]["ckpt"]) for a in ARCHS}

HORIZONS = [96, 192, 336, 720]
ETT4 = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

# paper Table 8 (paper_full.md L374-378) / Table 6 periodicity, frozen
PAPER_L = {"ETTh1": 2880, "ETTh2": 1728, "ETTm1": 2304, "ETTm2": 4032}
PAPER_P = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96}

# EXP-018 pinned Base overall anchor (bilinear, ETT-only, 4H equal weight
# per ds, then 4-ds equal weight over 16 groups) -- hard gate MG7/MG11.
BASE_OVERALL = {"overall_mse": 0.34439535439014435,
                "overall_mae": 0.37042006477713585}
SELF_TOL = 1e-6

# paper Table 17 ETT cells (paper_full.md L448-450; 3-decimal anchors).
# ONLY these cells are cited (correction #5: no six-dataset overall row --
# its bottom numbers carry OCR risk and are NOT used anywhere).
PAPER_T17_NOTE = ("paper Table 17, Appendix B.7 (paper_full.md L448-450); "
                  "3-decimal anchors, ETT cells only; per-ds 'avg' row is "
                  "the paper's own 4H equal-weight mean (same convention as "
                  "ours).  Six-dataset overall numbers from the table's "
                  "bottom row are deliberately NOT quoted (correction #5: "
                  "OCR-misalignment risk, e.g. 0.34617 vs Table 2 Huge "
                  "overall ~0.315); our ETT overall is self-computed.")
PAPER_T17 = {
    "ETTh1": {96:  {"mae_base": (0.353, 0.383), "mae_large": (0.346, 0.382), "mae_huge": (0.362, 0.384)},
              192: {"mae_base": (0.392, 0.410), "mae_large": (0.379, 0.406), "mae_huge": (0.407, 0.414)},
              336: {"mae_base": (0.407, 0.423), "mae_large": (0.391, 0.416), "mae_huge": (0.399, 0.419)},
              720: {"mae_base": (0.406, 0.441), "mae_large": (0.397, 0.433), "mae_huge": (0.395, 0.433)}},
    "ETTh2": {96:  {"mae_base": (0.271, 0.328), "mae_large": (0.286, 0.334), "mae_huge": (0.285, 0.333)},
              192: {"mae_base": (0.328, 0.367), "mae_large": (0.346, 0.375), "mae_huge": (0.337, 0.369)},
              336: {"mae_base": (0.345, 0.381), "mae_large": (0.356, 0.387), "mae_huge": (0.357, 0.388)},
              720: {"mae_base": (0.388, 0.422), "mae_large": (0.371, 0.409), "mae_huge": (0.379, 0.412)}},
    "ETTm1": {96:  {"mae_base": (0.341, 0.347), "mae_large": (0.344, 0.349), "mae_huge": (0.352, 0.351)},
              192: {"mae_base": (0.360, 0.360), "mae_large": (0.365, 0.363), "mae_huge": (0.360, 0.367)},
              336: {"mae_base": (0.377, 0.374), "mae_large": (0.381, 0.376), "mae_huge": (0.381, 0.383)},
              720: {"mae_base": (0.416, 0.405), "mae_large": (0.429, 0.411), "mae_huge": (0.440, 0.412)}},
    "ETTm2": {96:  {"mae_base": (0.228, 0.282), "mae_large": (0.225, 0.282), "mae_huge": (0.229, 0.282)},
              192: {"mae_base": (0.262, 0.305), "mae_large": (0.262, 0.305), "mae_huge": (0.265, 0.306)},
              336: {"mae_base": (0.293, 0.328), "mae_large": (0.299, 0.331), "mae_huge": (0.286, 0.324)},
              720: {"mae_base": (0.343, 0.370), "mae_large": (0.358, 0.377), "mae_huge": (0.355, 0.374)}},
}
# paper's own per-ds avg row (4H equal weight; reference only, recomputed
# from the cells above would give the same rounded values)
PAPER_T17_DS_AVG = {
    "ETTh1": {"mae_base": (0.390, 0.414), "mae_large": (0.378, 0.409), "mae_huge": (0.391, 0.412)},
    "ETTh2": {"mae_base": (0.333, 0.375), "mae_large": (0.340, 0.377), "mae_huge": (0.339, 0.375)},
    "ETTm1": {"mae_base": (0.374, 0.372), "mae_large": (0.379, 0.375), "mae_huge": (0.383, 0.378)},
    "ETTm2": {"mae_base": (0.282, 0.321), "mae_large": (0.286, 0.324), "mae_huge": (0.284, 0.322)},
}

DATASETS = {
    "ETTh1": dict(ctx=2880, period=24, root_path="./dataset/ETT-small/",
                  data_path="ETTh1.csv", data="ETTh1", minute=False),
    "ETTh2": dict(ctx=1728, period=24, root_path="./dataset/ETT-small/",
                  data_path="ETTh2.csv", data="ETTh2", minute=False),
    "ETTm1": dict(ctx=2304, period=96, root_path="./dataset/ETT-small/",
                  data_path="ETTm1.csv", data="ETTm1", minute=True),
    "ETTm2": dict(ctx=4032, period=96, root_path="./dataset/ETT-small/",
                  data_path="ETTm2.csv", data="ETTm2", minute=True),
}

FP_FILES = [
    "run.py", "exp/exp_long_term_forecasting.py", "exp/exp_basic.py",
    "models/VisionTS.py", "data_provider/data_factory.py",
    "data_provider/data_loader.py",
    "../visionts/model.py", "../visionts/models_mae.py",
]
# t45 files recorded in every run fingerprint (paths relative to ROOT)
T45_FP_FILES = ["repro_t45_mae_scale/infer_one_t45.py",
                "repro_t45_mae_scale/t45_common.py"]


# ---- split definition (identical to t43/t44) ----
def month_rows(ds):
    """K: rows per 30-day month block.  hourly K=720, 15T-minutely K=2880."""
    return 30 * 24 * (4 if DATASETS[ds]["minute"] else 1)


def split_rows(ds):
    """Absolute CSV-row boundaries: train [0,12K); val [12K,16K);
    TEST TARGET = [16K,20K)."""
    K = month_rows(ds)
    return {"train_end": 12 * K, "val_start": 12 * K, "val_end": 16 * K,
            "test_start": 16 * K, "test_end": 20 * K}


def expected_n_windows(ds, h):
    # test split len(data_x) = 4K + L -> N = 4K - h + 1 (L- and arch-independent)
    return 4 * month_rows(ds) - h + 1


# ---- geometry, formula-driven (correction #3) ----
def expected_geometry(ds, h, P, ctx=None, arch="mae_base"):
    """Replicates visionts/model.py VisionTS.update_config verbatim for the
    given arch.  The full 224x224 image has a num_patch x num_patch patch
    grid; the FIRST npi patch COLUMNS are visible.  input_resize size tuple
    is (224, patch_size*npi): element 1 is the image height, element 2 is
    the VISIBLE WIDTH in pixels (= patch_size*npi exactly, since
    224/num_patch == patch_size is integral for every arch)."""
    if ctx is None:
        ctx = PAPER_L[ds]
    fact = ARCH_FACTS[arch]
    patch = fact["patch_size"]
    num_patch = fact["num_patch"]
    pred = h
    pad_left = (-ctx) % P
    pad_right = (-pred) % P
    align_const = 0.4
    input_ratio = (pad_left + ctx) / (pad_left + ctx + pad_right + pred)
    npi_raw = int(input_ratio * num_patch * align_const)
    npi = npi_raw if npi_raw >= 1 else 1
    npo = num_patch - npi
    adjust = npi / num_patch
    image_size = 224
    width_vis = patch * npi          # visible WIDTH (correction #3)
    scale_x = ((pad_left + ctx) // P) / int(image_size * adjust)
    out_w = int(round(image_size * scale_x))
    return {
        "arch": arch, "patch_size": patch, "num_patch": num_patch,
        "periodicity": P, "context_len": ctx, "pred_len": pred,
        "pad_left": pad_left, "pad_right": pad_right,
        "num_patch_input": npi, "npi_raw": npi_raw,
        "num_patch_output": npo,
        "mask_ratio": 1.0 - npi / num_patch,
        "input_resize_size": (image_size, width_vis),
        "input_width_vis": width_vis,
        "input_width_div_patch": width_vis % patch,   # must be 0
        "scale_x": scale_x,
        "output_resize_size": (P, out_w),
    }


def geometry_table():
    """Static expected geometry for every (ds, H) x every arch (48 rows)."""
    rows = []
    for arch in ARCHS:
        for ds in ETT4:
            P = PAPER_P[ds]
            for h in HORIZONS:
                g = expected_geometry(ds, h, P, PAPER_L[ds], arch)
                g.update({"ds": ds, "H": h})
                rows.append(g)
    return rows


# static asserts (correction #3): visible width is patch-aligned for every
# arch; the int() formula itself yields npi>=1 everywhere (the official
# max(..,1) clamp is never triggered by the frozen L/H grid)
for _g in geometry_table():
    assert _g["npi_raw"] >= 1, "clamp would trigger: %s" % _g
    assert _g["context_len"] <= 16 * month_rows(_g["ds"]), _g
    assert _g["pad_left"] == 0, _g
    assert _g["input_width_vis"] % _g["patch_size"] == 0, _g
    assert _g["input_width_vis"] == _g["input_resize_size"][1], _g
    assert _g["input_resize_size"][0] == 224, _g
GEOMETRY_OK_STATIC = True

GEOMETRY_TABLE_CSV = os.path.join(LOG_DIR, "geometry_table_t45.csv")


# ---- deterministic truth recomputation (t43/t44 pattern, test split) ----
def std_array(ds):
    """f64 read -> StandardScaler fit on rows [0,12K) -> transform all ->
    f32 cast.  Mirrors Dataset_ETT_{hour,minute}.__read_data__ exactly
    (features='M': columns 1:, scaler on train block).  Independent of
    L and arch."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    c = DATASETS[ds]
    df_raw = pd.read_csv(os.path.join(ROOT, c["root_path"], c["data_path"]))
    cols_data = df_raw[df_raw.columns[1:]]
    sp = split_rows(ds)
    scaler = StandardScaler()
    scaler.fit(cols_data.values[: sp["train_end"]])
    data = scaler.transform(cols_data.values)
    return data.astype(np.float32)


def truth_from_std(ds, h):
    """Test targets: sliding H-step windows of the standardized data ->
    (N, H, 7) float32, window i covering absolute rows [16K+i, 16K+i+H).
    IDENTICAL for every L and every arch."""
    std = std_array(ds)
    sp = split_rows(ds)
    start, end = sp["test_start"], sp["test_end"]
    n = end - start - h + 1
    out = np.empty((n, h, std.shape[1]), dtype=np.float32)
    for i in range(n):
        out[i] = std[start + i: start + i + h]
    return out


# ---- hashes / json ----
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def framework_md5_dict():
    return {f: md5_of(os.path.join(ROOT, f)) for f in FP_FILES}


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
        f.write("\n")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def read_fingerprint(path):
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def write_text(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---- paths (correction #2: dirs single-purpose; logs embed ds/H/arch) ----
def save_parent(arch):
    return {"mae_base": SAVE_REPLAY,      # base runs in THIS round are
            "mae_large": SAVE_LARGE,      # replays only; the 16 real Base
            "mae_huge": SAVE_HUGE}[arch]  # groups are EXP-001 reuse


def run_dir(ds, h, arch):
    return os.path.join(save_parent(arch), "%s_%d" % (ds, h))


def log_path(ds, h, arch):
    return os.path.join(LOG_DIR, "t45_%s_%s_%d.log" % (ARCH_SHORT[arch], ds, h))


def base_dir(ds, h):
    return os.path.join(EXP001_BASE, "%s_%d" % (ds, h))


def p1_dir(ds, h):
    return os.path.join(EXP030_BASE, "%s_%d" % (ds, h))


def group_ids():
    return [(ds, h) for ds in ETT4 for h in HORIZONS]


# ---- fingerprints ----
def config_lines(ds, h, arch):
    c = DATASETS[ds]
    return [
        "dataset=%s pred_len=%d seq_len=%d periodicity=%d arch=%s"
        % (ds, h, PAPER_L[ds], PAPER_P[ds], arch),
        "norm_const=0.4 align_const=0.4 batch_size=32 train_epochs=0 seed=2021",
        "root_path=%s data_path=%s data=%s features=M"
        % (c["root_path"], c["data_path"], c["data"]),
        "ckpt_sha256=" + CKPTS[arch]["sha256"],
    ]


def fingerprint_lines(ds, h, arch, extra_files=()):
    lines = list(config_lines(ds, h, arch))
    for f in FP_FILES:
        lines.append("md5:%s=%s" % (f, md5_of(os.path.join(ROOT, f))))
    for f in list(extra_files) + T45_FP_FILES:
        lines.append("md5:%s=%s" % (f, md5_of(os.path.join(ROOT, f))))
    return lines


# ---- metric (exact utils.metrics import) ----
def load_metric():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from utils.metrics import metric
    return metric


def parse_log_metrics(txt):
    import re
    m = re.search(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", txt)
    if not m:
        return None
    return {"mse": float(m.group(1)), "mae": float(m.group(2))}


def parse_resize_ints(s):
    """'(224, 80)' / '[224, 80]' -> (224, 80); robust to torchvision str()."""
    import re
    nums = re.findall(r"\d+", str(s))
    return tuple(int(x) for x in nums)


def geometry_matches(observed, expected):
    """observed: dict as recorded by the update_config wrapper (resize sizes
    as strings); expected: expected_geometry dict.  Returns (ok, notes).
    Checks (correction #3): patch grid + visible columns + the visible
    WIDTH tuple, not a fixed-height divisibility."""
    notes = []
    ok = True
    for k in ("arch", "patch_size", "num_patch", "periodicity", "context_len",
              "pred_len", "pad_left", "pad_right", "num_patch_input"):
        o, e = observed.get(k), expected.get(k)
        if o != e:
            ok = False
            notes.append("%s observed=%s expected=%s" % (k, o, e))
    for k in ("scale_x", "mask_ratio"):
        o, e = observed.get(k), expected[k]
        if o is None or abs(float(o) - float(e)) > 1e-6:
            ok = False
            notes.append("%s observed=%s expected=%s" % (k, o, e))
    o_in = parse_resize_ints(observed.get("input_resize_size"))
    if o_in != tuple(expected["input_resize_size"]):
        ok = False
        notes.append("input_resize_size observed=%s expected=%s (elem2 must "
                     "be the visible width patch*npi)" % (o_in,
                                                          expected["input_resize_size"]))
    elif o_in[1] % expected["patch_size"] != 0:
        ok = False
        notes.append("visible width %d not patch-aligned" % o_in[1])
    o_out = parse_resize_ints(observed.get("output_resize_size"))
    if o_out != tuple(expected["output_resize_size"]):
        ok = False
        notes.append("output_resize_size observed=%s expected=%s"
                     % (o_out, expected["output_resize_size"]))
    if "npi_raw" in observed and observed["npi_raw"] != expected["npi_raw"]:
        ok = False
        notes.append("npi_raw observed=%s expected=%s"
                     % (observed["npi_raw"], expected["npi_raw"]))
    return ok, notes


GEOMETRY_KEYS = ("arch", "patch_size", "num_patch", "periodicity",
                 "context_len", "pred_len", "pad_left", "pad_right",
                 "num_patch_input", "scale_x", "input_resize_size",
                 "output_resize_size", "mask_ratio")


def geometry_subset(rec):
    """Geometry-only projection of an update_config record (drops
    norm_const/align_const/t so records from different archs can be
    compared where the formula predicts identity)."""
    return {k: rec[k] for k in GEOMETRY_KEYS if k in rec}
