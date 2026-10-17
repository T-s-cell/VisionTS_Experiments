"""Frozen constants, pins, paths, and verification for EXP-028
(VisionTS x timemmd_common_801010_v1, LayerNorm full-shot fine-tune).

Data/protocol pins are identical to EXP-027 (repro_timemmd_common_v1).
This package adds: the fixed per-domain P (EXP-027 selected_p.json, sha
pinned), the LN training config, and the EXP-027 reference constants used
for the primary comparison. Any mismatch is a hard stop (tier-1).
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_NAME = "timemmd_common_801010_v1"

# ---- model / inference constants (inherited from EXP-027) ----
SEED = 2021
ARCH = "mae_base"
FINETUNE_TYPE = "none"          # models are built fully frozen; LN training
NORM_CONST = 0.4                # selectively flips requires_grad afterwards
ALIGN_CONST = 0.4
INTERPOLATION = "bilinear"
FP64 = False
BATCH = 256                     # test/val inference batch AND effective
ATOL = 1e-6                     # train batch (microbatch accumulation)
RTOL = 1e-5

# ---- LN training config (directive EXP-028) ----
MICROBATCH = 32
LN_LR = 1e-4
LN_BETAS = (0.9, 0.999)
LN_EPS = 1e-8
LN_WD = 0.0
EPOCHS = 1
EXPECTED_LN_TENSORS = 84        # 42 LayerNorms x (weight, bias)
EXPECTED_LN_SCALARS = 55808

# ---- pinned sha256 (identical to EXP-027) ----
PROTOCOL_JSON_SHA256 = ("a111948a37cf90c92abbd63a239e8f79"
                        "f54128b5fd04aaff1a52520e7fe9f69a")
SPLITS_JSON_SHA256 = ("a29f83a0ff4a81862cb3325608328af54"
                      "427e1661ea2ca249dbafd5daaa69f0c")
SCALER_JSON_SHA256 = ("cb11b98be8d76899f2bec6a02eb755d5"
                      "d4fb2736046113eda1ea977b7797ae1c")
WEIGHTS_SHA256 = ("c551c37b63a221a3e6d7d7f49e96f34078c9b"
                  "183f788bedc0387493521bade0e")
WEIGHTS_FILENAME = "mae_visualize_vit_base.pth"
WEIGHTS_BYTES = 447725789

# ---- EXP-027 frozen references (primary comparison + P source) ----
EXP027_SELECTED_P_SHA256 = ("5ea33ae9ffc6754eee8a1302968cd184f"
                            "d5eb2f1a9bbbdc01e9bfa8766843532")
EXP027_OVERALL_MSE = 1.6287341283427343
EXP027_OVERALL_MAE = 0.799104650815328

# ---- acceptance totals (directive EXP-028) ----
TEST_WINDOWS_TOTAL = 7388
VAL_WINDOWS_TOTAL = 7448
TRAIN_WINDOWS_TOTAL = 55836
TRAIN_STEPS_TOTAL = 236         # sum over groups of ceil(n_train / 256)

# ---- frozen domain table + fixed P (= EXP-027 selected, verified) ----
ORDER = ["Agriculture", "Climate", "Economy", "Energy", "Environment",
         "Health", "Security", "SocialGood", "Traffic"]
TABLE = {
    "Agriculture": dict(L=192, H=[6, 8, 10, 12], P=[1, 3, 6, 12]),
    "Climate": dict(L=192, H=[6, 8, 10, 12], P=[1, 3, 6, 12]),
    "Economy": dict(L=192, H=[6, 8, 10, 12], P=[1, 3, 6, 12]),
    "Energy": dict(L=1056, H=[12, 24, 36, 48], P=[1, 4, 52]),
    "Environment": dict(L=528, H=[48, 96, 192, 336], P=[1, 7, 30, 365]),
    "Health": dict(L=96, H=[12, 24, 36, 48], P=[1, 4, 52]),
    "Security": dict(L=220, H=[6, 8, 10, 12], P=[1, 3, 6, 12]),
    "SocialGood": dict(L=192, H=[6, 8, 10, 12], P=[1, 3, 6, 12]),
    "Traffic": dict(L=96, H=[6, 8, 10, 12], P=[1, 3, 6, 12]),
}
FIXED_P = {"Agriculture": 6, "Climate": 3, "Economy": 12, "Energy": 4,
           "Environment": 7, "Health": 1, "Security": 6, "SocialGood": 3,
           "Traffic": 12}
PRECHECK = [("Energy", 48), ("Environment", 336), ("Health", 48),
            ("Security", 12)]

# TaTS reference overall (secondary comparison anchor, EXP-026)
TATS_OVERALL_MSE = 1.6709424058596294
TATS_OVERALL_MAE = 0.8187253458632363


def _env(name, default):
    return os.environ.get(name, default)


def protocol_dir():
    return Path(_env(
        "VTS_TIMEMMD_PROTOCOL",
        "/dev_data/wlt/VisionTS_Experiments/frozen_protocol/"
        + PROTOCOL_NAME))


def out_root():
    return Path(_env(
        "VTS_TIMEMMD_LN_OUT",
        "/dev_data/wlt/VisionTS_logs/timemmd_common_v1_ln_fullshot"))


def exp027_out():
    return Path(_env(
        "VTS_TIMEMMD_EXP027_OUT",
        "/dev_data/wlt/VisionTS_logs/timemmd_common_v1_zeroshot"))


def tats_ref_dir():
    return Path(_env(
        "VTS_TIMEMMD_TATS_REF",
        "/dev_data/wlt/VisionTS_Experiments/tats_ref"))


def ckpt_dir():
    return Path(_env("VTS_TIMEMMD_CKPT_DIR", str(REPO_ROOT / "ckpt")))


def test_npz_name(domain, h):
    return "test__%s__H%d.npz" % (domain, h)


def replay_npz_name(domain, h):
    return "replay__%s__H%d.npz" % (domain, h)


def val_npz_name(domain, h):
    return "val__%s__H%d.npz" % (domain, h)


def psearch_val_npz_name(domain, p, h):
    """EXP-027 psearch array naming (3-arg; ours above is 2-arg)."""
    return "val__%s__P%d__H%d.npz" % (domain, p, h)


def ckpt_npz_name(domain, h):
    return "ln_ckpt__%s__H%d.npz" % (domain, h)


def ckpt_meta_name(domain, h):
    return "ln_ckpt_meta__%s__H%d.json" % (domain, h)


def train_log_name(domain, h):
    return "train_log__%s__H%d.json" % (domain, h)


def group_id(domain, h):
    return "%s_H%d" % (domain, h)


def all_groups():
    return [(dom, h) for dom in ORDER for h in TABLE[dom]["H"]]


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_steps(n_train, effective_batch=BATCH):
    return -(-int(n_train) // int(effective_batch))


def verify_frozen(pdir=None):
    """Tier-1 gate: protocol pins, all 51 protocol files, frozen table vs
    splits.json, acceptance totals (train/val/test + 236 steps), MAE
    weights, AND the EXP-027 selected_p.json (sha pin + per-domain P
    equality with FIXED_P). Raises SystemExit on ANY mismatch."""
    from repro_timemmd_common_ln_v1.common import die
    pdir = Path(pdir) if pdir else protocol_dir()

    for name, pin in [("protocol.json", PROTOCOL_JSON_SHA256),
                      ("splits.json", SPLITS_JSON_SHA256),
                      ("scaler.json", SCALER_JSON_SHA256)]:
        got = sha256_file(pdir / name)
        if got != pin:
            die("[TIER-1] %s sha mismatch: got %s want %s"
                % (name, got, pin))

    proto = json.load(open(pdir / "protocol.json"))
    splits = json.load(open(pdir / "splits.json"))
    files = proto["files_sha256"]
    if len(files) != 51:
        die("[TIER-1] protocol files_sha256 count %d != 51" % len(files))
    for rel, want in sorted(files.items()):
        got = sha256_file(pdir / rel)
        if got != want:
            die("[TIER-1] protocol file drift: %s" % rel)

    if set(splits.keys()) != set(ORDER):
        die("[TIER-1] splits domains mismatch: %s" % sorted(splits.keys()))
    for dom in ORDER:
        t, s = TABLE[dom], splits[dom]
        if t["L"] != s["L"] or list(t["H"]) != list(s["H"]):
            die("[TIER-1] table vs splits mismatch for %s: L %s/%s H %s/%s"
                % (dom, t["L"], s["L"], t["H"], s["H"]))

    train_total = val_total = test_total = steps_total = 0
    for d in ORDER:
        for h in TABLE[d]["H"]:
            wc = splits[d]["window_counts"][str(h)]
            train_total += int(wc["train"])
            val_total += int(wc["val"])
            test_total += int(wc["test"])
            steps_total += expected_steps(wc["train"])
    if val_total != VAL_WINDOWS_TOTAL:
        die("[TIER-1] val windows %d != %d" % (val_total, VAL_WINDOWS_TOTAL))
    if test_total != TEST_WINDOWS_TOTAL:
        die("[TIER-1] test windows %d != %d"
            % (test_total, TEST_WINDOWS_TOTAL))
    if train_total != TRAIN_WINDOWS_TOTAL:
        die("[TIER-1] train windows %d != %d"
            % (train_total, TRAIN_WINDOWS_TOTAL))
    if steps_total != TRAIN_STEPS_TOTAL:
        die("[TIER-1] train optimizer steps %d != %d"
            % (steps_total, TRAIN_STEPS_TOTAL))

    wpath = ckpt_dir() / WEIGHTS_FILENAME
    if not wpath.exists():
        die("[TIER-1] MAE weights missing: %s" % wpath)
    if wpath.stat().st_size != WEIGHTS_BYTES:
        die("[TIER-1] MAE weights size %d != %d"
            % (wpath.stat().st_size, WEIGHTS_BYTES))
    wsha = sha256_file(wpath)
    if wsha != WEIGHTS_SHA256:
        die("[TIER-1] MAE weights sha mismatch: %s" % wsha)

    sel_path = exp027_out() / "selected_p.json"
    if not sel_path.exists():
        die("[TIER-1] EXP-027 selected_p.json missing: %s" % sel_path)
    sel_sha = sha256_file(sel_path)
    if sel_sha != EXP027_SELECTED_P_SHA256:
        die("[TIER-1] selected_p.json sha mismatch: got %s want %s"
            % (sel_sha, EXP027_SELECTED_P_SHA256))
    sel = json.load(open(sel_path))["selection"]
    for dom in ORDER:
        if int(sel[dom]["selected"]) != FIXED_P[dom]:
            die("[TIER-1] fixed P drift for %s: selected_p says %s, "
                "package pins %d"
                % (dom, sel[dom]["selected"], FIXED_P[dom]))

    return {"protocol": proto, "splits": splits,
            "protocol_dir": str(pdir), "weights_sha256": wsha,
            "selected_p_path": str(sel_path), "selected_p_sha256": sel_sha,
            "train_windows": train_total, "val_windows": val_total,
            "test_windows": test_total, "steps_total": steps_total}


def package_code_md5s():
    """md5 of every .py in this package + visionts/ (code guard)."""
    import hashlib
    out = {}
    for base in [Path(__file__).resolve().parent, REPO_ROOT / "visionts"]:
        for p in sorted(base.glob("*.py")):
            h = hashlib.md5()
            h.update(p.read_bytes())
            out[str(p.relative_to(REPO_ROOT))] = h.hexdigest()
    return out


def repo_commit():
    txt = REPO_ROOT / "COMMIT.txt"
    if txt.exists():
        return txt.read_text().strip()
    import subprocess
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True,
        text=True).stdout.strip()


if __name__ == "__main__":
    try:
        info = verify_frozen()
    except SystemExit as e:
        print(e)
        sys.exit(1)
    print("FROZEN OK: 51 protocol files, train=%d val=%d test=%d "
          "steps=%d, weights %s..., selected_p pinned"
          % (info["train_windows"], info["val_windows"],
             info["test_windows"], info["steps_total"],
             info["weights_sha256"][:12]))
