"""Frozen constants, pins, paths, and verification for this campaign.

Everything here is fixed by the directive + EXP-026 frozen protocol
timemmd_common_801010_v1. Any mismatch is a hard stop (tier-1).
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_NAME = "timemmd_common_801010_v1"

# ---- model / run constants (directive) ----
SEED = 2021
ARCH = "mae_base"
FINETUNE_TYPE = "none"          # fully frozen: requires_grad=False everywhere
NORM_CONST = 0.4
ALIGN_CONST = 0.4
INTERPOLATION = "bilinear"
FP64 = False
BATCH = 256
ATOL = 1e-6
RTOL = 1e-5

# ---- pinned sha256 ----
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

# ---- acceptance totals (directive) ----
TEST_WINDOWS_TOTAL = 7388
VAL_WINDOWS_TOTAL = 7448

# ---- frozen domain table (directive section 2) ----
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
# precheck groups (largest P, longest L, non-divisible periods, tail batch)
PRECHECK = [("Energy", 48, 52), ("Environment", 336, 365),
            ("Health", 48, 52), ("Security", 12, 12)]

# TaTS reference overall (36-group equal weight, standardized space) —
# recomputed comparison anchor, asserted during evaluation
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
        "VTS_TIMEMMD_OUT",
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


def val_npz_name(domain, p, h):
    return "val__%s__P%d__H%d.npz" % (domain, p, h)


def group_id(domain, h):
    return "%s_H%d" % (domain, h)


def all_groups():
    return [(dom, h) for dom in ORDER for h in TABLE[dom]["H"]]


def all_psearch_combos():
    return [(dom, p, h) for dom in ORDER for p in TABLE[dom]["P"]
            for h in TABLE[dom]["H"]]


def combo_id(domain, p, h):
    return "%s_P%d_H%d" % (domain, p, h)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen(pdir=None):
    """Tier-1 gate: verify protocol pins, all 51 protocol files, the frozen
    table vs splits.json, acceptance totals, and MAE weights. Returns
    dict(protocol, splits). Raises SystemExit on ANY mismatch."""
    from repro_timemmd_common_v1.common import die
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

    val_total = sum(int(splits[d]["window_counts"][str(h)]["val"])
                    for d in ORDER for h in TABLE[d]["H"])
    test_total = sum(int(splits[d]["window_counts"][str(h)]["test"])
                     for d in ORDER for h in TABLE[d]["H"])
    if val_total != VAL_WINDOWS_TOTAL:
        die("[TIER-1] val windows %d != %d" % (val_total, VAL_WINDOWS_TOTAL))
    if test_total != TEST_WINDOWS_TOTAL:
        die("[TIER-1] test windows %d != %d"
            % (test_total, TEST_WINDOWS_TOTAL))

    wpath = ckpt_dir() / WEIGHTS_FILENAME
    if not wpath.exists():
        die("[TIER-1] MAE weights missing: %s" % wpath)
    if wpath.stat().st_size != WEIGHTS_BYTES:
        die("[TIER-1] MAE weights size %d != %d"
            % (wpath.stat().st_size, WEIGHTS_BYTES))
    wsha = sha256_file(wpath)
    if wsha != WEIGHTS_SHA256:
        die("[TIER-1] MAE weights sha mismatch: %s" % wsha)

    return {"protocol": proto, "splits": splits,
            "protocol_dir": str(pdir), "weights_sha256": wsha,
            "val_windows": val_total, "test_windows": test_total}


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
    print("FROZEN OK: %d protocol files, val=%d test=%d, weights %s..."
          % (51, info["val_windows"], info["test_windows"],
             info["weights_sha256"][:12]))
