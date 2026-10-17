"""Build 3 audit packs (<50MB each) with sha256 sidecars.

  code_protocol : package + visionts source, code_state.json (md5 manifest,
                  commit, weights location), frozen protocol (51 files)
  logs_metrics  : preflight/selected_p/psearch/status/verify/replay/
                  evaluation/summary + per-group metrics jsons
                  (NO prediction arrays)
  pred_npz      : 36 cached test npz + replay report

Named VisionTS_timemmd_common_<content>_<YYYYMMDD>.tar.gz, written to
--out_dir (default /home/wlt/MMTS/temp when present, else <out>/packs).
"""
import argparse
import datetime
import io
import tarfile
from pathlib import Path

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import dump_json, \
    setup_deterministic_env, sha256_of

LIMIT = 50 * 1024 * 1024


def build_pack(out_dir, tag, stage_dir, extra_files):
    date = datetime.date.today().strftime("%Y%m%d")
    name = "VisionTS_timemmd_common_%s_%s" % (tag, date)
    tpath = Path(out_dir) / (name + ".tar.gz")
    with tarfile.open(tpath, "w:gz") as tar:
        root = Path(stage_dir)
        for p in sorted(root.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(Path(name) / p.relative_to(root)))
        for f in extra_files:
            f = Path(f)
            if not f.exists():
                raise SystemExit("[FATAL] pack source missing: %s" % f)
            if f.is_dir():
                for p in sorted(f.rglob("*")):
                    if p.is_file():
                        tar.add(p, arcname=str(Path(name) /
                                               f.name /
                                               p.relative_to(f)))
            else:
                tar.add(f, arcname=str(Path(name) / f.name))
    size = tpath.stat().st_size
    if size > LIMIT:
        tpath.unlink()
        raise SystemExit("[FATAL] pack %s is %.1fMB > 50MB"
                         % (tpath.name, size / 2**20))
    sha = sha256_of(tpath)
    (Path(out_dir) / (tpath.name + ".sha256")).write_text(
        "%s  %s\n" % (sha, tpath.name))
    print("PACK %s  %.2fMB  sha256=%s" % (tpath.name, size / 2**20, sha))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())
    default_local = Path("/home/wlt/MMTS/temp")
    out_dir = Path(args.out_dir) if args.out_dir else (
        default_local if default_local.exists() else out_root / "packs")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"repo_commit": P.repo_commit(),
                "code_md5": P.package_code_md5s(),
                "weights": {"path": str(P.ckpt_dir() / P.WEIGHTS_FILENAME),
                            "bytes": P.WEIGHTS_BYTES,
                            "sha256": pinfo["weights_sha256"]},
                "protocol_dir": str(pinfo["protocol_dir"]),
                "protocol_name": P.PROTOCOL_NAME}
    dump_json(out_root / "code_state.json", manifest)

    stage = out_root / "packs_stage"
    code_stage = stage / "code"
    code_stage.mkdir(parents=True, exist_ok=True)
    for rel in manifest["code_md5"]:
        src = P.REPO_ROOT / rel
        dst = code_stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    (code_stage / "code_state.json").write_text(
        (out_root / "code_state.json").read_text())

    logs_stage = stage / "logs"
    if logs_stage.exists():
        import shutil
        shutil.rmtree(logs_stage)
    (logs_stage / "runs").mkdir(parents=True)
    for jf in sorted(out_root.glob("*.json")) + \
            sorted(out_root.glob("*.md")):
        (logs_stage / jf.name).write_text(jf.read_text())
    for gdir in sorted((out_root / "runs").iterdir()):
        if gdir.is_dir():
            dst = logs_stage / "runs" / gdir.name
            dst.mkdir(parents=True, exist_ok=True)
            for jf in sorted(gdir.glob("*.json")):
                (dst / jf.name).write_text(jf.read_text())

    build_pack(out_dir, "code_protocol", code_stage, [
        pinfo["protocol_dir"],
    ])
    build_pack(out_dir, "logs_metrics", logs_stage, [
        out_root / "preflight_report.json",
        out_root / "selected_p.json",
        out_root / "psearch" / "status.json",
        out_root / "status.json",
        out_root / "guards.json",
    ])
    build_pack(out_dir, "pred_npz", out_root / "runs", [
        out_root / "replay_report.json",
    ])
    print("AUDIT PACKS DONE -> %s" % out_dir)


if __name__ == "__main__":
    main()
