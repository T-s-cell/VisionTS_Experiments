#!/usr/bin/env python3
"""Environment-drift report (the deliverable "环境差异清单").

Compares:
  OLD env : EXP-016 execution environment, from the archived
            env_ref/zeta_visionts_pip_freeze.txt + visionts_ref/logs/
            infer_summary.json (python/torch/gpu recorded at run time)
  NEW env : the CURRENT python environment (pip freeze via
            sys.executable -m pip), current GPU/driver.

Outputs results/env_report.json with:
  - python / torch / torchvision / numpy / gpu / driver lines;
  - per-package diff table over BOTH pip freezes (only differing-or-new
    packages listed in `diff`, key packages always in `key_packages`);
  - pandas_fallback: recorded if the EXP-016 pandas pin could not be used;
  - pandas_csv_parse_check: pandas read_csv(Idx/Date/Hist/Pred) vs the
    stdlib json/csv loader (load_windows) on all three CSVs — identity
    sets equal and Hist/Pred arrays bitwise equal as float64.

No GPU work. Exits nonzero if the pandas parse check fails."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (HERE, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C
import protocol as P

KEY_PACKAGES = ("numpy", "torch", "torchvision", "timm", "pandas",
                "pillow", "einops", "scipy", "scikit-learn", "transformers",
                "safetensors", "huggingface-hub")


def parse_freeze(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        if "==" in line:
            name, ver = line.split("==", 1)
            out[name.strip().lower()] = ver.strip()
        elif " @ " in line:
            name = line.split(" @ ", 1)[0].strip().lower()
            out[name] = line.split(" @ ", 1)[1].strip()
    return out


def gpu_info():
    import torch
    g = {"cuda_available": torch.cuda.is_available()}
    if g["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        g["gpu_name"] = p.name
        g["compute_capability"] = f"{p.major}.{p.minor}"
        g["total_mem_gib"] = round(p.total_memory / (1 << 30), 1)
    for nvidia_sm in ("/usr/lib/wsl/lib/nvidia-smi",
                      "/usr/bin/nvidia-smi"):
        if os.path.isfile(nvidia_sm):
            try:
                out = subprocess.run(
                    [nvidia_sm, "--query-gpu=driver_version",
                     "--format=csv,noheader"], capture_output=True,
                    text=True, timeout=30)
                g["driver_version"] = out.stdout.strip().splitlines()[0]
            except Exception:
                pass
            break
    return g


def pandas_parse_check():
    import pandas as pd
    rep = {"pandas_version": pd.__version__, "subsets": {}}
    from load_cgtsf import load_windows
    ok = True
    for ds in P.ORDER:
        wins = load_windows(ds)   # stdlib loader, includes full data gates
        df = pd.read_csv(P.csv_path(ds))
        for col in ("Idx", "Date", "Hist", "Pred"):
            if col not in df.columns:
                sys.exit(f"pandas check {ds}: missing column {col}")
        if len(df) != len(wins):
            sys.exit(f"pandas check {ds}: rows {len(df)} != {len(wins)}")
        n_bad_id = n_bad_val = 0
        for i, w in enumerate(wins):
            ix_p = df.at[i, "Idx"]
            dt_p = df.at[i, "Date"]
            if int(ix_p) != w["idx"] or str(dt_p) != w["date"]:
                n_bad_id += 1
                continue
            h = json.loads(df.at[i, "Hist"])
            t = json.loads(df.at[i, "Pred"])
            if not (len(h) == len(w["hist"])
                    and len(t) == len(w["target"])):
                n_bad_val += 1
                continue
            import numpy as np
            if not (np.array_equal(np.asarray(h, dtype=np.float64),
                                   w["hist"])
                    and np.array_equal(np.asarray(t, dtype=np.float64),
                                       w["target"])):
                n_bad_val += 1
        ds_ok = n_bad_id == 0 and n_bad_val == 0
        ok &= ds_ok
        rep["subsets"][ds] = {"rows": len(df),
                              "identity_mismatches": n_bad_id,
                              "value_mismatches": n_bad_val,
                              "ok": bool(ds_ok)}
        print(f"[env-report] pandas parse {ds}: id_bad={n_bad_id} "
              f"val_bad={n_bad_val} -> {'OK' if ds_ok else 'FAIL'}")
    rep["ok"] = bool(ok)
    return rep


def main():
    zeta_freeze = parse_freeze(open(os.path.join(
        HERE, "env_ref", "zeta_visionts_pip_freeze.txt")).read())
    cur = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True, check=True)
    cur_freeze = parse_freeze(cur.stdout)
    os.makedirs(P.LOGS_DIR, exist_ok=True)
    with open(os.path.join(P.LOGS_DIR, "pip_freeze_diag_env.txt"), "w") as f:
        f.write(cur.stdout)

    old_env = C.load_json(os.path.join(P.VTS_REF_DIR, "logs",
                                       "infer_summary.json"))["env"]
    diff = {}
    for name in sorted(set(zeta_freeze) | set(cur_freeze)):
        a, b = zeta_freeze.get(name), cur_freeze.get(name)
        if a != b:
            diff[name] = {"exp016_zeta": a, "diag_env": b}
    key = {k: {"exp016_zeta": zeta_freeze.get(k),
               "diag_env": cur_freeze.get(k)}
           for k in KEY_PACKAGES}

    rep = {
        "old_env_runtime_record": old_env,
        "new_env_runtime_record": C.env_info(),
        "gpu_new": gpu_info(),
        "key_packages": key,
        "diff_count": len(diff),
        "diff": diff,
    }
    fallback_note = os.environ.get("DIAG_PANDAS_FALLBACK_NOTE")
    if fallback_note:
        rep["pandas_fallback_note"] = fallback_note
    rep["pandas_csv_parse_check"] = pandas_parse_check()
    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    C.dump_json(rep, os.path.join(P.RESULTS_DIR, "env_report.json"))
    print(f"[env-report] diff packages: {len(diff)}; key: "
          + ", ".join(f"{k}={v['exp016_zeta']}->{v['diag_env']}"
                      for k, v in key.items()
                      if v["exp016_zeta"] != v["diag_env"]))
    print(f"[env-report] RESULT: "
          f"{'PASS' if rep['pandas_csv_parse_check']['ok'] else 'FAIL'}")
    sys.exit(0 if rep["pandas_csv_parse_check"]["ok"] else 1)


if __name__ == "__main__":
    main()
