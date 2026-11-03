"""EXP-037 G4 diagnostic 3/3 (eta, GPU; not part of run_t37_all.sh).
Anchor-gap exclusion: replay both subsets through the visionts package as
of commit 67466d3 (the commit whose era matches the precomputed monash.csv
anchor rows), extracted to /tmp/t37_oldpkg via `git archive 67466d3 visionts`.
Result (2026-11-04): old-code MAE identical to current code on both subsets
(fred 1807.417940070353 rel-to-anchor 4.555e-02; nn5 3.513941406127
rel-to-anchor 3.579e-05) -> the fred anchor 1893.668241959648 is unreachable
from ANY code state on this env; it is stale w.r.t. the author's runtime
environment (torch/GPU numerics), amplified by FRED-MD's heterogeneous scale.
Regenerate /tmp/t37_oldpkg first if missing:
  git archive 67466d3 visionts | mkdir -p /tmp/t37_oldpkg && tar -x -C /tmp/t37_oldpkg
"""
import sys, os, math
sys.path.insert(0, "/tmp/t37_oldpkg")
sys.path.insert(0, "/dev_data/wlt/VisionTS_Experiments/eval_gluonts")
os.chdir("/dev_data/wlt/VisionTS_Experiments/eval_gluonts")
import numpy as np, einops, torch
import run
import visionts
from visionts import VisionTS
print("visionts pkg:", visionts.__file__, "| has VisionTSpp:", hasattr(visionts, "VisionTSpp"))

device = "cuda:0"
model = VisionTS("mae_base", ckpt_dir="../ckpt/").to(device)

def forecast_fp(model, train_list, test_list, batch_size, device, periodicity):
    groups = {}
    for i in range(len(train_list)):
        groups.setdefault(len(train_list[i]), [[], [], []])
        groups[len(train_list[i])][0].append(i)
        groups[len(train_list[i])][1].append(train_list[i])
        groups[len(train_list[i])][2].append(test_list[i])
    out = {}
    for tl, (idx, tr, te) in groups.items():
        conv = lambda a: torch.FloatTensor(einops.rearrange(np.array(a), 'b t -> b t 1')).to(device)
        ctr, cte = conv(tr), conv(te)
        model.update_config(context_len=ctr.shape[1], pred_len=cte.shape[1], periodicity=periodicity)
        for bi in range(int(math.ceil(len(ctr) / batch_size))):
            b = ctr[bi*batch_size:(bi+1)*batch_size]
            p = model(b, fp64=True)
            for i in range(len(p)):
                out[idx[bi*batch_size+i]] = p[i, :, 0].detach().cpu().numpy()
    return np.stack([out[i] for i in range(len(train_list))])

for ds, P, ctx, anchor in (("fred_md", 1, 300, 1893.668241959648),
                           ("nn5_daily_without_missing", 7, 1001, 3.514067169133817)):
    test_data, metadata = run.get_gluonts_test_dataset(ds)
    data_train = [run.imputation_nan(x['target']) for x in test_data.input]
    data_test = [x['target'] for x in test_data.label]
    train = [x[-ctx:] for x in data_train]
    fc = forecast_fp(model, train, data_test, 512, device, P)
    mae = np.abs(np.asarray(data_test) - fc).mean()
    print("%s P=%d OLD-CODE MAE=%.12f anchor=%.12f rel=%.3e" % (ds, P, mae, anchor, abs(mae-anchor)/anchor))
