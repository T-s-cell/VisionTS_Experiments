"""EXP-037 G4 diagnostic 2/3 (eta, GPU; not part of run_t37_all.sh).
Anchor-gap exclusion: fp64 vs fp32 inference on both subsets' test split.
Result (2026-11-04): fred_md fp64 1807.417940070 vs fp32 1807.418212891
(rel ~1.5e-7); nn5 fp64 3.513941406 vs fp32 3.513941050 (rel ~1e-7) —
precision cannot explain the 4.56% fred anchor gap.
"""
import sys, os, math
sys.path.insert(0, "/dev_data/wlt/VisionTS_Experiments/eval_gluonts")
os.chdir("/dev_data/wlt/VisionTS_Experiments/eval_gluonts")
import numpy as np, einops, torch
import run
from visionts import VisionTS

device = "cuda:0"
model = VisionTS("mae_base", ckpt_dir="../ckpt/").to(device)

def forecast_fp(model, train_list, test_list, batch_size, device, periodicity, fp64):
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
            p = model(b, fp64=fp64)
            for i in range(len(p)):
                out[idx[bi*batch_size+i]] = p[i, :, 0].detach().cpu().numpy()
    return np.stack([out[i] for i in range(len(train_list))])

for ds, P, ctx in (("fred_md", 1, 300), ("nn5_daily_without_missing", 7, 1001)):
    test_data, metadata = run.get_gluonts_test_dataset(ds)
    data_train = [run.imputation_nan(x['target']) for x in test_data.input]
    data_test = [x['target'] for x in test_data.label]
    train = [x[-ctx:] for x in data_train]
    for fp64 in (True, False):
        fc = forecast_fp(model, train, data_test, 512, device, P, fp64)
        mae = np.abs(np.asarray(data_test) - fc).mean()
        print("%s P=%d ctx=%d fp64=%s TEST MAE=%.9f" % (ds, P, ctx, fp64, mae))
