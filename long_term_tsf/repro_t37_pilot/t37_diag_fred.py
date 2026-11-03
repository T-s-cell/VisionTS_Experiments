"""EXP-037 G4 diagnostic 1/3 (eta, GPU; not part of run_t37_all.sh).
FRED-MD anchor-gap exclusion: run the FULL official Table 5 monthly search
space P in {12,6,3,1} on validation (does P=1 stay the winner?), then test
forecasts at P=1 and P=12, then a bitwise determinism rerun of P=1.
Result (2026-11-04): P=1 wins validation 1422.974381 vs 2375.821657 (P=12) /
1923.210327 (P=6) / 2229.507898 (P=3) -> autotune outcome unchanged even
without the pandas-2.2 candidate collapse; test P=1 MAE 1807.417940 ==
formal run; rerun bitwise deterministic.
"""
import sys, os
sys.path.insert(0, "/dev_data/wlt/VisionTS_Experiments/eval_gluonts")
os.chdir("/dev_data/wlt/VisionTS_Experiments/eval_gluonts")
import numpy as np
import run
from visionts import VisionTS, freq_to_seasonality_list

device = "cuda:0"
model = VisionTS("mae_base", ckpt_dir="../ckpt/").to(device)
test_data, metadata = run.get_gluonts_test_dataset("fred_md")
data_train = [run.imputation_nan(x['target']) for x in test_data.input]
data_test = [x['target'] for x in test_data.label]
pred_len = len(data_test[0])
print("metadata.freq =", metadata.freq)
print("seasonality_list =", freq_to_seasonality_list(metadata.freq, run.POSSIBLE_SEASONALITIES))
for P in (12, 6, 3, 1):
    ctx = run.convert_context_len(1000, 300, P)
    val_train = [x[-ctx-pred_len:-pred_len] for x in data_train]
    val_test = [x[-pred_len:] for x in data_train]
    fc = run.forecast(model, val_train, val_test, 512, device, P)
    mae = np.abs(np.asarray(val_test) - fc).mean()
    print("VALID P=%d ctx=%d mae=%.6f" % (P, ctx, mae))
fc_cache = {}
for P in (1, 12):
    ctx = run.convert_context_len(1000, 300, P)
    train = [x[-ctx:] for x in data_train]
    fc = run.forecast(model, train, data_test, 512, device, P)
    fc_cache[P] = fc
    mae = np.abs(np.asarray(data_test) - fc).mean()
    mse = ((np.asarray(data_test) - np.stack(fc))**2).mean()
    print("TEST P=%d ctx=%d MAE=%.6f MSE=%.2f" % (P, ctx, mae, mse))
fc2 = run.forecast(model, [x[-300:] for x in data_train], data_test, 512, device, 1)
print("determinism P=1 rerun:", np.array_equal(np.stack(fc2), np.stack(fc_cache[1])))
