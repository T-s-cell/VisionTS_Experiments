# TimesX LN fine-tuning (protocol timesx_ln_v1) — results

- Runs: 285/285 frozen; selection rule: best_ft (val std-MSE, epochs 1-10, earliest on tie).
- Aggregation: window mean -> variable -> 19 domains merged per (method, seed) -> domain equal-weight -> overall; 3-seed mean±std(ddof=1); standardized by per-window input std.
- Z0 = fixed-P1 zero-shot recomputed on the NEW split; N0 = last-value. Both deterministic singles.

## Overall (standardized)

| method | MSE | MAE |
|---|---|---|
| Z0 | 3.654700 | 1.049081 |
| N0 | 3.766393 | 1.096217 |
| F0 | 3.546316+/-0.004496 | 1.007067+/-0.000412 |
| F1 | 3.561971+/-0.001681 | 1.006676+/-0.000059 |
| F2 | 3.569082+/-0.005394 | 1.007169+/-0.000768 |
| F3 | 3.562551+/-0.001666 | 1.006842+/-0.000059 |
| F4 | 3.569654+/-0.005355 | 1.007353+/-0.000761 |

## Improvement/degradation vs Z0

| method | level | metric | improved | degraded | equal |
|---|---|---|---|---|---|
| F0 | domain(19) | mse | 13 | 6 | 0 |
| F0 | var(190) | mse | 138 | 52 | 0 |
| F0 | domain(19) | mae | 14 | 5 | 0 |
| F0 | var(190) | mae | 134 | 56 | 0 |
| F1 | domain(19) | mse | 11 | 8 | 0 |
| F1 | var(190) | mse | 131 | 59 | 0 |
| F1 | domain(19) | mae | 14 | 5 | 0 |
| F1 | var(190) | mae | 132 | 58 | 0 |
| F2 | domain(19) | mse | 11 | 8 | 0 |
| F2 | var(190) | mse | 131 | 59 | 0 |
| F2 | domain(19) | mae | 14 | 5 | 0 |
| F2 | var(190) | mae | 133 | 57 | 0 |
| F3 | domain(19) | mse | 11 | 8 | 0 |
| F3 | var(190) | mse | 131 | 59 | 0 |
| F3 | domain(19) | mae | 14 | 5 | 0 |
| F3 | var(190) | mae | 132 | 58 | 0 |
| F4 | domain(19) | mse | 11 | 8 | 0 |
| F4 | var(190) | mse | 131 | 59 | 0 |
| F4 | domain(19) | mae | 14 | 5 | 0 |
| F4 | var(190) | mae | 133 | 57 | 0 |

## Pre-registered comparisons (overall)

| comparison | metric | delta | relative |
|---|---|---|---|
| F1-F0 | mse | +0.015654 | -0.44% |
| F1-F0 | mae | -0.000392 | +0.04% |
| F2-F1 | mse | +0.007111 | -0.20% |
| F2-F1 | mae | +0.000494 | -0.05% |
| F3-F1 | mse | +0.000581 | -0.02% |
| F3-F1 | mae | +0.000167 | -0.02% |
| F4-F2 | mse | +0.000572 | -0.02% |
| F4-F2 | mae | +0.000184 | -0.02% |
| F4-F3 | mse | +0.007102 | -0.20% |
| F4-F3 | mae | +0.000511 | -0.05% |
| F0-Z0 | mse | -0.108383 | +2.97% |
| F0-Z0 | mae | -0.042013 | +4.00% |
| F1-Z0 | mse | -0.092729 | +2.54% |
| F1-Z0 | mae | -0.042405 | +4.04% |
| F2-Z0 | mse | -0.085618 | +2.34% |
| F2-Z0 | mae | -0.041911 | +4.00% |
| F3-Z0 | mse | -0.092148 | +2.52% |
| F3-Z0 | mae | -0.042238 | +4.03% |
| F4-Z0 | mse | -0.085046 | +2.33% |
| F4-Z0 | mae | -0.041727 | +3.98% |

## Fallback (selected_with_fallback, separate table)

- fallback_to_zeroshot triggered in 32 of 285 runs (per-domain breakdown in fallback_summary.csv).

## Interpretation limits

- Three-seed std describes training randomness only; overlapping windows are not independent samples.
- The new split shares data with the earlier zero-shot study; this is an exploratory method comparison, not a blind test.
- No causal attribution beyond the pre-registered contrasts.
