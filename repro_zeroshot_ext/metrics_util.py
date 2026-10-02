#!/usr/bin/env python3
"""Dual-scale MSE/MAE with zero-variance denominator fallback.

Rules (frozen):
  - pred/true are NEVER modified; mu/std are only recorded or used as scalars.
  - standardized metric = raw / denom**k (mu cancels in pred-true differences),
    denom = window std when >= 1e-8, else the variable's frozen calib_std.
  - if both degenerate: standardized metrics are None + std_defined=False;
    raw metrics stay. Never zero, never silently dropped.
"""
EPS_STD = 1e-8


def sample_metrics(pred, true, win_std, calib_std):
    """pred, true: equal-length sequences of raw values for one sample."""
    n = len(true)
    diffs = [p - t for p, t in zip(pred, true)]
    raw_mse = sum(d * d for d in diffs) / n
    raw_mae = sum(abs(d) for d in diffs) / n

    if win_std is not None and win_std >= EPS_STD:
        denom, src = win_std, "window"
    elif calib_std is not None and calib_std >= EPS_STD:
        denom, src = calib_std, "calib_std"
    else:
        denom, src = None, "undefined"

    if denom is None:
        return {"raw_mse": raw_mse, "raw_mae": raw_mae, "std_mse": None,
                "std_mae": None, "denom_source": src, "std_defined": False}
    return {"raw_mse": raw_mse, "raw_mae": raw_mae,
            "std_mse": raw_mse / (denom * denom), "std_mae": raw_mae / denom,
            "denom_source": src, "std_defined": True}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
