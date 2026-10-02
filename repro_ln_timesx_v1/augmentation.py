#!/usr/bin/env python3
"""FreqMask augmentation (spec sec 5.1) — FrAug-style joint input+target transform.

Per training sample presentation:
  coin p=0.5 decides whether to augment (drawn ONCE from the run's aug stream);
  s = concat(x[0:96], y[0:12]) (108 steps, same training sample);
  S = rfft(s) (55 bins); S[0] (DC) always kept; k=1..54 zeroed independently
  w.p. 0.10; s_aug = irfft(S, n=108); split back 96/12.
Retry up to 3 times on non-finite output or augmented-input std < 1e-8, then
fall back to the ORIGINAL sample (d is then computed from the original input).
Original samples that are already constant skip augmentation entirely.
Values are never clipped. The aug RNG is independent of the base window-index
stream and is shared by F2/F4 (same seed -> identical augmented presentation).
"""
import numpy as np


class FreqMask:
    def __init__(self, prob, mask_rate, max_retry, std_eps, max_saved_examples=8):
        self.prob = prob
        self.mask_rate = mask_rate
        self.max_retry = max_retry
        self.std_eps = std_eps
        self.max_saved = max_saved_examples
        self.reset_stats()

    def reset_stats(self):
        self.stats = {"presented": 0, "coin_aug": 0, "applied": 0,
                      "skipped_constant": 0, "fallback": 0,
                      "mask_draws": 0, "masked_bins": 0}
        self.saved = []

    def __call__(self, x, y, rng, tag=""):
        """x, y: float64 arrays (96, 12). Returns (x_used, y_used, augmented).

        The caller computes d from the RETURNED x (original or augmented)."""
        self.stats["presented"] += 1
        if float(np.std(x)) < self.std_eps:
            self.stats["skipped_constant"] += 1
            return x, y, False
        if rng.random() >= self.prob:
            return x, y, False
        self.stats["coin_aug"] += 1
        for _attempt in range(self.max_retry):
            S = np.fft.rfft(np.concatenate([x, y]))
            draws = rng.random(len(S) - 1) < self.mask_rate
            S[1:] = np.where(draws, 0.0 + 0.0j, S[1:])
            s_aug = np.fft.irfft(S, n=len(x) + len(y))
            xa, ya = s_aug[:len(x)], s_aug[len(x):]
            if np.isfinite(s_aug).all() and float(np.std(xa)) >= self.std_eps:
                self.stats["applied"] += 1
                self.stats["mask_draws"] += int(len(draws))
                self.stats["masked_bins"] += int(draws.sum())
                if len(self.saved) < self.max_saved:
                    self.saved.append({
                        "tag": tag,
                        "masked_bins": int(draws.sum()),
                        "x_first8": [float(v) for v in x[:8]],
                        "y": [float(v) for v in y],
                        "x_aug_first8": [float(v) for v in xa[:8]],
                        "y_aug": [float(v) for v in ya]})
                return xa, ya, True
        self.stats["fallback"] += 1
        return x, y, False

    def masked_rate(self):
        if self.stats["mask_draws"] == 0:
            return None
        return self.stats["masked_bins"] / self.stats["mask_draws"]


def self_test():
    """Shape/DC/joint-transform/constant/fallback checks (preflight reuses)."""
    ok = []

    rng = np.random.default_rng(0)
    x = rng.normal(size=96) * 5 + 10
    y = rng.normal(size=12)
    fm = FreqMask(1.0, 0.10, 3, 1e-8)
    xa, ya, aug = fm(x, y, rng)
    ok.append(("shape preserved", xa.shape == (96,) and ya.shape == (12,)))
    s_mean = float(np.mean(np.concatenate([x, y])))
    saug_mean = float(np.mean(np.concatenate([xa, ya])))
    ok.append(("DC/mean preserved", abs(s_mean - saug_mean) < 1e-9))

    # joint transform: with p=1 and high mask rate the target MUST change
    fm_hi = FreqMask(1.0, 0.9, 3, 1e-8)
    rng2 = np.random.default_rng(1)
    _xa, ya_hi, aug_hi = fm_hi(x, y, rng2)
    ok.append(("target transformed jointly", aug_hi and not np.array_equal(ya_hi, y)))

    # p=0 -> identical original
    fm_off = FreqMask(0.0, 0.10, 3, 1e-8)
    xb, yb, aug_b = fm_off(x, y, np.random.default_rng(2))
    ok.append(("disabled -> bitwise original",
               not aug_b and np.array_equal(xb, x) and np.array_equal(yb, y)))

    # constant sample skips
    xc = np.full(96, 3.0)
    fm_c = FreqMask(1.0, 0.10, 3, 1e-8)
    _xc2, _yc2, aug_c = fm_c(xc, y, np.random.default_rng(3))
    ok.append(("constant skips augmentation",
               not aug_c and fm_c.stats["skipped_constant"] == 1))

    return ok


if __name__ == "__main__":
    for name, passed in self_test():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        assert passed, name
    print("augmentation self-test: ALL PASS")
