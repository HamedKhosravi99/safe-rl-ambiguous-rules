"""One-sided betting confidence sequence for the mean of i.i.d. observations
in [0, B] (Waudby-Smith and Ramdas, 2024, "Estimating means of bounded random
variables by betting", predictable plug-in hedged capital, c = 1/2).

For a candidate mean m the capital process K_t(m) = prod_{i<=t} (1 - lambda_i(m) (X_i - m))
with X_i = W_i / B in [0, 1] is a nonnegative supermartingale when the true
mean is m or larger, so {m : K_t(m) < 1/alpha} is a (1 - alpha) confidence
set at every t simultaneously.  betting_upper() returns its supremum in the
original units (the one-sided upper bound) and the first t at which the null
"mean >= d" is rejected (anytime-valid, so it is a legitimate stopping rule).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

C_CAP = 0.5


def _grid(n_geom: int = 4000, n_lin: int = 1001) -> np.ndarray:
    return np.unique(np.concatenate((np.geomspace(1e-7, 1.0, n_geom), np.linspace(0.0, 1.0, n_lin))))


def betting_upper(w, B: float, alpha: float = 0.05, d: Optional[float] = None, grid: Optional[np.ndarray] = None) -> dict:
    x = np.asarray(w, dtype=float) / float(B)
    n = x.size
    if n == 0:
        return dict(upper=float(B), first_crossing=None, n=0)
    t = np.arange(1, n + 1, dtype=float)
    mu_hat = (0.5 + np.cumsum(x)) / (t + 1.0)
    sig2 = (0.25 + np.cumsum((x - mu_hat) ** 2)) / (t + 1.0)
    sig2_prev = np.concatenate(([0.25], sig2[:-1]))
    lam = np.sqrt(2.0 * math.log(1.0 / alpha) / (sig2_prev * t * np.log1p(t)))
    lam = np.minimum(lam, C_CAP)
    g = _grid() if grid is None else np.asarray(grid, dtype=float)
    m0 = None if d is None else min(max(d / float(B), 0.0), 1.0)
    if m0 is not None and not np.any(np.isclose(g, m0)):
        g = np.unique(np.append(g, m0))
    i0 = int(np.argmin(np.abs(g - m0))) if m0 is not None else None
    cap = C_CAP / np.maximum(1.0 - g, 1e-12)
    thresh = math.log(1.0 / alpha)
    logK = np.zeros(g.size)
    first = None
    for i in range(n):
        lm = np.minimum(lam[i], cap)
        logK += np.log1p(-lm * (x[i] - g))
        if first is None and i0 is not None and logK[i0] >= thresh:
            first = i + 1
    ok = logK < thresh
    upper = float(g[ok].max()) if ok.any() else 0.0
    return dict(upper=upper * float(B), first_crossing=first, n=int(n), alpha=alpha,
                rejects_d=(None if m0 is None else bool(logK[i0] >= thresh)))


if __name__ == "__main__":
    # smoke test against the C-MAPSS-like shape: 97% zeros, rare charged episodes, B = 53.3
    rng = np.random.default_rng(0)
    B = 53.333
    for n in (300, 20000, 50000):
        w = np.where(rng.random(n) < 0.03, rng.choice([1 / 0.6, 2 / 0.6, 1 / 3.92], size=n), 0.0)
        r = betting_upper(w, B, alpha=0.025, d=0.05)
        mean = w.mean(); v = w.var(ddof=1); ln = math.log(2 / 0.025)
        bern = mean + math.sqrt(2 * v * ln / n) + 7 * B * ln / (3 * (n - 1))
        print(f"n={n:6d} mean={mean:.4f} betting_upper={r['upper']:.4f} bernstein={bern:.4f} first_crossing={r['first_crossing']}")
    # coverage check: 400 replications at n=300 with true mean known
    miss = 0; reps = 400
    for _ in range(reps):
        w = np.where(rng.random(300) < 0.03, 2 / 0.6, 0.0)
        if betting_upper(w, B, alpha=0.05)["upper"] < 0.03 * 2 / 0.6:
            miss += 1
    print(f"coverage check: miss rate {miss/reps:.3f} (alpha 0.05)")
