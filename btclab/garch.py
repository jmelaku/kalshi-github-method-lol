"""GARCH(1,1) volatility model for 1-minute BTC returns, with intraday seasonality.

Why GARCH: BTC volatility clusters. A quiet hour tends to be followed by a
quiet hour and a violent one by another, but vol also mean-reverts toward a
long-run level. GARCH(1,1) captures both and gives a forecast of variance for
every future minute, so a contract settling in 20 minutes and one settling in
20 hours each get the right amount of variance.

Model (on de-seasonalized returns e_t = r_t / s(hour of t)):

    h_t = omega + alpha * e_{t-1}^2 + beta * h_{t-1}
    e_t = sqrt(h_t) * z_t,   z_t ~ Student-t(nu), unit variance

  * Seasonality s(hour): BTC trades more violently during US/EU hours. We
    estimate a robust hour-of-day factor from mean |r| and shrink it toward 1.
  * Variance targeting: omega = target * (1 - alpha - beta), where target is
    the long-run variance. It blends the sample variance with Deribit implied
    vol, so the model mean-reverts toward what the options market expects.
  * alpha, beta, nu are fit by maximum likelihood (Student-t).
  * The recursion runs through scipy.signal.lfilter, so a fit over a week of
    1-minute data takes well under a second.

The standardized residuals z_t are kept: the Monte Carlo pricer bootstraps
them so simulated paths have BTC's real shock distribution, not an assumed one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import optimize, signal, special

MIN_PER_YEAR = 365.0 * 24 * 60


def hour_of_day(ts: np.ndarray) -> np.ndarray:
    return ((np.asarray(ts, dtype=np.int64) // 3600) % 24).astype(int)


def seasonal_factors(returns: np.ndarray, hours: np.ndarray, shrink_k: float = 150.0) -> np.ndarray:
    """Hour-of-day volatility multipliers, normalized so mean(s^2) = 1.

    Uses mean |r| (robust to jumps) and shrinks each hour toward 1 by
    n / (n + shrink_k), so a few days of data can't produce wild factors.
    """
    a = np.abs(returns)
    overall = a.mean()
    s = np.ones(24)
    if overall <= 0:
        return s
    for h in range(24):
        m = hours == h
        n = int(m.sum())
        if n:
            raw = a[m].mean() / overall
            w = n / (n + shrink_k)
            s[h] = 1.0 + w * (raw - 1.0)
    return s / math.sqrt(np.mean(s**2))


def _t_loglik(e2: np.ndarray, h: np.ndarray, nu: float) -> float:
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    return float(np.sum(c - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(e2 / (h * (nu - 2)))))


def garch_filter(e: np.ndarray, omega: float, alpha: float, beta: float, h0: float) -> np.ndarray:
    """Conditional variances h_t for t = 0..n (h[n] is the one-step-ahead forecast)."""
    x = np.empty(e.size + 1)
    x[0] = h0
    x[1:] = omega + alpha * e * e
    # h_t = beta * h_{t-1} + x_t, with h_0 = x_0
    h = signal.lfilter([1.0], [1.0, -beta], x)
    return h


@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    nu: float
    target: float  # long-run per-minute variance of de-seasonalized returns
    season: np.ndarray  # 24 hour-of-day multipliers
    h_next: float  # variance forecast for the next minute (de-seasonalized)
    resid: np.ndarray  # standardized residuals z_t
    loglik: float
    n: int
    last_ts: int
    extra: dict = field(default_factory=dict)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def halflife_min(self) -> float:
        p = self.persistence
        return math.log(0.5) / math.log(p) if 0 < p < 1 else math.inf

    def annual_vol_now(self) -> float:
        return math.sqrt(self.h_next * MIN_PER_YEAR)

    def annual_vol_longrun(self) -> float:
        return math.sqrt(self.target * MIN_PER_YEAR)

    def update(self, r_new: Sequence[float], ts_new: Sequence[int]) -> None:
        """Roll the filter forward with new 1-minute returns (no refit)."""
        for r, ts in zip(r_new, ts_new):
            e = r / self.season[int(ts // 3600) % 24]
            self.h_next = self.omega + self.alpha * e * e + self.beta * self.h_next
            self.last_ts = int(ts)

    def expected_variance_path(self, steps: int) -> np.ndarray:
        """E[h_{T+k}] for k = 1..steps (de-seasonalized)."""
        k = np.arange(steps)
        return self.target + self.persistence**k * (self.h_next - self.target)


def fit_garch(
    returns: Sequence[float],
    timestamps: Sequence[int],
    implied_annual_vol: Optional[float] = None,
    implied_weight: float = 0.5,
    start: Optional[tuple[float, float, float]] = None,
) -> GarchFit:
    """Fit GARCH(1,1)-t with variance targeting on 1-minute log returns.

    `timestamps` are the unix seconds of each return's bar (used for seasonality).
    """
    r = np.asarray(returns, dtype=float)
    ts = np.asarray(timestamps, dtype=np.int64)
    if r.size < 500:
        raise ValueError("need at least 500 one-minute returns to fit GARCH")
    r = r - r.mean()
    hrs = hour_of_day(ts)
    season = seasonal_factors(r, hrs)
    e = r / season[hrs]

    sample_var = float(np.mean(e * e))
    target = sample_var
    if implied_annual_vol and implied_annual_vol > 0:
        iv_var = implied_annual_vol**2 / MIN_PER_YEAR
        target = math.exp((1 - implied_weight) * math.log(sample_var) + implied_weight * math.log(iv_var))
    e2 = e * e

    def unpack(x):
        # persistence p in (0.5, 0.99999), alpha share in (0, 1), nu in (2.1, 60)
        p = 0.5 + 0.49999 * special.expit(x[0])
        a_share = special.expit(x[1])
        alpha = p * a_share * 0.5  # alpha <= p/2 keeps the fit sensible
        beta = p - alpha
        nu = 2.1 + 57.9 * special.expit(x[2])
        return alpha, beta, nu

    def nll(x):
        alpha, beta, nu = unpack(x)
        omega = target * (1 - alpha - beta)
        h = garch_filter(e, omega, alpha, beta, sample_var)[:-1]
        if np.any(h <= 0) or not np.all(np.isfinite(h)):
            return 1e12
        return -_t_loglik(e2, h, nu)

    x0 = np.array(start) if start is not None else np.array([3.0, -0.5, -1.5])
    best = None
    for guess in (x0, np.array([5.0, -1.0, -2.0]), np.array([2.0, 0.0, -1.0])):
        res = optimize.minimize(nll, guess, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 600})
        if best is None or res.fun < best.fun:
            best = res
        if start is not None:
            break  # warm start: one solve is enough
    alpha, beta, nu = unpack(best.x)
    omega = target * (1 - alpha - beta)
    h = garch_filter(e, omega, alpha, beta, sample_var)
    z = e / np.sqrt(h[:-1])
    z = (z - z.mean()) / z.std()  # exact unit variance for the bootstrap
    return GarchFit(
        omega=omega,
        alpha=alpha,
        beta=beta,
        nu=nu,
        target=target,
        season=season,
        h_next=float(h[-1]),
        resid=z,
        loglik=-float(best.fun),
        n=int(r.size),
        last_ts=int(ts[-1]),
        extra={"x": best.x, "sample_annual_vol": math.sqrt(sample_var * MIN_PER_YEAR)},
    )
