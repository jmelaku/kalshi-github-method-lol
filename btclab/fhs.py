"""Filtered Historical Simulation (FHS) pricer.

For each settlement time we simulate N price paths minute by minute:

    e_k = sqrt(h_k) * z_k              z_k bootstrapped from BTC's own GARCH residuals
    r_k = s(hour_k) * e_k * sqrt(u)    s = intraday seasonality, u = vol-level uncertainty
    h_{k+1} = omega + alpha * e_k^2 + beta * h_k

and read every contract's probability off the simulated distribution of the
settlement price. Compared with a closed-form lognormal or Student-t:

  * Tails come from real BTC shocks, and they thin out at longer horizons
    automatically, because simulated vol clusters and mean-reverts like the
    real thing. No hand-tuned tail table.
  * Variance is right for each contract's exact horizon and time of day.
  * Vol-level uncertainty: each path draws a scale u ~ lognormal(mean 1, sd eta),
    because the current vol estimate is itself uncertain. The result is a
    predictive distribution, not a point estimate.
  * Antithetic pairs (z, -z) and one shared simulation for all settlement
    times in a cycle reduce Monte Carlo noise and cost.

Each probability comes with an uncertainty `sd`, which combines Monte Carlo
error and sensitivity to the vol level (paths with low vs high u). The engine
haircuts edges by k * sd, which counters the winner's curse: the trades that
look best are disproportionately the ones where the model is most wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .garch import GarchFit
from .pricing import Contract


@dataclass
class ProbEstimate:
    p: float
    sd: float  # uncertainty of p (MC error + vol-level sensitivity)


@dataclass
class TerminalDist:
    """Simulated log returns to one settlement time."""

    spot: float
    log_ret: np.ndarray  # shape (N,)
    vol_scale: np.ndarray  # u per path, shape (N,)

    def prob(self, contract: Contract) -> ProbEstimate:
        lo, hi = contract.bounds()
        st = self.spot * np.exp(self.log_ret)
        hit = (st > lo) & (st < hi)
        n = hit.size
        p = float(hit.mean())
        mc_var = max(p * (1 - p), 1.0 / n) / n
        # Sensitivity to the vol level: compare low-vol and high-vol quartiles of u.
        q1, q3 = np.quantile(self.vol_scale, [0.25, 0.75])
        lo_p = float(hit[self.vol_scale <= q1].mean())
        hi_p = float(hit[self.vol_scale >= q3].mean())
        # Quartile means sit about 1.27 sd apart for a normal; halve the
        # difference and scale so this approximates one sd of p across u.
        vol_sd = abs(hi_p - lo_p) / 2.54
        return ProbEstimate(p, math.sqrt(mc_var + vol_sd**2))


class FHSPricer:
    def __init__(
        self,
        n_paths: int = 20_000,
        vol_uncertainty: float = 0.15,
        basis_usd: float = 20.0,
        seed: Optional[int] = None,
    ):
        if n_paths % 2:
            n_paths += 1
        self.n = n_paths
        self.eta = vol_uncertainty
        self.basis_usd = basis_usd
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        fit: GarchFit,
        spot: float,
        now_ts: float,
        horizons_s: list[float],
    ) -> list[TerminalDist]:
        """One shared simulation; returns a distribution for each horizon (seconds)."""
        if not horizons_s:
            return []
        steps = [max(1, int(math.ceil(h / 60.0))) for h in horizons_s]
        kmax = max(steps)
        want = {}
        for i, k in enumerate(steps):
            want.setdefault(k, []).append(i)

        half = self.n // 2
        u_half = np.exp(self.rng.normal(-0.5 * self.eta**2, self.eta, half))
        u = np.concatenate([u_half, u_half])
        su = np.sqrt(u)
        h = np.full(self.n, fit.h_next)
        cum = np.zeros(self.n)
        out: list[Optional[TerminalDist]] = [None] * len(horizons_s)
        z_pool = fit.resid
        minute0 = int(now_ts // 60)
        for k in range(1, kmax + 1):
            z_half = z_pool[self.rng.integers(0, z_pool.size, half)]
            z = np.concatenate([z_half, -z_half])  # antithetic
            e = np.sqrt(h) * z
            hour = ((minute0 + k) // 60) % 24
            cum += fit.season[hour] * e * su
            h = fit.omega + fit.alpha * e * e + fit.beta * h
            if k in want:
                r = cum.copy()
                if self.basis_usd > 0:
                    r += self.rng.normal(0.0, self.basis_usd / spot, self.n)
                r -= math.log(np.mean(np.exp(r)))  # martingale: E[S_T] = S_0
                for i in want[k]:
                    out[i] = TerminalDist(spot, r, u)
        return out  # type: ignore[return-value]
