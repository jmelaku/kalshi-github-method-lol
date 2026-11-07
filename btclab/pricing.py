"""Fair value for Kalshi BTC contracts.

Every Kalshi BTC contract is a binary (digital) option on the settlement index
at a fixed time: "above K", "below K", or "between K1 and K2". Its fair price
is the probability of that outcome.

We model the log return from now to settlement, r = ln(S_T / S_0), as

    lognormal:  r ~ Normal(mu, s^2)
    fat-tailed: r ~ mu + s * sqrt((nu-2)/nu) * StudentT(nu)

with s = sigma * sqrt(tau) and mu = -s^2/2 (zero drift in price).
Both versions have the same standard deviation, so they differ only in the
shape of the tails. Short-horizon BTC returns have fat tails, so the
Student-t model typically prices far-out brackets higher than lognormal.

Basis risk: Kalshi settles on the CF Benchmarks Bitcoin Real-Time Index
(BRTI), while our live spot comes from a single exchange. `basis_usd` adds
that uncertainty (in dollars, 1 standard deviation) to the variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from scipy import stats

SECONDS_PER_YEAR = 365.0 * 24 * 3600  # crypto trades 24/7


@dataclass(frozen=True)
class Contract:
    """The payoff definition of one Kalshi market.

    strike_type follows Kalshi's enum: greater, greater_or_equal, less,
    less_or_equal, between. For "between", floor and cap are both set.
    """

    strike_type: str
    floor: Optional[float] = None
    cap: Optional[float] = None

    def bounds(self) -> tuple[float, float]:
        """Return (lo, hi) such that YES pays when lo < S_T < hi."""
        st = self.strike_type
        if st in ("greater", "greater_or_equal"):
            if self.floor is None:
                raise ValueError("greater contract needs floor")
            return self.floor, math.inf
        if st in ("less", "less_or_equal"):
            if self.cap is None:
                raise ValueError("less contract needs cap")
            return 0.0, self.cap
        if st == "between":
            if self.floor is None or self.cap is None:
                raise ValueError("between contract needs floor and cap")
            return self.floor, self.cap
        raise ValueError(f"unsupported strike_type: {st}")


@dataclass(frozen=True)
class ModelParams:
    sigma: float  # annualized volatility, e.g. 0.55 for 55%
    model: str = "student_t"  # or "lognormal"
    nu: float = 4.0  # Student-t degrees of freedom (lower = fatter tails)
    basis_usd: float = 0.0  # 1-sd uncertainty between our spot and the settlement index


def return_std(spot: float, tau_years: float, p: ModelParams) -> float:
    """Total standard deviation of the log return to settlement."""
    if tau_years <= 0:
        return 0.0
    var = p.sigma**2 * tau_years
    if p.basis_usd > 0:
        var += (p.basis_usd / spot) ** 2
    return math.sqrt(var)


def prob_above(spot: float, strike: float, tau_years: float, p: ModelParams) -> float:
    """P(S_T > strike)."""
    if strike <= 0:
        return 1.0
    if math.isinf(strike):
        return 0.0
    s = return_std(spot, tau_years, p)
    x = math.log(strike / spot)
    if s == 0:
        return 1.0 if x < 0 else 0.0
    mu = -0.5 * s * s
    if p.model == "lognormal":
        return float(stats.norm.sf((x - mu) / s))
    if p.model == "student_t":
        if p.nu <= 2:
            raise ValueError("nu must be > 2 so variance is finite")
        scale = s * math.sqrt((p.nu - 2.0) / p.nu)
        return float(stats.t.sf((x - mu) / scale, df=p.nu))
    raise ValueError(f"unknown model: {p.model}")


def fair_prob(contract: Contract, spot: float, tau_years: float, p: ModelParams) -> float:
    """Probability that the contract settles YES."""
    lo, hi = contract.bounds()
    pr = prob_above(spot, lo, tau_years, p) - prob_above(spot, hi, tau_years, p)
    return min(1.0, max(0.0, pr))


def tau_from_seconds(seconds: float) -> float:
    return max(seconds, 0.0) / SECONDS_PER_YEAR


def implied_sigma(
    contract: Contract,
    market_price: float,
    spot: float,
    tau_years: float,
    model: str = "lognormal",
    nu: float = 4.0,
) -> Optional[float]:
    """Volatility that makes the model price equal market_price.

    Only well-defined for monotone contracts ("above"/"below"), where price is
    monotone in sigma on each side of the money. Returns None if no root.
    """
    if contract.strike_type == "between" or not 0 < market_price < 1:
        return None

    def f(sig: float) -> float:
        return fair_prob(contract, spot, tau_years, ModelParams(sig, model, nu)) - market_price

    lo, hi = 1e-4, 10.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-7:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)
