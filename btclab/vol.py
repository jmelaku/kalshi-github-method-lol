"""Volatility estimation.

Short-dated Kalshi contracts (hours to a day) care about volatility over the
next few hours, so we combine:

  * EWMA realized vol from recent 1-minute returns (reacts fast), and
  * Deribit's DVOL index, the 30-day implied vol (forward-looking, smoother).

sigma = w * realized + (1 - w) * implied

Weight w shrinks toward implied as the horizon grows. Both inputs are
annualized decimals (0.55 = 55%).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

MINUTES_PER_YEAR = 365.0 * 24 * 60


def log_returns(closes: Sequence[float]) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    if c.size < 2:
        return np.array([])
    if np.any(c <= 0):
        raise ValueError("prices must be positive")
    return np.diff(np.log(c))


def ewma_vol(
    closes: Sequence[float],
    bar_minutes: float = 1.0,
    halflife_bars: float = 60.0,
) -> float:
    """Annualized EWMA volatility from evenly spaced closes."""
    r = log_returns(closes)
    if r.size == 0:
        raise ValueError("need at least 2 prices")
    lam = 0.5 ** (1.0 / halflife_bars)
    weights = lam ** np.arange(r.size - 1, -1, -1)  # newest return gets weight 1
    var_per_bar = float(np.sum(weights * r * r) / np.sum(weights))
    bars_per_year = MINUTES_PER_YEAR / bar_minutes
    return math.sqrt(var_per_bar * bars_per_year)


def realized_vol(closes: Sequence[float], bar_minutes: float = 1.0) -> float:
    """Plain (equal-weight) annualized realized vol."""
    r = log_returns(closes)
    if r.size == 0:
        raise ValueError("need at least 2 prices")
    return math.sqrt(float(np.mean(r * r)) * MINUTES_PER_YEAR / bar_minutes)


def blend(
    realized: Optional[float],
    implied: Optional[float],
    horizon_hours: float,
    realized_weight_at_0h: float = 0.7,
    halflife_hours: float = 12.0,
) -> float:
    """Blend realized and implied vol; implied gains weight at longer horizons."""
    if realized is None and implied is None:
        raise ValueError("need realized or implied vol")
    if implied is None:
        return float(realized)
    if realized is None:
        return float(implied)
    w = realized_weight_at_0h * 0.5 ** (max(horizon_hours, 0.0) / halflife_hours)
    return w * realized + (1.0 - w) * implied
