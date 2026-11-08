"""Options-implied pricer: read Kalshi probabilities off Deribit's BTC option prices.

Deribit lists BTC options expiring daily, weekly, monthly and quarterly, each at
08:00 UTC. Their prices imply a full risk-neutral distribution for BTC at every
expiry: a digital ("pays $1 if S_T > K") is minus the slope of the call price in
strike, and the volatility smile carries the market's view of skew and fat tails.

For a Kalshi contract settling at time T:

  1. Each Deribit expiry gives a smile: implied vol by log-moneyness k = ln(K/F),
     from Deribit's mark IVs (calls and puts averaged per strike).
  2. Each smile is smoothed (a cubic in k fit near the money, flat beyond), since
     raw marks have kinks and stale far strikes that would imply negative density.
  3. Total implied variance w = iv^2 * t is interpolated between the two expiries
     around T. Hourly contracts sit before Deribit's first expiry, and there are
     two standard ways to carry a smile to a shorter horizon: at the same
     log-moneyness k ("sticky moneyness", skew fades at short horizons) or at the
     same standardized moneyness k / sqrt(t) (skew keeps its shape in sd units).
     We price with both, use the average, and count half their difference as
     model uncertainty. Between two listed expiries the two agree closely.
  4. Interpolation runs on a *variance clock*, not calendar time: the hour-of-day
     seasonality from the GARCH fit says how much of a day's variance falls in
     each hour, so a contract spanning the quiet Asian night gets less variance
     than one spanning the US open. The options set the level; the clock sets
     the timing.
  5. P(S_T > K) = -dC/dK by central difference on undiscounted Black-76 call
     prices, so the smile's skew is included. Brackets are differences of these.

Uncertainty: each probability is repriced with every implied vol bumped up and
down by `iv_rel_sd` (relative), and half the difference is the `sd` the engine
haircuts, combined with the interpolation disagreement above. `basis_usd` adds variance for the gap between our spot and BRTI.

Caveats: these are risk-neutral probabilities, which can differ slightly from
real-world ones (a small risk premium, negligible over hours). Kalshi's hourly
contracts sit well inside Deribit's first expiry, so for them the result leans
on interpolation; whether that beats the GARCH model is exactly what side-by-side
paper trading is for.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

from .pricing import Contract

YEAR_S = 365.0 * 86400.0
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_NAME = re.compile(r"^BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])$")


def expiry_ts(code: str) -> float:
    """Deribit expiry code like '9NOV26' -> unix seconds at 08:00 UTC."""
    m = re.match(r"^(\d{1,2})([A-Z]{3})(\d{2})$", code)
    if not m:
        raise ValueError(f"bad expiry code {code!r}")
    d, mon, y = int(m.group(1)), _MONTHS[m.group(2)], 2000 + int(m.group(3))
    return datetime(y, mon, d, 8, 0, tzinfo=timezone.utc).timestamp()


@dataclass
class Smile:
    expiry_ts: float
    forward: float  # Deribit's synthetic future for this expiry
    k: np.ndarray  # log-moneyness ln(K/F), sorted
    iv: np.ndarray  # implied vol (decimal) at each k
    coef: Optional[np.ndarray] = None  # smoothing cubic in k, valid on [k_lo, k_hi]
    k_lo: float = 0.0
    k_hi: float = 0.0

    def smooth(self, years: float, width_sd: float = 4.0, min_points: int = 6) -> "Smile":
        """Fit a cubic in k to the strikes within `width_sd` ATM sds of the money."""
        atm = float(np.interp(0.0, self.k, self.iv))
        half = width_sd * max(atm, 0.05) * math.sqrt(max(years, 1e-6))
        m = np.abs(self.k) <= half
        if m.sum() >= min_points:
            self.coef = np.polyfit(self.k[m], self.iv[m], 3)
            self.k_lo, self.k_hi = float(self.k[m].min()), float(self.k[m].max())
        return self

    def iv_at(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        if self.coef is None:
            return np.interp(k, self.k, self.iv)  # flat beyond the listed strikes
        # Inside the fitted range use the cubic; beyond it continue in a straight
        # line with the edge slope, so the smile has no kink (a kink in the smile
        # is a jump in the implied probability).
        kc = np.clip(k, self.k_lo, self.k_hi)
        iv = np.polyval(self.coef, kc)
        d = np.polyder(self.coef)
        iv = iv + np.where(k < self.k_lo, np.polyval(d, self.k_lo) * (k - self.k_lo), 0.0)
        iv = iv + np.where(k > self.k_hi, np.polyval(d, self.k_hi) * (k - self.k_hi), 0.0)
        return np.clip(iv, 0.05, 5.0)


def variance_clock(season: Optional[Sequence[float]]):
    """Returns f(t0, t1) -> effective years of variance between two unix times.

    With hour-of-day seasonality multipliers s[h] (from the GARCH fit), one
    minute in hour h counts as s[h]^2 / mean(s^2) minutes. Without it, plain
    calendar time.
    """
    if season is None:
        return lambda t0, t1: max(t1 - t0, 0.0) / YEAR_S
    w = np.asarray(season, dtype=float) ** 2
    w = w / w.mean()
    day = float(w.sum()) * 3600.0  # weighted seconds in a full day (= 86400)
    cum = np.concatenate([[0.0], np.cumsum(w) * 3600.0])  # weighted seconds from 00:00 to each hour

    def at(t: float) -> float:
        days, rem = divmod(t, 86400.0)
        h = int(rem // 3600)
        return days * day + cum[h] + w[h] * (rem - h * 3600.0)

    return lambda t0, t1: max(at(t1) - at(t0), 0.0) / YEAR_S


@dataclass
class ProbEstimate:
    p: float
    sd: float


class OptionSurface:
    def __init__(self, smiles: list[Smile], index_price: float, now_ts: float, clock=None):
        if not smiles:
            raise ValueError("no usable option expiries")
        self.smiles = sorted(smiles, key=lambda s: s.expiry_ts)
        self.index_price = index_price
        self.now_ts = now_ts
        self.clock = clock or variance_clock(None)
        self._tau = [self.clock(now_ts, s.expiry_ts) for s in self.smiles]

    @classmethod
    def from_deribit(cls, rows: list[dict], now_ts: float, clock=None, min_strikes: int = 5,
                     min_hours: float = 1.0) -> "OptionSurface":
        """Build from Deribit get_book_summary_by_currency(kind=option) rows.

        Only options with a live bid and ask are used; far strikes nobody quotes
        carry stale marks. Expiries closer than `min_hours` are dropped: their
        marks are noisy and they carry almost no variance information.
        """
        by_exp: dict[str, dict] = {}
        index = []
        quoted = [r for r in rows if (r.get("bid_price") or 0) > 0 and (r.get("ask_price") or 0) > 0]
        for r in quoted:
            m = _NAME.match(r.get("instrument_name", ""))
            iv = r.get("mark_iv")
            fwd = r.get("underlying_price")
            if not m or not iv or iv <= 0 or not fwd:
                continue
            code = m.group(1) + m.group(2) + m.group(3)
            e = by_exp.setdefault(code, {"fwd": [], "strikes": {}})
            e["fwd"].append(float(fwd))
            e["strikes"].setdefault(float(m.group(4)), []).append(float(iv) / 100.0)
            if r.get("estimated_delivery_price"):
                index.append(float(r["estimated_delivery_price"]))
        smiles = []
        for code, e in by_exp.items():
            t = expiry_ts(code)
            if t - now_ts < min_hours * 3600 or len(e["strikes"]) < min_strikes:
                continue
            fwd = float(np.median(e["fwd"]))
            ks = sorted(e["strikes"])
            k = np.log(np.asarray(ks) / fwd)
            iv = np.asarray([np.mean(e["strikes"][x]) for x in ks])
            smiles.append(Smile(t, fwd, k, iv).smooth((t - now_ts) / YEAR_S))
        idx = float(np.median(index)) if index else (smiles[0].forward if smiles else 0.0)
        return cls(smiles, idx, now_ts, clock)

    # ---- interpolation ----
    def _bracket(self, tau: float) -> tuple[int, int, float]:
        """Indices of the smiles around variance-time tau and the weight on the second."""
        taus = self._tau
        if tau <= taus[0]:
            return 0, 0, 0.0
        if tau >= taus[-1]:
            n = len(taus) - 1
            return n, n, 0.0
        j = int(np.searchsorted(taus, tau))
        i = j - 1
        return i, j, (tau - taus[i]) / (taus[j] - taus[i])

    def _fwd_ratio(self, t_ts: float) -> float:
        """Forward / index at time t: linear in calendar time from 1 at now (carry is tiny)."""
        pts_t = [self.now_ts] + [s.expiry_ts for s in self.smiles]
        pts_r = [1.0] + [s.forward / self.index_price for s in self.smiles]
        return float(np.interp(t_ts, pts_t, pts_r))

    def total_variance(self, k: np.ndarray, tau: float, iv_scale: float = 1.0,
                       standardized: bool = False) -> np.ndarray:
        """Implied total variance at log-moneyness k and variance-time tau (years).

        standardized=False looks each smile up at the same k; True at the same
        k / sqrt(tau), i.e. at k * sqrt(tau_i / tau) on smile i.
        """
        i, j, a = self._bracket(tau)
        si, sj = self.smiles[i], self.smiles[j]
        ki = k * math.sqrt(self._tau[i] / tau) if standardized else k
        wi = (iv_scale * si.iv_at(ki)) ** 2 * self._tau[i]
        if i == j:
            # Outside the listed expiries: keep that smile's variance *rate*.
            return wi * (tau / self._tau[i])
        kj = k * math.sqrt(self._tau[j] / tau) if standardized else k
        wj = (iv_scale * sj.iv_at(kj)) ** 2 * self._tau[j]
        return wi + a * (wj - wi)

    def prob_above(self, spot: float, strikes: np.ndarray, t_ts: float, basis_usd: float = 0.0,
                   iv_scale: float = 1.0, standardized: bool = False) -> np.ndarray:
        """Risk-neutral P(S_T > K) for each strike, skew included."""
        strikes = np.asarray(strikes, dtype=float)
        tau = self.clock(self.now_ts, t_ts)
        if tau <= 0:
            return (spot > strikes).astype(float)
        fwd = spot * self._fwd_ratio(t_ts)
        extra = (basis_usd / spot) ** 2 if basis_usd > 0 else 0.0
        h = 1e-4 * fwd

        def call(kk: np.ndarray) -> np.ndarray:
            w = self.total_variance(np.log(kk / fwd), tau, iv_scale, standardized) + extra
            sw = np.sqrt(np.maximum(w, 1e-14))
            d1 = (np.log(fwd / kk) + 0.5 * w) / sw
            return fwd * norm.cdf(d1) - kk * norm.cdf(d1 - sw)

        p = (call(strikes - h) - call(strikes + h)) / (2 * h)
        return np.clip(p, 0.0, 1.0)

    def curve(self, spot: float, t_ts: float, basis_usd: float = 0.0, iv_scale: float = 1.0,
              standardized: bool = False, n: int = 801, width_sd: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        """P(S_T > K) on a strike grid, forced non-increasing.

        A probability curve that ever rises with strike would mean negative
        density (an arbitrage in the interpolated smile). Small violations can
        appear when a steep short-dated smile is stretched onto a shorter
        horizon; a running minimum removes them.
        """
        tau = self.clock(self.now_ts, t_ts)
        sd = max(self.atm_vol(t_ts), 0.05) * math.sqrt(max(tau, 1e-9)) + basis_usd / spot
        grid = spot * np.exp(np.linspace(-width_sd * sd, width_sd * sd, n))
        p = self.prob_above(spot, grid, t_ts, basis_usd, iv_scale, standardized)
        return grid, np.minimum.accumulate(p)

    def prob(self, contract: Contract, spot: float, t_ts: float, basis_usd: float = 0.0,
             iv_rel_sd: float = 0.10, cache: Optional[dict] = None) -> ProbEstimate:
        """Probability of YES with an uncertainty sd.

        Pass the same `cache` dict for all contracts in one cycle: every contract
        with the same settlement time then reads off the same curves.
        """
        lo, hi = contract.bounds()
        cache = {} if cache is None else cache

        def p_at(scale: float, standardized: bool) -> float:
            key = (t_ts, scale, standardized)
            if key not in cache:
                cache[key] = self.curve(spot, t_ts, basis_usd, scale, standardized)
            grid, pa = cache[key]

            def above(k: float) -> float:
                if k <= 0:
                    return 1.0
                if k == math.inf:
                    return 0.0
                return float(np.interp(k, grid, pa, left=1.0, right=0.0))

            return min(max(above(lo) - above(hi), 0.0), 1.0)

        pk, pz = p_at(1.0, False), p_at(1.0, True)
        p = 0.5 * (pk + pz)
        vol_sd = (abs(p_at(1.0 + iv_rel_sd, False) - p_at(1.0 - iv_rel_sd, False))
                  + abs(p_at(1.0 + iv_rel_sd, True) - p_at(1.0 - iv_rel_sd, True))) / 4.0
        interp_sd = abs(pk - pz) / 2.0
        return ProbEstimate(p, math.sqrt(vol_sd**2 + interp_sd**2))

    def atm_vol(self, t_ts: float) -> float:
        """Annualized at-the-money implied vol to time t (on the variance clock)."""
        tau = self.clock(self.now_ts, t_ts)
        if tau <= 0:
            return 0.0
        return float(math.sqrt(self.total_variance(np.asarray([0.0]), tau)[0] / tau))
