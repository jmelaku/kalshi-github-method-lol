"""Offline simulator: a realistic fake BTC price path plus a fake Kalshi.

Purpose: run the full engine (pricing -> scanning -> risk -> paper fills ->
settlement -> report) without network access, and measure whether the
engine's *expected* P&L matches its *realized* P&L over many events.

The simulated BTC has the features that matter for pricing short-dated
binaries, with realistic magnitudes:

  * Stochastic volatility: log-vol mean-reverts (half-life ~1 day), so vol
    ranges roughly 30%-90% around a 50% base rather than wandering off.
  * Intraday seasonality: vol peaks around US trading hours.
  * Fat-tailed minute shocks (Student-t, nu=4) plus occasional jumps.
  * Implied vol (DVOL) that tracks the current vol regime, with noise and a
    small premium over realized, as on Deribit.

Simulated market makers are decent but imperfect: they price with a lognormal
model using a slow 4-hour EWMA of realized vol (ignoring seasonality and
jumps), overprice cheap tail brackets (favorite-longshot bias), and add
quote noise and a spread. Those flaws are the "edge" here, so a profit in the
simulator is NOT evidence of an edge on the real Kalshi. What the simulator
CAN show is whether the engine's probabilities are honest: expected and
realized P&L should agree within statistical error.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from .book import Quote
from .kalshi import MarketInfo
from .pricing import Contract, ModelParams, fair_prob, tau_from_seconds
from .vol import ewma_vol

MIN_PER_YEAR = 365 * 24 * 60


class PricePath:
    """Minute-by-minute BTC path with stochastic vol, seasonality, fat tails and jumps."""

    def __init__(
        self,
        start: datetime,
        minutes: int,
        s0: float = 80_000.0,
        vol: float = 0.50,
        seed: int = 7,
        nu: float = 4.0,
        vol_of_vol: float = 0.35,  # stationary sd of log-vol
        vol_halflife_min: float = 24 * 60,
        season_amp: float = 0.25,
        jumps_per_day: float = 2.0,
        jump_sd: float = 0.004,
    ):
        rng = np.random.default_rng(seed)
        kappa = math.log(2) / vol_halflife_min
        eps = rng.normal(0, vol_of_vol * math.sqrt(2 * kappa), minutes)
        x = np.empty(minutes)
        x[0] = rng.normal(0, vol_of_vol)
        for i in range(1, minutes):
            x[i] = x[i - 1] * (1 - kappa) + eps[i]
        base = vol * np.exp(x - 0.5 * vol_of_vol**2)  # E[base^2] ~ vol^2

        t0 = int(start.timestamp())
        ts = t0 + 60 * np.arange(1, minutes + 1)
        hours = (ts // 3600) % 24
        season = 1.0 + season_amp * np.cos(2 * np.pi * (hours - 15) / 24)  # peak ~15:00 UTC
        season /= math.sqrt(np.mean((1.0 + season_amp * np.cos(2 * np.pi * (np.arange(24) - 15) / 24)) ** 2))

        z = rng.standard_t(nu, minutes) * math.sqrt((nu - 2) / nu)
        jumps = (rng.random(minutes) < jumps_per_day / 1440) * rng.normal(0, jump_sd, minutes)
        sig_min = base * season / math.sqrt(MIN_PER_YEAR)
        r = sig_min * z + jumps
        self.start = start
        self.t0 = t0
        self.prices = s0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
        self.base_vol = np.concatenate([[base[0]], base])  # regime vol (no seasonality) per minute
        self.true_vol = float(vol)

    def index(self, t: datetime) -> int:
        i = int((t - self.start).total_seconds() // 60)
        return max(0, min(i, len(self.prices) - 1))

    def at(self, t: datetime) -> float:
        return float(self.prices[self.index(t)])

    def closes_before(self, t: datetime, minutes: int) -> list[float]:
        i = self.index(t)
        return [float(x) for x in self.prices[max(0, i - minutes) : i + 1]]

    def candles_before(self, t: datetime, minutes: int) -> tuple[list[int], list[float]]:
        i = self.index(t)
        j = max(0, i - minutes)
        ts = [self.t0 + 60 * k for k in range(j, i + 1)]
        return ts, [float(x) for x in self.prices[j : i + 1]]


class SimClock:
    def __init__(self, now: datetime):
        self.now = now


class SimFeed:
    def __init__(self, path: PricePath, clock: SimClock, seed: int = 11):
        self.path, self.clock = path, clock
        self.rng = np.random.default_rng(seed)

    def spot(self) -> float:
        return self.path.at(self.clock.now)

    def recent_closes(self, minutes: int = 240) -> list[float]:
        return self.path.closes_before(self.clock.now, minutes)

    def recent_candles(self, minutes: int = 240) -> tuple[list[int], list[float]]:
        return self.path.candles_before(self.clock.now, minutes)

    def dvol(self) -> Optional[float]:
        # 30-day implied vol: mostly long-run level, partly current regime,
        # a 5% premium over realized, plus noise.
        cur = float(self.path.base_vol[self.path.index(self.clock.now)])
        iv = math.exp(0.6 * math.log(self.path.true_vol) + 0.4 * math.log(cur))
        return 1.05 * iv * float(np.exp(self.rng.normal(0, 0.04)))


class SimKalshi:
    """Hourly BTC range events ($250 brackets) plus above/below strikes, listed 6h ahead."""

    def __init__(
        self,
        path: PricePath,
        clock: SimClock,
        n_brackets: int = 24,
        width: float = 250.0,
        hours_ahead: int = 6,
        mm_vol_mult: float = 1.0,
        mm_halflife_min: float = 240.0,
        longshot_premium: float = 0.02,
        noise: float = 0.012,
        half_spread: float = 0.02,
        seed: int = 3,
    ):
        self.path, self.clock = path, clock
        self.n, self.width, self.hours_ahead = n_brackets, width, hours_ahead
        self.mm_vol_mult, self.mm_hl = mm_vol_mult, mm_halflife_min
        self.longshot, self.noise, self.half = longshot_premium, noise, half_spread
        self.rng = np.random.default_rng(seed)
        self.markets: dict[str, MarketInfo] = {}
        self._mm_vol_cache: tuple[Optional[datetime], float] = (None, 0.5)

    def _ensure_events(self) -> None:
        now = self.clock.now
        base = now.replace(minute=0, second=0, microsecond=0)
        for h in range(1, self.hours_ahead + 1):
            close = base + timedelta(hours=h)
            ev = f"KXBTC-SIM{close:%y%m%d%H}"
            if any(m.event_ticker == ev for m in self.markets.values()):
                continue
            center = round(self.path.at(now) / self.width) * self.width
            n = self.n + 4 * (h - 1)  # wider ladder for further events
            lo0 = center - n // 2 * self.width
            for i in range(n):
                lo = lo0 + i * self.width
                self._add(f"{ev}-B{int(lo)}", ev, Contract("between", lo, lo + self.width - 0.01), close)
            for k in (center - 1500, center - 750, center, center + 750, center + 1500):
                self._add(f"{ev}-T{int(k)}", ev, Contract("greater", float(k), None), close)

    def _add(self, ticker: str, ev: str, c: Contract, close: datetime) -> None:
        self.markets[ticker] = MarketInfo(ticker, ev, c, close, "active", "", {})

    def _mm_vol(self) -> float:
        when, v = self._mm_vol_cache
        if when != self.clock.now:
            closes = self.path.closes_before(self.clock.now, 1440)
            v = ewma_vol(closes, 1.0, self.mm_hl) * self.mm_vol_mult
            self._mm_vol_cache = (self.clock.now, v)
        return v

    def _mm_quote(self, m: MarketInfo) -> Quote:
        now = self.clock.now
        secs = m.seconds_to_close(now)
        p = fair_prob(m.contract, self.path.at(now), tau_from_seconds(secs), ModelParams(self._mm_vol(), "lognormal"))
        if p < 0.15:
            p += self.longshot * (1 - p / 0.15)  # favorite-longshot bias on cheap brackets
        p = min(0.995, max(0.005, p + self.rng.normal(0, self.noise)))
        bid = math.floor((p - self.half) * 100) / 100
        ask = math.ceil((p + self.half) * 100) / 100
        size = float(self.rng.integers(20, 200))
        q = Quote()
        if 0.01 <= bid <= 0.99:
            q.yes_bids = [(bid, size)]
        if 0.01 <= ask <= 0.99:
            q.no_bids = [(round(1 - ask, 2), size)]
        return q

    # ---- Exchange interface ----
    def btc_markets(self) -> list[MarketInfo]:
        self._ensure_events()
        out = []
        for m in self.markets.values():
            if m.close_time > self.clock.now:
                q = self._mm_quote(m)
                m.raw = {
                    "ticker": m.ticker,
                    "yes_bid_dollars": f"{q.yes_bid:.4f}" if q.yes_bid else None,
                    "yes_ask_dollars": f"{q.yes_ask:.4f}" if q.yes_ask else None,
                    "yes_bid_size_fp": f"{q.no_ask_size:.2f}",
                    "yes_ask_size_fp": f"{q.yes_ask_size:.2f}",
                }
                m._quote = q  # type: ignore[attr-defined]
                out.append(m)
        return out

    def orderbook(self, ticker: str, depth: int = 10) -> Quote:
        m = self.markets[ticker]
        return getattr(m, "_quote", None) or self._mm_quote(m)

    def market(self, ticker: str) -> dict:
        m = self.markets.get(ticker)
        if not m:
            return {}
        if self.clock.now < m.close_time:
            return {"ticker": ticker, "result": ""}
        s = self.path.at(m.close_time)
        lo, hi = m.contract.bounds()
        return {"ticker": ticker, "result": "yes" if lo < s < hi else "no"}

    def create_order(self, ticker: str, buy: str, count: int, price: float, **kw) -> dict:
        raise RuntimeError("the simulator only supports paper mode")


def run_simulation(
    hours: int = 72,
    step_min: int = 5,
    seed: int = 7,
    ledger_path: str = ":memory:",
    limits=None,
    cfg=None,
    warmup_days: float = 4.0,
):
    """Run the engine over a simulated period. Returns (ledger, summary)."""
    from .engine import Engine, EngineConfig
    from .ledger import Ledger
    from .risk import RiskLimits, RiskManager

    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    warmup = int(warmup_days * 1440)
    path = PricePath(start - timedelta(minutes=warmup), hours * 60 + warmup + 8 * 60, seed=seed)
    clock = SimClock(start)
    ex = SimKalshi(path, clock, seed=seed + 1)
    feed = SimFeed(path, clock, seed=seed + 2)
    ledger = Ledger(ledger_path)
    risk = RiskManager(limits or RiskLimits(kill_switch_file="__no_kill_switch_in_sim__"))
    cfg = cfg or EngineConfig(basis_usd=0.0, mc_seed=seed)
    eng = Engine(ex, feed, ledger, risk, cfg=cfg)

    cycles = trades = 0
    while clock.now < start + timedelta(hours=hours):
        r = eng.cycle(clock.now)
        trades += len(r.trades)
        cycles += 1
        clock.now += timedelta(minutes=step_min)
        eng.settle(clock.now)
    clock.now += timedelta(hours=8)  # let the last events settle
    eng.settle(clock.now)
    return ledger, {"cycles": cycles, "trades": trades}
