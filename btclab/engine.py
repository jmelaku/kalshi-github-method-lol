"""The trading loop: price every BTC market, trade the best edges, settle, repeat.

Pricing pipeline each cycle (default `pricer="fhs"`):

  1. GARCH(1,1)-t with intraday seasonality, fit by maximum likelihood on the
     last few days of 1-minute returns (refit every few hours, filtered
     forward every cycle). Long-run variance is anchored to Deribit DVOL.
  2. Filtered historical simulation: one Monte Carlo run per cycle, from now
     to the furthest settlement time, bootstrapping BTC's own GARCH residuals.
     Every contract is priced off the same simulated paths (consistent
     prices across brackets) with an uncertainty estimate.
  3. Disagreement shrinkage: fair value = market mid + beta * (model - mid),
     with beta learned from settled results per disagreement size and
     contract type (cluster-robust, starts at a conservative prior of 0.5).
  4. Edges are haircut by k * uncertainty before any decision.

The closed-form pricers ("lognormal", "student_t") are kept for comparison.

The engine talks to two small interfaces, so the same code runs against live
Kalshi + Coinbase or the offline simulator:

  Exchange: btc_markets(), orderbook(ticker), market(ticker), create_order(...)
  Feed:     spot(), recent_closes(minutes), recent_candles(minutes), dvol()

Cost control: one paginated GET /markets call returns top-of-book prices for
every market, so full order books are fetched only for markets that already
look mispriced.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol

import numpy as np

from .book import Quote, quote_from_market
from .calibrate import ShrinkTable, fit_shrink_from_ledger
from .fees import FeeModel
from .fhs import FHSPricer
from .garch import GarchFit, fit_garch
from .kalshi import MarketInfo
from .ledger import Ledger
from .pricing import ModelParams, fair_prob, tau_from_seconds
from .risk import Book, RiskManager
from .scanner import Opportunity, Snapshot, evaluate_prob, rank
from .vol import blend, ewma_vol

log = logging.getLogger("btclab")


class Exchange(Protocol):
    def btc_markets(self) -> list[MarketInfo]: ...
    def orderbook(self, ticker: str, depth: int = 10) -> Quote: ...
    def market(self, ticker: str) -> dict: ...
    def create_order(self, ticker: str, buy: str, count: int, price: float, **kw) -> dict: ...


class Feed(Protocol):
    def spot(self) -> float: ...
    def recent_closes(self, minutes: int = 240) -> list[float]: ...
    def recent_candles(self, minutes: int = 240) -> tuple[list[int], list[float]]: ...
    def dvol(self) -> Optional[float]: ...


def nu_for_horizon(hours: float, table: list[list[float]]) -> float:
    """Interpolate the Student-t tail parameter for a horizon (closed-form pricer only).

    `table` is [[hours, nu], ...]; 0 means lognormal. Interpolates 1/nu in log
    time and holds the end values flat. Returns math.inf for "use lognormal".
    """
    pts = sorted((max(h, 1e-6), (1.0 / n if n and n > 0 else 0.0)) for h, n in table)
    h = max(hours, 1e-6)
    if h <= pts[0][0]:
        inv = pts[0][1]
    elif h >= pts[-1][0]:
        inv = pts[-1][1]
    else:
        inv = pts[-1][1]
        for (h0, i0), (h1, i1) in zip(pts, pts[1:]):
            if h0 <= h <= h1:
                w = (math.log(h) - math.log(h0)) / (math.log(h1) - math.log(h0))
                inv = i0 + w * (i1 - i0)
                break
    return math.inf if inv < 1.0 / 200 else 1.0 / inv


@dataclass
class EngineConfig:
    pricer: str = "fhs"  # "fhs" (GARCH + Monte Carlo), "student_t" or "lognormal"
    # --- FHS / GARCH ---
    n_paths: int = 20_000
    vol_uncertainty: float = 0.15  # sd of log vol-level scale per path
    garch_lookback_min: int = 4 * 1440
    garch_refit_min: int = 240
    implied_weight: float = 0.5  # weight of DVOL in the long-run variance target
    haircut_k: float = 1.0  # decisions use edge - k * sd
    mc_seed: Optional[int] = None
    # --- shrinkage toward the market price, learned from settled results ---
    calibrate: bool = True
    shrink_prior_beta: float = 0.5  # trust in model-vs-market disagreement before any data
    shrink_prior_sd: float = 0.25
    shrink_min_days: int = 10  # settled days before learned estimates are used at all
    shrink_z_se: float = 1.0  # use the posterior minus this many standard errors
    shrink_prior_obs: float = 0.0  # deprecated (kept so old config files still load)
    # Above/below contracts are traded only once settled data prove an edge.
    evidence_classes: list = field(default_factory=lambda: ["threshold"])
    evidence_min_days: int = 20
    evidence_min_events: int = 30
    # With no market reference price at all, never trade on the raw model.
    require_market_mid: bool = True
    calib_refit_cycles: int = 60
    calib_min_markets: int = 0  # deprecated (kept so old config files still load)
    # --- closed-form pricers ---
    nu_by_horizon: list = field(default_factory=lambda: [[0.25, 8.0], [1.0, 10.0], [6.0, 20.0], [24.0, 50.0]])
    vol_halflife_min: float = 60.0
    vol_lookback_min: int = 240
    # --- general ---
    basis_usd: float = 20.0
    record_snapshots: bool = True
    snapshot_interval_s: float = 300.0  # at most one stored snapshot per market per interval
    mode: str = "paper"  # "paper" or "live"
    max_trades_per_cycle: int = 5
    model: Optional[str] = None  # deprecated alias for `pricer`


@dataclass
class CycleResult:
    spot: float
    sigma_realized: float
    sigma_implied: Optional[float]
    markets_priced: int
    books_fetched: int
    opportunities: list[Opportunity] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    garch: Optional[dict] = None
    blend: Optional[dict] = None


class Engine:
    def __init__(
        self,
        exchange: Exchange,
        feed: Feed,
        ledger: Ledger,
        risk: RiskManager,
        fees: FeeModel = FeeModel(),
        cfg: Optional[EngineConfig] = None,
    ):
        cfg = cfg or EngineConfig()
        if cfg.model and cfg.model != cfg.pricer:
            cfg.pricer = cfg.model
        self.x, self.feed, self.ledger, self.risk, self.fees, self.cfg = exchange, feed, ledger, risk, fees, cfg
        if cfg.mode not in ("paper", "live"):
            raise ValueError("mode must be paper or live")
        if cfg.pricer not in ("fhs", "student_t", "lognormal"):
            raise ValueError("pricer must be fhs, student_t or lognormal")
        self.fit: Optional[GarchFit] = None
        self._fit_time: Optional[float] = None
        self.fhs = FHSPricer(cfg.n_paths, cfg.vol_uncertainty, cfg.basis_usd, cfg.mc_seed)
        self.shrink = (ShrinkTable.prior(cfg.shrink_prior_beta, tuple(cfg.evidence_classes)) if cfg.calibrate
                       else ShrinkTable.prior(1.0))
        self._cycles = 0

    # ---------- volatility model ----------
    def _ensure_garch(self, now_ts: float, implied: Optional[float]) -> GarchFit:
        cfg = self.cfg
        stale = self._fit_time is None or now_ts - self._fit_time >= cfg.garch_refit_min * 60
        if self.fit is None or stale:
            ts, closes = self.feed.recent_candles(cfg.garch_lookback_min)
            c = np.asarray(closes, dtype=float)
            r = np.diff(np.log(c))
            start = tuple(self.fit.extra["x"]) if self.fit is not None else None
            self.fit = fit_garch(r, ts[1:], implied, cfg.implied_weight, start=start)
            self._fit_time = now_ts
            log.info(
                "GARCH refit: alpha=%.4f beta=%.4f nu=%.1f halflife=%.0fmin vol now %.1f%% long-run %.1f%%",
                self.fit.alpha, self.fit.beta, self.fit.nu, self.fit.halflife_min,
                100 * self.fit.annual_vol_now(), 100 * self.fit.annual_vol_longrun(),
            )
        else:
            ts, closes = self.feed.recent_candles(180)
            c = np.asarray(closes, dtype=float)
            r = np.diff(np.log(c))
            new = [(ri, ti) for ri, ti in zip(r, ts[1:]) if ti > self.fit.last_ts]
            if new:
                self.fit.update([a for a, _ in new], [b for _, b in new])
        return self.fit

    def params_for(self, secs: float, realized: float, implied: Optional[float]) -> ModelParams:
        """Closed-form pricer parameters."""
        hours = secs / 3600.0
        sigma = blend(realized, implied, hours)
        if self.cfg.pricer == "lognormal":
            return ModelParams(sigma, "lognormal", basis_usd=self.cfg.basis_usd)
        nu = nu_for_horizon(hours, self.cfg.nu_by_horizon)
        if nu == float("inf"):
            return ModelParams(sigma, "lognormal", basis_usd=self.cfg.basis_usd)
        return ModelParams(sigma, "student_t", nu, self.cfg.basis_usd)

    def _maybe_refit_shrink(self) -> None:
        if not self.cfg.calibrate:
            return
        if self._cycles % self.cfg.calib_refit_cycles == 1:
            c = self.cfg
            self.shrink = fit_shrink_from_ledger(
                self.ledger.db, beta0=c.shrink_prior_beta, prior_sd=c.shrink_prior_sd, z_se=c.shrink_z_se,
                min_clusters=c.shrink_min_days, evidence_classes=tuple(c.evidence_classes),
                evidence_min_clusters=c.evidence_min_days, evidence_min_events=c.evidence_min_events,
            )

    # ---------- main loop ----------
    def cycle(self, now: Optional[datetime] = None, trade: bool = True) -> CycleResult:
        now = now or datetime.now(timezone.utc)
        now_ts = now.timestamp()
        self._cycles += 1
        spot = self.feed.spot()
        try:
            implied = self.feed.dvol()
        except Exception as e:  # implied vol is optional
            log.warning("dvol unavailable: %s", e)
            implied = None

        l = self.risk.l
        markets = [
            m for m in self.x.btc_markets() if l.min_secs_to_close <= m.seconds_to_close(now) <= l.max_secs_to_close
        ]

        # ---- probabilities for every market ----
        probs: dict[str, tuple[float, float]] = {}
        if self.cfg.pricer == "fhs":
            fit = self._ensure_garch(now_ts, implied)
            sigma_now = fit.annual_vol_now()
            closes = sorted({m.close_time for m in markets})
            dists = self.fhs.simulate(fit, spot, now_ts, [(c - now).total_seconds() for c in closes])
            by_close = dict(zip(closes, dists))
            for m in markets:
                est = by_close[m.close_time].prob(m.contract)
                probs[m.ticker] = (est.p, est.sd)
            model_name = "fhs"
            garch_info = {
                "alpha": round(fit.alpha, 4),
                "beta": round(fit.beta, 4),
                "nu": round(fit.nu, 1),
                "vol_now": round(sigma_now, 4),
                "vol_longrun": round(fit.annual_vol_longrun(), 4),
                "halflife_min": round(fit.halflife_min, 1),
            }
        else:
            sigma_now = ewma_vol(self.feed.recent_closes(self.cfg.vol_lookback_min), 1.0, self.cfg.vol_halflife_min)
            for m in markets:
                p = fair_prob(
                    m.contract, spot, tau_from_seconds(m.seconds_to_close(now)),
                    self.params_for(m.seconds_to_close(now), sigma_now, implied),
                )
                probs[m.ticker] = (p, 0.0)
            model_name = self.cfg.pricer
            garch_info = None

        self._maybe_refit_shrink()
        res = CycleResult(spot, sigma_now, implied, len(markets), 0, garch=garch_info,
                          blend={"rows": self.shrink.rows, "events": self.shrink.events})
        snaps: list[Snapshot] = []
        opps: list[Opportunity] = []
        k = self.cfg.haircut_k

        for m in markets:
            p, sd = probs[m.ticker]
            q = quote_from_market(m.raw)
            if self.cfg.require_market_mid and q.mid is None:
                continue  # empty book: nothing to trade against and no reference price
            used = self.shrink.apply(p, q.mid, m.contract.strike_type)
            snap, top = evaluate_prob(m, q, spot, p, sd, sigma_now, model_name, self.fees, now, used, k)
            if any(o.edge_adj >= l.min_edge_usd for o in top):
                # Looks mispriced at top of book: confirm against the full book.
                book = self.x.orderbook(m.ticker)
                res.books_fetched += 1
                used = self.shrink.apply(p, book.mid, m.contract.strike_type)
                snap, top = evaluate_prob(m, book, spot, p, sd, sigma_now, model_name, self.fees, now, used, k)
            snaps.append(snap)
            opps.extend(top)

        if self.cfg.record_snapshots:
            self.ledger.add_snapshots(snaps, self.cfg.snapshot_interval_s)
        res.opportunities = rank(opps)

        if trade:
            traded_markets: set[str] = set()
            for o in res.opportunities:
                if len(res.trades) >= self.cfg.max_trades_per_cycle:
                    break
                if o.ticker in traded_markets:
                    continue  # never buy both sides of one market in a cycle
                state = Book(self.ledger.market_contracts(o.ticker), self.ledger.exposure(),
                             self.ledger.daily_pnl(now.date().isoformat()), self.ledger.exposure(o.event_ticker))
                n, why = self.risk.size(o, state)
                if n <= 0:
                    res.skipped.append((o.ticker, why))
                    if why.startswith(("kill switch", "daily loss")):
                        break
                    continue
                t = self._execute(o, n, now)
                if t:
                    res.trades.append(t)
                    traded_markets.add(o.ticker)
        return res

    def _execute(self, o: Opportunity, n: int, now: datetime) -> Optional[dict]:
        if self.cfg.mode == "paper":
            fee = self.fees.taker_fee(o.price, n)
            tid = self.ledger.add_trade(o, n, o.price, fee, "paper", ts=now.isoformat())
            return {"id": tid, "ticker": o.ticker, "side": o.side, "count": n, "price": o.price, "fee": fee,
                    "edge": o.edge, "edge_adj": o.edge_adj}

        # Live: immediate-or-cancel limit at the ask we priced, so we never pay more.
        resp = self.x.create_order(o.ticker, o.side, n, o.price, time_in_force="immediate_or_cancel")
        filled = float(resp.get("fill_count") or 0)
        if filled <= 0:
            log.info("no fill on %s %s @ %.2f", o.ticker, o.side, o.price)
            return None
        yes_px = float(resp.get("average_fill_price") or (o.price if o.side == "yes" else 1 - o.price))
        px = yes_px if o.side == "yes" else round(1.0 - yes_px, 4)
        fee = float(resp.get("average_fee_paid") or 0) * filled
        tid = self.ledger.add_trade(o, filled, px, fee, "live", resp.get("order_id", ""), now.isoformat())
        return {"id": tid, "ticker": o.ticker, "side": o.side, "count": filled, "price": px, "fee": fee,
                "edge": o.edge, "edge_adj": o.edge_adj}

    def settle(self, now: Optional[datetime] = None) -> int:
        """Look up results for markets we traded or priced; returns markets settled."""
        now = now or datetime.now(timezone.utc)
        tickers = self.ledger.unsettled_tickers(now)
        if not tickers:
            return 0
        if hasattr(self.x, "results"):
            results = self.x.results(tickers)  # batched lookup (live client)
        else:
            results = {t: ((self.x.market(t) or {}).get("result") or "") for t in tickers}
        done = 0
        for t, r in results.items():
            if r in ("yes", "no"):
                self.ledger.set_result(t, r, now.isoformat())
                done += 1
        return done
