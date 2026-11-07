"""Compare fair value against the Kalshi book and find edge after fees.

For a contract with fair YES probability p (uncertainty sd):

  buy YES at ask a:  EV per contract = p       - a - fee(a)
  buy NO  at ask b:  EV per contract = (1 - p) - b - fee(b)

`edge` is the model's expected value per contract. `edge_adj` subtracts
k * sd, and trading decisions use `edge_adj`. That haircut matters: the
markets where the model disagrees most with the price are disproportionately
the ones where the model is wrong (winner's curse), so raw edges on the
trades we actually pick are biased upward.

The scanner also proposes maker quotes (resting bids around fair value) for a
future market-making mode; those are logged, not traded.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from .book import Quote
from .fees import FeeModel
from .kalshi import MarketInfo
from .pricing import ModelParams, fair_prob, tau_from_seconds


@dataclass
class Snapshot:
    ts: str
    ticker: str
    event_ticker: str
    strike_type: str
    floor: Optional[float]
    cap: Optional[float]
    secs_to_close: float
    spot: float
    sigma: float
    model: str
    fair_p: float  # raw model probability
    fair_sd: float  # its uncertainty
    fair_used: float  # after optional model/market calibration blend
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    maker_yes_bid: Optional[float]
    maker_no_bid: Optional[float]

    @property
    def market_mid(self) -> Optional[float]:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return 0.5 * (self.yes_bid + self.yes_ask)


@dataclass
class Opportunity:
    ticker: str
    event_ticker: str
    side: str  # "yes" or "no": the side we would buy
    price: float  # ask we would pay for that side
    fair: float  # fair value of that side
    fee_per_contract: float
    edge: float  # model EV per contract after fees, dollars
    available: float  # contracts at that price
    secs_to_close: float
    fair_sd: float = 0.0
    haircut_k: float = 1.0

    @property
    def edge_adj(self) -> float:
        """Edge after the uncertainty haircut; used for decisions."""
        return self.edge - self.haircut_k * self.fair_sd

    @property
    def fair_adj(self) -> float:
        return self.fair - self.haircut_k * self.fair_sd

    @property
    def edge_frac(self) -> float:
        return self.edge_adj / self.price

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(edge_adj=self.edge_adj, edge_frac=self.edge_frac)
        return d


def _floor_cent(x: float) -> float:
    return math.floor(x * 100 + 1e-9) / 100


def maker_quotes(fair: float, margin: float) -> tuple[Optional[float], Optional[float]]:
    """Resting YES bid and NO bid at `margin` inside fair value."""
    yb = _floor_cent(fair - margin)
    nb = _floor_cent((1 - fair) - margin)
    return (yb if 0.01 <= yb <= 0.99 else None, nb if 0.01 <= nb <= 0.99 else None)


def evaluate_prob(
    m: MarketInfo,
    q: Quote,
    spot: float,
    fair_p: float,
    fair_sd: float,
    sigma: float,
    model: str,
    fees: FeeModel,
    now: Optional[datetime] = None,
    fair_used: Optional[float] = None,
    haircut_k: float = 1.0,
    maker_margin: float = 0.03,
) -> tuple[Snapshot, list[Opportunity]]:
    now = now or datetime.now(timezone.utc)
    secs = m.seconds_to_close(now)
    p = fair_p if fair_used is None else fair_used
    myb, mnb = maker_quotes(p, maker_margin)
    snap = Snapshot(
        ts=now.isoformat(),
        ticker=m.ticker,
        event_ticker=m.event_ticker,
        strike_type=m.contract.strike_type,
        floor=m.contract.floor,
        cap=m.contract.cap,
        secs_to_close=secs,
        spot=spot,
        sigma=sigma,
        model=model,
        fair_p=fair_p,
        fair_sd=fair_sd,
        fair_used=p,
        yes_bid=q.yes_bid,
        yes_ask=q.yes_ask,
        maker_yes_bid=myb,
        maker_no_bid=mnb,
    )
    opps = []
    for side, ask, fair, avail in (
        ("yes", q.yes_ask, p, q.yes_ask_size),
        ("no", q.no_ask, 1.0 - p, q.no_ask_size),
    ):
        if ask is None or not 0 < ask < 1:
            continue
        fee = fees.fee_per_contract(ask, taker=True)
        edge = fair - ask - fee
        if edge > 0:
            opps.append(Opportunity(m.ticker, m.event_ticker, side, ask, fair, fee, edge, avail, secs, fair_sd, haircut_k))
    return snap, opps


def evaluate(
    m: MarketInfo,
    q: Quote,
    spot: float,
    params: ModelParams,
    fees: FeeModel,
    now: Optional[datetime] = None,
    maker_margin: float = 0.03,
) -> tuple[Snapshot, list[Opportunity]]:
    """Closed-form pricing (lognormal / Student-t), no uncertainty estimate."""
    now = now or datetime.now(timezone.utc)
    p = fair_prob(m.contract, spot, tau_from_seconds(m.seconds_to_close(now)), params)
    return evaluate_prob(m, q, spot, p, 0.0, params.sigma, params.model, fees, now, maker_margin=maker_margin)


def rank(opps: list[Opportunity]) -> list[Opportunity]:
    return sorted(opps, key=lambda o: o.edge_adj, reverse=True)
