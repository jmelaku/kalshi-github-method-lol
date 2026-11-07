"""Position sizing and hard risk limits.

Every order passes through `RiskManager.size`, which returns 0 contracts (and
a reason) whenever any limit would be breached. The same checks apply in paper
and live mode, so paper results reflect what live trading would have done.

Sizing uses fractional Kelly on the haircut probability. Buying a contract at
cost c (price + fee) that pays $1 with probability q has Kelly fraction
(q - c) / (1 - c) of bankroll.
Full Kelly is far too aggressive when q is itself an estimate, so we use a
small fraction (default 10%) and hard caps on top.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .scanner import Opportunity


@dataclass
class RiskLimits:
    bankroll_usd: float = 500.0
    kelly_fraction: float = 0.10
    min_edge_usd: float = 0.02  # EV per contract after fees
    min_edge_frac: float = 0.05  # EV relative to price paid
    min_price: float = 0.03  # skip extreme longshots/near-certainties
    max_price: float = 0.97
    min_secs_to_close: float = 300  # no trading in the last 5 minutes
    max_secs_to_close: float = 36 * 3600
    max_contracts_per_order: int = 25
    max_contracts_per_market: int = 50
    max_exposure_usd: float = 200.0  # total cost of open positions
    max_event_exposure_usd: float = 60.0  # cost of positions in one event (they share one price outcome)
    max_daily_loss_usd: float = 50.0
    kill_switch_file: str = "STOP"  # create this file to halt all trading


@dataclass
class Book:
    """Current state the risk manager needs."""

    market_contracts: float  # contracts already held in this market
    exposure_usd: float  # cost basis of all open positions
    daily_pnl_usd: float  # realized P&L today (negative = loss)
    event_exposure_usd: float = 0.0  # cost basis of open positions in this market's event


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.l = limits

    def halted(self, state: Book) -> str:
        if os.path.exists(self.l.kill_switch_file):
            return f"kill switch file '{self.l.kill_switch_file}' present"
        if state.daily_pnl_usd <= -self.l.max_daily_loss_usd:
            return f"daily loss limit hit ({state.daily_pnl_usd:.2f})"
        return ""

    def size(self, o: Opportunity, state: Book) -> tuple[int, str]:
        l = self.l
        why = self.halted(state)
        if why:
            return 0, why
        if not l.min_price <= o.price <= l.max_price:
            return 0, "price outside allowed band"
        if not l.min_secs_to_close <= o.secs_to_close <= l.max_secs_to_close:
            return 0, "outside trading window"
        # Decisions use the uncertainty-haircut edge, not the raw model edge.
        if o.edge_adj < l.min_edge_usd or o.edge_frac < l.min_edge_frac:
            return 0, "edge below threshold"

        cost = o.price + o.fee_per_contract
        kelly = max(0.0, (o.fair_adj - cost) / (1.0 - cost)) if cost < 1 else 0.0
        n = math.floor(l.kelly_fraction * kelly * l.bankroll_usd / cost)
        n = min(
            n,
            l.max_contracts_per_order,
            int(l.max_contracts_per_market - state.market_contracts),
            int((l.max_exposure_usd - state.exposure_usd) / cost),
            int((l.max_event_exposure_usd - state.event_exposure_usd) / cost),
            int(o.available),
        )
        if n <= 0:
            return 0, "size limits leave no room"
        return n, "ok"
