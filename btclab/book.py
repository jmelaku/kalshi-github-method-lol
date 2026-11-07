"""Kalshi order book normalization.

Kalshi's order book endpoint returns bids only, for both sides:

    {"orderbook_fp": {"yes_dollars": [["0.1500", "100.00"], ...],
                      "no_dollars":  [["0.8500", "50.00"], ...]}}

A NO bid at q is the same as a YES offer at 1 - q, so

    best YES ask = 1 - best NO bid
    best NO  ask = 1 - best YES bid

Getting this backwards silently inverts every signal, so all code goes through
`Quote`. Older/alternate response shapes (integer cents, "orderbook" with
"yes"/"no" keys) are also accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def _to_dollars(x: Any) -> float:
    v = float(x)
    return v / 100.0 if v > 1.0 else v  # integer cents -> dollars


@dataclass
class Quote:
    yes_bids: list[tuple[float, float]] = field(default_factory=list)  # (price, size), best first
    no_bids: list[tuple[float, float]] = field(default_factory=list)

    @property
    def yes_bid(self) -> Optional[float]:
        return self.yes_bids[0][0] if self.yes_bids else None

    @property
    def no_bid(self) -> Optional[float]:
        return self.no_bids[0][0] if self.no_bids else None

    @property
    def yes_ask(self) -> Optional[float]:
        return round(1.0 - self.no_bid, 4) if self.no_bid is not None else None

    @property
    def no_ask(self) -> Optional[float]:
        return round(1.0 - self.yes_bid, 4) if self.yes_bid is not None else None

    @property
    def yes_ask_size(self) -> float:
        return self.no_bids[0][1] if self.no_bids else 0.0

    @property
    def no_ask_size(self) -> float:
        return self.yes_bids[0][1] if self.yes_bids else 0.0

    @property
    def mid(self) -> Optional[float]:
        """Midpoint of YES bid and ask. With one side empty, the empty side is
        treated as the boundary ($1 ask or $0 bid), so a deep in- or
        out-of-the-money market still has a reference price."""
        if self.yes_bid is None and self.yes_ask is None:
            return None
        bid = self.yes_bid if self.yes_bid is not None else 0.0
        ask = self.yes_ask if self.yes_ask is not None else 1.0
        return round(0.5 * (bid + ask), 4)

    @property
    def spread(self) -> Optional[float]:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return round(self.yes_ask - self.yes_bid, 4)

    def depth_to_buy_yes(self, max_price: float) -> float:
        """Contracts of YES available at or below max_price."""
        return sum(s for p, s in self.no_bids if 1.0 - p <= max_price + 1e-9)

    def depth_to_buy_no(self, max_price: float) -> float:
        return sum(s for p, s in self.yes_bids if 1.0 - p <= max_price + 1e-9)


def _levels(raw: Any) -> list[tuple[float, float]]:
    out = []
    for lvl in raw or []:
        price, size = lvl[0], lvl[1]
        p = _to_dollars(price)
        s = float(size)
        if 0 < p < 1 and s > 0:
            out.append((p, s))
    out.sort(key=lambda t: t[0], reverse=True)  # best (highest) bid first
    return out


def parse_orderbook(resp: dict) -> Quote:
    """Accept the current `orderbook_fp` shape and older variants."""
    if "orderbook_fp" in resp:
        ob = resp["orderbook_fp"] or {}
        return Quote(_levels(ob.get("yes_dollars")), _levels(ob.get("no_dollars")))
    ob = resp.get("orderbook", resp) or {}
    yes = ob.get("yes_dollars", ob.get("yes"))
    no = ob.get("no_dollars", ob.get("no"))
    return Quote(_levels(yes), _levels(no))


def quote_from_market(m: dict) -> Quote:
    """Top-of-book quote from a market object's *_dollars fields (no depth)."""

    def g(key: str) -> Optional[float]:
        v = m.get(f"{key}_dollars", m.get(key))
        if v in (None, ""):
            return None
        p = _to_dollars(v)
        return p if 0 < p < 1 else None

    yb, ya = g("yes_bid"), g("yes_ask")
    q = Quote()
    if yb is not None:
        q.yes_bids = [(yb, float(m.get("yes_bid_size_fp") or 1))]
    if ya is not None:
        q.no_bids = [(round(1.0 - ya, 4), float(m.get("yes_ask_size_fp") or 1))]
    return q
