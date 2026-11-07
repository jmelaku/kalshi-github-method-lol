"""Kalshi fee model.

General trading fee (Kalshi fee schedule):

    fee = round_up(multiplier * C * P * (1 - P))

where C is the number of contracts and P the price in dollars. The standard
taker multiplier is 0.07. Resting (maker) orders pay nothing unless the series
charges maker fees, in which case the multiplier is 0.0175.

Kalshi computes the model fee, rounds it up to $0.000001, and then aligns the
balance change to the member's precision grid ($0.0001 for direct members,
$0.01 for accounts cleared through an FCM). We default to $0.01 rounding,
which is the conservative choice for estimating edge.

Published check: 100 contracts at $0.55 -> 0.07*100*0.55*0.45 = 1.7325 -> $1.74.

Fees differ by series (Kalshi lists many non-standard series). Always compare
against `fee_cost` on real fills and adjust `FeeModel` if needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TAKER_MULTIPLIER = 0.07
MAKER_MULTIPLIER = 0.0175


def _round_up(x: float, step: float) -> float:
    # The small epsilon guards against float noise like 1.7300000000000002.
    return math.ceil(x / step - 1e-9) * step


def raw_fee(price: float, contracts: float, multiplier: float) -> float:
    """Unrounded model fee in dollars."""
    if not 0.0 < price < 1.0:
        raise ValueError(f"price must be in (0, 1), got {price}")
    if contracts < 0:
        raise ValueError("contracts must be non-negative")
    return multiplier * contracts * price * (1.0 - price)


@dataclass(frozen=True)
class FeeModel:
    taker_multiplier: float = TAKER_MULTIPLIER
    maker_multiplier: float = 0.0  # set to MAKER_MULTIPLIER for series with maker fees
    precision: float = 0.01  # 0.0001 for direct members

    def taker_fee(self, price: float, contracts: float) -> float:
        return round(_round_up(raw_fee(price, contracts, self.taker_multiplier), self.precision), 6)

    def maker_fee(self, price: float, contracts: float) -> float:
        if self.maker_multiplier == 0:
            return 0.0
        return round(_round_up(raw_fee(price, contracts, self.maker_multiplier), self.precision), 6)

    def fee(self, price: float, contracts: float, taker: bool) -> float:
        return self.taker_fee(price, contracts) if taker else self.maker_fee(price, contracts)

    def fee_per_contract(self, price: float, taker: bool, lot: int = 100) -> float:
        """Average fee per contract for a typical lot (rounding spread over the lot)."""
        return self.fee(price, lot, taker) / lot
