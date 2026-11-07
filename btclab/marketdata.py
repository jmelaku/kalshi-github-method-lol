"""Free BTC market data: spot and 1-minute candles (Coinbase), implied vol (Deribit DVOL).

None of these need an account or API key.

Note: Kalshi settles on the CF Benchmarks BRTI index, which aggregates several
exchanges. Coinbase spot is a close proxy, not the settlement price; the
pricing model's `basis_usd` covers the difference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

COINBASE = "https://api.exchange.coinbase.com"
KRAKEN = "https://api.kraken.com/0/public"
DERIBIT = "https://www.deribit.com/api/v2/public"


@dataclass
class Candle:
    ts: int  # unix seconds, bar open
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketData:
    def __init__(self, session: Optional[requests.Session] = None, timeout_s: float = 10.0):
        self.s = session or requests.Session()
        self.s.headers.setdefault("User-Agent", "btclab/0.1")
        self.timeout_s = timeout_s

    def _get(self, url: str, params: Optional[dict] = None):
        r = self.s.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def spot(self) -> float:
        """Last BTC-USD price; falls back to Kraken if Coinbase fails."""
        try:
            return float(self._get(f"{COINBASE}/products/BTC-USD/ticker")["price"])
        except Exception:
            data = self._get(f"{KRAKEN}/Ticker", {"pair": "XBTUSD"})
            return float(next(iter(data["result"].values()))["c"][0])

    def candles(self, start_ts: int, end_ts: int, granularity: int = 60) -> list[Candle]:
        """Coinbase candles between start and end (unix seconds), oldest first.

        Coinbase returns at most 300 bars per call, so we page.
        """
        out: dict[int, Candle] = {}
        step = 300 * granularity
        t = start_ts
        while t < end_ts:
            t2 = min(t + step, end_ts)
            rows = self._get(
                f"{COINBASE}/products/BTC-USD/candles",
                {"granularity": granularity, "start": _iso(t), "end": _iso(t2)},
            )
            for ts, lo, hi, op, cl, vol in rows:
                out[int(ts)] = Candle(int(ts), op, hi, lo, cl, vol)
            t = t2
            time.sleep(0.15)  # stay well under public rate limits
        return [out[k] for k in sorted(out)]

    def recent_closes(self, minutes: int = 240) -> list[float]:
        return self.recent_candles(minutes)[1]

    def recent_candles(self, minutes: int = 240) -> tuple[list[int], list[float]]:
        """(timestamps, closes) of the last `minutes` 1-minute bars, oldest first.

        Bars missing from Coinbase (no trades that minute) are forward-filled so
        the series is evenly spaced, which the GARCH model assumes.
        """
        now = int(time.time()) // 60 * 60
        bars = self.candles(now - minutes * 60, now, 60)
        if not bars:
            return [], []
        ts, closes = [], []
        last = bars[0].close
        by_ts = {c.ts: c.close for c in bars}
        for t in range(bars[0].ts, bars[-1].ts + 60, 60):
            last = by_ts.get(t, last)
            ts.append(t)
            closes.append(last)
        return ts, closes

    def dvol(self) -> Optional[float]:
        """Latest Deribit DVOL (30-day BTC implied vol) as a decimal, e.g. 0.52."""
        now_ms = int(time.time() * 1000)
        data = self._get(
            f"{DERIBIT}/get_volatility_index_data",
            {"currency": "BTC", "start_timestamp": now_ms - 3 * 3600_000, "end_timestamp": now_ms, "resolution": "60"},
        )
        rows = (data.get("result") or {}).get("data") or []
        return float(rows[-1][4]) / 100.0 if rows else None


def _iso(ts: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
