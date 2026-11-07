"""Kalshi REST client.

Authentication (Kalshi API keys):
  message   = timestamp_ms + METHOD + path      (path WITHOUT the query string)
  signature = base64(RSA-PSS-SHA256(message)),  MGF1(SHA256), salt = digest length
  headers   = KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, KALSHI-ACCESS-TIMESTAMP

Orders use the V2 endpoint POST /portfolio/events/orders, which trades the YES
leg only: side="bid" buys YES, side="ask" sells YES. Buying NO at price q is
expressed as selling YES at 1 - q. Verify this with a 1-contract order before
trading size; see README "Going live".

Environment variables:
  KALSHI_API_KEY_ID        key id (uuid)
  KALSHI_PRIVATE_KEY_PATH  path to the PEM private key
  KALSHI_BASE_URL          defaults to production
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from .book import Quote, parse_orderbook
from .pricing import Contract

PROD_URL = "https://external-api.kalshi.com/trade-api/v2"
DEMO_URL = "https://external-api.demo.kalshi.co/trade-api/v2"

BTC_SERIES = ("KXBTC", "KXBTCD")  # range brackets, above/below


class KalshiError(RuntimeError):
    pass


def load_private_key(path: str) -> rsa.RSAPrivateKey:
    with open(path, "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise KalshiError("expected an RSA private key")
    return key


def sign(private_key: rsa.RSAPrivateKey, timestamp_ms: str, method: str, path: str) -> str:
    path = path.split("?", 1)[0]
    msg = f"{timestamp_ms}{method.upper()}{path}".encode()
    sig = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


@dataclass
class MarketInfo:
    ticker: str
    event_ticker: str
    contract: Contract
    close_time: datetime
    status: str
    result: str
    raw: dict

    def seconds_to_close(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (self.close_time - now).total_seconds()


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_market(m: dict) -> Optional[MarketInfo]:
    """Turn a Kalshi market object into MarketInfo; None if we can't price it."""
    st = m.get("strike_type")
    if st not in ("greater", "greater_or_equal", "less", "less_or_equal", "between"):
        return None
    try:
        floor = float(m["floor_strike"]) if m.get("floor_strike") is not None else None
        cap = float(m["cap_strike"]) if m.get("cap_strike") is not None else None
        contract = Contract(st, floor, cap)
        contract.bounds()
        close = parse_time(m["close_time"])
    except (KeyError, ValueError, TypeError):
        return None
    return MarketInfo(
        ticker=m["ticker"],
        event_ticker=m.get("event_ticker", ""),
        contract=contract,
        close_time=close,
        status=m.get("status", ""),
        result=m.get("result", "") or "",
        raw=m,
    )


class KalshiClient:
    def __init__(
        self,
        base_url: str = PROD_URL,
        key_id: Optional[str] = None,
        private_key: Optional[rsa.RSAPrivateKey] = None,
        min_interval_s: float = 0.12,
        timeout_s: float = 10.0,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id
        self.private_key = private_key
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self._last = 0.0

    @classmethod
    def from_env(cls) -> "KalshiClient":
        key_id = os.environ.get("KALSHI_API_KEY_ID")
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        pk = load_private_key(key_path) if key_path else None
        return cls(os.environ.get("KALSHI_BASE_URL", PROD_URL), key_id, pk)

    @property
    def authenticated(self) -> bool:
        return bool(self.key_id and self.private_key)

    def _path_prefix(self) -> str:
        # "https://host/trade-api/v2" -> "/trade-api/v2"
        return "/" + self.base_url.split("://", 1)[1].split("/", 1)[1]

    def _headers(self, method: str, path: str) -> dict:
        h = {"Accept": "application/json"}
        if self.authenticated:
            ts = str(int(time.time() * 1000))
            h["KALSHI-ACCESS-KEY"] = self.key_id  # type: ignore[assignment]
            h["KALSHI-ACCESS-TIMESTAMP"] = ts
            h["KALSHI-ACCESS-SIGNATURE"] = sign(self.private_key, ts, method, self._path_prefix() + path)  # type: ignore[arg-type]
        return h

    def request(self, method: str, path: str, params: Optional[dict] = None, body: Optional[dict] = None) -> dict:
        for attempt in range(5):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            resp = self.session.request(
                method,
                self.base_url + path,
                params={k: v for k, v in (params or {}).items() if v is not None},
                json=body,
                headers=self._headers(method, path),
                timeout=self.timeout_s,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(min(8.0, 0.5 * 2**attempt))
                continue
            if resp.status_code >= 400:
                raise KalshiError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
            return resp.json() if resp.content else {}
        raise KalshiError(f"{method} {path}: gave up after retries")

    # ---- market data ----
    def iter_markets(self, **params: Any) -> Iterator[dict]:
        cursor = None
        while True:
            data = self.request("GET", "/markets", {**params, "limit": 1000, "cursor": cursor})
            yield from data.get("markets", [])
            cursor = data.get("cursor")
            if not cursor:
                return

    def btc_markets(self, series: tuple[str, ...] = BTC_SERIES, status: str = "open") -> list[MarketInfo]:
        out = []
        for s in series:
            for m in self.iter_markets(series_ticker=s, status=status):
                mi = parse_market(m)
                if mi:
                    out.append(mi)
        return out

    def market(self, ticker: str) -> dict:
        return self.request("GET", f"/markets/{ticker}").get("market", {})

    def results(self, tickers: list[str], batch: int = 100) -> dict[str, str]:
        """Settlement result per ticker ('yes', 'no' or ''), 100 tickers per request."""
        out: dict[str, str] = {}
        for i in range(0, len(tickers), batch):
            chunk = tickers[i : i + batch]
            for m in self.iter_markets(tickers=",".join(chunk)):
                out[m["ticker"]] = m.get("result") or ""
        return out

    def orderbook(self, ticker: str, depth: int = 10) -> Quote:
        return parse_orderbook(self.request("GET", f"/markets/{ticker}/orderbook", {"depth": depth}))

    # ---- account ----
    def balance(self) -> dict:
        return self.request("GET", "/portfolio/balance")

    def positions(self) -> dict:
        return self.request("GET", "/portfolio/positions")

    # ---- orders (V2) ----
    def create_order(
        self,
        ticker: str,
        buy: str,  # "yes" or "no"
        count: int,
        price: float,  # price of the side you are buying, in dollars
        time_in_force: str = "immediate_or_cancel",
        post_only: bool = False,
    ) -> dict:
        if buy not in ("yes", "no"):
            raise ValueError("buy must be 'yes' or 'no'")
        if not 0.01 <= price <= 0.99:
            raise ValueError("price must be between 0.01 and 0.99")
        side, yes_price = ("bid", price) if buy == "yes" else ("ask", round(1.0 - price, 4))
        body = {
            "ticker": ticker,
            "client_order_id": str(uuid.uuid4()),
            "side": side,
            "count": f"{int(count)}",
            "price": f"{yes_price:.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": post_only,
        }
        return self.request("POST", "/portfolio/events/orders", body=body)
