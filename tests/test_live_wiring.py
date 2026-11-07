"""Run the real KalshiClient through the engine against canned API responses."""

from datetime import datetime, timedelta, timezone

from btclab.engine import Engine, EngineConfig
from btclab.kalshi import KalshiClient
from btclab.ledger import Ledger
from btclab.risk import RiskLimits, RiskManager

NOW = datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc)
CLOSE = (NOW + timedelta(hours=2)).isoformat().replace("+00:00", "Z")


class Resp:
    def __init__(self, data, code=200):
        self._d, self.status_code = data, code
        self.content = b"x"
        self.text = str(data)

    def json(self):
        return self._d


class FakeKalshiHTTP:
    """Serves GET /markets (paged), /markets/{t}/orderbook and /markets/{t}."""

    def __init__(self):
        self.paths = []
        self.markets = [
            # "BTC above 80,000" priced at 0.20 YES ask while spot is 80,000: very cheap.
            {"ticker": "KXBTCD-A", "event_ticker": "KXBTCD-E", "strike_type": "greater", "floor_strike": 80000,
             "close_time": CLOSE, "status": "active", "yes_bid_dollars": "0.1800", "yes_ask_dollars": "0.2000",
             "yes_ask_size_fp": "30.00", "yes_bid_size_fp": "30.00"},
            # A fairly priced far bracket.
            {"ticker": "KXBTC-B", "event_ticker": "KXBTC-E", "strike_type": "between", "floor_strike": 90000,
             "cap_strike": 90249.99, "close_time": CLOSE, "status": "active",
             "yes_bid_dollars": "", "yes_ask_dollars": "0.0100"},
            {"ticker": "WEIRD", "strike_type": "custom", "close_time": CLOSE},
        ]

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        path = url.split("/trade-api/v2", 1)[1]
        self.paths.append(path)
        if path == "/markets" and params.get("tickers"):
            # batched settlement lookup
            return Resp({"markets": [{"ticker": t, "result": "yes"} for t in params["tickers"].split(",")], "cursor": ""})
        if path == "/markets":
            series = params.get("series_ticker")
            ms = [m for m in self.markets if m["ticker"].startswith(series + "-") or (series == "KXBTC" and m["ticker"] == "WEIRD")]
            if params.get("cursor") is None:
                return Resp({"markets": ms[:1], "cursor": "p2" if len(ms) > 1 else ""})
            return Resp({"markets": ms[1:], "cursor": ""})
        if path.endswith("/orderbook"):
            return Resp({"orderbook_fp": {"yes_dollars": [["0.1800", "30.00"]], "no_dollars": [["0.8000", "25.00"]]}})
        if path.startswith("/markets/"):
            return Resp({"market": {"ticker": path.split("/")[2], "result": "yes"}})
        return Resp({}, 404)


import math


class FlatFeed:
    def spot(self):
        return 80_000.0

    def recent_closes(self, minutes=240):
        import math

        return [80_000 * math.exp(0.0005 * ((-1) ** i)) for i in range(minutes)]

    def recent_candles(self, minutes=240):
        import numpy as np

        rng = np.random.default_rng(0)
        n = max(minutes, 600)
        r = rng.normal(0, 0.5 / math.sqrt(525_600), n)
        closes = 80_000 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
        closes *= 80_000 / closes[-1]
        t_end = int(NOW.timestamp()) // 60 * 60
        ts = [t_end - 60 * (n - i) for i in range(n + 1)]
        return ts, closes.tolist()

    def dvol(self):
        return 0.5


def test_engine_with_real_client_and_canned_api(tmp_path):
    http = FakeKalshiHTTP()
    client = KalshiClient(session=http, min_interval_s=0)
    led = Ledger(":memory:")
    risk = RiskManager(RiskLimits(kill_switch_file=str(tmp_path / "STOP")))
    # Plumbing test: let the engine trade the above/below contract without the evidence gate.
    eng = Engine(client, FlatFeed(), led, risk, cfg=EngineConfig(mc_seed=1, n_paths=10_000, basis_usd=0.0, evidence_classes=[]))

    r = eng.cycle(NOW)
    assert r.markets_priced == 2  # custom strike type skipped, pagination followed
    assert r.books_fetched == 1  # only the mispriced market's book is fetched
    assert "/markets/KXBTCD-A/orderbook" in http.paths
    assert r.opportunities[0].ticker == "KXBTCD-A" and r.opportunities[0].side == "yes"
    assert len(r.trades) == 1 and r.trades[0]["count"] > 0

    assert eng.settle(NOW + timedelta(hours=3)) >= 1
    rep = led.report()
    assert rep["settled"] == 1 and rep["realized_pnl"] > 0


def test_threshold_contracts_need_evidence_by_default(tmp_path):
    client = KalshiClient(session=FakeKalshiHTTP(), min_interval_s=0)
    risk = RiskManager(RiskLimits(kill_switch_file=str(tmp_path / "STOP")))
    eng = Engine(client, FlatFeed(), Ledger(":memory:"), risk, cfg=EngineConfig(mc_seed=1, n_paths=10_000, basis_usd=0.0))
    r = eng.cycle(NOW)
    assert r.trades == []  # the mispriced above/below contract is not traded without proof
