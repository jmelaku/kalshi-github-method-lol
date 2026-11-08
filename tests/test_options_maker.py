"""Options-implied pricer, composite spot, and paper maker (resting order) mode."""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from btclab.engine import Engine, EngineConfig
from btclab.fees import FeeModel
from btclab.kalshi import MarketInfo
from btclab.ledger import Ledger
from btclab.marketdata import MarketData
from btclab.options import OptionSurface, expiry_ts, variance_clock
from btclab.pricing import Contract, ModelParams, fair_prob
from btclab.risk import RiskLimits, RiskManager

NOW = datetime(2026, 11, 8, 20, 0, tzinfo=timezone.utc)
SPOT = 80_000.0


def deribit_rows(iv_fn, forward=SPOT, expiries=("9NOV26", "10NOV26", "13NOV26", "27NOV26")):
    """Fake get_book_summary_by_currency rows: calls and puts with mark IV from iv_fn(k)."""
    rows = []
    for code in expiries:
        for strike in range(60_000, 100_001, 1_000):
            iv = 100.0 * iv_fn(math.log(strike / forward))
            for cp in ("C", "P"):
                rows.append({"instrument_name": f"BTC-{code}-{strike}-{cp}", "mark_iv": iv,
                             "underlying_price": forward, "estimated_delivery_price": forward,
                             "bid_price": 0.01, "ask_price": 0.02})
    return rows


def test_expiry_code_parses_to_0800_utc():
    assert expiry_ts("9NOV26") == datetime(2026, 11, 9, 8, 0, tzinfo=timezone.utc).timestamp()
    assert expiry_ts("25DEC26") == datetime(2026, 12, 25, 8, 0, tzinfo=timezone.utc).timestamp()


def test_flat_smile_matches_lognormal():
    """With the same implied vol at every strike, options pricing is plain lognormal."""
    surf = OptionSurface.from_deribit(deribit_rows(lambda k: 0.50), NOW.timestamp())
    for hours in (3, 20, 60):
        t = NOW.timestamp() + hours * 3600
        tau = hours / (365 * 24)
        for c in (Contract("greater", 81_000), Contract("between", 79_500, 79_750), Contract("less", None, 78_000)):
            want = fair_prob(c, SPOT, tau, ModelParams(0.50, "lognormal"))
            got = surf.prob(c, SPOT, t, iv_rel_sd=0.0)
            assert got.p == pytest.approx(want, abs=2e-3)
            assert got.sd == pytest.approx(0.0, abs=1e-3)  # both interpolations agree on a flat smile


def test_put_skew_puts_median_above_forward_and_brackets_partition():
    skew = lambda k: 0.45 - 1.5 * k + 4.0 * k * k  # downside strikes pricier, like BTC
    surf = OptionSurface.from_deribit(deribit_rows(skew), NOW.timestamp())
    t = NOW.timestamp() + 24 * 3600
    assert surf.prob(Contract("greater", SPOT), SPOT, t).p > 0.5
    cache: dict = {}
    edges = np.arange(70_000, 90_001, 250)
    total = sum(surf.prob(Contract("between", a, b), SPOT, t, cache=cache).p for a, b in zip(edges, edges[1:]))
    total += surf.prob(Contract("less", None, 70_000), SPOT, t, cache=cache).p
    total += surf.prob(Contract("greater", 90_000), SPOT, t, cache=cache).p
    assert total == pytest.approx(1.0, abs=1e-6)
    grid, pa = surf.curve(SPOT, t)
    assert np.all(np.diff(pa) <= 1e-12)  # never rises with strike: no negative density


def test_iv_bump_gives_uncertainty():
    surf = OptionSurface.from_deribit(deribit_rows(lambda k: 0.50), NOW.timestamp())
    est = surf.prob(Contract("between", 79_750, 80_000), SPOT, NOW.timestamp() + 6 * 3600, iv_rel_sd=0.10)
    assert 0.002 < est.sd < 0.05


def test_variance_clock_weights_busy_hours():
    season = np.ones(24)
    season[14:21] = 2.0  # US hours twice as volatile
    clock = variance_clock(season)
    day0 = NOW.replace(hour=0).timestamp()
    assert clock(day0, day0 + 86400) == pytest.approx(86400 / (365 * 86400))  # a full day is a full day
    assert clock(day0 + 15 * 3600, day0 + 16 * 3600) > 2 * clock(day0 + 3 * 3600, day0 + 4 * 3600)
    assert variance_clock(None)(day0, day0 + 3600) == pytest.approx(3600 / (365 * 86400))


class FakeSession:
    def __init__(self, fail=()):
        self.fail = set(fail)
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        data = {
            "coinbase": {"bid": "80000", "ask": "80002", "price": "80001"},
            "kraken": {"result": {"XXBTZUSD": {"b": ["80030"], "a": ["80032"], "c": ["80031"]}}},
            "bitstamp": {"bid": "80010", "ask": "80012"},
            "gemini": {"bid": "79990", "ask": "79992"},
        }
        name = next(n for n in data if n in url)
        if name in self.fail:
            raise ConnectionError(name)

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return data[name]

        return R()


def test_composite_spot_is_median_and_falls_back():
    md = MarketData(session=FakeSession(), spot_source="composite")
    assert md.spot() == pytest.approx((80_001 + 80_011) / 2)  # median of 79991, 80001, 80011, 80031
    assert MarketData(session=FakeSession(fail={"kraken", "bitstamp"}), spot_source="composite").spot() == \
        pytest.approx((79_991 + 80_001) / 2)
    lone = MarketData(session=FakeSession(fail={"kraken", "bitstamp", "gemini"}), spot_source="composite")
    assert lone.spot() == pytest.approx(80_001)  # one source left: plain Coinbase
    assert MarketData(session=FakeSession(), spot_source="coinbase").spot() == pytest.approx(80_001)


# ---------- engine: options pricer and resting orders ----------

class Feed:
    def __init__(self, rows):
        self.rows = rows

    def spot(self):
        return SPOT

    def dvol(self):
        return 0.5

    def recent_closes(self, minutes=240):
        return self.recent_candles(minutes)[1]

    def recent_candles(self, minutes=240):
        rng = np.random.default_rng(0)
        n = max(minutes, 600)
        r = rng.normal(0, 0.5 / math.sqrt(525_600), n)
        closes = SPOT * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
        closes *= SPOT / closes[-1]
        t_end = int(NOW.timestamp()) // 60 * 60
        return [t_end - 60 * (n - i) for i in range(n + 1)], closes.tolist()

    def option_summaries(self):
        return self.rows


class Exchange:
    """One bracket market whose top-of-book the test controls."""

    def __init__(self):
        self.contract = Contract("between", 79_750, 80_250)
        self.yes_bid, self.yes_ask = 0.10, 0.40

    def btc_markets(self):
        raw = {"yes_bid_dollars": self.yes_bid, "yes_ask_dollars": self.yes_ask,
               "yes_bid_size_fp": 100, "yes_ask_size_fp": 100}
        return [MarketInfo("KXBTC-X", "EV1", self.contract, NOW + timedelta(hours=3), "active", "", raw)]

    def orderbook(self, ticker, depth=10):
        from btclab.book import quote_from_market

        return quote_from_market(self.btc_markets()[0].raw)

    def market(self, ticker):
        return {}

    def create_order(self, *a, **kw):
        raise AssertionError("paper mode must not send orders")


def engine(tmp_path, ex, **cfg):
    led = Ledger(":memory:")
    risk = RiskManager(RiskLimits(kill_switch_file=str(tmp_path / "STOP"), bankroll_usd=1000))
    c = EngineConfig(pricer="options", basis_usd=0.0, calibrate=False, **cfg)
    return Engine(ex, Feed(deribit_rows(lambda k: 0.50)), led, risk, FeeModel(maker_multiplier=0.0175), c), led


def test_options_pricer_runs_in_engine(tmp_path):
    ex = Exchange()
    eng, led = engine(tmp_path, ex)
    r = eng.cycle(NOW, trade=False)
    assert r.markets_priced == 1 and r.garch["options_expiries"] == 4
    want = fair_prob(ex.contract, SPOT, 3 / (365 * 24), ModelParams(0.50, "lognormal"))
    snap = led.db.execute("SELECT fair_p, model FROM snapshots").fetchone()
    assert snap["model"] == "options"
    assert snap["fair_p"] == pytest.approx(want, abs=0.03)  # seasonality reshapes time a little


def test_resting_bid_fills_only_when_a_seller_reaches_it(tmp_path):
    ex = Exchange()  # YES bid 0.10, ask 0.40: wide market, fair is about 0.5
    eng, led = engine(tmp_path, ex, execution="maker")
    r = eng.cycle(NOW)
    assert r.maker["placed"] == 1 and r.trades == []
    o = led.resting_orders()[0]
    assert o["side"] == "yes" and o["price"] == pytest.approx(0.11) and o["improved"] == 1  # one tick above the bid

    ex.yes_ask = 0.30  # sellers come down, but not to our bid
    r = eng.cycle(NOW + timedelta(minutes=1))
    assert r.maker["filled"] == 0 and len(led.resting_orders()) == 1

    ex.yes_ask = 0.11  # a seller at our price: we improved the bid, so we'd be first in line
    r = eng.cycle(NOW + timedelta(minutes=2))
    assert r.maker["filled"] == 1
    t = led.db.execute("SELECT * FROM trades").fetchone()
    assert t["liquidity"] == "maker" and t["price"] == pytest.approx(0.11)
    assert t["fee"] == pytest.approx(FeeModel(maker_multiplier=0.0175).maker_fee(0.11, t["count"]))
    assert led.order_stats()["orders_filled"] == 1


def test_resting_bid_cancelled_when_fair_value_moves(tmp_path):
    ex = Exchange()
    eng, led = engine(tmp_path, ex, execution="maker")
    eng.cycle(NOW)
    assert len(led.resting_orders()) == 1
    ex.contract = Contract("between", 89_750, 90_250)  # market now far from spot: YES is worth ~0
    eng.cycle(NOW + timedelta(minutes=1))
    reasons = [r["reason"] for r in led.db.execute("SELECT reason FROM orders WHERE status='cancelled'")]
    assert any("edge below threshold" in x for x in reasons)


def test_resting_orders_count_toward_risk_limits(tmp_path):
    ex = Exchange()
    eng, led = engine(tmp_path, ex, execution="maker")
    eng.cycle(NOW)
    o = led.resting_orders()[0]
    assert led.resting_exposure() == pytest.approx(o["price"] * o["count"])
    assert led.resting_exposure("EV1") == pytest.approx(o["price"] * o["count"])
    assert led.resting_contracts("KXBTC-X") == o["count"]


def test_maker_mode_is_paper_only(tmp_path):
    with pytest.raises(ValueError):
        engine(tmp_path, Exchange(), execution="both", mode="live")


def test_resting_budget_caps_total_resting_cost(tmp_path):
    ex = Exchange()
    eng, led = engine(tmp_path, ex, execution="maker", maker_max_exposure_usd=0.5)
    eng.cycle(NOW)
    assert led.resting_exposure() <= 0.5 + 1e-9
    o = led.resting_orders()[0]
    assert o["count"] == int(0.5 / o["price"])


def test_compare_uses_common_start_and_same_markets(tmp_path):
    from btclab.compare import compare

    a, b = Ledger(str(tmp_path / "a.db")), Ledger(str(tmp_path / "b.db"))
    snap = "INSERT INTO snapshots (ts,ticker,fair_p,yes_bid,yes_ask,secs_to_close) VALUES (?,?,?,?,?,?)"
    a.db.execute(snap, ("2026-11-07T00:00:00+00:00", "OLD", 0.9, 0.4, 0.5, 3600))  # before b existed
    for led, p in ((a, 0.6), (b, 0.8)):
        led.db.execute(snap, ("2026-11-08T00:00:00+00:00", "M1", p, 0.4, 0.5, 3600))
        led.db.execute(snap, ("2026-11-08T00:00:00+00:00", "M2", p, 0.4, 0.5, 3600))
        led.db.commit()
        led.set_result("M1", "yes")
        led.set_result("M2", "yes")
    a.set_result("OLD", "no")
    since, sa, sb = compare(str(tmp_path / "a.db"), str(tmp_path / "b.db"))
    assert since.startswith("2026-11-08")
    assert sa["accuracy_points"] == sb["accuracy_points"] == 2  # OLD excluded from both
    assert sb["logloss_model"] < sa["logloss_model"]  # 0.8 on YES outcomes beats 0.6
    assert sb["edge_over_market"] > sa["edge_over_market"] > 0
