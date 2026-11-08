from datetime import datetime, timedelta, timezone

import pytest

from btclab.book import Quote
from btclab.fees import FeeModel
from btclab.kalshi import MarketInfo
from btclab.ledger import Ledger
from btclab.pricing import Contract, ModelParams
from btclab.risk import Book, RiskLimits, RiskManager
from btclab.scanner import Opportunity, evaluate
from btclab.sim import run_simulation

NOW = datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc)


def mk(ticker="T", st="greater", floor=80_000.0, cap=None, mins=60):
    return MarketInfo(ticker, "EV", Contract(st, floor, cap), NOW + timedelta(minutes=mins), "active", "", {})


def opp(**kw):
    d = dict(ticker="T", event_ticker="EV", side="yes", price=0.40, fair=0.50, fee_per_contract=0.02,
             edge=0.08, available=100, secs_to_close=3600)
    d.update(kw)
    return Opportunity(**d)


def test_evaluate_finds_cheap_yes():
    q = Quote(yes_bids=[(0.30, 50)], no_bids=[(0.65, 40)])  # YES ask = 0.35
    snap, opps = evaluate(mk(), q, 80_000, ModelParams(0.5, "lognormal"), FeeModel(), NOW)
    assert snap.fair_p == pytest.approx(0.5, abs=0.01)
    assert [o.side for o in opps] == ["yes"]
    o = opps[0]
    assert o.price == 0.35 and o.available == 40
    assert o.edge == pytest.approx(snap.fair_p - 0.35 - FeeModel().fee_per_contract(0.35, True))


def test_evaluate_no_edge_in_fair_market():
    q = Quote(yes_bids=[(0.48, 50)], no_bids=[(0.48, 50)])
    _, opps = evaluate(mk(), q, 80_000, ModelParams(0.5, "lognormal"), FeeModel(), NOW)
    assert opps == []


def test_risk_blocks_and_sizes(tmp_path):
    rm = RiskManager(RiskLimits(kill_switch_file=str(tmp_path / "STOP")))
    empty = Book(0, 0, 0)
    n, why = rm.size(opp(), empty)
    assert n > 0 and why == "ok"
    assert rm.size(opp(edge=0.005), empty)[0] == 0
    assert rm.size(opp(secs_to_close=60), empty)[0] == 0
    assert rm.size(opp(price=0.99), empty)[0] == 0
    assert rm.size(opp(), Book(0, 0, -60))[1].startswith("daily loss")
    assert rm.size(opp(), Book(50, 0, 0))[0] == 0  # market cap reached
    assert rm.size(opp(available=2), empty)[0] <= 2
    (tmp_path / "STOP").write_text("")
    assert rm.size(opp(), empty)[1].startswith("kill switch")


def test_ledger_settlement_pnl():
    led = Ledger(":memory:")
    led.add_trade(opp(side="yes", price=0.40), 10, 0.40, 0.17, "paper")
    led.add_trade(opp(side="no", price=0.55), 4, 0.55, 0.07, "paper")
    assert led.exposure() == pytest.approx(10 * 0.40 + 0.17 + 4 * 0.55 + 0.07)
    led.set_result("T", "yes", NOW.isoformat())
    r = led.report()
    # YES wins: +0.60*10 - 0.17 ; NO loses: -0.55*4 - 0.07
    assert r["realized_pnl"] == pytest.approx(6.0 - 0.17 - 2.2 - 0.07, abs=0.01)
    assert r["settled"] == 2 and led.exposure() == 0


def test_simulation_end_to_end():
    from btclab.engine import EngineConfig

    led, s = run_simulation(hours=6, seed=1, step_min=10, cfg=EngineConfig(basis_usd=0.0, mc_seed=1, n_paths=4000))
    r = led.report()
    assert s["cycles"] == 36
    assert r["trades"] == r["settled"]  # everything settles
    assert r["calibration_points"] > 0
    assert "pnl_gap_z" in r or r["events"] < 2


def test_haircut_reduces_decision_edge():
    o = opp(edge=0.05, fair_sd=0.02, haircut_k=1.5)
    assert o.edge_adj == pytest.approx(0.02)
    assert o.fair_adj == pytest.approx(0.47)


def test_pnl_check_and_settlement_scope():
    from btclab.scanner import Snapshot

    led = Ledger(":memory:")
    snap = lambda t, secs: Snapshot(NOW.isoformat(), t, "EV", "greater", 1.0, None, secs, 1, 0.5, "fhs", 0.5, 0.0, 0.5,
                                    0.4, 0.6, None, None)
    led.add_snapshots([snap("PAST", 60), snap("FUTURE", 7200)])
    assert led.unsettled_tickers(NOW + timedelta(minutes=5)) == ["PAST"]  # only finished markets are checked
    for i in range(20):
        o = opp(ticker=f"T{i}", event_ticker=f"E{i}", fair=0.5, price=0.40)
        led.add_trade(o, 10, 0.40, 0.0, "paper")
        led.set_result(f"T{i}", "yes" if i % 2 else "no")
    chk = led.pnl_check()
    assert chk["events"] == 20
    # Expected +0.10*10 per trade = $20 total; realized: 10 wins of +6, 10 losses of -4 = +20.
    assert chk["pnl_minus_expected"] == pytest.approx(0.0, abs=1e-6)


def test_current_pnl_marks_open_trades_at_bid():
    from btclab.ledger import Ledger

    led = Ledger(":memory:")
    ins = "INSERT INTO trades (ts,mode,ticker,event_ticker,side,price,count,fee,status,pnl) VALUES (?,?,?,?,?,?,?,?,?,?)"
    led.db.execute(ins, ("t", "paper", "A", "E", "yes", 0.30, 10, 0.10, "settled", 6.90))
    led.db.execute(ins, ("t", "paper", "B", "E", "yes", 0.20, 10, 0.10, "open", None))  # bid 0.25
    led.db.execute(ins, ("t", "paper", "C", "E", "no", 0.40, 5, 0.05, "open", None))   # NO bid = 1 - 0.70
    led.db.execute(ins, ("t", "paper", "D", "E", "yes", 0.50, 2, 0.02, "open", None))  # no snapshot
    snap = "INSERT INTO snapshots (ts,ticker,yes_bid,yes_ask) VALUES (?,?,?,?)"
    led.db.execute(snap, ("2026-01-01T00:00:00+00:00", "B", 0.10, 0.15))  # older quote, ignored
    led.db.execute(snap, ("2026-01-01T00:05:00+00:00", "B", 0.25, 0.27))
    led.db.execute(snap, ("2026-01-01T00:05:00+00:00", "C", 0.68, 0.70))
    c = led.current_pnl()
    assert c["realized_pnl"] == 6.90 and c["settled_trades"] == 1 and c["open_trades"] == 3
    assert c["open_cost"] == round(2.10 + 2.05 + 1.02, 2)
    assert c["open_value"] == round(2.50 + 1.50 + 1.00, 2)
    assert c["unrealized_pnl"] == round(5.00 - 5.17, 2)
    assert c["current_pnl"] == round(6.90 - 0.17, 2)
    assert c["unpriced_open_trades"] == 1
    assert led.current_pnl("live")["open_trades"] == 0


def test_paper_cannot_rebuy_contracts_it_already_took(tmp_path):
    """Kalshi keeps showing a seller's size after a paper buy; paper must treat it as used up."""
    from btclab.engine import Engine, EngineConfig

    class Ex:
        def btc_markets(self):
            raw = {"yes_bid_dollars": 0.20, "yes_ask_dollars": 0.30, "yes_bid_size_fp": 50, "yes_ask_size_fp": 6}
            return [MarketInfo("KXBTCD-A", "EV", Contract("greater", 80_000.0), NOW + timedelta(hours=2),
                               "active", "", raw)]

        def orderbook(self, ticker, depth=10):
            return Quote(yes_bids=[(0.20, 50)], no_bids=[(0.70, 6)])  # 6 YES offered at 0.30

        def market(self, ticker):
            return {}

    from tests.test_live_wiring import FlatFeed

    def run(deplete):
        led = Ledger(":memory:")
        risk = RiskManager(RiskLimits(kill_switch_file=str(tmp_path / "STOP"), bankroll_usd=10_000))
        cfg = EngineConfig(mc_seed=1, n_paths=4_000, basis_usd=0.0, evidence_classes=[], calibrate=False,
                           paper_deplete_book=deplete)
        eng = Engine(Ex(), FlatFeed(), led, risk, cfg=cfg)
        for i in range(3):
            eng.cycle(NOW + timedelta(seconds=i))
        return led.market_contracts("KXBTCD-A")

    assert run(True) == 6  # only what the seller actually offered
    assert run(False) > 6  # the old behavior re-bought the same 6 every cycle
