"""Command line interface.

  python -m btclab price    --spot 80000 --strike 81000 --hours 3 --vol 0.5
  python -m btclab validate --days 14          (real Coinbase candles)
  python -m btclab validate --synthetic        (offline)
  python -m btclab demo                        (full offline simulation)
  python -m btclab scan                        (live prices, no trading)
  python -m btclab run --interval 60           (paper trading loop)
  python -m btclab run --live                  (real orders; see README)
  python -m btclab settle
  python -m btclab report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone

from .engine import Engine, EngineConfig
from .fees import FeeModel
from .ledger import Ledger
from .pricing import Contract, ModelParams, fair_prob
from .risk import RiskLimits, RiskManager

DEFAULT_DB = "data/btclab.db"


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so no extra dependency is needed."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _subset(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise SystemExit(f"unknown {cls.__name__} keys in config: {sorted(unknown)}")
    return cls(**d)


def load_config(path: str) -> tuple[RiskLimits, EngineConfig, FeeModel]:
    if not os.path.exists(path):
        return RiskLimits(), EngineConfig(), FeeModel()
    cfg = json.load(open(path))
    return (
        _subset(RiskLimits, cfg.get("risk", {})),
        _subset(EngineConfig, cfg.get("engine", {})),
        _subset(FeeModel, cfg.get("fees", {})),
    )


def _live_stack(args, mode: str):
    from .kalshi import KalshiClient
    from .marketdata import MarketData

    limits, ecfg, fees = load_config(args.config)
    ecfg.mode = mode
    client = KalshiClient.from_env()
    return Engine(client, MarketData(), Ledger(args.db), RiskManager(limits), fees, ecfg), client


def cmd_price(a) -> None:
    hours = a.hours
    if a.cap is not None and a.strike is not None:
        c = Contract("between", a.strike, a.cap)
    elif a.below:
        c = Contract("less", None, a.strike)
    else:
        c = Contract("greater", a.strike, None)
    tau = hours / (365 * 24)
    for model in ("lognormal", "student_t"):
        p = fair_prob(c, a.spot, tau, ModelParams(a.vol, model, a.nu))
        print(f"{model:<10} P(YES) = {p:.4f}   fair YES = ${p:.2f}   fair NO = ${1 - p:.2f}")


def _print_cycle(r) -> None:
    iv = f"{r.sigma_implied:.1%}" if r.sigma_implied else "n/a"
    print(f"spot ${r.spot:,.2f}  vol now {r.sigma_realized:.1%}  implied {iv}  "
          f"markets {r.markets_priced}  books fetched {r.books_fetched}")
    if r.garch:
        g = r.garch
        print(f"  GARCH: alpha {g['alpha']}  beta {g['beta']}  tail nu {g['nu']}  "
              f"vol half-life {g['halflife_min']:.0f}m  long-run vol {g['vol_longrun']:.1%}")
    if r.blend is not None:
        b = r.blend
        src = f"learned from {b['rows']} settled snapshots in {b['events']} events" if b["rows"] else "prior only (no settled data yet)"
        print(f"  shrinkage toward market: {src}")
    if not r.opportunities:
        print("  no positive-edge opportunities after fees")
    for o in r.opportunities[:10]:
        print(f"  {o.ticker:<32} buy {o.side.upper():<3} @ {o.price:.2f}  fair {o.fair:.3f}±{o.fair_sd:.3f}  "
              f"edge {o.edge:+.3f} -> {o.edge_adj:+.3f}/ct after haircut  avail {o.available:.0f}  "
              f"{o.secs_to_close / 60:.0f}m left")
    for t in r.trades:
        print(f"  TRADE #{t['id']} {t['ticker']} {t['side'].upper()} x{t['count']:g} @ {t['price']:.2f} fee {t['fee']:.2f}")


def cmd_scan(a) -> None:
    eng, _ = _live_stack(a, "paper")
    _print_cycle(eng.cycle(trade=False))


def cmd_run(a) -> None:
    mode = "live" if a.live else "paper"
    eng, client = _live_stack(a, mode)
    if mode == "live":
        if not client.authenticated:
            sys.exit("live mode needs KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
        if os.environ.get("BTCLAB_ENABLE_LIVE") != "1":
            sys.exit("live mode is locked. Set BTCLAB_ENABLE_LIVE=1 once you have reviewed your risk limits.")
        l = eng.risk.l
        print(f"LIVE TRADING. Max exposure ${l.max_exposure_usd}, max daily loss ${l.max_daily_loss_usd}, "
              f"max {l.max_contracts_per_order} contracts/order. Create a file named '{l.kill_switch_file}' to stop.")
        if not a.yes and input("Type LIVE to continue: ").strip() != "LIVE":
            sys.exit("aborted")
    end = time.time() + a.hours * 3600 if a.hours else float("inf")
    while time.time() < end:
        try:
            print(f"\n[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {mode}")
            _print_cycle(eng.cycle())
            n = eng.settle()
            if n:
                print(f"  settled {n} markets")
        except KeyboardInterrupt:
            break
        except Exception as e:  # keep running through transient API errors
            logging.exception("cycle failed: %s", e)
        time.sleep(a.interval)


def cmd_settle(a) -> None:
    eng, _ = _live_stack(a, "paper")
    print(f"settled {eng.settle()} markets")


def _print_report(led: Ledger, mode=None, breakdown: bool = False) -> None:
    r = led.report(mode)
    print(json.dumps({k: v for k, v in r.items() if k != "calibration_buckets"}, indent=2))
    z = r.get("pnl_gap_z")
    if z is not None:
        verdict = ("consistent with the model's probabilities" if abs(z) < 2
                   else "OUTSIDE the 95% band: the model's probabilities may be off")
        print(f"\nRealized minus expected P&L: {r['pnl_minus_expected']:+.2f} "
              f"(95% band {r['pnl_95pct_band'][0]:+.0f} to {r['pnl_95pct_band'][1]:+.0f}), z = {z:+.2f}: {verdict}.")
    if r.get("calibration_buckets"):
        print("\nCalibration (spaced snapshots >= 10 min before close):")
        print(f"  {'bucket':<9} {'n':>7} {'model':>7} {'market':>7} {'actual':>7}")
        for b in r["calibration_buckets"]:
            print(f"  {b['bucket']:<9} {b['n']:>7} {b['model_avg']:>7.3f} {b['market_avg']:>7.3f} {b['actual_freq']:>7.3f}")
    if breakdown:
        print("\nP&L by side and fair value:")
        for b in led.breakdown(mode):
            print(f"  {b['side']:<3} fair {b['fair_bucket']}  trades {b['trades']:>4}  expected {b['expected']:>8.2f}  "
                  f"realized {b['realized']:>8.2f}  win {b['win_rate']:.2f}")


def cmd_report(a) -> None:
    _print_report(Ledger(a.db), a.mode, a.breakdown)


def cmd_pnl(a) -> None:
    c = Ledger(a.db).current_pnl(a.mode)
    print(f"current P&L {c['current_pnl']:+.2f}   (realized {c['realized_pnl']:+.2f} on {c['settled_trades']} settled, "
          f"open {c['unrealized_pnl']:+.2f} on {c['open_trades']} open trades)")
    if c["open_trades"]:
        note = f"   {c['unpriced_open_trades']} open trades had no quote and are carried at cost" if c["unpriced_open_trades"] else ""
        print(f"  open positions cost {c['open_cost']:.2f}, worth {c['open_value']:.2f} at the bid "
              f"(quotes as of {(c['quotes_as_of'] or 'n/a')[:16].replace('T', ' ')} UTC){note}")


def cmd_validate(a) -> None:
    from . import validate

    if a.synthetic:
        from .sim import PricePath

        path = PricePath(datetime(2026, 1, 1, tzinfo=timezone.utc), a.days * 1440, seed=a.seed)
        closes = path.prices.tolist()
        stamps = [path.t0 + 60 * k for k in range(len(closes))]
        print(f"synthetic path with stochastic vol, seasonality and jumps: {a.days} days of 1-minute bars")
    else:
        from .marketdata import MarketData

        print(f"downloading {a.days} days of Coinbase 1-minute candles...")
        stamps, closes = MarketData().recent_candles(a.days * 1440)
    if a.fit:
        table, detail = validate.fit_nu_by_horizon(closes, width=a.width)
        for h, res in detail:
            print(f"horizon {h} min")
            best = min(res, key=lambda r: r.log_loss)
            for r in res:
                print("  " + r.row() + ("   <- best" if r is best else ""))
        print("\nFor the closed-form pricers only, paste into config.json under \"engine\":")
        print(json.dumps({"nu_by_horizon": table}))
        return
    window = 4 * 1440
    if len(closes) < window + a.horizon + 60:
        sys.exit(f"need more than {window // 1440 + 1} days of data; use --days 10 or more")
    print(f"horizon {a.horizon} min, ${a.width:g} brackets, forecasts every 30 min after a 4-day warm-up\n")
    models = (("lognormal", 0.0), ("student_t", 4.0), ("student_t", 10.0))
    for r in validate.run(closes, horizon_min=a.horizon, width=a.width, models=models, start_index=window):
        print(r.row())
    print("fitting GARCH and simulating (out-of-sample, ~1 minute)...")
    print(validate.run_fhs(closes, stamps, horizon_min=a.horizon, width=a.width).row())
    print("\nLower log loss is better; a calibrated model covers ~0.50 and ~0.90.")


def cmd_demo(a) -> None:
    from .engine import EngineConfig
    from .sim import run_simulation

    print(f"Simulating {a.hours}h of hourly BTC bracket markets (offline, about a minute)...")
    cfg = EngineConfig(pricer=a.pricer, basis_usd=0.0, mc_seed=a.seed, n_paths=10_000)
    ledger, s = run_simulation(hours=a.hours, seed=a.seed, step_min=10, ledger_path=a.db or ":memory:", cfg=cfg)
    print(f"{s['cycles']} scan cycles, {s['trades']} paper trades\n")
    _print_report(ledger, breakdown=True)
    if s["trades"] < 5:
        print("\nFew or no trades is expected. The engine trusts only half of any disagreement with the market"
              " until 10 days of settled results prove more, and skips above/below contracts until 20 days do."
              " After the market makers' spread and Kalshi's fees, little edge survives in this simulation."
              " Try `--hours 240`, or `verify`, for a longer test.")
    print("\nNote: the simulated market makers have built-in flaws, so profit here shows the machinery works,"
          " not that the edge is real. Paper trade on live Kalshi prices for that.")


def cmd_verify(a) -> None:
    """Multi-seed honesty check: is expected P&L an unbiased forecast of realized P&L?"""
    import math as _m

    from .engine import EngineConfig
    from .sim import run_simulation

    print(f"{a.seeds} independent simulations x {a.days} days (a few minutes each)\n")
    print(f"{'seed':>4} {'trades':>7} {'events':>7} {'expected':>9} {'realized':>9} {'gap':>8} {'z':>6}")
    tot_e = tot_r = 0.0
    zs = []
    for seed in range(1, a.seeds + 1):
        cfg = EngineConfig(basis_usd=0.0, mc_seed=seed, n_paths=10_000)
        led, _ = run_simulation(hours=24 * a.days, seed=seed, step_min=10, cfg=cfg)
        r = led.report()
        e, re_ = r["expected_pnl_settled"], r["realized_pnl"]
        z = r.get("pnl_gap_z")
        tot_e += e
        tot_r += re_
        if z is not None:
            zs.append(z)
        print(f"{seed:>4} {r['trades']:>7} {r.get('events', 0):>7} {e:>9.2f} {re_:>9.2f} {re_ - e:>+8.2f} "
              f"{'' if z is None else f'{z:+.2f}':>6}")
    if zs:
        combined = sum(zs) / _m.sqrt(len(zs))
        print(f"\ntotal expected {tot_e:.2f}, realized {tot_r:.2f}; combined z = {combined:+.2f} "
              f"({'consistent' if abs(combined) < 2 else 'NOT consistent'} with honest probabilities)")


def main(argv=None) -> None:
    load_dotenv()
    logging.basicConfig(level=os.environ.get("BTCLAB_LOG", "WARNING"), format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="btclab", description="Price and trade Kalshi BTC markets")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--db", default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("price", help="fair value calculator")
    p.add_argument("--spot", type=float, required=True)
    p.add_argument("--strike", type=float, required=True, help="threshold, or bracket floor with --cap")
    p.add_argument("--cap", type=float, help="bracket cap (makes it a range contract)")
    p.add_argument("--below", action="store_true", help="contract pays if price ends below strike")
    p.add_argument("--hours", type=float, required=True)
    p.add_argument("--vol", type=float, default=0.5, help="annualized vol, 0.5 = 50%%")
    p.add_argument("--nu", type=float, default=4.0)
    p.set_defaults(fn=cmd_price)

    sub.add_parser("scan", help="price live markets once, no trading").set_defaults(fn=cmd_scan)

    p = sub.add_parser("run", help="trading loop (paper unless --live)")
    p.add_argument("--live", action="store_true")
    p.add_argument("--yes", action="store_true", help="skip the LIVE confirmation prompt")
    p.add_argument("--interval", type=float, default=60.0, help="seconds between scans")
    p.add_argument("--hours", type=float, default=0, help="stop after N hours (0 = run forever)")
    p.set_defaults(fn=cmd_run)

    sub.add_parser("settle", help="record results for finished markets").set_defaults(fn=cmd_settle)

    p = sub.add_parser("report", help="P&L and calibration")
    p.add_argument("--mode", choices=["paper", "live"])
    p.add_argument("--breakdown", action="store_true", help="P&L by side and fair-value bucket")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("pnl", help="current profit: realized plus open positions at the bid (offline, instant)")
    p.add_argument("--mode", choices=["paper", "live"])
    p.set_defaults(fn=cmd_pnl)

    p = sub.add_parser("validate", help="calibration study on historical candles")
    p.add_argument("--days", type=int, default=14, help="days of history (at least 6)")
    p.add_argument("--horizon", type=int, default=60, help="minutes ahead")
    p.add_argument("--width", type=float, default=250.0)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--fit", action="store_true", help="choose tail parameter per horizon and print config")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("verify", help="multi-seed check that expected P&L is honest")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--days", type=int, default=10)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("demo", help="offline end-to-end simulation")
    p.add_argument("--hours", type=int, default=72)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pricer", choices=["fhs", "student_t", "lognormal"], default="fhs")
    p.set_defaults(fn=cmd_demo, db=None)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
