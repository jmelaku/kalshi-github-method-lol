# CLAUDE.md: Kalshi BTC Lab handoff

This file briefs Claude Code on this project: what it is, how it works, why it's built this way, what's verified, what isn't, and what to do next. Read it fully before changing anything.

## 1. What this is and who it's for

A Python system that **prices and trades Kalshi's Bitcoin prediction markets**. Kalshi lists contracts like "BTC between $80,000 and $80,249.99 at 3pm ET" (range brackets, series `KXBTC`) and "BTC above $81,000 at 5pm" (thresholds, series `KXBTCD`). Each pays $1 if true, else $0, so each is a binary option whose fair price is a probability.

The owner is a student building this for two reasons:
1. **Resume project** for SWE / quant / PM roles. It pairs with an earlier project of theirs, "Apex Exchange" (a matching engine). Rigor, honest evaluation, and a clean story matter as much as profit.
2. **Actually trading BTC on Kalshi**, carefully: paper first, then small live size.

They're comfortable running terminal commands but new to this codebase. Explain what you change and why in plain language.

### Owner's environment
- MacBook Air (Apple Silicon), project at `~/Downloads/kalshi-btc-lab`.
- **System Python 3.9.6** in a venv at `.venv`. Code MUST stay 3.9-compatible: every module uses `from __future__ import annotations`; don't use `match`, runtime `X | Y` unions, `zip(strict=)`, or other 3.10+ features. (`pyproject.toml` says >=3.10; that's stale, 3.9 works and is tested.)
- A harmless `NotOpenSSLWarning` (LibreSSL) appears on every run. Ignore it.
- All 46 tests pass on their machine (`python -m pytest`).
- They have NOT yet connected a Kalshi API key or run anything against live Kalshi.

## 2. Current status

| Area | Status |
|---|---|
| Pricing engine (GARCH + Monte Carlo) | Built, unit-tested, validated in simulation |
| Shrinkage / honesty machinery | Built, tested, verified in simulation (see §6) |
| Simulator, CLI, ledger, reports | Built and tested |
| Kalshi REST client | Written from Kalshi's docs; tested only against canned responses. **Never run against the live API.** |
| Live trading | Implemented, locked behind multiple guards, never exercised |

**Immediate next step for the owner:** fill in `.env` (API key ID + path to private key `.pem`), copy `config.example.json` to `config.json`, run `python -m btclab scan`. That first scan is the first contact with real Kalshi and Coinbase data. Expect possible small breakages (§7); fix them carefully.

## 3. Commands

All run from the project root with the venv active (`source .venv/bin/activate`).

```
python -m pytest                     # 46 tests, offline
python -m btclab price --spot 80000 --strike 81000 --hours 3 --vol 0.5   # closed-form calculator
python -m btclab validate --days 14  # out-of-sample model test on real Coinbase candles (needs network, >=6 days)
python -m btclab validate --synthetic --days 10   # same, offline on simulated data
python -m btclab demo [--hours 72]   # offline end-to-end simulation + report
python -m btclab verify [--seeds 4 --days 10]     # multi-seed honesty check (slow: minutes per seed)
python -m btclab scan                # price live markets once, no trading
python -m btclab run [--interval 60] [--hours N]  # paper trading loop
python -m btclab run --live          # real orders (locked, see §8)
python -m btclab settle              # record results of finished markets
python -m btclab report [--breakdown] [--mode paper|live]
python -m btclab pnl [--mode paper|live]           # current profit: realized + open positions at the bid (instant, offline)
```

Global flags: `--config config.json` (default) and `--db data/btclab.db` (default).

Env vars (`.env`, loaded by a tiny built-in loader in `cli.py`): `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_BASE_URL` (default prod `https://external-api.kalshi.com/trade-api/v2`; demo `https://external-api.demo.kalshi.co/trade-api/v2`), `BTCLAB_ENABLE_LIVE` (must be `1` for live), `BTCLAB_LOG` (e.g. `INFO`).

## 4. Architecture

```
btclab/
  garch.py       GARCH(1,1)-t fit by MLE, hour-of-day seasonality, variance targeting toward implied vol
  fhs.py         Filtered historical simulation (Monte Carlo) pricer; returns p and sd per contract
  calibrate.py   ShrinkTable: how much of (model - market) to trust, learned from settled data
                 (also an older logistic Blend, now only a diagnostic)
  pricing.py     Contract definition + closed-form lognormal / Student-t pricing (comparison baselines)
  vol.py         EWMA realized vol, realized/implied blend (used by closed-form pricers)
  fees.py        Kalshi fee schedule: round_up(0.07*C*P*(1-P)) taker; maker 0 or 0.0175
  book.py        Order book normalization. Kalshi returns BIDS ONLY: yes_ask = 1 - best_no_bid
  kalshi.py      REST client: RSA-PSS signing, pagination, retries, V2 orders, batched results
  marketdata.py  Coinbase candles/spot (Kraken fallback), Deribit DVOL. All free, no keys.
  scanner.py     Snapshot + Opportunity: edge after fees, edge_adj after uncertainty haircut
  risk.py        Fractional Kelly + hard limits (per order/market/event/total, daily loss, STOP file)
  engine.py      The loop. Talks to `Exchange` and `Feed` protocols, so live and sim share code.
  ledger.py      SQLite: snapshots, trades, results, markets; reports and honesty checks
  validate.py    Out-of-sample calibration studies on candles
  sim.py         Realistic simulator (stochastic vol, seasonality, fat tails, jumps, imperfect market makers)
  cli.py         argparse CLI
tests/           46 tests (see §9)
docs/RESUME_AND_INTERVIEWS.md   resume bullets + interview prep (owner-facing)
README.md        user-facing docs, including the full version history and verification table
```

### One engine cycle (`Engine.cycle`), default `pricer="fhs"`
1. Get spot (Coinbase) and DVOL (Deribit; optional).
2. List open BTC markets via paginated `GET /markets` for `KXBTC` and `KXBTCD`. Keep those within `[min_secs_to_close, max_secs_to_close]` (default 5 min to 36 h).
3. **GARCH:** refit every `garch_refit_min` (240) on `garch_lookback_min` (5760 = 4 days) of 1-minute candles, warm-started; otherwise filter forward with new returns since `fit.last_ts`.
4. **Monte Carlo:** one simulation from now to the furthest settlement time, 20k paths (antithetic), bootstrapping GARCH standardized residuals through the GARCH recursion with seasonality, plus a per-path vol-level scale `u ~ lognormal(sd=vol_uncertainty)`. Snapshots are taken at each needed horizon, with a martingale correction and basis noise (`basis_usd`). Each contract gets `p` and `sd` (Monte Carlo error plus sensitivity across low vs. high `u` quartiles).
5. **Shrinkage:** `fair_used = mid + beta(class, d_bin) * (p - mid)` with `d = p - mid`; see §5. A market with an empty book is skipped.
6. **Scan:** for each side, `edge = fair - ask - fee`, `edge_adj = edge - haircut_k * sd`. Only if top-of-book shows `edge_adj >= min_edge` is the full order book fetched (cost control) and the market re-evaluated.
7. Store snapshots (thinned to one per market per `snapshot_interval_s` = 300 s), and upsert into the `markets` table (ticker, close_ts, result).
8. **Trade:** rank by `edge_adj`; at most one trade per market per cycle and `max_trades_per_cycle` total; `RiskManager.size` must return > 0. Paper fills at the ask; live sends an IOC limit at that ask.
9. `settle()`: indexed lookup of unsettled markets past close, then batched `GET /markets?tickers=...` (100 per call), then `set_result` marks P&L.

## 5. Why it's built this way (the history matters, so don't undo these lessons)

The README has the full v1–v5 story. The short version:

- **v1, closed-form Student-t (ν = 4) for all horizons:** lost money in simulation. Minute-level fat tails average out over an hour, so it overpriced near-spot brackets. Hence horizon-aware tails, and later the Monte Carlo.
- **v2, GARCH + Monte Carlo:** beat the simulated market on log loss in every run (about 0.155 vs. 0.173) with excellent calibration, **but realized P&L fell far short of expected** (6 seeds × 10 days: expected $4,915, realized $620). This is the **winner's curse**: the trades it picks are where it disagrees most with the market, and those are disproportionately model errors. Small disagreements (2–4¢) were real; large ones were mostly not. Threshold (above/below) contracts were unreliable in both directions.
- **v3, disagreement shrinkage, clustered by event:** gaps fell from hundreds to tens of dollars but stayed negative. Trade inspection showed the evidence gate for thresholds had opened on noise: **consecutive hourly events share one price path, so events aren't independent.**
- **v4, clustered by day:** the gate still opened after about 5 days of agreeing evidence. Cluster-robust standard errors are unreliable with few clusters. Also found: a one-sided book (no YES ask) made `mid` None, so the code silently used the raw model. Fixed: the empty side counts as $0 or $1.
- **v5 (current), Bayesian shrinkage with minimum evidence:**
  - per (class ∈ {between, threshold}) × (10 signed disagreement bins), OLS slope `beta_hat` of `(y - mid)` on `d`, with **day-clustered** robust SE and a G/(G−1) correction;
  - fewer than `shrink_min_days` (10) distinct days: data ignored, `beta = prior` (0.5 for brackets);
  - otherwise a normal-normal update with prior `N(0.5, 0.25²)`; the beta used is the posterior mean minus `shrink_z_se` (1.0) posterior sds, clipped to [0, 1];
  - **thresholds:** prior 0, and they stay at 0 unless ≥ 20 days, ≥ 30 events, and `beta_hat > 2·se`. Rationale: thresholds depend on BTC's *direction*, where a vol model has no special insight; brackets mostly depend on *dispersion*.

**Invariant to protect:** the engine should never trust its own probabilities more than the settled evidence supports. Changes that make it trade more must be justified by the honesty checks (§6), not by bigger simulated profit.

## 6. Verification results (simulation, current version)

- 6 seeds × 10 days: 18 trades, expected +$5.10, realized −$11.69, day-clustered z = −1.74, consistent with honest probabilities. It trades rarely because brackets use the 0.5 prior for the first 10 days and thresholds are gated.
- Higher-power test on every snapshot where the final engine saw edge above 2¢ after fees: **1,378 opportunities on 39 independent days, expected +2.59¢ per contract, realized +2.94¢, z = +0.37.** Below that threshold it underestimates its edge, which is the safe direction.
- **Interpretation:** after honest accounting, there is little edge against competent simulated market makers after their spread and fees. The simulator's market makers have deliberate flaws (4-hour EWMA vol, no seasonality or jumps, longshot bias, quote noise), so **simulated profit is never evidence of a real edge.** Only live paper trading can show that.

Honesty tooling: `Ledger.pnl_check()` (realized minus expected per event, with z), `Ledger.calibration()` (model vs. market log loss/Brier and buckets on spaced snapshots ≥ 10 min before close), `Ledger.breakdown()` (expected vs. realized by side and fair-value bucket), and `btclab verify`.

## 7. Untested assumptions about the live Kalshi API (likely first breakages)

The build environment couldn't reach Kalshi. Check these on the first `scan`, and fix them in `kalshi.py` / `book.py` without changing the math:

1. **Base URL:** `https://external-api.kalshi.com/trade-api/v2` (the older host `api.elections.kalshi.com` may also work).
2. **Signing:** message = `timestamp_ms + METHOD + path`, where the path includes `/trade-api/v2` and excludes the query string; RSA-PSS, SHA-256, MGF1(SHA-256), salt = digest length; base64. Headers `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-SIGNATURE`, `KALSHI-ACCESS-TIMESTAMP`. The client signs every request when a key is configured.
3. **Series tickers** `KXBTC` and `KXBTCD` (`kalshi.BTC_SERIES`). Confirm on kalshi.com.
4. **Market fields used:** `ticker`, `event_ticker`, `strike_type` (greater / greater_or_equal / less / less_or_equal / between), `floor_strike`, `cap_strike`, `close_time` (ISO), `status`, `result`, `yes_bid_dollars`, `yes_ask_dollars`, `yes_bid_size_fp`, `yes_ask_size_fp`. `book.quote_from_market` also accepts integer-cent `yes_bid`/`yes_ask`.
5. **Order book:** `GET /markets/{ticker}/orderbook` returning `{"orderbook_fp": {"yes_dollars": [[price_str, size_str], ...], "no_dollars": [...]}}`. Older `{"orderbook": {"yes": [[cents, size]], "no": ...}}` is also parsed.
6. **Batched results:** `GET /markets?tickers=a,b,c`.
7. **Orders (V2):** `POST /portfolio/events/orders`, YES leg only: `side="bid"` buys YES; buying NO at q is sent as `side="ask"` at `1 - q`. Body: `ticker, client_order_id, side, count (string), price (string, dollars, 4dp), time_in_force="immediate_or_cancel", self_trade_prevention_type="taker_at_cross", post_only`. Response fields read: `fill_count`, `average_fill_price`, `average_fee_paid`, `order_id`. **The NO mapping must be verified with a single 1-contract order before any size.**
8. **Settlement index:** Kalshi settles on CF Benchmarks BRTI, not Coinbase. `basis_usd` (20) adds that uncertainty. Read the market rules and adjust if needed.
9. **Fees:** some series have non-standard fees. Compare real fills to `FeeModel`.

## 8. Safety rules (non-negotiable)

- **Never commit secrets.** `.env`, `*.pem`, `*.key`, `data/`, and `*.db` are in `.gitignore`. Keep it that way. Never print key contents.
- **Live trading needs all of:** the `--live` flag, `KALSHI_API_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH`, `BTCLAB_ENABLE_LIVE=1`, and typing `LIVE` at the prompt (or `--yes`). Don't weaken these guards. Don't place real orders on the owner's behalf; they run live mode themselves.
- **Kill switch:** a file named `STOP` in the working directory halts trading on the next cycle.
- Risk limits live in `config.json` → `risk` (per-order, per-market, `max_event_exposure_usd`, total exposure, daily loss, price band 0.03–0.97, no trading in the last 5 minutes). Paper and live use identical checks.
- The owner's rule for going live: hundreds of settled paper trades across 100+ events, realized P&L positive **and** within the 95% band of expected (|z| < 2), and model log loss better than the market's.

## 9. Tests and conventions

- `tests/test_models.py`: GARCH filter vs. loop, GARCH recovers known parameters, Monte Carlo matches closed-form lognormal within its error bars, partition sums to 1, martingale, multi-horizon sharing, vol uncertainty raises sd, logistic blend recovery, shrinkage learns reliability and gates thresholds, data ignored before 10 days, day clustering more conservative than event clustering.
- `tests/test_pricing.py`, `test_fees.py`, `test_book_and_client.py` (bids-only book, signing verification, NO-as-YES-ask order mapping), `test_engine.py` (evaluate, risk, ledger P&L, settlement scope, end-to-end sim), `test_live_wiring.py` (real `KalshiClient` against a fake HTTP session; threshold gate blocks trading by default).
- Keep tests offline and fast (the suite runs in about 5 s). Fix seeds (`mc_seed`, sim seeds) for determinism.
- When changing pricing or shrinkage, run `python -m btclab verify` (slow) or a multi-seed simulation and check that z stays within ±2. Don't tune parameters to make simulated profit look better.
- SQLite schema changes: add columns through the `MIGRATIONS` dict in `ledger.py`, so the owner's existing database upgrades in place.
- Config keys map 1:1 to the dataclasses `EngineConfig`, `RiskLimits`, and `FeeModel`. Unknown keys raise an error on load, so add new fields to the dataclass first. Deprecated keys (`calib_min_markets`, `shrink_prior_obs`, `model`) are kept so old configs still load.

## 10. Known limitations and roadmap

- Polling only; WebSocket order books would cut latency and API calls.
- No maker (resting-quote) mode yet. Maker quotes are computed and logged but never placed. Adverse selection would need tracking.
- No event-day vol (FOMC/CPI) or weekend effects in GARCH; seasonality is hour-of-day only.
- Monte Carlo cost grows with the furthest horizon (36 h ≈ 2,160 steps × 20k paths); fine at a 60 s interval on a laptop.
- `validate` with real data has no historical DVOL, so it tests GARCH without the implied-vol anchor.
- Dashboard (live fair-value curve vs. Kalshi, P&L, calibration) not built.

## 11. How to help the owner right now

1. Help them create `.env` and `config.json` and run `python -m btclab scan`. Debug any API mismatch (§7) with small, targeted fixes plus a test that reproduces the real response shape.
2. Then start paper trading (`python -m btclab run --interval 60`) and keep the laptop awake (`caffeinate -i`). Check `report --breakdown` periodically.
3. Explain results in plain language. "No positive-edge opportunities" is a normal, frequent outcome, especially in the first 10 days, while the shrinkage uses its conservative prior.
4. Keep the README verification table and `docs/RESUME_AND_INTERVIEWS.md` accurate if results or behavior change. Never fabricate numbers.
