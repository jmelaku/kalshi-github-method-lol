# Kalshi BTC Lab

A pricing and trading system for Kalshi's Bitcoin markets. For each contract ("BTC between $80,000 and $80,249.99 at 3pm", "BTC above $81,000 at 5pm"), it estimates a fair probability with a GARCH volatility model and Monte Carlo simulation on BTC's own historical shocks. It then measures how much of its disagreement with the market is real, from settled results, and trades only edges that survive Kalshi's fees, an uncertainty haircut, and hard risk limits.

It runs on a laptop using free data, and it starts in paper mode.

```
 Coinbase 1m candles ──► GARCH(1,1)-t + intraday seasonality ──┐   (MLE, refit every 4h,
 Deribit DVOL ─────────► long-run variance anchor ──────────────┤    filtered every cycle)
                                                                ▼
                     Filtered historical simulation: 20,000 paths, bootstrapped residuals,
                     vol-level uncertainty, antithetic draws, one run for all settlement times
                                                                │  p ± sd for every contract
 Kalshi GET /markets ─► top of book ──┐                         ▼
 Kalshi orderbook (only if mispriced) ┴──► shrink toward market: fair = mid + β(d, type)·(p − mid)
                                           β learned from settled results (cluster-robust)
                                                                │
                                   edge − k·sd after fees ──► risk manager ──► paper / live order
                                                                │  (Kelly, per-market, per-event,
                     SQLite ledger ◄────────────────────────────┘   exposure, daily loss, kill switch)
       snapshots · trades · results ──► report: realized vs expected P&L with 95% band,
                                        log loss vs market, calibration, P&L by bucket
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                       # 59 tests, no network needed
python -m btclab demo                  # offline simulation, about a minute
python -m btclab price --spot 80000 --strike 81000 --hours 3 --vol 0.5
```

## The workflow, in order

| Step | Command | What it tells you |
|---|---|---|
| 1. Validate on real history | `python -m btclab validate --days 14` | Out-of-sample log loss and interval coverage of the GARCH + Monte Carlo pricer vs. simple models, on real Coinbase data. |
| 2. Look at live prices | `python -m btclab scan` | Fair value ± uncertainty vs. Kalshi for every open BTC market, with edges before and after the haircut. No trading. |
| 3. Paper trade | `python -m btclab run --interval 60` | Logs the trades it *would* make at real prices, with every risk check applied, and learns its shrinkage table from settled markets. Run it for weeks. |
| 4. Score it | `python -m btclab report --breakdown` | Realized vs. expected P&L with a 95% band, model vs. market log loss, calibration table, P&L by side and bucket. |
| (any time) | `python -m btclab compare data/btclab.db data/options.db` | Side-by-side scorecard of two paper bots since the newer one started: P&L, z, and model-vs-market log loss on the same settled markets. |
| (any time) | `python -m btclab pnl` | Current profit in one line: realized P&L plus open positions valued at what they would sell for now. Instant, offline. |
| (any time) | `python -m btclab verify` | Reruns the multi-seed simulation honesty check from the table below. |
| 5. Go live, small | `python -m btclab run --live` | Real orders. See the checklist below. |

Copy `config.example.json` to `config.json` and `.env.example` to `.env` before steps 2–5.

**The rule for going live:** after at least a few hundred settled trades across 100+ events, realized P&L must be positive *and* inside the report's 95% band around expected P&L (|z| < 2). Model log loss should also beat the market's. If realized is below the band, the model is overconfident and its "edge" is model error. Keep paper trading.

## How the pricing works

### 1. Volatility: GARCH(1,1) with Student-t shocks and intraday seasonality

BTC volatility clusters and mean-reverts. On de-seasonalized 1-minute returns:

```
h_t = ω + α·e²_{t−1} + β·h_{t−1},    e_t = √h_t · z_t,   z_t ~ Student-t(ν)
```

- **Fit by maximum likelihood** (α, β, ν) on the last 4 days of 1-minute data. The recursion runs through a linear filter, so a fit takes a fraction of a second. It refits every 4 hours (warm-started) and filters forward every cycle.
- **Variance targeting with implied vol:** ω is set so the long-run variance is a blend of sample variance and Deribit's DVOL. Forecasts for longer horizons mean-revert toward what the options market expects.
- **Intraday seasonality:** an hour-of-day vol multiplier, estimated robustly from mean |r| and shrunk toward 1.

### 2. Probabilities: filtered historical simulation (Monte Carlo)

For each cycle, one simulation runs from now to the furthest settlement time. Each minute draws a shock **from BTC's own standardized GARCH residuals** (bootstrap), scales it by the GARCH variance and that hour's seasonal factor, and updates the variance. As a result:

- Tails come from real BTC shocks and thin out at longer horizons on their own, because simulated vol clusters and mean-reverts. There's no assumed distribution shape.
- Every contract in every event is priced off the same paths, so bracket prices are mutually consistent and sum to 1.
- **Vol-level uncertainty:** each path draws a scale factor for the vol level, because the vol estimate is itself uncertain. Each probability comes with an uncertainty `sd` (Monte Carlo error plus vol sensitivity).
- Antithetic draws reduce noise; a martingale correction keeps E[S_T] = S_0; `basis_usd` adds noise because Kalshi settles on the CF Benchmarks BRTI index, not on Coinbase.

### 3. How much of the disagreement with the market is real?

A model can be well calibrated overall and still lose money, because **the trades it picks are exactly where it disagrees most with the market, and those are disproportionately where the model is wrong.** (The winner's curse; see "What building it taught me" below.)

So the engine never trades on raw model probabilities. It uses:

```
fair = mid + β(d, contract type) · d,     d = p_model − mid
```

β is estimated from settled markets, separately for each disagreement size and contract type, by regressing outcome − mid on d:
- **Bayesian prior, day-level evidence:** range brackets start at β = 0.5 (trust half of any disagreement). Learned estimates are ignored until 10 distinct days have settled, then combined with the prior weighted by their uncertainty.
- **Cluster-robust standard errors by day.** Brackets in one event share one price outcome, and hourly events that overlap in time ride the same price path, so neither markets nor events are independent. (My first version clustered by event; its errors were far too small and the evidence gate opened on noise. More below.)
- **Lower confidence bound:** the β used is the estimate minus one standard error, clipped to [0, 1]. The engine never trusts the model more than it claims.
- **Above/below contracts must prove themselves.** They depend on the *direction* of BTC, where a volatility model has no special insight, so they start at β = 0. They're traded only once 20+ settled days show a positive β at 2 day-clustered standard errors. In practice that means at least three weeks of paper trading.
- **One-sided books:** if only a bid or only an ask exists, the missing side counts as $0 or $1 for the reference midpoint, so the shrinkage still applies. With an empty book the market is skipped.

### 4. Decision and sizing

Edge on buying YES at ask *a* is `fair − a − fee(a)`, and decisions use `edge − k·sd`. Fees follow Kalshi's schedule: `round_up(0.07 × C × P × (1 − P))` for takers. Size is 10% Kelly on the haircut probability, then capped per order, per market, per event, and in total.

## A/B test: options-implied pricing, composite spot, resting bids

`config.options.example.json` turns on three opt-in upgrades. Run it as a second paper bot with its own database next to the baseline, then compare:

```bash
cp config.options.example.json config.options.json
nohup caffeinate -i .venv/bin/python -u -m btclab --config config.options.json --db data/options.db run --interval 60 > data/options.log 2>&1 &
python -m btclab compare data/btclab.db data/options.db
```

1. **Options-implied pricer** (`pricer: "options"`, `btclab/options.py`). Reads probabilities off Deribit's BTC option marks: one smile per expiry (smoothed cubic in log-moneyness, quoted strikes only), total implied variance interpolated between expiries on a variance clock that uses the GARCH hour-of-day seasonality, and P(S > K) = -dC/dK from Black-76 so the skew is included. Each settlement time gets one probability curve, forced non-increasing (no negative density), and every contract reads off it. Hourly contracts sit before Deribit's first expiry, so the smile is carried down two standard ways (same moneyness, same standardized moneyness); the price is their average and half their gap goes into the uncertainty sd, along with a ±10% implied-vol bump. No Monte Carlo: 500 contracts price in about 0.03 s.
2. **Composite spot** (`spot_source: "composite"`). Median mid of Coinbase, Kraken, Bitstamp and Gemini, four BRTI constituents, instead of Coinbase alone. Falls back to Coinbase if fewer than two respond.
3. **Resting bids** (`execution: "both"`, paper only). Takes clear edges as before; elsewhere rests one bid per market, one tick above the best bid, capped so the haircut edge still clears the thresholds after the maker fee. A bid counts as filled when a later cycle shows the ask at or below it (we improved the bid) or strictly below it (we joined a bid). Fills are scored with the model's fair value at detection, so adverse selection shows up in expected vs. realized P&L. Resting bids count against every risk limit as if filled, and their total cost is capped by `maker_max_exposure_usd` so they can't crowd out taker trades. `report` adds fill rate and maker-vs-taker P&L.

Changing three things at once means the comparison says whether the bundle helps, not which part. The snapshots record each model's probabilities, so the pricer's accuracy can still be judged on its own.

## What building it taught me

**v1, fixed fat tails.** A Student-t (ν = 4) for every contract expected +$261 in simulation and lost $54. Minute-level fat tails mostly average out over an hour, so it overpriced brackets near spot. Fitting tails per horizon fixed that particular error, but 48-hour tests were too short to see what came next.

**v2, much longer tests exposed the real problem.** Over 10-day simulations (six seeds), the GARCH + Monte Carlo model beat the simulated market on log loss in every run (≈0.15 vs. ≈0.17), with near-perfect calibration in every probability bucket. Yet realized P&L fell short of expected by more than chance in five of six runs. The breakdown showed why: small disagreements (2–4¢) were almost fully real, but large ones were mostly model error, and above/below contracts were unreliable in both directions. The trades it chose were systematically its worst predictions.

**v3, disagreement shrinkage + evidence gate.** This cut the gaps from hundreds of dollars to tens, but every seed still came in below expected. Inspecting individual trades showed the cause: the evidence gate for above/below contracts had *opened* mid-run, at full trust, and those trades lost. The statistics were wrong. I had clustered standard errors by event, but consecutive hourly events share the same price path, so ten "independent" events were really one or two observations. A separate edge case let one-sided order books skip the shrinkage entirely.

**v4, day-clustered errors + one-sided book fix.** Still below expected. With day clustering, the gate now opened after just 5 days of agreeing evidence. That's a known failure: cluster-robust errors are unreliable with few clusters. The bracket trust level also jumped to about 0.9 on a few days of data.

**v5, Bayesian shrinkage with minimum evidence.** Learned estimates are ignored until 10 settled days exist (20 for above/below), and then combined with the prior by a proper normal-normal update. See the verification results below.

The general lesson: **calibration and log loss are not enough. Check expected vs. realized P&L on the trades you actually take, with error bars, broken down by bucket.** That's what `report` does.

## Verification (simulation)

Independent seeds × 10 simulated days each. "Gap" is realized minus expected P&L; z uses day-clustered errors (|z| < 2 means consistent with honest probabilities).

| Version | Trades | Expected P&L | Realized P&L | Gap | z |
|---|---|---|---|---|---|
| v2 GARCH + Monte Carlo, raw model (6 seeds) | 5,855 | $4,915 | $620 | −$4,295 | five of six seeds below −2 |
| v3 + shrinkage, event-clustered (seeds 2, 4–6) | 151 | $97 | −$120 | −$217 | −3.5 worst seed |
| **v5 final (6 seeds)** | **18** | **$5** | **−$12** | **−$17** | **−1.7 (pooled)** |

The trade count is small, so the strongest test is on every snapshot where the final engine saw an edge above its 2¢ trading threshold: **1,378 opportunities on 39 independent days, expected +2.59¢ per contract, realized +2.94¢, z = +0.37.** Below the threshold it *underestimates* its edge, which is the safe direction. The GARCH + Monte Carlo model also beat the simulated market's log loss in all six seeds (≈0.155 vs. ≈0.173).

**The honest bottom line:** once the engine stopped trusting its own disagreements more than the evidence supports, it found little edge against competent simulated market makers after their spread and Kalshi's fees. Most of the earlier "edge" was the model being wrong. On real Kalshi markets, only paper trading will tell. The engine starts conservative and learns how far to trust itself from your own settled results.

The simulator (`sim.py`) has stochastic volatility, intraday seasonality, fat-tailed shocks, and jumps. Its market makers are decent but imperfect: they use a 4-hour EWMA vol, ignore seasonality and jumps, overprice cheap longshots, and quote with noise. **A profit in simulation is not evidence of an edge on the real Kalshi.** What the simulator *can* test is whether the engine's expected P&L is honest.

## Cost control

- Data is free: Coinbase, Kraken fallback, Deribit public endpoints.
- One paginated `GET /markets` call gives top-of-book for every BTC market. Full order books are fetched only for markets already mispriced after the haircut.
- Settlement checks only markets past their close, 100 tickers per request, via an indexed table, so they stay fast as history grows.
- Snapshots are thinned to one per market per 5 minutes.
- The client spaces requests and backs off on HTTP 429/5xx.

## Risk controls (all in `config.json`)

Fractional Kelly sizing (10%) on the haircut probability, with hard caps: contracts per order, contracts per market, **exposure per event** (brackets in one event are one bet on one price), total exposure, a daily loss limit, a price band that skips 1–3¢ longshots, and no trading in the last 5 minutes before close. **Create a file named `STOP` in the project folder to halt all trading immediately.** Paper and live mode use the same checks.

## Going live checklist

1. Paper results meet the rule above.
2. Create a Kalshi API key, save the `.pem` outside the repo, and fill in `.env`.
3. Start with a small `max_exposure_usd` and `max_daily_loss_usd`.
4. **Verify the order mapping with one contract.** Orders use Kalshi's V2 endpoint (`POST /portfolio/events/orders`), which trades the YES leg only. This code buys NO at price *q* by sending `side="ask"` at `1 − q`. Set `max_contracts_per_order` to 1, let it buy one NO contract, and confirm in the Kalshi app that you hold one NO contract at the expected price.
5. Check `fee_cost` on your first fills against `FeeModel`. Some series have non-standard fees; adjust `taker_multiplier` if they differ.
6. Set `BTCLAB_ENABLE_LIVE=1` and run `python -m btclab run --live`.

## Assumptions to verify against live data

The client was written from Kalshi's current documentation, but parts could not be tested against the live API from the build environment:

- **Series tickers** `KXBTC` (ranges) and `KXBTCD` (above/below). Confirm on kalshi.com, and add other series in `kalshi.BTC_SERIES`.
- **Settlement source and time:** each market's rules name the exact BRTI method. Tune `basis_usd` if settlement differs from Coinbase.
- **Market data auth:** the client signs every request when a key is configured. If unauthenticated reads are refused, add your key even for paper mode.

## Project layout

```
btclab/
  garch.py       GARCH(1,1)-t MLE, intraday seasonality, variance targeting, filtering
  fhs.py         filtered historical simulation pricer with uncertainty estimates
  calibrate.py   disagreement shrinkage (cluster-robust, evidence gate), logistic blend diagnostic
  pricing.py     closed-form binary-option pricing (lognormal, Student-t), implied vol
  vol.py         EWMA realized vol, implied/realized blend
  fees.py        Kalshi fee schedule and rounding
  book.py        order book normalization (Kalshi returns bids only)
  kalshi.py      REST client with RSA-PSS signing, V2 orders, batched settlement lookups
  marketdata.py  Coinbase / Kraken / Bitstamp / Gemini / Deribit public data, composite spot
  options.py     options-implied pricer: Deribit smiles, variance-clock interpolation, digitals
  scanner.py     fair value vs. book, edge after fees and haircut, maker quotes
  risk.py        Kelly sizing and hard limits
  engine.py      the loop: fit, simulate, shrink, scan, size, execute, settle
  ledger.py      SQLite: snapshots, trades, resting orders, results; honesty check, calibration, breakdowns
  compare.py     side-by-side scorecard of two databases over a common window
  validate.py    out-of-sample calibration study on historical candles
  sim.py         simulator: stochastic vol, seasonality, jumps, imperfect market makers
  cli.py         command line
tests/           59 tests: GARCH parameter recovery, Monte Carlo vs. closed form, shrinkage,
                 fees, order book, signing, risk, ledger, API wiring, end-to-end simulation,
                 options pricer vs. lognormal, skew and partitions, composite spot, resting bids, compare
```

## Roadmap

- WebSocket order book stream instead of polling.
- Live resting orders (paper maker mode exists; live needs order amend/cancel and fill reconciliation), plus inventory skew.
- Event-day vol (FOMC, CPI) and weekend effects in the GARCH model.
- Hedge bracket exposure with Kalshi's BTC perpetuals.
- Dashboard: live fair-value curve vs. Kalshi prices, P&L, and calibration plot.

## Disclaimer

This is software, not financial advice. Trading involves real risk of loss.
