"""btclab: price and trade Kalshi's Bitcoin markets.

Modules
-------
fees       Kalshi fee model (taker / maker, rounding).
pricing    Fair probability for each Kalshi BTC contract (lognormal and fat-tailed).
vol        Volatility estimation (EWMA realized vol, implied-vol blend).
book       Order book normalization (Kalshi returns YES bids and NO bids only).
kalshi     Kalshi REST client with RSA-PSS request signing.
marketdata Free BTC spot / candle / implied-vol feeds (Coinbase, Deribit).
scanner    Compares fair value to the book and ranks opportunities after fees.
risk       Position sizing and hard risk limits.
ledger     SQLite store for snapshots, paper/live trades, settlement, reports.
validate   Calibration study of the pricing model on historical candles.
sim        Synthetic market generator so everything runs offline.
cli        Command-line entry point (`python -m btclab ...`).
"""

__version__ = "0.1.0"
