"""SQLite store for snapshots, trades, results, and performance reports.

Tables
  snapshots  every priced market per scan: fair value vs the book
  trades     paper and live fills
  results    settled outcome per ticker (yes/no)

The report answers the questions that matter, with error bars:
  1. Are the model's probabilities honest? Realized vs expected P&L, with a
     z-score computed from event-level variance (events, not trades, are the
     independent units: brackets in one event all depend on one price).
  2. Is the model better calibrated than the market? Log loss and Brier on
     spaced snapshots taken well before close, plus calibration buckets.
  3. Where does P&L come from? Breakdown by side and fair-value bucket.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .scanner import Opportunity, Snapshot

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
  ts TEXT, ticker TEXT, event_ticker TEXT, strike_type TEXT, floor REAL, cap REAL,
  secs_to_close REAL, spot REAL, sigma REAL, model TEXT, fair_p REAL,
  fair_sd REAL, fair_used REAL,
  yes_bid REAL, yes_ask REAL, maker_yes_bid REAL, maker_no_bid REAL
);
CREATE INDEX IF NOT EXISTS snap_ticker ON snapshots(ticker);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, mode TEXT, ticker TEXT, event_ticker TEXT, side TEXT,
  price REAL, count REAL, fee REAL, fair REAL, edge REAL, fair_sd REAL, edge_adj REAL,
  order_id TEXT, status TEXT DEFAULT 'open', result TEXT, pnl REAL
);
CREATE TABLE IF NOT EXISTS results (ticker TEXT PRIMARY KEY, result TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS markets (ticker TEXT PRIMARY KEY, event_ticker TEXT, close_ts REAL, result TEXT);
CREATE INDEX IF NOT EXISTS markets_open ON markets(result, close_ts);
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, mode TEXT, ticker TEXT, event_ticker TEXT, side TEXT,
  price REAL, count REAL, fair REAL, fair_sd REAL, improved INTEGER,
  status TEXT DEFAULT 'resting', done_ts TEXT, reason TEXT, trade_id INTEGER
);
CREATE INDEX IF NOT EXISTS orders_status ON orders(status);
"""

# Columns added after v0.1; older databases get them via ALTER TABLE.
MIGRATIONS = {
    "snapshots": {"fair_sd": "REAL", "fair_used": "REAL"},
    "trades": {"fair_sd": "REAL", "edge_adj": "REAL", "liquidity": "TEXT"},
}


class Ledger:
    def __init__(self, path: str = "data/btclab.db"):
        import os

        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        for table, cols in MIGRATIONS.items():
            have = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for c, typ in cols.items():
                if c not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {c} {typ}")
        if self.db.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 0:
            # One-time backfill for databases created before the markets table existed.
            for r in self.db.execute(
                "SELECT ticker, event_ticker, ts, secs_to_close FROM snapshots WHERE rowid IN "
                "(SELECT MAX(rowid) FROM snapshots GROUP BY ticker)"
            ).fetchall():
                close = datetime.fromisoformat(r["ts"]).timestamp() + (r["secs_to_close"] or 0)
                self.db.execute("INSERT OR IGNORE INTO markets VALUES (?,?,?,NULL)", (r["ticker"], r["event_ticker"], close))
            self.db.execute("UPDATE markets SET result = (SELECT result FROM results WHERE results.ticker = markets.ticker)")
        self.db.commit()
        self._last_snap: dict[str, float] = {}

    @classmethod
    def readonly(cls, path: str) -> "Ledger":
        """Open an existing database for reading only (dashboards, while a bot writes to it)."""
        led = cls.__new__(cls)
        led.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        led.db.row_factory = sqlite3.Row
        led._last_snap = {}
        return led

    def has_table(self, name: str) -> bool:
        return self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

    # ---- writes ----
    def add_snapshots(self, snaps: list[Snapshot], min_interval_s: float = 0.0) -> int:
        """Store snapshots, at most one per market every `min_interval_s`. Returns rows written."""
        keep = []
        for s in snaps:
            t = datetime.fromisoformat(s.ts).timestamp()
            if t - self._last_snap.get(s.ticker, -1e18) >= min_interval_s:
                keep.append(s)
                self._last_snap[s.ticker] = t
        self.db.executemany(
            "INSERT OR IGNORE INTO markets VALUES (?,?,?,NULL)",
            [(s.ticker, s.event_ticker, datetime.fromisoformat(s.ts).timestamp() + s.secs_to_close) for s in snaps],
        )
        if keep:
            cols = list(asdict(keep[0]).keys())
            self.db.executemany(
                f"INSERT INTO snapshots ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [tuple(asdict(s).values()) for s in keep],
            )
        self.db.commit()
        return len(keep)

    def add_trade(
        self,
        o: Opportunity,
        count: float,
        price: float,
        fee: float,
        mode: str,
        order_id: str = "",
        ts: Optional[str] = None,
        liquidity: str = "taker",
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO trades (ts,mode,ticker,event_ticker,side,price,count,fee,fair,edge,fair_sd,edge_adj,order_id,"
            "liquidity) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ts or datetime.now(timezone.utc).isoformat(),
                mode,
                o.ticker,
                o.event_ticker,
                o.side,
                price,
                count,
                fee,
                o.fair,
                # Expected value at the price actually paid (may differ from the scanned ask).
                o.fair - price - fee / max(count, 1e-9),
                o.fair_sd,
                o.edge_adj,
                order_id,
                liquidity,
            ),
        )
        self.db.commit()
        return int(cur.lastrowid)

    # ---- resting (maker) orders, paper only ----
    def add_order(self, o: Opportunity, count: float, mode: str, improved: bool, ts: str) -> int:
        cur = self.db.execute(
            "INSERT INTO orders (ts,mode,ticker,event_ticker,side,price,count,fair,fair_sd,improved) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ts, mode, o.ticker, o.event_ticker, o.side, o.price, count, o.fair, o.fair_sd, int(improved)),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def resting_orders(self) -> list:
        return self.db.execute("SELECT * FROM orders WHERE status='resting' ORDER BY id").fetchall()

    def close_order(self, order_id: int, status: str, ts: str, reason: str = "", trade_id: Optional[int] = None) -> None:
        self.db.execute("UPDATE orders SET status=?, done_ts=?, reason=?, trade_id=? WHERE id=?",
                        (status, ts, reason, trade_id, order_id))
        self.db.commit()

    def resting_contracts(self, ticker: str) -> float:
        r = self.db.execute("SELECT COALESCE(SUM(count),0) FROM orders WHERE status='resting' AND ticker=?", (ticker,))
        return float(r.fetchone()[0])

    def resting_exposure(self, event_ticker: Optional[str] = None) -> float:
        """Cost if every resting order filled (risk limits count it as already spent)."""
        if event_ticker is None:
            r = self.db.execute("SELECT COALESCE(SUM(price*count),0) FROM orders WHERE status='resting'")
        else:
            r = self.db.execute("SELECT COALESCE(SUM(price*count),0) FROM orders WHERE status='resting' "
                                "AND event_ticker=?", (event_ticker,))
        return float(r.fetchone()[0])

    def order_stats(self) -> dict:
        r = self.db.execute(
            "SELECT COUNT(*) n, SUM(status='filled') filled, SUM(status='cancelled') cancelled, "
            "SUM(status='resting') resting FROM orders"
        ).fetchone()
        n = r["n"] or 0
        done = (r["filled"] or 0) + (r["cancelled"] or 0)
        return {"orders_placed": n, "orders_filled": r["filled"] or 0, "orders_cancelled": r["cancelled"] or 0,
                "orders_resting": r["resting"] or 0,
                "fill_rate": round((r["filled"] or 0) / done, 3) if done else None}

    def liquidity_breakdown(self, mode: Optional[str] = None) -> list[dict]:
        """Expected vs realized P&L for maker fills vs taker trades.

        For maker fills, `expected` uses the model's fair value when the fill
        was detected, so if fills mostly happen when the market moves against
        us (adverse selection), it shows up as smaller or negative edge here.
        """
        where = "AND mode=?" if mode else ""
        args = (mode,) if mode else ()
        rows = self.db.execute(
            f"SELECT COALESCE(liquidity,'taker') liq, COUNT(*) n, SUM(edge*count) exp, SUM(pnl) pnl, "
            f"AVG(pnl>0) win, SUM(fee) fees FROM trades WHERE status='settled' {where} GROUP BY liq ORDER BY liq",
            args,
        ).fetchall()
        return [{"liquidity": r["liq"], "trades": r["n"], "expected": round(r["exp"], 2),
                 "realized": round(r["pnl"], 2), "win_rate": round(r["win"], 3), "fees": round(r["fees"], 2)}
                for r in rows]

    def set_result(self, ticker: str, result: str, ts: Optional[str] = None) -> int:
        """Record a settlement and close all trades in that market. Returns trades settled."""
        if result not in ("yes", "no"):
            raise ValueError("result must be yes or no")
        ts = ts or datetime.now(timezone.utc).isoformat()
        self.db.execute("INSERT OR REPLACE INTO results VALUES (?,?,?)", (ticker, result, ts))
        self.db.execute("UPDATE markets SET result=? WHERE ticker=?", (result, ticker))
        rows = self.db.execute("SELECT * FROM trades WHERE ticker=? AND status='open'", (ticker,)).fetchall()
        for r in rows:
            won = r["side"] == result
            pnl = (1.0 - r["price"]) * r["count"] - r["fee"] if won else -r["price"] * r["count"] - r["fee"]
            self.db.execute(
                "UPDATE trades SET status='settled', result=?, pnl=? WHERE id=?", (result, round(pnl, 6), r["id"])
            )
        self.db.commit()
        return len(rows)

    # ---- reads used by the risk manager ----
    def taken_at(self, ticker: str, side: str, price: float, mode: str = "paper") -> float:
        """Contracts our own taker trades already bought at this exact ask.

        Paper fills never touch the real book, so Kalshi keeps showing the
        seller's full size. A real order would have used those contracts up.
        """
        r = self.db.execute(
            "SELECT COALESCE(SUM(count),0) FROM trades WHERE ticker=? AND side=? AND mode=? "
            "AND ABS(price-?) < 1e-6 AND COALESCE(liquidity,'taker')='taker'",
            (ticker, side, mode, price),
        )
        return float(r.fetchone()[0])

    def market_contracts(self, ticker: str) -> float:
        r = self.db.execute("SELECT COALESCE(SUM(count),0) FROM trades WHERE ticker=? AND status='open'", (ticker,))
        return float(r.fetchone()[0])

    def exposure(self, event_ticker: Optional[str] = None) -> float:
        """Cost basis of open positions, overall or within one event."""
        if event_ticker is None:
            r = self.db.execute("SELECT COALESCE(SUM(price*count+fee),0) FROM trades WHERE status='open'")
        else:
            r = self.db.execute(
                "SELECT COALESCE(SUM(price*count+fee),0) FROM trades WHERE status='open' AND event_ticker=?",
                (event_ticker,),
            )
        return float(r.fetchone()[0])

    def daily_pnl(self, day: Optional[str] = None) -> float:
        day = day or datetime.now(timezone.utc).date().isoformat()
        r = self.db.execute(
            "SELECT COALESCE(SUM(t.pnl),0) FROM trades t JOIN results r ON t.ticker=r.ticker "
            "WHERE t.status='settled' AND substr(r.ts,1,10)=?",
            (day,),
        )
        return float(r.fetchone()[0])

    def unsettled_tickers(self, now: Optional[datetime] = None) -> list[str]:
        """Markets we priced or hold, with no result yet, whose close time has passed.

        Indexed lookup, so it stays fast as history grows, and only finished
        markets are checked, so settlement costs one API lookup per market.
        """
        now = now or datetime.now(timezone.utc)
        out = {r[0] for r in self.db.execute(
            "SELECT ticker FROM markets WHERE result IS NULL AND close_ts <= ?", (now.timestamp(),))}
        # Open trades in markets we never snapshotted: always check.
        out.update(r[0] for r in self.db.execute(
            "SELECT DISTINCT ticker FROM trades WHERE status='open' AND ticker NOT IN (SELECT ticker FROM markets)"))
        return sorted(out)

    # ---- reporting ----
    def report(self, mode: Optional[str] = None) -> dict:
        where = "WHERE mode=?" if mode else ""
        args = (mode,) if mode else ()
        t = self.db.execute(
            f"SELECT COUNT(*) n, SUM(status='settled') settled, COALESCE(SUM(pnl),0) pnl, "
            f"SUM(pnl>0) wins, "
            f"COALESCE(SUM(CASE WHEN status='settled' THEN edge*count END),0) exp_pnl_settled, "
            f"COALESCE(SUM(CASE WHEN status='settled' THEN price*count+fee END),0) cost_settled, "
            f"COALESCE(SUM(fee),0) fees FROM trades {where}",
            args,
        ).fetchone()
        settled = t["settled"] or 0
        out = {
            "trades": t["n"],
            "settled": settled,
            "realized_pnl": round(t["pnl"], 2),
            "expected_pnl_settled": round(t["exp_pnl_settled"], 2),
            "fees_paid": round(t["fees"], 2),
            "win_rate": round((t["wins"] or 0) / settled, 3) if settled else None,
            "return_on_cost": round(t["pnl"] / t["cost_settled"], 3) if t["cost_settled"] else None,
        }
        out.update(self.pnl_check(mode))
        out.update(self.calibration())
        return out

    def current_pnl(self, mode: Optional[str] = None) -> dict:
        """Realized P&L plus open positions marked at what they'd sell for now.

        Open trades are valued at the latest stored snapshot's bid for their side
        (YES at yes_bid, NO at 1 - yes_ask), so no network call is needed. That is
        a conservative liquidation value; a missing bid counts as 0.
        """
        where, args = ("AND mode=?", (mode,)) if mode else ("", ())
        realized, settled = self.db.execute(
            f"SELECT COALESCE(SUM(pnl),0), COUNT(*) FROM trades WHERE status='settled' {where}", args
        ).fetchone()
        rows = self.db.execute(
            "SELECT t.side, t.price, t.count, t.fee, s.yes_bid, s.yes_ask, s.ts snap_ts FROM trades t "
            "LEFT JOIN snapshots s ON s.rowid = (SELECT MAX(rowid) FROM snapshots WHERE ticker = t.ticker) "
            f"WHERE t.status='open' {where}",
            args,
        ).fetchall()
        cost = value = 0.0
        unpriced = 0
        last_ts = None
        for r in rows:
            cost += r["price"] * r["count"] + r["fee"]
            if r["snap_ts"] is None:
                unpriced += 1
                value += r["price"] * r["count"]  # no quote stored: carry at cost
                continue
            bid = r["yes_bid"] if r["side"] == "yes" else (1.0 - r["yes_ask"] if r["yes_ask"] is not None else None)
            value += (bid or 0.0) * r["count"]
            last_ts = max(last_ts or r["snap_ts"], r["snap_ts"])
        unrealized = value - cost
        return {
            "realized_pnl": round(realized, 2),
            "settled_trades": settled,
            "open_trades": len(rows),
            "open_cost": round(cost, 2),
            "open_value": round(value, 2),
            "unrealized_pnl": round(unrealized, 2),
            "current_pnl": round(realized + unrealized, 2),
            "unpriced_open_trades": unpriced,
            "quotes_as_of": last_ts,
        }

    def pnl_check(self, mode: Optional[str] = None) -> dict:
        """Is realized P&L consistent with expected? Uses events as independent units.

        For each event, gap = realized - expected. Under honest probabilities
        the gaps average zero, so z = sum(gap) / (sd(gap) * sqrt(n_events))
        should usually lie in [-2, 2].
        """
        where = "AND mode=?" if mode else ""
        args = (mode,) if mode else ()
        rows = self.db.execute(
            f"SELECT event_ticker, SUM(pnl) pnl, SUM(edge*count) exp FROM trades "
            f"WHERE status='settled' {where} GROUP BY event_ticker",
            args,
        ).fetchall()
        n = len(rows)
        if n < 2:
            return {"events": n}
        gap = np.array([r["pnl"] - r["exp"] for r in rows])
        sd = float(gap.std(ddof=1))
        total = float(gap.sum())
        half_band = 1.96 * sd * math.sqrt(n)
        return {
            "events": n,
            "pnl_minus_expected": round(total, 2),
            "pnl_95pct_band": [round(-half_band, 2), round(half_band, 2)],
            "pnl_gap_z": round(total / (sd * math.sqrt(n)), 2) if sd > 0 else None,
        }

    def calibration(self, buckets: int = 10) -> dict:
        """Model vs market on spaced snapshots taken at least 10 minutes before close."""
        from .calibrate import training_rows

        pm, pk, y = training_rows(self.db)
        n = int(y.size)
        if n == 0:
            return {"calibration_points": 0}

        def ll(p):
            p = np.clip(p, 1e-4, 1 - 1e-4)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        table = []
        idx = np.minimum((pm * buckets).astype(int), buckets - 1)
        for b in range(buckets):
            m = idx == b
            if m.any():
                table.append(
                    {
                        "bucket": f"{b / buckets:.1f}-{(b + 1) / buckets:.1f}",
                        "n": int(m.sum()),
                        "model_avg": round(float(pm[m].mean()), 3),
                        "market_avg": round(float(pk[m].mean()), 3),
                        "actual_freq": round(float(y[m].mean()), 3),
                    }
                )
        return {
            "calibration_points": n,
            "logloss_model": round(ll(pm), 5),
            "logloss_market": round(ll(pk), 5),
            "brier_model": round(float(np.mean((pm - y) ** 2)), 5),
            "brier_market": round(float(np.mean((pk - y) ** 2)), 5),
            "calibration_buckets": table,
        }

    def breakdown(self, mode: Optional[str] = None) -> list[dict]:
        """Expected vs realized P&L by side and fair-value bucket."""
        where = "AND mode=?" if mode else ""
        args = (mode,) if mode else ()
        rows = self.db.execute(
            f"SELECT side, MIN(CAST(fair*5 AS INT),4) b, COUNT(*) n, SUM(count) contracts, "
            f"SUM(edge*count) exp, SUM(pnl) pnl, AVG(pnl>0) win, AVG(price) px, AVG(fair) fair "
            f"FROM trades WHERE status='settled' {where} GROUP BY side, b ORDER BY side, b",
            args,
        ).fetchall()
        return [
            {
                "side": r["side"],
                "fair_bucket": f"{r['b'] / 5:.1f}-{(r['b'] + 1) / 5:.1f}",
                "trades": r["n"],
                "contracts": r["contracts"],
                "expected": round(r["exp"], 2),
                "realized": round(r["pnl"], 2),
                "win_rate": round(r["win"], 3),
                "avg_price": round(r["px"], 3),
                "avg_fair": round(r["fair"], 3),
            }
            for r in rows
        ]

    def trades(self, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))]
