"""Side-by-side scorecard for two paper-trading databases (an A/B test).

Each bot writes its own SQLite ledger. To compare them fairly:

  * only activity after a common start time counts (default: when the later
    of the two databases began), so a head start doesn't count as skill;
  * model accuracy (log loss vs the market) is measured on the same settled
    markets for both, spaced snapshots at least 10 minutes before close.

Lower log loss is better. "edge over market" = market log loss minus model log
loss, so positive means the model's probabilities beat Kalshi's prices.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np


def _open(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def first_snapshot(db: sqlite3.Connection) -> Optional[str]:
    r = db.execute("SELECT MIN(ts) FROM snapshots").fetchone()
    return r[0] if r else None


def _accuracy_rows(db, since: str, tickers: Optional[set], spacing_min: float = 30.0, min_secs: float = 600.0):
    rows = db.execute(
        "SELECT s.ticker, s.ts, s.fair_p, s.yes_bid, s.yes_ask, r.result FROM snapshots s "
        "JOIN results r ON s.ticker = r.ticker WHERE s.ts >= ? AND s.yes_bid IS NOT NULL "
        "AND s.yes_ask IS NOT NULL AND s.secs_to_close >= ? ORDER BY s.ticker, s.ts",
        (since, min_secs),
    ).fetchall()
    out, last = [], {}
    for r in rows:
        if tickers is not None and r["ticker"] not in tickers:
            continue
        t = datetime.fromisoformat(r["ts"])
        prev = last.get(r["ticker"])
        if prev is not None and (t - prev).total_seconds() < spacing_min * 60:
            continue
        last[r["ticker"]] = t
        out.append((r["fair_p"], 0.5 * (r["yes_bid"] + r["yes_ask"]), 1.0 if r["result"] == "yes" else 0.0))
    return out


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def settled_tickers(db, since: str) -> set:
    return {r[0] for r in db.execute(
        "SELECT DISTINCT s.ticker FROM snapshots s JOIN results r ON s.ticker = r.ticker WHERE s.ts >= ?", (since,))}


def scorecard(db, since: str, tickers: Optional[set] = None) -> dict:
    t = db.execute(
        "SELECT COUNT(*) n, SUM(status='settled') settled, COALESCE(SUM(pnl),0) pnl, "
        "COALESCE(SUM(CASE WHEN status='settled' THEN edge*count END),0) exp, COALESCE(SUM(fee),0) fees, "
        "SUM(pnl>0) wins FROM trades WHERE ts >= ?", (since,)).fetchone()
    has_liq = any(c[1] == "liquidity" for c in db.execute("PRAGMA table_info(trades)"))
    maker = db.execute("SELECT COUNT(*) FROM trades WHERE ts >= ? AND liquidity='maker'", (since,)).fetchone()[0] \
        if has_liq else 0
    ev = db.execute(
        "SELECT SUM(pnl) - SUM(edge*count) gap FROM trades WHERE status='settled' AND ts >= ? GROUP BY event_ticker",
        (since,)).fetchall()
    gaps = np.array([r[0] for r in ev], dtype=float)
    z = None
    if gaps.size >= 2 and gaps.std(ddof=1) > 0:
        z = float(gaps.sum() / (gaps.std(ddof=1) * math.sqrt(gaps.size)))
    acc = _accuracy_rows(db, since, tickers)
    out = {
        "trades": t["n"], "maker_fills": maker, "settled": t["settled"] or 0,
        "realized_pnl": round(t["pnl"], 2), "expected_pnl": round(t["exp"], 2), "fees": round(t["fees"], 2),
        "win_rate": round((t["wins"] or 0) / t["settled"], 3) if t["settled"] else None,
        "events": int(gaps.size), "pnl_gap_z": round(z, 2) if z is not None else None,
        "accuracy_points": len(acc),
    }
    if acc:
        a = np.array(acc)
        lm, lk = _ll(a[:, 0], a[:, 2]), _ll(a[:, 1], a[:, 2])
        out.update(logloss_model=round(lm, 4), logloss_market=round(lk, 4), edge_over_market=round(lk - lm, 4))
    return out


def compare(path_a: str, path_b: str, since: Optional[str] = None) -> tuple[str, dict, dict]:
    a, b = _open(path_a), _open(path_b)
    if since is None:
        starts = [s for s in (first_snapshot(a), first_snapshot(b)) if s]
        since = max(starts) if starts else "1970-01-01"
    common = settled_tickers(a, since) & settled_tickers(b, since)
    return since, scorecard(a, since, common), scorecard(b, since, common)
