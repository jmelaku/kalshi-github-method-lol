"""Live terminal dashboard: `python -m btclab watch`.

Reads each bot's SQLite database read-only (the bots keep writing to them) and
redraws every few seconds: account value, P&L and returns, open positions,
resting bids, recent results, and model accuracy. Nothing here touches Kalshi
or any network; it only reads files on this laptop.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .ledger import Ledger

# Bots store at most one snapshot per market every 5 minutes (snapshot_interval_s),
# so "no new snapshot for 7 minutes" is the earliest we can say a bot has stopped.
STALE_S = 420


@dataclass
class Bot:
    name: str
    db: str
    config: str


def default_bots() -> list[Bot]:
    bots = [Bot("Baseline · GARCH + Monte Carlo", "data/btclab.db", "config.json"),
            Bot("Options · Deribit-implied", "data/options.db", "config.options.json")]
    return [b for b in bots if os.path.exists(b.db)]


def _bankroll(config: str) -> float:
    try:
        return float(json.load(open(config)).get("risk", {}).get("bankroll_usd", 500.0))
    except (OSError, ValueError):
        return 500.0


def _local(ts: Optional[float]) -> str:
    if ts is None:
        return "?"
    return datetime.fromtimestamp(ts).strftime("%a %-I:%M%p").replace("AM", "am").replace("PM", "pm")


def _iso_ts(s: Optional[str]) -> Optional[float]:
    return datetime.fromisoformat(s).timestamp() if s else None


def _money(v: Optional[float], sign: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{'+' if sign and v > 0 else '-' if v < 0 else ''}${abs(v):,.2f}"


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:+.1%}"


def _pnl_text(v: Optional[float], fmt=_money) -> Text:
    if v is None:
        return Text("n/a", style="dim")
    return Text(fmt(v), style="green" if v > 0.005 else "red" if v < -0.005 else "")


def _contract(row) -> str:
    """'$75,125–75,375' for a bracket, '> $81,000' / '< $78,000' for above/below."""
    st, lo, hi = row["strike_type"], row["floor"], row["cap"]
    if st == "between" and lo is not None and hi is not None:
        return f"${lo:,.0f}–{hi:,.0f}"
    if st in ("greater", "greater_or_equal") and lo is not None:
        return f"> ${lo:,.0f}"
    if st in ("less", "less_or_equal") and hi is not None:
        return f"< ${hi:,.0f}"
    return "?"


class BotView:
    """Collects one bot's numbers. Slow accuracy stats are cached between redraws."""

    def __init__(self, bot: Bot, slow_every_s: float = 60.0):
        self.bot = bot
        self.bankroll = _bankroll(bot.config)
        self.slow_every_s = slow_every_s
        self._slow: dict = {}
        self._slow_at = 0.0

    def collect(self, now: Optional[float] = None) -> dict:
        now = now or time.time()
        led = Ledger.readonly(self.bot.db)
        try:
            d = self._fast(led, now)
            if now - self._slow_at >= self.slow_every_s or not self._slow:
                self._slow = {**led.pnl_check(), **{k: v for k, v in led.calibration().items()
                                                   if k != "calibration_buckets"}}
                self._slow_at = now
            d.update(self._slow)
            return d
        finally:
            led.db.close()

    def _fast(self, led: Ledger, now: float) -> dict:
        db = led.db
        c = led.current_pnl()
        t = db.execute(
            "SELECT COUNT(*) n, SUM(status='settled') settled, SUM(pnl>0) wins, "
            "COALESCE(SUM(CASE WHEN status='settled' THEN price*count+fee END),0) cost, "
            "COALESCE(SUM(CASE WHEN status='settled' THEN edge*count END),0) exp, "
            "COALESCE(SUM(fee),0) fees FROM trades").fetchone()
        today = datetime.now().strftime("%Y-%m-%d")  # local day
        tt = db.execute("SELECT COUNT(*) FROM trades WHERE substr(ts,1,10) >= ?", (today,)).fetchone()[0]
        last = db.execute("SELECT MAX(ts) FROM snapshots").fetchone()[0]
        last_ts = _iso_ts(last)
        positions = db.execute(
            "SELECT t.ticker, t.side, SUM(t.count) qty, SUM(t.price*t.count)/SUM(t.count) avg_px, "
            "SUM(t.price*t.count+t.fee) cost, s.yes_bid, s.yes_ask, s.strike_type, s.floor, s.cap, m.close_ts "
            "FROM trades t "
            "LEFT JOIN snapshots s ON s.rowid = (SELECT MAX(rowid) FROM snapshots WHERE ticker=t.ticker) "
            "LEFT JOIN markets m ON m.ticker = t.ticker "
            "WHERE t.status='open' GROUP BY t.ticker, t.side ORDER BY m.close_ts, t.ticker").fetchall()
        pos = []
        for p in positions:
            bid = p["yes_bid"] if p["side"] == "yes" else (1 - p["yes_ask"] if p["yes_ask"] is not None else None)
            value = (bid or 0.0) * p["qty"] if p["yes_bid"] is not None or p["yes_ask"] is not None else None
            pos.append({"what": _contract(p), "side": p["side"].upper(), "qty": p["qty"], "avg": p["avg_px"],
                        "bid": bid, "pnl": None if value is None else value - p["cost"],
                        "settles": _local(p["close_ts"])})
        resting = []
        if led.has_table("orders"):
            for r in db.execute(
                    "SELECT o.side, o.price, o.count, o.fair, o.ts, s.strike_type, s.floor, s.cap, m.close_ts "
                    "FROM orders o "
                    "LEFT JOIN snapshots s ON s.rowid = (SELECT MAX(rowid) FROM snapshots WHERE ticker=o.ticker) "
                    "LEFT JOIN markets m ON m.ticker = o.ticker "
                    "WHERE o.status='resting' ORDER BY o.id DESC").fetchall():
                resting.append({"what": _contract(r), "side": r["side"].upper(), "qty": r["count"], "px": r["price"],
                                "fair": r["fair"], "age_min": (now - _iso_ts(r["ts"])) / 60,
                                "settles": _local(r["close_ts"])})
        recent = []
        for r in db.execute(
                "SELECT t.side, t.price, t.count, t.pnl, t.result, s.strike_type, s.floor, s.cap, m.close_ts "
                "FROM trades t "
                "LEFT JOIN snapshots s ON s.rowid = (SELECT MAX(rowid) FROM snapshots WHERE ticker=t.ticker) "
                "LEFT JOIN markets m ON m.ticker = t.ticker "
                "WHERE t.status='settled' ORDER BY m.close_ts DESC, t.id DESC LIMIT 6").fetchall():
            recent.append({"what": _contract(r), "side": r["side"].upper(), "qty": r["count"], "px": r["price"],
                           "won": r["side"] == r["result"], "pnl": r["pnl"], "settled": _local(r["close_ts"])})
        settled = t["settled"] or 0
        return {
            "bankroll": self.bankroll,
            "account": self.bankroll + c["current_pnl"],
            "current": c["current_pnl"], "realized": c["realized_pnl"], "unrealized": c["unrealized_pnl"],
            "ret_bankroll": c["current_pnl"] / self.bankroll if self.bankroll else None,
            "ret_cost": c["realized_pnl"] / t["cost"] if t["cost"] else None,
            "expected": t["exp"], "fees": t["fees"],
            "trades": t["n"], "settled": settled, "open": c["open_trades"], "open_cost": c["open_cost"],
            "win_rate": (t["wins"] or 0) / settled if settled else None,
            "today_trades": tt, "today_pnl": led.daily_pnl(datetime.now(timezone.utc).date().isoformat()),
            "last_ts": last_ts, "running": last_ts is not None and now - last_ts < STALE_S,
            "positions": pos, "resting": resting, "recent": recent,
            "resting_cost": sum(r["px"] * r["qty"] for r in resting),
        }


def _kpis(d: dict) -> Table:
    k = Table.grid(padding=(0, 2))
    k.add_column(style="dim", no_wrap=True)
    k.add_column(justify="right", no_wrap=True)
    k.add_column(style="dim", no_wrap=True)
    k.add_column(justify="right", no_wrap=True)
    z = d.get("pnl_gap_z")
    lm, lk = d.get("logloss_model"), d.get("logloss_market")
    acc = Text("n/a", style="dim")
    if lm is not None and lk is not None:
        better = lk - lm
        acc = Text(f"{'beats' if better > 0 else 'trails'} market by {abs(better):.4f}",
                   style="green" if better > 0 else "red")
    rows = [
        ("Account value", Text(_money(d["account"], sign=False), style="bold"), "Started with", _money(d["bankroll"], False)),
        ("Total P&L", _pnl_text(d["current"]), "Return on account", _pnl_text(d["ret_bankroll"], _pct)),
        ("Realized (final)", _pnl_text(d["realized"]), "Return on money bet", _pnl_text(d["ret_cost"], _pct)),
        ("Open (at the bid)", _pnl_text(d["unrealized"]), "Model expected", _money(d["expected"])),
        ("Today's realized", _pnl_text(d["today_pnl"]), "Fees paid", _money(d["fees"], False)),
        ("Trades", f"{d['trades']}  ({d['settled']} settled · {d['open']} open)", "Win rate",
         "n/a" if d["win_rate"] is None else f"{d['win_rate']:.0%}"),
        ("Money in open bets", _money(d["open_cost"], False), "Resting bids",
         f"{len(d['resting'])}  ({_money(d['resting_cost'], False)})" if d["resting"] else "none"),
        ("z (honesty check)", Text("n/a" if z is None else f"{z:+.2f}", style="" if z is None or abs(z) < 2 else "red"),
         "Accuracy", acc),
    ]
    for r in rows:
        k.add_row(*r)
    return k


def _table(title: str, cols: list[tuple], rows: list[list], empty: str) -> Table:
    t = Table(title=title, title_justify="left", title_style="bold", box=box.SIMPLE_HEAD, expand=True,
              pad_edge=False, show_edge=False)
    for name, kw in cols:
        t.add_column(name, **kw)
    if not rows:
        t.add_row(Text(empty, style="dim"), *[""] * (len(cols) - 1))
    for r in rows:
        t.add_row(*r)
    return t


def render_bot(view: BotView, d: dict, max_rows: int = 8) -> Panel:
    status = (Text("● running", style="green") if d["running"]
              else Text("● stopped", style="red")) + Text(f"   last update {_local(d['last_ts'])}", style="dim")
    R = {"justify": "right", "no_wrap": True}
    pos = _table("Open positions", [("Contract", {"no_wrap": True}), ("Side", {}), ("Qty", R), ("Paid", R),
                                    ("Bid now", R), ("P&L", R), ("Settles", {"no_wrap": True})],
                 [[p["what"], p["side"], f"{p['qty']:g}", f"{p['avg']:.2f}",
                   "n/a" if p["bid"] is None else f"{p['bid']:.2f}", _pnl_text(p["pnl"]), p["settles"]]
                  for p in d["positions"][:max_rows]], "no open positions")
    parts = [status, Text(""), _kpis(d), Text(""), pos]
    if d["resting"] or view.bot.config.endswith("options.json"):
        parts.append(_table("Resting bids (waiting for a seller)",
                            [("Contract", {"no_wrap": True}), ("Side", {}), ("Qty", R), ("Bid", R),
                             ("Fair", R), ("Waiting", R), ("Settles", {"no_wrap": True})],
                            [[r["what"], r["side"], f"{r['qty']:g}", f"{r['px']:.2f}", f"{r['fair']:.2f}",
                              f"{r['age_min']:.0f}m", r["settles"]] for r in d["resting"][:max_rows]],
                            "none"))
    parts.append(_table("Recent results", [("Contract", {"no_wrap": True}), ("Side", {}), ("Qty", R),
                                           ("Paid", R), ("Result", {}), ("P&L", R), ("Settled", {"no_wrap": True})],
                        [[r["what"], r["side"], f"{r['qty']:g}", f"{r['px']:.2f}",
                          Text("won", style="green") if r["won"] else Text("lost", style="red"),
                          _pnl_text(r["pnl"]), r["settled"]] for r in d["recent"]],
                        "nothing settled yet"))
    return Panel(Group(*parts), title=Text(view.bot.name, style="bold"), title_align="left",
                 border_style="grey50", padding=(1, 2))


def render(views: list[BotView], width: int, now: Optional[float] = None):
    now = now or time.time()
    panels = []
    for v in views:
        try:
            panels.append(render_bot(v, v.collect(now)))
        except Exception as e:  # e.g. database briefly locked: show it, try again next redraw
            panels.append(Panel(Text(f"couldn't read {v.bot.db}: {e}", style="red"), title=v.bot.name))
    head = Text.assemble(("Kalshi BTC Lab", "bold"), ("  ·  paper trading, no real money  ·  ", "dim"),
                         (datetime.fromtimestamp(now).strftime("%a %b %-d, %-I:%M:%S %p"), ""),
                         ("   ctrl+c to close", "dim"))
    if len(panels) > 1 and width >= 170:
        grid = Table.grid(expand=True, padding=(0, 1))
        for _ in panels:
            grid.add_column(ratio=1)
        grid.add_row(*panels)
        return Group(head, grid)
    return Group(head, *panels)


def watch(bots: Optional[list[Bot]] = None, refresh_s: float = 5.0, once: bool = False) -> None:
    bots = bots or default_bots()
    console = Console()
    if not bots:
        console.print("No bot databases found in data/. Start a bot first: python -m btclab run")
        return
    views = [BotView(b) for b in bots]
    if once:
        console.print(render(views, console.width))
        return
    with Live(render(views, console.width), console=console, screen=True, auto_refresh=False) as live:
        try:
            while True:
                time.sleep(refresh_s)
                live.update(render(views, console.width), refresh=True)
        except KeyboardInterrupt:
            pass
