"""Learn how much to trust the model vs. the market, from settled results.

The market price carries information the model doesn't have (order flow,
news, other traders' models). Instead of assuming either is right, we fit

    P(YES) = sigmoid(a + b * logit(p_model) + c * logit(p_market_mid))

by maximum likelihood on settled markets. (a, b, c) = (0, 1, 0) means "trust
the model completely"; b < 1 means the model is overconfident; c > 0 means
the market adds information. The fit is regularized toward (0, 1, 0), so with
little data it barely moves, and it is only switched on after `min_markets`
settled markets.

Training rows are snapshots spaced at least `spacing_min` apart per market
and at least `min_secs` before close, so near-certain last-minute snapshots
don't dominate and one market can't contribute hundreds of rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from scipy import optimize, special

EPS = 1e-4


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Blend:
    a: float = 0.0
    b: float = 1.0
    c: float = 0.0
    n: int = 0
    active: bool = False
    logloss_model: Optional[float] = None
    logloss_market: Optional[float] = None
    logloss_blend: Optional[float] = None

    def apply(self, p_model: float, p_mid: Optional[float]) -> float:
        if not self.active or p_mid is None:
            return p_model
        x = self.a + self.b * float(logit(p_model)) + self.c * float(logit(p_mid))
        return float(special.expit(x))


def training_rows(db, spacing_min: float = 30.0, min_secs: float = 600.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = db.execute(
        "SELECT s.ticker, s.ts, s.fair_p, s.yes_bid, s.yes_ask, r.result FROM snapshots s "
        "JOIN results r ON s.ticker = r.ticker "
        "WHERE s.yes_bid IS NOT NULL AND s.yes_ask IS NOT NULL AND s.secs_to_close >= ? "
        "ORDER BY s.ticker, s.ts",
        (min_secs,),
    ).fetchall()
    pm, pk, y = [], [], []
    last: dict[str, datetime] = {}
    for r in rows:
        t = datetime.fromisoformat(r["ts"])
        prev = last.get(r["ticker"])
        if prev is not None and (t - prev).total_seconds() < spacing_min * 60:
            continue
        last[r["ticker"]] = t
        pm.append(r["fair_p"])
        pk.append(0.5 * (r["yes_bid"] + r["yes_ask"]))
        y.append(1.0 if r["result"] == "yes" else 0.0)
    return np.array(pm), np.array(pk), np.array(y)


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_blend(
    p_model: np.ndarray,
    p_mid: np.ndarray,
    y: np.ndarray,
    l2: float = 5.0,
    min_markets: int = 300,
) -> Blend:
    n = int(y.size)
    if n < 30:
        return Blend(n=n)
    lm, lk = logit(p_model), logit(p_mid)
    prior = np.array([0.0, 1.0, 0.0])

    def nll(th):
        z = th[0] + th[1] * lm + th[2] * lk
        # log(1 + e^z) - y z, computed stably
        return float(np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum((th - prior) ** 2))

    def grad(th):
        z = th[0] + th[1] * lm + th[2] * lk
        g = special.expit(z) - y
        return np.array([g.sum(), (g * lm).sum(), (g * lk).sum()]) + 2 * l2 * (th - prior)

    res = optimize.minimize(nll, prior, jac=grad, method="L-BFGS-B")
    a, b, c = (float(v) for v in res.x)
    pb = special.expit(a + b * lm + c * lk)
    return Blend(
        a=a,
        b=b,
        c=c,
        n=n,
        active=n >= min_markets,
        logloss_model=_logloss(p_model, y),
        logloss_market=_logloss(p_mid, y),
        logloss_blend=_logloss(pb, y),
    )


def fit_from_ledger(db, **kw) -> Blend:
    pm, pk, y = training_rows(db)
    return fit_blend(pm, pk, y, **kw)


# ---------------------------------------------------------------------------
# Disagreement shrinkage: how much of (model - market) is real?
# ---------------------------------------------------------------------------
#
# The logistic blend above treats every snapshot alike, but most snapshots are
# near-certain far brackets where model and market agree. Trading only happens
# where they disagree, so what matters is: given the model says p and the
# market mid is m, what is E[outcome - m]?
#
#     E[y - m | d] = beta(d, contract class) * d,     d = p_model - m
#
# beta is estimated separately for each disagreement bin and contract class
# ("between" brackets vs "above/below" thresholds), because reliability is not
# constant: small disagreements tend to be real edge, large ones are often the
# model being wrong (stale vol after a regime change, etc.).
#
# Estimation details that keep this honest:
#   * Bayesian prior: range brackets start at beta0 = 0.5 (trust half of any
#     disagreement). Data is ignored until 10 distinct days have settled, then
#     combined with the prior weighted by its day-clustered uncertainty.
#   * Cluster-robust standard errors by DAY. Brackets in one event share one
#     price outcome, and hourly events that overlap in time share the same
#     price path, so neither markets nor events are independent observations.
#     (Clustering by event made the errors far too small: the evidence gate
#     opened on noise in simulation testing.) With few days of data the
#     errors are large, which is the honest answer.
#   * The beta actually used is the posterior minus `z_se` standard errors,
#     clipped to [0, 1]. We never trust the model *more* than it claims.

D_EDGES = (-1.0, -0.10, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.10, 1.0)
D_CENTERS = (-0.15, -0.08, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.08, 0.15)
CLASSES = ("between", "threshold")


def contract_class(strike_type: str) -> str:
    return "between" if strike_type == "between" else "threshold"


def d_bin(d: float) -> int:
    for i in range(len(D_EDGES) - 1):
        if d < D_EDGES[i + 1]:
            return i
    return len(D_EDGES) - 2


@dataclass
class ShrinkTable:
    beta: dict  # (class, bin) -> beta used
    beta_raw: dict  # (class, bin) -> point estimate
    se: dict  # (class, bin) -> cluster-robust standard error
    n: dict  # (class, bin) -> rows
    rows: int = 0
    events: int = 0
    beta0: float = 0.5
    events_by_bin: dict = None  # (class, bin) -> distinct events

    @classmethod
    def prior(cls, beta0: float = 0.5, evidence_classes: tuple = ()) -> "ShrinkTable":
        keys = [(c, b) for c in CLASSES for b in range(len(D_CENTERS))]
        start = {k: (0.0 if k[0] in evidence_classes else beta0) for k in keys}
        return cls(start, {k: beta0 for k in keys}, {k: None for k in keys}, {k: 0 for k in keys},
                   beta0=beta0, events_by_bin={k: 0 for k in keys})

    def apply(self, p_model: float, p_mid: Optional[float], strike_type: str) -> float:
        if p_mid is None:
            return p_model
        d = p_model - p_mid
        b = self.beta[(contract_class(strike_type), d_bin(d))]
        return min(1.0, max(0.0, p_mid + b * d))

    def as_rows(self) -> list[dict]:
        out = []
        for (c, b), v in sorted(self.beta.items()):
            out.append({"class": c, "d_bin": f"{D_EDGES[b]:+.2f}..{D_EDGES[b + 1]:+.2f}", "n": self.n[(c, b)],
                        "events": (self.events_by_bin or {}).get((c, b), 0),
                        "beta_est": round(self.beta_raw[(c, b)], 3),
                        "se": None if self.se[(c, b)] is None else round(self.se[(c, b)], 3), "beta_used": round(v, 3)})
        return out


def fit_shrink(
    d: np.ndarray,
    g: np.ndarray,
    klass: np.ndarray,
    cluster: np.ndarray,
    beta0: float = 0.5,
    prior_sd: float = 0.25,
    z_se: float = 1.0,
    min_clusters: int = 10,
    evidence_classes: tuple = ("threshold",),
    evidence_min_clusters: int = 20,
    evidence_min_events: int = 30,
    evidence_z: float = 2.0,
    events: Optional[np.ndarray] = None,
    prior_obs: Optional[float] = None,  # deprecated, ignored
) -> ShrinkTable:
    """Fit beta per (class, disagreement bin). g = outcome - market mid.

    `cluster` groups rows that are not independent; the engine uses the UTC
    day, because every market settling on one day rides one price path.
    `events` (optional) only counts distinct events per bin.

    Estimation, per bin:
      1. OLS slope beta_hat of g on d, with a cluster-robust standard error
         (small-sample factor G/(G-1)).
      2. Fewer than `min_clusters` days: data ignored, beta = prior beta0.
         Cluster-robust errors are unreliable with few clusters, and in
         testing a handful of agreeing days produced false confidence.
      3. Otherwise, normal-normal Bayesian update of the prior
         N(beta0, prior_sd^2) with N(beta_hat, se^2); the beta used is the
         posterior mean minus `z_se` posterior sds, clipped to [0, 1].
      4. Classes in `evidence_classes` (above/below contracts) have prior 0 and
         stay at 0 unless there are >= `evidence_min_clusters` days and
         >= `evidence_min_events` events and beta_hat > `evidence_z` * se.
    """
    t = ShrinkTable.prior(beta0, tuple(evidence_classes))
    ev = events if events is not None else cluster
    t.rows = int(d.size)
    t.events = int(len(set(ev.tolist()))) if d.size else 0
    if d.size == 0:
        return t
    bins = np.array([d_bin(x) for x in d])
    for c in CLASSES:
        gated = c in evidence_classes
        b0 = 0.0 if gated else beta0
        for b in range(len(D_CENTERS)):
            m = (klass == c) & (bins == b)
            n = int(m.sum())
            t.n[(c, b)] = n
            if n == 0:
                continue
            dd, gg, cl = d[m], g[m], cluster[m]
            t.events_by_bin[(c, b)] = len(set(ev[m].tolist()))
            sdd = float(np.sum(dd * dd))
            beta_hat = float(np.sum(dd * gg)) / sdd
            u = dd * (gg - beta_hat * dd)
            _, inv = np.unique(cl, return_inverse=True)
            sc = np.bincount(inv, weights=u)
            n_cl = sc.size
            se = math.sqrt((n_cl / (n_cl - 1) if n_cl > 1 else 1e6) * float(np.sum(sc * sc))) / sdd
            t.beta_raw[(c, b)] = beta_hat
            t.se[(c, b)] = se
            if n_cl < min_clusters:
                used = b0
            else:
                wd, wp = 1.0 / max(se, 1e-9) ** 2, 1.0 / prior_sd**2
                post = (wd * beta_hat + wp * b0) / (wd + wp)
                post_sd = 1.0 / math.sqrt(wd + wp)
                used = post - z_se * post_sd
            if gated and (n_cl < evidence_min_clusters or t.events_by_bin[(c, b)] < evidence_min_events
                          or beta_hat - evidence_z * se <= 0):
                used = 0.0
            t.beta[(c, b)] = float(min(1.0, max(0.0, used)))
    return t


def shrink_rows(db, min_secs: float = 300.0, max_days: float = 30.0) -> tuple[np.ndarray, ...]:
    """Settled snapshots with a two-sided market, from the last `max_days`.

    Returns (d, g, class, day cluster, event).
    """
    rows = db.execute(
        "SELECT s.fair_p, s.yes_bid, s.yes_ask, s.strike_type, s.event_ticker, s.ts, r.result FROM snapshots s "
        "JOIN results r ON s.ticker = r.ticker "
        "WHERE s.yes_bid IS NOT NULL AND s.yes_ask IS NOT NULL AND s.secs_to_close >= ?",
        (min_secs,),
    ).fetchall()
    if not rows:
        e = np.array([])
        o = np.array([], dtype=object)
        return e, e, o, o, o
    latest = max(datetime.fromisoformat(r["ts"]) for r in rows)
    keep = [r for r in rows if (latest - datetime.fromisoformat(r["ts"])).total_seconds() <= max_days * 86400]
    p = np.array([r["fair_p"] for r in keep])
    m = 0.5 * (np.array([r["yes_bid"] for r in keep]) + np.array([r["yes_ask"] for r in keep]))
    y = np.array([1.0 if r["result"] == "yes" else 0.0 for r in keep])
    ok = (m > 0.02) & (m < 0.98)
    klass = np.array([contract_class(r["strike_type"]) for r in keep], dtype=object)
    day = np.array([r["ts"][:10] for r in keep], dtype=object)
    ev = np.array([r["event_ticker"] for r in keep], dtype=object)
    return (p - m)[ok], (y - m)[ok], klass[ok], day[ok], ev[ok]


def fit_shrink_from_ledger(db, **kw) -> ShrinkTable:
    d, g, k, day, ev = shrink_rows(db)
    return fit_shrink(d, g, k, day, events=ev, **kw)
