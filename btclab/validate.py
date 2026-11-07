"""Is the pricing model calibrated? Test it on historical BTC candles.

This needs no Kalshi history. At many past moments we:
  1. estimate vol from the previous few hours of 1-minute closes,
  2. lay out $W-wide brackets around spot, like a Kalshi range event,
  3. predict each bracket's probability for the price H hours later,
  4. check which bracket BTC actually landed in.

Metrics (lower is better for both):
  log_loss  -mean(log p[actual bracket])
  brier     mean squared error across all brackets of the event
Plus interval coverage: how often the outcome fell inside the model's central
50% and 90% ranges (a calibrated model gives ~0.50 and ~0.90).

Comparing lognormal vs Student-t here is the core quant result: it shows
whether fat tails improve pricing, before risking a dollar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from .pricing import Contract, ModelParams, fair_prob
from .vol import ewma_vol

MIN_PER_YEAR = 365 * 24 * 60


@dataclass
class ValidationResult:
    model: str
    n: int
    log_loss: float
    brier: float
    cover50: float
    cover90: float

    def row(self) -> str:
        return (
            f"{self.model:<11} n={self.n:<5} log_loss={self.log_loss:.4f}  brier={self.brier:.5f}  "
            f"50%-cover={self.cover50:.3f}  90%-cover={self.cover90:.3f}"
        )


def _quantile(spot: float, tau: float, p: ModelParams, q: float) -> float:
    s = p.sigma * math.sqrt(tau)
    mu = -0.5 * s * s
    if p.model == "lognormal":
        z = stats.norm.ppf(q) * s
    else:
        z = stats.t.ppf(q, df=p.nu) * s * math.sqrt((p.nu - 2) / p.nu)
    return spot * math.exp(mu + z)


def run(
    closes: Sequence[float],
    horizon_min: int = 60,
    lookback_min: int = 240,
    step_min: int = 30,
    width: float = 250.0,
    n_brackets: int = 40,
    models: Sequence[tuple[str, float]] = (("lognormal", 0.0), ("student_t", 4.0), ("student_t", 3.0)),
    start_index: int = 0,
) -> list[ValidationResult]:
    c = np.asarray(closes, dtype=float)
    out = []
    for model, nu in models:
        ll = br = 0.0
        c50 = c90 = 0
        n = 0
        for i in range(max(lookback_min, start_index), len(c) - horizon_min, step_min):
            spot, final = c[i], c[i + horizon_min]
            sigma = ewma_vol(c[i - lookback_min : i + 1], 1.0, 60.0)
            p = ModelParams(sigma, model, nu or 4.0)
            tau = horizon_min / MIN_PER_YEAR
            center = round(spot / width) * width
            lo0 = center - n_brackets // 2 * width
            edges = [lo0 + k * width for k in range(n_brackets + 1)]
            probs = [fair_prob(Contract("between", edges[k], edges[k + 1]), spot, tau, p) for k in range(n_brackets)]
            # Open-ended tails so the event covers every outcome.
            probs = [fair_prob(Contract("less", None, edges[0]), spot, tau, p)] + probs
            probs.append(fair_prob(Contract("greater", edges[-1], None), spot, tau, p))
            hit = 0 if final < edges[0] else (len(probs) - 1 if final >= edges[-1] else 1 + int((final - edges[0]) // width))
            y = np.zeros(len(probs))
            y[hit] = 1.0
            pr = np.clip(np.asarray(probs), 1e-6, 1.0)
            ll += -math.log(pr[hit])
            br += float(np.sum((pr - y) ** 2))
            c50 += _quantile(spot, tau, p, 0.25) <= final <= _quantile(spot, tau, p, 0.75)
            c90 += _quantile(spot, tau, p, 0.05) <= final <= _quantile(spot, tau, p, 0.95)
            n += 1
        name = model if model == "lognormal" else f"t(nu={nu:g})"
        out.append(ValidationResult(name, n, ll / n, br / n, c50 / n, c90 / n) if n else ValidationResult(name, 0, *(math.nan,) * 4))
    return out


NU_GRID = (4.5, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)


def fit_nu_by_horizon(
    closes: Sequence[float],
    horizons_min: Sequence[int] = (15, 60, 360, 1440),
    width: float = 250.0,
) -> tuple[list[list[float]], list[tuple[int, list[ValidationResult]]]]:
    """Pick the tail parameter with the lowest log loss at each horizon.

    Returns (table for EngineConfig.nu_by_horizon, full results per horizon).
    A nu of 0 in the table means lognormal won.
    """
    models = [("lognormal", 0.0)] + [("student_t", nu) for nu in NU_GRID]
    table, detail = [], []
    for h in horizons_min:
        # Wider brackets at longer horizons keep the event a sensible size.
        w = width * max(1.0, math.sqrt(h / 60))
        res = run(closes, horizon_min=h, width=w, step_min=max(30, h // 4), models=models)
        res = [r for r in res if r.n > 0]
        if not res:
            continue
        best_i = min(range(len(res)), key=lambda i: res[i].log_loss)
        nu = 0.0 if models[best_i][0] == "lognormal" else models[best_i][1]
        table.append([round(h / 60, 4), nu])
        detail.append((h, res))
    return table, detail


def run_fhs(
    closes: Sequence[float],
    timestamps: Sequence[int],
    horizon_min: int = 60,
    fit_window_min: int = 4 * 1440,
    refit_every_min: int = 720,
    step_min: int = 30,
    width: float = 250.0,
    n_brackets: int = 40,
    n_paths: int = 8000,
    vol_uncertainty: float = 0.15,
    seed: int = 0,
) -> ValidationResult:
    """Out-of-sample test of the GARCH + Monte Carlo pricer.

    The GARCH model only ever sees data before each forecast time: it is refit
    every `refit_every_min` on the trailing window and filtered forward in
    between, so nothing leaks from the future. Historical implied vol isn't
    available here, so this is a slightly harder test than live trading.

    Indexing: r[j] = log(c[j+1] / c[j]) is the return ending at ts[j+1].
    """
    from .fhs import FHSPricer
    from .garch import fit_garch

    c = np.asarray(closes, dtype=float)
    ts = np.asarray(timestamps, dtype=np.int64)
    r = np.diff(np.log(c))
    pricer = FHSPricer(n_paths, vol_uncertainty, basis_usd=0.0, seed=seed)
    fit = None
    fit_at = filtered_to = 0
    ll = br = 0.0
    c50 = c90 = n = 0
    for i in range(fit_window_min, len(c) - horizon_min, step_min):
        if fit is None or i - fit_at >= refit_every_min:
            start = tuple(fit.extra["x"]) if fit is not None else None
            fit = fit_garch(r[i - fit_window_min : i], ts[i - fit_window_min + 1 : i + 1], start=start)
            fit_at = filtered_to = i
        elif filtered_to < i:
            fit.update(r[filtered_to:i], ts[filtered_to + 1 : i + 1])
            filtered_to = i
        spot, final = c[i], c[i + horizon_min]
        st = spot * np.exp(pricer.simulate(fit, spot, int(ts[i]), [horizon_min * 60])[0].log_ret)
        center = round(spot / width) * width
        edges = center - n_brackets // 2 * width + width * np.arange(n_brackets + 1)
        probs = np.histogram(st, bins=np.concatenate([[-np.inf], edges, [np.inf]]))[0] / st.size
        hit = 0 if final < edges[0] else (len(probs) - 1 if final >= edges[-1] else 1 + int((final - edges[0]) // width))
        y = np.zeros(len(probs))
        y[hit] = 1.0
        ll += -math.log(max(probs[hit], 1e-6))
        br += float(np.sum((probs - y) ** 2))
        q = np.quantile(st, [0.05, 0.25, 0.75, 0.95])
        c50 += q[1] <= final <= q[2]
        c90 += q[0] <= final <= q[3]
        n += 1
    return ValidationResult("garch+fhs", n, ll / n, br / n, c50 / n, c90 / n)
