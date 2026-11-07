"""Correctness tests for GARCH, the Monte Carlo pricer, and the calibration blend."""

import math

import numpy as np
import pytest
from scipy import special

from btclab.calibrate import fit_blend
from btclab.fhs import FHSPricer
from btclab.garch import GarchFit, fit_garch, garch_filter
from btclab.pricing import Contract, ModelParams, fair_prob

MIN_PER_YEAR = 365 * 24 * 60
T0 = 1_767_225_600  # 2026-01-01 00:00 UTC


def simulate_garch(n, omega, alpha, beta, nu, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
    r = np.empty(n)
    h = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = math.sqrt(h) * z[t]
        h = omega + alpha * r[t] ** 2 + beta * h
    return r


def test_garch_filter_matches_loop():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1e-3, 200)
    h = garch_filter(e, 1e-8, 0.05, 0.9, 1e-6)
    ref = [1e-6]
    for x in e:
        ref.append(1e-8 + 0.05 * x * x + 0.9 * ref[-1])
    assert np.allclose(h, ref)


def test_garch_recovers_parameters():
    target = 0.5**2 / MIN_PER_YEAR
    alpha, beta, nu = 0.04, 0.95, 5.0
    r = simulate_garch(20_000, target * (1 - alpha - beta), alpha, beta, nu, seed=3)
    f = fit_garch(r, T0 + 60 * np.arange(1, r.size + 1))
    assert f.persistence == pytest.approx(alpha + beta, abs=0.01)
    assert f.alpha == pytest.approx(alpha, abs=0.015)
    assert 3.5 < f.nu < 8.0
    assert f.annual_vol_longrun() == pytest.approx(0.5, rel=0.15)
    assert np.std(f.resid) == pytest.approx(1.0, abs=1e-6)


def test_garch_variance_forecast_mean_reverts():
    f = GarchFit(1e-9, 0.05, 0.9, 5, 2e-8, np.ones(24), 8e-8, np.zeros(10), 0.0, 10, T0)
    path = f.expected_variance_path(500)
    assert path[0] == pytest.approx(8e-8)
    assert path[-1] == pytest.approx(2e-8, rel=1e-3)
    assert np.all(np.diff(path) < 0)


def constant_vol_fit(sigma_annual):
    """A GARCH 'fit' with no vol dynamics and normal shocks: pure lognormal."""
    var = sigma_annual**2 / MIN_PER_YEAR
    z = np.random.default_rng(0).standard_normal(200_000)
    z = (z - z.mean()) / z.std()
    return GarchFit(var, 0.0, 0.0, 1e6, var, np.ones(24), var, z, 0.0, z.size, T0)


def test_fhs_matches_closed_form_lognormal():
    fit = constant_vol_fit(0.6)
    pr = FHSPricer(n_paths=200_000, vol_uncertainty=0.0, basis_usd=0.0, seed=2)
    spot, secs = 80_000.0, 2 * 3600
    d = pr.simulate(fit, spot, T0, [secs])[0]
    tau = secs / (365 * 24 * 3600)
    for c in (Contract("greater", 81_000, None), Contract("between", 79_500, 80_249.99), Contract("less", None, 78_000)):
        exact = fair_prob(c, spot, tau, ModelParams(0.6, "lognormal"))
        est = d.prob(c)
        assert est.p == pytest.approx(exact, abs=4 * est.sd + 0.002)


def test_fhs_partition_and_martingale():
    fit = constant_vol_fit(0.5)
    d = FHSPricer(n_paths=20_000, vol_uncertainty=0.2, basis_usd=15, seed=4).simulate(fit, 80_000.0, T0, [3600])[0]
    assert np.mean(80_000 * np.exp(d.log_ret)) == pytest.approx(80_000, rel=1e-9)
    edges = np.arange(76_000, 84_001, 250)
    total = d.prob(Contract("less", None, edges[0])).p + d.prob(Contract("greater", edges[-1], None)).p
    total += sum(d.prob(Contract("between", a, b)).p for a, b in zip(edges, edges[1:]))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_fhs_one_simulation_serves_many_horizons():
    fit = constant_vol_fit(0.5)
    d1, d6 = FHSPricer(n_paths=40_000, vol_uncertainty=0.0, basis_usd=0, seed=5).simulate(fit, 80_000.0, T0, [3600, 6 * 3600])
    assert np.std(d6.log_ret) / np.std(d1.log_ret) == pytest.approx(math.sqrt(6), rel=0.03)


def test_vol_uncertainty_shows_up_in_sd():
    fit = constant_vol_fit(0.5)
    c = Contract("greater", 82_000, None)
    lo = FHSPricer(20_000, 0.0, 0, seed=6).simulate(fit, 80_000.0, T0, [3 * 3600])[0].prob(c)
    hi = FHSPricer(20_000, 0.3, 0, seed=6).simulate(fit, 80_000.0, T0, [3 * 3600])[0].prob(c)
    assert hi.sd > 2 * lo.sd


def test_blend_recovers_overconfident_model():
    rng = np.random.default_rng(7)
    n = 20_000
    true_logit = rng.normal(0, 1.5, n)
    y = (rng.random(n) < special.expit(true_logit)).astype(float)
    p_model = special.expit(1.6 * true_logit)  # overconfident: should get b ~ 1/1.6
    p_mkt = special.expit(true_logit + rng.normal(0, 1.0, n))  # noisy but unbiased
    b = fit_blend(p_model, p_mkt, y, l2=1.0, min_markets=100)
    assert b.active
    assert b.logloss_blend < b.logloss_model
    assert b.b == pytest.approx(1 / 1.6, abs=0.08)
    assert b.c > 0


def test_blend_stays_at_prior_with_little_data():
    b = fit_blend(np.full(50, 0.3), np.full(50, 0.4), np.zeros(50), min_markets=300)
    assert not b.active
    assert b.apply(0.3, 0.4) == 0.3


def test_shrink_learns_reliability_and_gates_thresholds():
    from btclab.calibrate import fit_shrink

    rng = np.random.default_rng(11)
    n = 60_000
    d = rng.normal(0, 0.03, n)
    klass = np.where(rng.random(n) < 0.5, "between", "threshold").astype(object)
    true_beta = np.where(klass == "between", 0.9, 0.0)  # thresholds: disagreement is pure noise
    m = rng.uniform(0.2, 0.8, n)
    y = (rng.random(n) < m + true_beta * d).astype(float)
    events = (np.arange(n) // 20).astype(object)  # 20 correlated rows per event
    days = (np.arange(n) // 2000).astype(object)  # 30 days
    t = fit_shrink(d, y - m, klass, days, events=events)
    # Bins with 2-6 cent disagreements: large enough to trade, plenty of data.
    # (Sub-2-cent bins are inherently noisy, and such edges never clear fees.)
    # With 30 days of data on a true beta of 0.9, the conservative posterior
    # moves up from the 0.5 prior but never past the truth.
    for b in (2, 3, 6, 7):
        assert 0.5 < t.beta[("between", b)] < 0.9
    assert all(v == 0.0 for (c, _), v in t.beta.items() if c == "threshold")
    assert t.apply(0.55, 0.50, "between") > 0.525
    assert t.apply(0.55, 0.50, "greater") == 0.50


def test_day_clustering_is_more_conservative_than_event_clustering():
    from btclab.calibrate import fit_shrink

    rng = np.random.default_rng(3)
    n_days, per_day = 6, 3000
    rows = []
    for day in range(n_days):
        shock = rng.normal(0, 0.5)  # one shared price path per day drives every row
        d = rng.normal(0, 0.03, per_day)
        g = d * shock + rng.normal(0, 0.05, per_day)  # true beta = 0, noisy daily
        ev = day * 100 + rng.integers(0, 24, per_day)
        rows.append((d, g, np.full(per_day, day), ev))
    d, g, day, ev = (np.concatenate(x) for x in zip(*rows))
    k = np.full(d.size, "threshold", dtype=object)
    by_day = fit_shrink(d, g, k, day.astype(object), events=ev.astype(object))
    by_event = fit_shrink(d, g, k, ev.astype(object), events=ev.astype(object))
    b = 6  # +0.02..+0.04 bin
    assert by_day.se[("threshold", b)] > 2 * by_event.se[("threshold", b)]
    assert all(v == 0.0 for (c, _), v in by_day.beta.items() if c == "threshold")


def test_shrink_ignores_data_until_enough_days():
    from btclab.calibrate import fit_shrink

    rng = np.random.default_rng(5)
    n = 20_000
    d = rng.normal(0, 0.03, n)
    g = d * 1.0 + rng.normal(0, 0.05, n)  # strong apparent edge...
    days = (np.arange(n) // 4000).astype(object)  # ...but only 5 days of it
    k = np.where(rng.random(n) < 0.5, "between", "threshold").astype(object)
    t = fit_shrink(d, g, k, days)
    assert all(v == 0.5 for (c, _), v in t.beta.items() if c == "between")
    assert all(v == 0.0 for (c, _), v in t.beta.items() if c == "threshold")
