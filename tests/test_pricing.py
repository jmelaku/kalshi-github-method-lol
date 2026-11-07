import math

import numpy as np
import pytest
from scipy import stats

from btclab.engine import nu_for_horizon
from btclab.pricing import Contract, ModelParams, fair_prob, implied_sigma, prob_above

TAU_1H = 1 / (365 * 24)


@pytest.mark.parametrize("model", ["lognormal", "student_t"])
def test_partition_sums_to_one(model):
    p = ModelParams(0.6, model, 4.0)
    spot, w = 80_000.0, 250.0
    edges = [spot - 40 * w + i * w for i in range(81)]
    total = fair_prob(Contract("less", None, edges[0]), spot, TAU_1H, p)
    total += sum(fair_prob(Contract("between", a, b), spot, TAU_1H, p) for a, b in zip(edges, edges[1:]))
    total += fair_prob(Contract("greater", edges[-1], None), spot, TAU_1H, p)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_lognormal_matches_black_scholes_digital():
    s, k, sig, tau = 80_000.0, 81_000.0, 0.5, 3 * TAU_1H
    d2 = (math.log(s / k) - 0.5 * sig**2 * tau) / (sig * math.sqrt(tau))
    assert prob_above(s, k, tau, ModelParams(sig, "lognormal")) == pytest.approx(stats.norm.cdf(d2), abs=1e-12)


def test_above_and_below_complement():
    p = ModelParams(0.5, "student_t", 5.0)
    a = fair_prob(Contract("greater", 82_000, None), 80_000, TAU_1H * 6, p)
    b = fair_prob(Contract("less", None, 82_000), 80_000, TAU_1H * 6, p)
    assert a + b == pytest.approx(1.0)


def test_monotone_in_strike():
    p = ModelParams(0.5, "lognormal")
    probs = [prob_above(80_000, k, TAU_1H, p) for k in range(78_000, 82_001, 500)]
    assert all(x > y for x, y in zip(probs, probs[1:]))


def test_fat_tails_price_far_strikes_higher():
    far = Contract("greater", 84_000, None)
    ln = fair_prob(far, 80_000, TAU_1H, ModelParams(0.5, "lognormal"))
    st = fair_prob(far, 80_000, TAU_1H, ModelParams(0.5, "student_t", 4.0))
    assert st > ln


def test_student_t_matches_monte_carlo():
    rng = np.random.default_rng(0)
    nu, sig, tau, s0 = 5.0, 0.6, 2 * TAU_1H, 80_000.0
    sd = sig * math.sqrt(tau)
    r = -0.5 * sd * sd + sd * math.sqrt((nu - 2) / nu) * rng.standard_t(nu, 400_000)
    st = s0 * np.exp(r)
    c = Contract("between", 80_500, 81_000)
    mc = float(np.mean((st > 80_500) & (st < 81_000)))
    assert fair_prob(c, s0, tau, ModelParams(sig, "student_t", nu)) == pytest.approx(mc, abs=0.003)


def test_expiry_is_deterministic():
    p = ModelParams(0.5, "lognormal")
    assert fair_prob(Contract("greater", 79_000, None), 80_000, 0.0, p) == 1.0
    assert fair_prob(Contract("greater", 81_000, None), 80_000, 0.0, p) == 0.0


def test_basis_widens_distribution():
    c = Contract("between", 79_990, 80_010)
    tight = fair_prob(c, 80_000, TAU_1H / 60, ModelParams(0.5, "lognormal"))
    wide = fair_prob(c, 80_000, TAU_1H / 60, ModelParams(0.5, "lognormal", basis_usd=30))
    assert wide < tight


def test_implied_sigma_roundtrip():
    c = Contract("greater", 81_000, None)
    price = fair_prob(c, 80_000, TAU_1H * 4, ModelParams(0.55, "lognormal"))
    assert implied_sigma(c, price, 80_000, TAU_1H * 4) == pytest.approx(0.55, abs=1e-4)


def test_nu_interpolation():
    table = [[0.25, 8.0], [1.0, 10.0], [24.0, 0.0]]
    assert nu_for_horizon(0.1, table) == pytest.approx(8.0)
    assert nu_for_horizon(1.0, table) == pytest.approx(10.0)
    assert nu_for_horizon(48, table) == math.inf  # lognormal
    assert 10.0 < nu_for_horizon(3.0, table) < math.inf
