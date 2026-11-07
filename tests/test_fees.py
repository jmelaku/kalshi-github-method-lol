import pytest

from btclab.fees import MAKER_MULTIPLIER, FeeModel


def test_published_example():
    # Kalshi fee schedule example: 100 contracts at $0.55 -> $1.74
    assert FeeModel().taker_fee(0.55, 100) == pytest.approx(1.74)


def test_rounds_up_to_cent():
    # 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> $0.02
    assert FeeModel().taker_fee(0.50, 1) == pytest.approx(0.02)


def test_direct_member_precision():
    assert FeeModel(precision=0.0001).taker_fee(0.55, 100) == pytest.approx(1.7325)


def test_fee_peaks_at_50c_and_is_symmetric():
    f = FeeModel(precision=1e-6)
    assert f.taker_fee(0.5, 100) > f.taker_fee(0.3, 100) > f.taker_fee(0.1, 100)
    assert f.taker_fee(0.3, 100) == pytest.approx(f.taker_fee(0.7, 100))


def test_maker_free_by_default_and_charged_when_configured():
    assert FeeModel().maker_fee(0.5, 100) == 0.0
    assert FeeModel(maker_multiplier=MAKER_MULTIPLIER).maker_fee(0.5, 100) == pytest.approx(0.44)


def test_rejects_bad_price():
    with pytest.raises(ValueError):
        FeeModel().taker_fee(1.0, 10)
