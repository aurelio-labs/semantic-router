import numpy as np
import pytest

from semantic_router.calibration import Calibrator, ece


def _sample(n: int = 600, seed: int = 0):
    # scores drawn from a known sigmoid so we can check the fit recovers it
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.2, 1.0, n)
    p = 1 / (1 + np.exp(-12 * (s - 0.6)))
    return s, (rng.uniform(size=n) < p).astype(float), p


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_fit_is_monotone_and_calibrated(method):
    s, y, _ = _sample()
    cal = Calibrator.fit(s, y, method)
    grid = np.linspace(0.2, 1.0, 50)
    assert np.all(np.diff(cal(grid)) >= -1e-9)
    assert ece(cal(s), y) < 0.05


def test_platt_recovers_the_generating_curve():
    s, y, p_true = _sample(n=3000)
    cal = Calibrator.fit(s, y, "platt")
    assert np.abs(cal(s) - p_true).mean() < 0.03


def test_isotonic_never_claims_certainty():
    s, y, _ = _sample()
    cal = Calibrator.fit(s, y, "isotonic")
    out = cal(np.array([-1.0, 0.0, 2.0]))
    assert 0 < out.min() and out.max() < 1


def test_fit_needs_both_classes():
    with pytest.raises(ValueError):
        Calibrator.fit(np.array([0.1, 0.9]), np.array([1.0, 1.0]))


def test_round_trip():
    s, y, _ = _sample()
    cal = Calibrator.fit(s, y, "isotonic")
    again = Calibrator.from_dict(cal.to_dict())
    assert np.allclose(cal(s), again(s))


def test_ece_of_perfect_calibration_is_zero():
    p = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8])
    y = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=float)
    assert ece(p, y, bins=5) == pytest.approx(0.0)
