from itertools import permutations

import numpy as np
import pytest

from semantic_router.calibration import Calibrator, ece
from semantic_router.route import Route
from semantic_router.routers.base import BaseRouter


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


@pytest.mark.parametrize("labels", [[0, 1], [1, 0]])
def test_isotonic_tied_scores_have_the_same_probability(labels):
    cal = Calibrator.fit(np.array([0.5, 0.5]), np.array(labels), "isotonic")
    assert float(cal(0.5)) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "scores, labels, expected",
    [
        ([0.5] * 4, [0, 0, 0, 1], [0.25, 0.25, 0.25]),
        # The first two score groups violate monotonicity: pool all five
        # samples, rather than taking an unweighted mean of their group means.
        ([0.1] * 3 + [0.5] * 2 + [0.9], [0, 1, 1, 0, 0, 1], [0.4, 0.4, 0.875]),
        # Pooling must cascade through more than two groups.
        ([0.1] * 2 + [0.5] * 3 + [0.9], [1, 1, 0, 1, 0, 0], [0.5, 0.5, 0.5]),
    ],
)
def test_isotonic_is_invariant_to_sample_permutations(scores, labels, expected):
    scores, labels = np.array(scores), np.array(labels)
    grid = np.linspace(-0.1, 1.1, 31)
    reference = Calibrator.fit(scores, labels, "isotonic")
    assert reference(np.array([0.1, 0.5, 0.9])) == pytest.approx(expected)
    assert np.all(np.diff(reference.params["x"]) > 0)
    assert np.all(np.diff(reference(grid)) >= 0)
    assert 0 < reference(grid).min() <= reference(grid).max() < 1
    restored = Calibrator.from_dict(reference.to_dict())
    assert restored(grid) == pytest.approx(reference(grid))
    for order in permutations(range(len(scores))):
        order = list(order)
        cal = Calibrator.fit(scores[order], labels[order], "isotonic")
        assert cal.params == reference.params
        assert cal(grid) == pytest.approx(reference(grid))


def test_isotonic_without_ties_preserves_interpolation_and_clipping():
    cal = Calibrator.fit(
        np.array([0.1, 0.4, 0.6, 0.9]), np.array([0, 1, 0, 1]), "isotonic"
    )
    grid = np.array([-1, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 2])
    assert cal(grid) == pytest.approx(
        [1 / 6, 1 / 6, 1 / 3, 0.5, 0.5, 0.5, 2 / 3, 5 / 6, 5 / 6]
    )


@pytest.mark.parametrize("labels", [[None, "a"], ["a", None]])
@pytest.mark.asyncio
async def test_isotonic_ties_do_not_change_router_threshold_decisions(labels):
    class ConstantRouter(BaseRouter):
        async def _scores(self, query):
            return {"a": 0.5}

    router = ConstantRouter(
        routes=[Route(name="a", utterances=["example"])], confidence_threshold=0.6
    )
    await router.afit(["first", "second"], labels, method="isotonic")
    rejected = await router.acall("query")
    assert rejected.name is None and rejected.calibrated
    accepted = await router.acall("query", threshold=0.5)
    assert accepted.name == "a" and accepted.calibrated
    assert accepted.confidence == pytest.approx(0.5)


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


def test_platt_never_learns_a_falling_curve():
    # higher scores are *less* often right here; a negative slope would reverse rankings
    s = np.array([0.9, 0.85, 0.8, 0.3, 0.25, 0.2])
    y = np.array([0, 0, 1, 1, 1, 0])
    cal = Calibrator.fit(s, y, "platt")
    assert cal.params["a"][0] >= 0
    p = cal(s)
    assert np.all(np.diff(p[np.argsort(s)]) >= 0)  # never decreasing in the score
