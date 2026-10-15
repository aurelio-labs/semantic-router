import asyncio

import numpy as np
import pytest

from semantic_router.encoders import DenseEncoder
from semantic_router.route import Route
from semantic_router.v2 import IndexedRouter

# each keyword owns an axis, so "refund" is near "charged" and far from "crash"
AXES = {
    "refund": 0,
    "charged": 0,
    "invoice": 0,
    "crash": 1,
    "error": 1,
    "bug": 1,
    "call": 2,
    "human": 2,
}


class FakeEncoder(DenseEncoder):
    name: str = "fake"

    def __call__(self, docs):
        out = []
        for d in docs:
            v = np.zeros(3)
            for w in str(d).lower().split():
                if w in AXES:
                    v[AXES[w]] += 1
            if not v.any():
                v[:] = 1
            out.append(list(v / np.linalg.norm(v)))
        return out


ROUTES = [
    Route(
        name="billing", utterances=["refund please", "charged twice", "invoice wrong"]
    ),
    Route(name="technical", utterances=["app crash", "error on save", "found a bug"]),
]


@pytest.fixture
def router():
    return IndexedRouter(FakeEncoder(), ROUTES)


def test_routes_by_similarity(router):
    assert router("i want a refund").name == "billing"
    assert router("it keeps crashing with an error").name == "technical"


def test_threshold_blocks_weak_matches(router):
    weak = router("hello there")
    assert weak.name is not None  # no bar: something always wins
    assert router("hello there", threshold=0.99).name is None


def test_limit_none_returns_every_route_ranked(router):
    picks = router("refund", limit=None)
    assert [p.name for p in picks] == ["billing", "technical"]
    assert picks[0].confidence > picks[1].confidence
    assert not picks[0].calibrated


def test_add_and_remove_routes(router):
    router.add(Route(name="support", utterances=["call a human", "speak to a person"]))
    assert router("can i call a human").name == "support"
    router.remove("support")
    assert router("can i call a human").name != "support"


def test_fit_sets_a_calibrator_and_reports(router):
    X = ["refund", "charged", "invoice", "crash", "error", "bug"] * 20 + ["hello"] * 30
    y = ["billing"] * 3 * 20 + ["technical"] * 3 * 20 + [None] * 30
    # ensure the query set has real negatives so both classes exist
    report = router.fit(X, y, "platt")
    assert report.n == len(X)
    assert router.calibrator is not None
    assert router("refund").calibrated
    assert 0.0 <= report.ece <= 1.0


def test_sync_call_works_inside_a_running_loop(router):
    async def inside():
        return router("refund")  # notebooks do exactly this

    assert asyncio.run(inside()).name == "billing"


def test_evaluate_applies_threshold(router):
    router.confidence_threshold = 0.99
    assert router.evaluate(["hello there"], [None]) == 1.0


def test_hybrid_router_rebuilds_sparse_index_on_add():
    from semantic_router.encoders import BM25Encoder

    r = IndexedRouter(FakeEncoder(), ROUTES, sparse_encoder=BM25Encoder(), alpha=0.5)
    assert r("i want a refund").name == "billing"
    r.add(Route(name="support", utterances=["call a human", "speak to a person"]))
    # the vocabulary changed on add; the old rows must have been re-encoded, not left stale
    assert r("i want a refund").name == "billing"
    assert r("can i call a human").name == "support"
    assert len(r.index) == 8


def test_sync_call_from_inside_the_router_loop_fails_loudly(router):
    from semantic_router.v2 import base

    async def naughty():
        return router("refund")  # sync call on the loop that must run it

    router(
        "refund"
    )  # any sync call creates the shared loop; do not depend on test order
    fut = asyncio.run_coroutine_threadsafe(naughty(), base._loop)
    with pytest.raises(RuntimeError, match="deadlock"):
        fut.result(timeout=5)


def test_fit_warns_when_training_on_stored_utterances(router, monkeypatch):
    from semantic_router.v2 import indexed

    warned: list[str] = []
    monkeypatch.setattr(indexed.logger, "warning", warned.append)
    X = ["refund please", "app crash", "hello"]  # first two are stored utterances
    router.fit(X, ["billing", "technical", None])
    assert warned and "2/3 training queries are stored utterances" in warned[0]
