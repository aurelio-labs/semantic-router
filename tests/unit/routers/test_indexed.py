import asyncio
from uuid import uuid4

import numpy as np
import pytest

from semantic_router.encoders import DenseEncoder
from semantic_router.route import Route
from semantic_router.routers import IndexedRouter

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
    from semantic_router.routers import base

    async def naughty():
        return router("refund")  # sync call on the loop that must run it

    router(
        "refund"
    )  # any sync call creates the shared loop; do not depend on test order
    fut = asyncio.run_coroutine_threadsafe(naughty(), base._loop)
    with pytest.raises(RuntimeError, match="deadlock"):
        fut.result(timeout=5)


X_FIT = ["refund", "charged", "invoice", "crash", "error", "bug"] * 20 + ["hello"] * 30
Y_FIT = ["billing"] * 60 + ["technical"] * 60 + [None] * 30


def test_an_empty_index_passed_in_is_the_one_used():
    from semantic_router.index import LocalIndex

    idx = LocalIndex()  # empty, so len() is 0 and falsy
    r = IndexedRouter(FakeEncoder(), ROUTES, index=idx)
    assert r.index is idx
    assert len(idx) == 6


def test_an_empty_qdrant_index_passed_in_is_the_one_used():
    pytest.importorskip("qdrant_client")
    from semantic_router.index import QdrantIndex

    idx = QdrantIndex(index_name=f"v1-{uuid4().hex[:8]}", dimensions=3)  # in memory
    r = IndexedRouter(FakeEncoder(), ROUTES, index=idx)
    assert r.index is idx
    assert len(idx) == 6
    assert r("refund").name == "billing"


def test_add_refits_the_calibrator(router):
    router.fit(X_FIT, Y_FIT)
    before = router.calibrator
    router.add(Route(name="support", utterances=["call a human", "speak to a person"]))
    assert router.calibrator is not None and router.calibrator is not before
    assert router("can i call a human").calibrated


def test_remove_refits_with_the_removed_routes_queries_as_no_route(router):
    router.fit(X_FIT, Y_FIT)
    before = router.calibrator
    router.remove("technical")
    assert router.calibrator is not None and router.calibrator is not before
    assert "technical" not in router._fit_data[1]


def test_a_route_change_drops_a_calibrator_it_cannot_refit(monkeypatch):
    from semantic_router.calibration import Calibrator
    from semantic_router.routers import base

    warned: list[str] = []
    monkeypatch.setattr(base.logger, "warning", warned.append)
    given = Calibrator("platt", {"a": [10.0], "b": [-8.0]})  # fitted elsewhere
    r = IndexedRouter(FakeEncoder(), ROUTES, calibrator=given)
    r.add(Route(name="support", utterances=["call a human"]))
    assert r.calibrator is None and not r("refund").calibrated
    assert warned and "no labelled queries to refit" in warned[0]


def test_fit_warns_when_training_on_stored_utterances(router, monkeypatch):
    from semantic_router.routers import indexed

    warned: list[str] = []
    monkeypatch.setattr(indexed.logger, "warning", warned.append)
    X = ["refund please", "app crash", "hello"]  # first two are stored utterances
    router.fit(X, ["billing", "technical", None])
    assert warned and "2/3 training queries are stored utterances" in warned[0]


def test_a_route_bar_overrides_the_router_bar():
    strict = [
        Route(name="billing", utterances=["refund please"], confidence_threshold=0.99),
        Route(name="technical", utterances=["app crash"]),
    ]
    r = IndexedRouter(encoder=FakeEncoder(), routes=strict, confidence_threshold=0.5)
    # "refund crash" sits halfway: 0.71 to each route, under billing's own 0.99 bar
    assert r("refund crash").name == "technical"
    assert r("refund").name == "billing"  # 1.0 clears it


def test_a_per_call_threshold_wins_over_route_and_router_bars():
    strict = [Route(name="billing", utterances=["refund"], confidence_threshold=0.99)]
    r = IndexedRouter(encoder=FakeEncoder(), routes=strict, confidence_threshold=0.99)
    assert r("refund crash", threshold=0.5).name == "billing"


def test_add_takes_one_route_or_a_list(router):
    router.add(routes=Route(name="support", utterances=["call a human"]))
    router.add(routes=[Route(name="other", utterances=["speak to a person"])])
    assert [r.name for r in router.routes] == [
        "billing",
        "technical",
        "support",
        "other",
    ]


def test_route_level_fields_are_stored_on_every_record():
    routes = [
        Route(
            name="billing",
            utterances=["refund please", "charged twice"],
            description="money owed",
            confidence_threshold=0.9,
            metadata={"team": "finance"},
        ),
        Route(name="technical", utterances=["app crash"]),
    ]
    r = IndexedRouter(encoder=FakeEncoder(), routes=routes)
    meta = {u.utterance: u.metadata for u in r.index.get_utterances()}
    assert meta["charged twice"] == {
        "team": "finance",
        "sr_description": "money owed",
        "sr_confidence_threshold": 0.9,
    }
    assert meta["app crash"] == {}  # unset fields are left out, not stored as None
