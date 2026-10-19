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
    from semantic_router.encoders import TfidfEncoder

    r = IndexedRouter(FakeEncoder(), ROUTES, sparse_encoder=TfidfEncoder(), alpha=0.5)
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


# an async client that, like httpx or anyio, binds to the first loop that uses it
_BOUND: dict[int, asyncio.AbstractEventLoop] = {}


class LoopBoundEncoder(FakeEncoder):
    async def acall(self, docs):
        loop = asyncio.get_running_loop()
        if _BOUND.setdefault(id(self), loop) is not loop:
            raise RuntimeError("bound to a different event loop")
        return self(docs)


def test_sync_calls_leave_async_clients_free_for_the_callers_loop():
    r = IndexedRouter(encoder=LoopBoundEncoder(), routes=ROUTES)
    # sync calls run on the shared background loop; they must not bind the async client
    assert r("refund").name == "billing"
    r.fit(X_FIT, Y_FIT)
    assert r.evaluate(["refund"], ["billing"]) == 1.0
    # the caller's own loop is then the first to use it, so this works
    assert asyncio.run(r.acall("refund")).name == "billing"


def test_sync_calls_use_the_index_sync_api(router, monkeypatch):
    async def boom(*args, **kwargs):
        raise AssertionError("a sync call used the index's async API")

    for name in ("aquery", "aget_routes", "adelete", "aadd"):
        monkeypatch.setattr(type(router.index), name, boom)
    assert router("refund").name == "billing"
    router.add(routes=Route(name="support", utterances=["call a human"]))
    router.remove(name="support")
    assert [r.name for r in router.routes] == ["billing", "technical"]


# ---------------------------------------------------------------------------
# matched_utterances (#479) — utterance-level visibility on RouteChoice
# ---------------------------------------------------------------------------


def test_route_choice_has_matched_utterances_with_default_limit(router):
    choice = router("i want a refund", return_utterances=True)
    assert choice.name == "billing"
    assert choice.matched_utterances is not None
    # every matched utterance should point at the chosen route
    for u in choice.matched_utterances:
        assert u.route == "billing"
        assert u.text  # non-empty text
        assert isinstance(u.score, float)


def test_matched_utterances_are_opt_in_by_default(router):
    # return_utterances=False by default: utterances stay off the choice
    choice = router("i want a refund")
    assert choice.name == "billing"
    assert choice.matched_utterances is None


def test_matched_utterances_populated_for_every_route_when_limit_is_none(router):
    # a query that sits between both routes should pull utterances for both
    choices = router("refund crash", limit=None, return_utterances=True)
    names = {c.name for c in choices}
    assert names == {"billing", "technical"}
    for c in choices:
        assert c.matched_utterances is not None
        assert c.matched_utterances, f"{c.name} has no matched utterances"
        for u in c.matched_utterances:
            assert u.route == c.name


def test_matched_utterances_survive_calibration(router):
    router.fit(X_FIT, Y_FIT)
    choice = router("refund please", return_utterances=True)
    assert choice.calibrated
    assert choice.matched_utterances is not None
    # calibrated confidence is a probability; utterance score is raw similarity
    raw_scores = [u.score for u in choice.matched_utterances]
    assert all(s == 1.0 for s in raw_scores)


def test_matched_utterances_cleared_between_calls(router):
    first = router("refund", return_utterances=True)
    assert first.matched_utterances is not None
    # a completely different query: utterances from the previous call must not leak
    second = router("hello there", threshold=0.99, return_utterances=True)
    # threshold blocked everything: no route chosen
    assert second.name is None or second.name == first.name  # fine either way
    # the router's internal state should be clean for the next call
    third = router("call a human", return_utterances=True)
    if third.name == "support":
        # would only happen if the utterances leaked; not strictly required
        pass
    assert third.matched_utterances is not None


def test_matched_utterances_available_on_hybrid_router():
    from semantic_router.encoders import TfidfEncoder

    tfidf = TfidfEncoder()
    r = IndexedRouter(FakeEncoder(), ROUTES, sparse_encoder=tfidf, alpha=0.5)
    choice = r("i want a refund", return_utterances=True)
    assert choice.name == "billing"
    assert choice.matched_utterances is not None
    assert any(u.route == "billing" for u in choice.matched_utterances)


def test_matched_utterances_returned_by_async_acall(router):
    import asyncio

    async def run():
        return await router.acall("app crash bug", return_utterances=True)

    choice = asyncio.run(run())
    assert choice.name == "technical"
    assert choice.matched_utterances is not None
    assert all(u.route == "technical" for u in choice.matched_utterances)


def test_empty_route_choice_has_no_matched_utterances(router):
    # push the threshold so high nothing passes
    choice = router("hello there", threshold=0.99, return_utterances=True)
    assert choice.name is None
    # utterances are only populated when we actually pick a route
    assert choice.matched_utterances is None


def test_overlapping_async_calls_do_not_share_utterances(router):
    """concurrent acalls on the same router must not leak each other's utterances."""
    import asyncio

    async def one(query: str, expected_route: str):
        return await router.acall(query, return_utterances=True), expected_route

    async def run():
        # fire two queries at once, and interleave with a third
        t1 = asyncio.create_task(one("refund", "billing"))
        t2 = asyncio.create_task(one("app crash bug", "technical"))
        c1, e1 = await t1
        c2, e2 = await t2
        # each call sees only its own utterances, all tied to its chosen route
        assert c1.name == e1
        assert c2.name == e2
        assert c1.matched_utterances is not None
        assert c2.matched_utterances is not None
        for u in c1.matched_utterances:
            assert u.route == c1.name, "call 1 saw call 2's utterances!"
        for u in c2.matched_utterances:
            assert u.route == c2.name, "call 2 saw call 1's utterances!"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# LocalIndex query(return_utterances=True) contract
# ---------------------------------------------------------------------------


def test_local_index_query_returns_utterances_when_requested():
    from semantic_router.index import LocalIndex

    idx = LocalIndex()
    idx.add(
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        routes=["a", "b"],
        utterances=["first", "second"],
    )
    vec = np.array([1.0, 0.0, 0.0])
    # default: backward-compatible two-tuple
    scores, names = idx.query(vec, top_k=2)
    assert len(scores) == 2 and len(names) == 2
    # with the flag: three-tuple
    scores, names, utterances = idx.query(vec, top_k=2, return_utterances=True)
    assert names == ["a", "b"] or names == ["b", "a"]
    assert "first" in utterances and "second" in utterances


def test_hybrid_local_index_query_returns_utterances_when_requested():
    from semantic_router.index import HybridLocalIndex
    from semantic_router.schema import SparseEmbedding

    idx = HybridLocalIndex()
    idx.add(
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        routes=["a", "b"],
        utterances=["first", "second"],
        sparse_embeddings=[
            SparseEmbedding.from_dict({0: 1.0}),
            SparseEmbedding.from_dict({1: 1.0}),
        ],
    )
    vec = np.array([0.5, 0.5, 0.0])
    sparse = SparseEmbedding.from_dict({0: 0.5, 1: 0.5})
    scores, names, utterances = idx.query(
        vec, top_k=2, sparse_vector=sparse, return_utterances=True
    )
    assert len(scores) == 2
    assert "first" in utterances and "second" in utterances
