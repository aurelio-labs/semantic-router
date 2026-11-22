import asyncio
from uuid import uuid4

import numpy as np
import pytest

from semantic_router.encoders import DenseEncoder
from semantic_router.index import LocalIndex
from semantic_router.route import Route
from semantic_router.routers import BaseRouter, Chain, IndexedRouter, Merge

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


def test_hybrid_router_built_on_a_populated_index_rebuilds_stale_sparse_rows():
    from semantic_router.encoders import TfidfEncoder
    from semantic_router.index import HybridLocalIndex

    class ConstantEncoder(DenseEncoder):
        name: str = "constant"

        def __call__(self, docs):
            return [[1.0, 0.0] for _ in docs]

    def build(routes, index):
        return IndexedRouter(
            ConstantEncoder(),
            routes,
            index=index,
            sparse_encoder=TfidfEncoder(),
            alpha=0.5,
            confidence_threshold=0.8,
        )

    shared = HybridLocalIndex()
    first = build([Route(name="existing", utterances=["zebra"])], shared)
    assert first("zebra").name == "existing"

    # a second router on the same index refits on the merged corpus; the stored
    # "zebra" row must be re-encoded with the new vocabulary, not left stale
    reopened = build([Route(name="new", utterances=["apple"])], shared)
    assert reopened("zebra").name == "existing"
    assert reopened("apple").name == "new"
    assert len(shared) == 2


# matched utterances (#479)


def test_matched_utterances_are_off_unless_asked(router):
    choice = router("i want a refund")
    assert choice.name == "billing"
    assert choice.matched_utterances is None


def test_matched_utterances_are_the_hits_behind_the_choice(router):
    choice = router("i want a refund", return_utterances=True)
    assert choice.name == "billing"
    texts = [u.text for u in choice.matched_utterances]
    # every billing utterance sits on the refund axis, so all three are hits
    assert sorted(texts) == sorted(str(u) for u in ROUTES[0].utterances)
    assert {u.route for u in choice.matched_utterances} == {"billing"}
    assert all(u.score == pytest.approx(1.0) for u in choice.matched_utterances)


def test_matched_utterances_come_best_first():
    mixed = Route(name="mixed", utterances=["refund", "refund crash", "crash"])
    choice = IndexedRouter(FakeEncoder(), [mixed])("refund", return_utterances=True)
    # LocalIndex returns its hits worst first; the choice lists them best first
    assert [u.text for u in choice.matched_utterances] == [
        "refund",
        "refund crash",
        "crash",
    ]


def test_a_route_with_no_hit_in_top_k_has_an_empty_list():
    r = IndexedRouter(FakeEncoder(), ROUTES, top_k=3)
    choices = r("i want a refund", limit=None, return_utterances=True)
    by_name = {c.name: c.matched_utterances for c in choices}
    assert len(by_name["billing"]) == 3
    # asked for, but none of technical's utterances made the top 3: [] rather than None
    assert by_name["technical"] == []


def test_matched_utterances_keep_the_raw_score_after_calibration(router):
    router.fit(X_FIT, Y_FIT)
    choice = router("refund please", return_utterances=True)
    assert choice.calibrated
    assert max(u.score for u in choice.matched_utterances) == pytest.approx(1.0)


def test_an_empty_choice_has_no_matched_utterances(router):
    choice = router("hello there", threshold=0.99, return_utterances=True)
    assert choice.name is None
    assert choice.matched_utterances is None


def test_hybrid_router_returns_matched_utterances():
    from semantic_router.encoders import TfidfEncoder

    r = IndexedRouter(FakeEncoder(), ROUTES, sparse_encoder=TfidfEncoder(), alpha=0.5)
    choice = r("i want a refund", return_utterances=True)
    assert choice.name == "billing"
    assert "refund please" in [u.text for u in choice.matched_utterances]


class SlowEncoder(FakeEncoder):
    async def acall(self, docs):
        # yield to the loop, so overlapping calls interleave between embed and search
        await asyncio.sleep(0.005)
        return self(docs)


def test_overlapping_calls_each_get_their_own_utterances():
    r = IndexedRouter(SlowEncoder(), ROUTES, top_k=3)
    queries = ["refund", "crash"] * 50

    async def go():
        return await asyncio.gather(
            *(r.acall(q, return_utterances=True) for q in queries)
        )

    want = {"refund": "billing", "crash": "technical"}
    for q, choice in zip(queries, asyncio.run(go())):
        assert choice.name == want[q]
        assert {u.route for u in choice.matched_utterances} == {want[q]}


class OldIndex(LocalIndex):
    """a custom index written before return_utterances existed."""

    calls: int = 0

    def query(self, vector, top_k=5, route_filter=None, sparse_vector=None):
        self.calls += 1
        return super().query(vector=vector, top_k=top_k, route_filter=route_filter)

    async def aquery(self, vector, top_k=5, route_filter=None, sparse_vector=None):
        return self.query(vector=vector, top_k=top_k, route_filter=route_filter)


def test_an_index_without_the_flag_still_routes_and_searches_once():
    index = OldIndex()
    r = IndexedRouter(FakeEncoder(), ROUTES, index=index)
    before = index.calls
    choice = r("i want a refund", return_utterances=True)
    assert choice.name == "billing"
    assert choice.matched_utterances is None  # the index can't say
    assert index.calls == before + 1


class BrokenIndex(LocalIndex):
    broken: bool = False
    calls: int = 0

    def query(self, *args, **kwargs):
        if not self.broken:
            return super().query(*args, **kwargs)
        self.calls += 1  # the search ran, then something inside it went wrong
        raise TypeError("a bug inside the index")


def test_a_type_error_inside_the_index_is_raised_not_searched_again():
    index = BrokenIndex()
    r = IndexedRouter(FakeEncoder(), ROUTES, index=index)
    index.broken = True
    with pytest.raises(TypeError, match="a bug inside the index"):
        r("i want a refund", return_utterances=True)
    assert index.calls == 1


class Keywords(BaseRouter):
    """a custom router as the docs write one: _scores only."""

    async def _scores(self, query):
        return {"billing": float("refund" in query), "technical": 0.0}


def test_a_custom_router_takes_the_flag_and_has_no_utterances():
    choice = Keywords(ROUTES)("refund", return_utterances=True)
    assert choice.name == "billing"
    assert choice.matched_utterances is None


def test_chain_gives_the_utterances_of_the_router_that_answered(router):
    chain = Chain(routers=[Keywords(ROUTES, confidence_threshold=0.5), router])
    # the keyword router is sure: no index search is behind its answer
    assert chain("refund", return_utterances=True).matched_utterances is None
    # it isn't sure about this one, so the indexed router answers, with its hits
    choice = chain("app crash", return_utterances=True)
    assert choice.name == "technical"
    assert "app crash" in [u.text for u in choice.matched_utterances]


def test_merge_gives_the_utterances_of_every_router_that_has_them(router):
    other = IndexedRouter(FakeEncoder(), ROUTES)
    merge = Merge(routers=[Keywords(ROUTES), router, other])
    choice = merge("found a bug", return_utterances=True)
    assert choice.name == "technical"
    texts = [u.text for u in choice.matched_utterances]
    # both indexed routers found it; the keyword router adds nothing
    assert texts.count("found a bug") == 2
    assert merge("found a bug").matched_utterances is None
