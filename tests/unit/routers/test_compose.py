import asyncio

from semantic_router.route import Route
from semantic_router.routers import Chain, Merge
from semantic_router.routers.base import BaseRouter, QueryResult

ROUTES = [Route(name="a", utterances=["a"]), Route(name="b", utterances=["b"])]


class Canned(BaseRouter):
    def __init__(self, scores, probabilistic=False, **kw):
        super().__init__(ROUTES, **kw)
        self.canned, self.probabilistic, self.calls = scores, probabilistic, 0

    async def _scores(self, query, return_utterances=False):
        self.calls += 1
        return QueryResult(scores=dict(self.canned))


def test_chain_falls_through_when_first_is_unsure():
    unsure = Canned({"a": 0.5, "b": 0.4}, confidence_threshold=0.8)
    sure = Canned({"a": 0.1, "b": 0.9}, probabilistic=True)
    pick = Chain(routers=[unsure, sure])("q")
    assert pick.name == "b" and unsure.calls == 1 and sure.calls == 1


def test_chain_stops_at_first_confident_router():
    first = Canned({"a": 0.95, "b": 0.05}, confidence_threshold=0.8)
    second = Canned({"a": 0.0, "b": 1.0})
    assert Chain(routers=[first, second])("q").name == "a" and second.calls == 0


def test_chain_is_calibrated_only_when_every_link_is():
    mixed = [Canned({"a": 1}), Canned({"a": 1}, probabilistic=True)]
    assert not Chain(routers=mixed).calibrated
    both = [Canned({"a": 1}, probabilistic=True), Canned({"a": 1}, probabilistic=True)]
    assert Chain(routers=both).calibrated


def test_merge_takes_a_weighted_mean():
    m = Merge(
        routers=[Canned({"a": 1.0, "b": 0.0}), Canned({"a": 0.0, "b": 1.0})],
        weights=[3, 1],
    )
    picks = m("q", limit=None)
    assert {p.name: round(p.confidence, 2) for p in picks} == {"a": 0.75, "b": 0.25}


def test_chain_checks_the_top_routes_own_bar():
    strict = [Route(name="a", utterances=["a"], confidence_threshold=0.99), ROUTES[1]]

    class Strict(Canned):
        def __init__(self, scores, **kw):
            super().__init__(scores, **kw)
            self.routes = list(strict)

    first = Strict({"a": 0.9, "b": 0.1}, confidence_threshold=0.5)
    second = Canned({"a": 0.0, "b": 1.0})
    # 0.9 clears the router's 0.5 but not route a's own 0.99, so the chain falls through
    assert Chain(routers=[first, second])("q").name == "b" and second.calls == 1


def test_chain_and_merge_read_their_routes_live():
    first = Canned({"a": 1.0, "b": 0.0})
    chain, merge = Chain(routers=[first]), Merge(routers=[first])
    first.routes = [*first.routes, Route(name="c", utterances=["c"])]
    assert [r.name for r in chain.routes] == ["a", "b", "c"]
    assert [r.name for r in merge.routes] == ["a", "b", "c"]


def test_a_flat_calibrator_never_reorders_routes():
    from semantic_router.calibration import Calibrator

    flat = Calibrator("platt", {"a": [0.0], "b": [0.0]})  # every score maps to 0.5
    r = Canned({"a": 0.2, "b": 0.9}, calibrator=flat)
    assert r("q").name == "b"
    assert Chain(routers=[r])("q").name == "b"


def test_an_empty_pick_on_a_calibrated_router_says_calibrated():
    r = Canned({"a": 0.2, "b": 0.1}, probabilistic=True, confidence_threshold=0.9)
    choice = r("q")
    assert choice.name is None and choice.calibrated


class Batched(BaseRouter):
    """scores from a table per query, and records the size of every batch it is asked"""

    def __init__(self, table, **kw):
        super().__init__(ROUTES, **kw)
        self.table, self.batches = table, []

    async def _scores(self, query, return_utterances=False):
        return QueryResult(scores=dict(self.table[query]))

    async def _scores_many(self, queries, return_utterances=False):
        self.batches.append(len(queries))
        return [QueryResult(scores=dict(self.table[q])) for q in queries]


def test_chain_batches_and_only_passes_on_the_unsure_queries():
    sure, unsure = {"a": 0.9, "b": 0.1}, {"a": 0.5, "b": 0.4}
    first = Batched({"q1": sure, "q2": unsure, "q3": sure}, confidence_threshold=0.8)
    second = Batched({q: {"a": 0.0, "b": 1.0} for q in ("q1", "q2", "q3")})
    chain = Chain(routers=[first, second])
    scored = asyncio.run(chain._scores_many(["q1", "q2", "q3"]))
    assert first.batches == [3] and second.batches == [1]  # one batch each
    assert [max(s.scores, key=s.scores.get) for s in scored] == [
        "a",
        "b",
        "a",
    ]  # order kept
    assert chain("q2").name == "b"  # a single query takes the same path


def test_merge_scores_the_whole_batch_once_per_router():
    one = Batched({"q1": {"a": 1.0, "b": 0.0}, "q2": {"a": 0.0, "b": 1.0}})
    two = Batched({"q1": {"a": 1.0, "b": 0.0}, "q2": {"a": 0.0, "b": 1.0}})
    merge = Merge(routers=[one, two])
    scored = asyncio.run(merge._scores_many(["q1", "q2"]))
    assert one.batches == [2] and two.batches == [2]
    assert [max(s.scores, key=s.scores.get) for s in scored] == ["a", "b"]
