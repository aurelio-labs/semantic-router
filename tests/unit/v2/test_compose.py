from semantic_router.route import Route
from semantic_router.v2 import Chain, Merge
from semantic_router.v2.base import BaseRouter

ROUTES = [Route(name="a", utterances=["a"]), Route(name="b", utterances=["b"])]


class Canned(BaseRouter):
    def __init__(self, scores, probabilistic=False, **kw):
        super().__init__(ROUTES, **kw)
        self.canned, self.probabilistic, self.calls = scores, probabilistic, 0

    async def _scores(self, query):
        self.calls += 1
        return dict(self.canned)


def test_chain_falls_through_when_first_is_unsure():
    unsure = Canned({"a": 0.5, "b": 0.4}, confidence_threshold=0.8)
    sure = Canned({"a": 0.1, "b": 0.9}, probabilistic=True)
    pick = Chain(unsure, sure)("q")
    assert pick.name == "b" and unsure.calls == 1 and sure.calls == 1


def test_chain_stops_at_first_confident_router():
    first = Canned({"a": 0.95, "b": 0.05}, confidence_threshold=0.8)
    second = Canned({"a": 0.0, "b": 1.0})
    assert Chain(first, second)("q").name == "a" and second.calls == 0


def test_chain_is_calibrated_only_when_every_link_is():
    assert not Chain(Canned({"a": 1}), Canned({"a": 1}, probabilistic=True)).calibrated
    assert Chain(
        Canned({"a": 1}, probabilistic=True), Canned({"a": 1}, probabilistic=True)
    ).calibrated


def test_merge_takes_a_weighted_mean():
    m = Merge(
        Canned({"a": 1.0, "b": 0.0}), Canned({"a": 0.0, "b": 1.0}), weights=[3, 1]
    )
    picks = m("q", limit=None)
    assert {p.name: round(p.confidence, 2) for p in picks} == {"a": 0.75, "b": 0.25}
