"""IndexedRouter end to end on every index backend, with a real local encoder
(all-MiniLM-L6-v2 through fastembed, downloaded once and cached; no API keys).

The index is the router's single source of truth: ``router.routes`` reads it, so two
routers on one index always agree on what routes exist.
"""

import pytest

from semantic_router.encoders import FastEmbedEncoder, TfidfEncoder
from semantic_router.index import (
    BaseIndex,
    HybridLocalIndex,
    LocalIndex,
    PineconeIndex,
    PostgresIndex,
    QdrantIndex,
)
from semantic_router.route import Route
from semantic_router.routers import IndexedRouter
from semantic_router.schema import RouteChoice

pytest.importorskip("fastembed")


class LocalTestEncoder(FastEmbedEncoder):
    name: str = "sentence-transformers/all-MiniLM-L6-v2"


POLITICS = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
    description="Opinions about politics and politicians.",
    metadata={"team": "trust-and-safety"},
)
CHITCHAT = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)
BILLING = Route(
    name="billing",
    utterances=[
        "I was charged twice for my order",
        "can I get a refund please",
        "my card payment failed",
    ],
    confidence_threshold=0.2,
)
ROUTES = [POLITICS, CHITCHAT]
N_UTTERANCES = sum(len(r.utterances) for r in ROUTES)


@pytest.fixture(scope="module")
def encoder() -> LocalTestEncoder:
    return LocalTestEncoder()


@pytest.fixture(
    params=[LocalIndex, HybridLocalIndex, PineconeIndex, PostgresIndex, QdrantIndex],
    ids=lambda cls: cls.__name__,
)
def index_cls(request):
    return request.param


@pytest.fixture
def index(index_cls, make_index) -> BaseIndex:
    return make_index(index_cls)


def make_router(index: BaseIndex, encoder, routes=ROUTES, **kwargs) -> IndexedRouter:
    if isinstance(index, HybridLocalIndex):
        # a fitted sparse encoder exercises the refit-and-rebuild path on add
        kwargs.setdefault("sparse_encoder", TfidfEncoder())
    return IndexedRouter(encoder=encoder, routes=routes, index=index, **kwargs)


def second_handle(index: BaseIndex, make_index) -> BaseIndex:
    """Another index object on the same stored index, as a second process would open
    it. The in-memory indexes can only be shared as the same object."""
    if isinstance(index, (PineconeIndex, PostgresIndex, QdrantIndex)):
        return make_index(type(index), name=index.index_name)
    return index


def as_dicts(routes: list[Route]) -> dict:
    return {
        r.name: {
            "utterances": sorted(map(str, r.utterances)),
            "description": r.description,
            "confidence_threshold": r.confidence_threshold,
            "metadata": r.metadata,
        }
        for r in routes
    }


def test_routes_read_back_from_the_index(index, encoder):
    router = make_router(index, encoder)

    assert as_dicts(router.routes) == as_dicts(ROUTES)
    assert len(index.get_utterances()) == N_UTTERANCES


def test_classifies_queries(index, encoder):
    router = make_router(index, encoder)

    assert router("don't you love politics?").name == "politics"
    assert router("how's the weather today?").name == "chitchat"


@pytest.mark.asyncio
async def test_acall_classifies_queries(index, encoder):
    router = make_router(index, encoder)

    choice = await router.acall("don't you love politics?")

    assert isinstance(choice, RouteChoice)
    assert choice.name == "politics"


def test_limit_none_ranks_every_route(index, encoder):
    router = make_router(index, encoder)

    choices = router("don't you love politics?", limit=None)

    assert [c.name for c in choices] == ["politics", "chitchat"]
    assert choices[0].confidence >= choices[1].confidence


def test_add_route(index, encoder):
    router = make_router(index, encoder)
    router.add(routes=BILLING)

    assert {r.name for r in router.routes} == {"politics", "chitchat", "billing"}
    assert router("I'd like my money back").name == "billing"


def test_remove_route(index, encoder):
    router = make_router(index, encoder)
    router.remove(name="politics")

    assert [r.name for r in router.routes] == ["chitchat"]
    assert {u.route for u in index.get_utterances()} == {"chitchat"}
    assert router("don't you love politics?").name == "chitchat"


def test_remove_a_route_that_is_not_there(index, encoder):
    router = make_router(index, encoder)
    router.remove(name="not-a-route")

    assert as_dicts(router.routes) == as_dicts(ROUTES)


def test_route_threshold_is_read_from_the_index(index, encoder):
    strict = POLITICS.model_copy(update={"confidence_threshold": 0.99})
    router = make_router(index, encoder, routes=[strict, CHITCHAT])

    # politics is the top route but can't clear its own 0.99 bar on a paraphrase
    assert router("don't you love politics?").name != "politics"
    stored = {r.name: r for r in router.routes}
    assert stored["politics"].confidence_threshold == 0.99


def test_router_threshold_applies_when_the_route_has_none(index, encoder):
    router = make_router(index, encoder, confidence_threshold=0.99)

    assert router("don't you love politics?") == RouteChoice()


def test_constructing_twice_on_one_index_does_not_duplicate(index, encoder, make_index):
    first = make_router(index, encoder)
    sparse = {"sparse_encoder": first.sparse_encoder} if first.sparse_encoder else {}
    make_router(second_handle(index, make_index), encoder, **sparse)

    assert len(index.get_utterances()) == N_UTTERANCES


def test_a_router_on_a_filled_index_needs_no_routes(index, encoder, make_index):
    first = make_router(index, encoder)
    sparse = {"sparse_encoder": first.sparse_encoder} if first.sparse_encoder else {}

    second = make_router(second_handle(index, make_index), encoder, routes=(), **sparse)

    assert as_dicts(second.routes) == as_dicts(ROUTES)
    assert second("don't you love politics?").name == "politics"


def test_routes_another_router_adds_show_up(index, encoder, make_index):
    first = make_router(index, encoder)
    sparse = {"sparse_encoder": first.sparse_encoder} if first.sparse_encoder else {}
    second = make_router(second_handle(index, make_index), encoder, **sparse)

    first.add(routes=BILLING)

    assert "billing" in {r.name for r in second.routes}
    assert second("I'd like my money back").name == "billing"


def test_fit_and_evaluate(index, encoder, request):
    if isinstance(index, HybridLocalIndex):
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="at the default alpha=0.3, TF-IDF carries 70% of the score and is "
                "too noisy on this small corpus (raw top-1 4/6; 6/6 from alpha=0.7). "
                "the default alpha is an open question in thread 10714",
            )
        )
    router = make_router(index, encoder)
    X = [
        "what do you think about the government",
        "who should I vote for in the election",
        "what's the weather going to be like",
        "how has your day been",
        "what is the weather like on mars",  # no route should match
    ]
    y = ["politics", "politics", "chitchat", "chitchat", None]

    report = router.fit(X=X, y=y)

    assert report.n == 5
    assert report.accuracy == 1.0  # over the queries that have a route
    assert router.calibrated
    # no threshold is set, so some route always wins and the None query is a miss
    assert router.evaluate(X=X, y=y) == 0.8
