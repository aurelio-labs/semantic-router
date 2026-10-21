"""RedisIndex against the local Redis Stack container. Route filters, deletes and
namespaces must match names exactly: a raw RediSearch TAG folds case and splits
on commas, so `Chat` used to match `chat`, and `a` matched `a,b`."""

import asyncio
import uuid

import numpy as np
import pytest

pytest.importorskip("redis", reason="redis not installed")

from redis.exceptions import ConnectionError as RedisConnectionError

from semantic_router.index.redis import RedisIndex
from semantic_router.schema import Metric

ROUTES = ["Chat", "chat", "a,b", "a", "what?", "path/to", "x y", "über"]


def _index(**kwargs) -> RedisIndex:
    # unique index (and key prefix) per test so the suite runs in parallel;
    # conftest drops it afterwards
    return RedisIndex(index_name=f"test-{uuid.uuid4().hex}", **kwargs)


def _vectors(n: int, dims: int = 8) -> list[list[float]]:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, dims))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).tolist()


def _add_routes(index: RedisIndex) -> list[list[float]]:
    vectors = _vectors(len(ROUTES))
    index.add(
        embeddings=vectors,
        routes=ROUTES,
        utterances=[f"utterance for {r}" for r in ROUTES],
    )
    return vectors


@pytest.mark.parametrize("route", ROUTES)
def test_route_filter_matches_only_that_route(route):
    index = _index()
    vectors = _add_routes(index)
    query = np.array(vectors[ROUTES.index(route)])
    _, routes = index.query(vector=query, top_k=len(ROUTES), route_filter=[route])
    assert routes == [route]


@pytest.mark.parametrize("route", ["Chat", "a,b", "a"])
def test_delete_removes_only_that_route(route):
    index = _index()
    _add_routes(index)
    index.delete(route_name=route)
    assert sorted({u.route for u in index.get_utterances()}) == sorted(
        set(ROUTES) - {route}
    )


def test_namespaces_differing_in_case_are_separate_tenants():
    name = f"test-{uuid.uuid4().hex}"
    upper = RedisIndex(index_name=name, namespace="Acme")
    lower = RedisIndex(index_name=name, namespace="acme")
    vectors = _vectors(2)
    upper.add(embeddings=vectors[:1], routes=["billing"], utterances=["refund"])
    lower.add(embeddings=vectors[1:], routes=["support"], utterances=["help"])
    assert [u.route for u in upper.get_utterances()] == ["billing"]
    _, routes = lower.query(vector=np.array(vectors[0]), top_k=5)
    assert routes == ["support"]
    lower.delete(route_name="billing")
    assert [u.route for u in upper.get_utterances()] == ["billing"]


@pytest.mark.parametrize("metric", [Metric.COSINE, Metric.DOTPRODUCT, Metric.EUCLIDEAN])
def test_scores_match_the_metric(metric):
    index = _index(metric=metric)
    a, b = (np.array(v) for v in _vectors(2))
    b = b * 2.0  # not unit length, so dot product and cosine differ
    index.add(embeddings=[b.tolist()], routes=["b"], utterances=["b"])
    scores, _ = index.query(vector=a, top_k=1)
    expected = {
        Metric.COSINE: a @ b / np.linalg.norm(a) / np.linalg.norm(b),
        Metric.DOTPRODUCT: a @ b,
        # the Euclidean distance, as QdrantIndex and PostgresIndex return it
        Metric.EUCLIDEAN: np.linalg.norm(a - b),
    }[metric]
    assert scores[0] == pytest.approx(expected, abs=1e-5)


def test_sync_and_async_writes_upsert_the_same_records():
    index = _index()
    vectors = _vectors(2)
    routes, utterances = ["billing", "support"], ["refund", "help"]
    index.add(embeddings=vectors, routes=routes, utterances=utterances)

    async def run() -> int:
        await index.aadd(embeddings=vectors, routes=routes, utterances=utterances)
        _, found = await index.aquery(vector=np.array(vectors[1]), top_k=1)
        assert found == ["support"]
        assert await index.adelete(route_name="billing")
        return await index.alen()

    assert asyncio.run(run()) == 1
    assert [u.route for u in index.get_utterances()] == ["support"]


def test_unreachable_server_raises_instead_of_looking_empty():
    # nothing listens on this port; an outage must not read as "no index"
    index = RedisIndex(index_name=f"test-{uuid.uuid4().hex}", port=6390)
    with pytest.raises(RedisConnectionError):
        index.query(vector=np.array(_vectors(1)[0]), top_k=1)
    with pytest.raises(RedisConnectionError):
        index.is_ready()


def test_index_names_that_extend_each_other_keep_their_own_records():
    # RediSearch matches keys by prefix: `name:` used to cover `name:v2:` too
    name = f"test-{uuid.uuid4().hex}"
    base, v2 = RedisIndex(index_name=name), RedisIndex(index_name=f"{name}:v2")
    vectors = _vectors(2)
    base.add(embeddings=vectors[:1], routes=["base"], utterances=["one"])
    v2.add(embeddings=vectors[1:], routes=["v2"], utterances=["two"])
    assert [u.route for u in base.get_utterances()] == ["base"]
    base.delete_index()
    assert [u.route for u in v2.get_utterances()] == ["v2"]


def test_vectors_of_another_size_raise_instead_of_vanishing():
    # Redis stores a wrong-size vector without indexing it, so it was never found
    name = f"test-{uuid.uuid4().hex}"
    RedisIndex(index_name=name).add(
        embeddings=_vectors(1, dims=8), routes=["eight"], utterances=["one"]
    )
    with pytest.raises(ValueError, match="8-dimensional"):
        RedisIndex(index_name=name).add(
            embeddings=_vectors(1, dims=4), routes=["four"], utterances=["two"]
        )
    reader = RedisIndex(index_name=name)
    reader.query(vector=np.array(_vectors(1, dims=8)[0]), top_k=1)
    assert reader.dimensions == 8  # a read-only handle learns them
