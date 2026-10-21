"""RedisIndex against the local Redis Stack container. Route filters, deletes and
namespaces must match names exactly: a raw RediSearch TAG folds case and splits
on commas, so `Chat` used to match `chat`, and `a` matched `a,b`."""

import asyncio
import hashlib
import uuid

import numpy as np
import pytest

pytest.importorskip("redis", reason="redis not installed")

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import NoPermissionError

from semantic_router.encoders import DenseEncoder
from semantic_router.index.redis import RedisIndex
from semantic_router.route import Route
from semantic_router.routers import SemanticRouter
from semantic_router.schema import ConfigParameter, Metric

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


def _keys(index: RedisIndex) -> list:
    return list(index.client.scan_iter(match=f"{index._key_prefix()}*"))


def test_a_wrong_size_vector_writes_nothing():
    # Redis stores a wrong-size vector without indexing it, and that hash
    # survives delete_index() to come back in a new index of that size
    index = _index()
    index.add(embeddings=_vectors(1, dims=8), routes=["eight"], utterances=["one"])
    with pytest.raises(ValueError, match="8-dimensional"):
        index.add(embeddings=_vectors(1, dims=4), routes=["four"], utterances=["two"])
    with pytest.raises(ValueError, match="8-dimensional"):
        index.add(
            embeddings=_vectors(1, dims=8) + _vectors(1, dims=4),
            routes=["eight", "four"],
            utterances=["three", "four"],
        )
    with pytest.raises(ValueError, match="8-dimensional"):
        asyncio.run(
            index.aadd(
                embeddings=_vectors(1, dims=4), routes=["four"], utterances=["five"]
            )
        )
    assert len(_keys(index)) == 1


def test_a_mixed_batch_on_a_new_index_writes_nothing():
    index = _index()
    with pytest.raises(ValueError, match="8-dimensional"):
        index.add(
            embeddings=_vectors(1, dims=8) + _vectors(1, dims=4),
            routes=["eight", "four"],
            utterances=["one", "two"],
        )
    assert _keys(index) == []


class _HashEncoder(DenseEncoder):
    """Deterministic 8-d vectors: syncing doesn't depend on what they mean."""

    name: str = "hash"

    def __call__(self, docs: list) -> list[list[float]]:
        return [
            [b / 255 for b in hashlib.sha256(str(d).encode()).digest()[:8]]
            for d in docs
        ]


ROUTER_ROUTES = [
    Route(name="greet", utterances=["hi", "hello"]),
    Route(name="bye", utterances=["bye", "see you"]),
]


def _router(index: RedisIndex) -> SemanticRouter:
    return SemanticRouter(
        encoder=_HashEncoder(),
        routes=[r.model_copy() for r in ROUTER_ROUTES],
        index=index,
    )


def test_tenants_keep_their_own_sync_state():
    # sr_hash and sr_lock used to be shared, so tenant b looked synced after a
    name = f"test-{uuid.uuid4().hex}"
    a = RedisIndex(index_name=name, namespace="org-a")
    b = RedisIndex(index_name=name, namespace="org-b")
    router_a, router_b = _router(a), _router(b)
    router_a.sync("local")
    assert not router_b.is_synced()
    router_b.sync("local")
    assert len(b.get_utterances()) == 4
    a.lock(value=True)
    assert not b._is_locked()


def test_delete_index_drops_the_sync_state():
    name = f"test-{uuid.uuid4().hex}"
    index = RedisIndex(index_name=name)
    _router(index).sync("local")
    index.lock(value=True)
    index.delete_index()
    reopened = RedisIndex(index_name=name)
    router = _router(reopened)
    assert not router.is_synced()
    assert not reopened._is_locked()
    router.sync("local")
    assert len(reopened.get_utterances()) == 4

    # and the same through adelete_index()
    reopened._write_config(ConfigParameter(field="sr_hash", value="stale"))
    asyncio.run(reopened.adelete_index())
    assert RedisIndex(index_name=name)._read_hash().value == ""


@pytest.mark.parametrize("metric", [Metric.DOTPRODUCT, Metric.EUCLIDEAN])
def test_reopening_with_another_metric_raises(metric):
    # the scores would come back under the wrong metric
    name = f"test-{uuid.uuid4().hex}"
    RedisIndex(index_name=name, metric=Metric.COSINE).add(
        embeddings=_vectors(1), routes=["r"], utterances=["u"]
    )
    with pytest.raises(ValueError, match="cosine"):
        RedisIndex(index_name=name, metric=metric).query(
            vector=np.array(_vectors(1)[0]), top_k=1
        )
    with pytest.raises(ValueError, match="cosine"):
        RedisIndex(index_name=name, metric=metric).add(
            embeddings=_vectors(1), routes=["r2"], utterances=["u2"]
        )


def test_ids_dont_collide_across_colons():
    name = f"test-{uuid.uuid4().hex}"
    a, ab = (RedisIndex(index_name=name, namespace=ns) for ns in ("a", "a:b"))
    vectors = _vectors(2)
    a.add(embeddings=vectors[:1], routes=["b:c"], utterances=["d"])
    ab.add(embeddings=vectors[1:], routes=["c"], utterances=["d"])
    assert [(u.route, u.utterance) for u in a.get_utterances()] == [("b:c", "d")]

    index = _index()
    index.add(embeddings=vectors, routes=["a:b", "a"], utterances=["c", "b:c"])
    assert len(index.get_utterances()) == 2


def test_an_empty_route_filter_matches_nothing():
    index = _index()
    vectors = _add_routes(index)
    query = np.array(vectors[0])
    assert index.query(vector=query, top_k=5, route_filter=[])[1] == []
    assert asyncio.run(index.aquery(vector=query, top_k=5, route_filter=[]))[1] == []


def test_an_index_named_like_the_config_keys_is_refused():
    # its key prefix would pick up every other index's sync state
    with pytest.raises(ValueError, match="sr_config"):
        RedisIndex(index_name="sr_config")


def test_a_missing_index_reads_as_empty():
    # the one server error that means "no index yet", in each Redis's wording
    index = _index()
    query = np.array(_vectors(1)[0])
    assert index.query(vector=query, top_k=1)[1] == []
    assert not index.is_ready()
    assert len(index) == 0

    async def run():
        assert (await index.aquery(vector=query, top_k=1))[1] == []
        assert not await index.ais_ready()
        assert await index.alen() == 0

    asyncio.run(run())


@pytest.fixture
def no_ft_info_user():
    """A Redis user that may run every command except FT.INFO."""
    admin = _index()
    name = f"test-{uuid.uuid4().hex}"
    admin.client.acl_setuser(
        name,
        enabled=True,
        passwords=["+pw"],
        keys=["*"],
        channels=["*"],
        commands=["+@all", "-ft.info"],
    )
    yield name
    admin.client.acl_deluser(name)


def test_other_server_errors_raise_instead_of_looking_empty(no_ft_info_user):
    # a permission error used to read as "no index", so query returned [],
    # is_ready() False and len() 0
    index = _index(username=no_ft_info_user, password="pw")
    query = np.array(_vectors(1)[0])
    for call in (
        lambda: index.query(vector=query, top_k=1),
        index.is_ready,
        lambda: len(index),
    ):
        with pytest.raises(NoPermissionError):
            call()

    async def run():
        for acall in (
            lambda: index.aquery(vector=query, top_k=1),
            index.ais_ready,
            index.alen,
        ):
            with pytest.raises(NoPermissionError):
                await acall()

    asyncio.run(run())
