"""The record contract every index backend keeps, run against the real backends.

The index is the single source of truth for routes. IndexedRouter writes one record
per (route, utterance), with the route-level fields in reserved ``sr_`` metadata keys,
and ``get_routes`` rebuilds every route from those records alone.
"""

import hashlib

import numpy as np
import pytest

from semantic_router.index import (
    BaseIndex,
    HybridLocalIndex,
    LocalIndex,
    PineconeIndex,
    PostgresIndex,
    QdrantIndex,
)
from semantic_router.route import Route
from semantic_router.routers.indexed import _record_metadata
from semantic_router.schema import SparseEmbedding

DIMS = 3

ROUTES = [
    Route(
        name="politics",
        utterances=["isn't politics the best thing ever", "what are your opinions"],
        description="Talk about politics.",
        confidence_threshold=0.8,
        metadata={"team": "a"},
    ),
    # its name starts with another route's name, so a delete must match names exactly
    Route(name="politics_extra", utterances=["who will win the election"]),
    Route(name="chitchat", utterances=["how's the weather today?"]),
]
N_RECORDS = sum(len(r.utterances) for r in ROUTES)


def _seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def records(index: BaseIndex, routes: list[Route]) -> dict:
    """The add() arguments IndexedRouter builds for these routes."""
    pairs = [(r, u) for r in routes for u in r.utterances]
    vectors = [np.random.default_rng(_seed(u)).standard_normal(DIMS) for _, u in pairs]
    kwargs: dict = {
        "embeddings": [(v / np.linalg.norm(v)).tolist() for v in vectors],
        "routes": [r.name for r, _ in pairs],
        "utterances": [u for _, u in pairs],
        "metadata_list": [_record_metadata(r) for r, _ in pairs],
    }
    if isinstance(index, HybridLocalIndex):
        kwargs["sparse_embeddings"] = [
            SparseEmbedding.from_dict({_seed(u) % 50: 1.0}) for _, u in pairs
        ]
    return kwargs


def as_dicts(routes: list[Route]) -> dict:
    return {
        r.name: {
            "utterances": sorted(r.utterances),
            "description": r.description,
            "confidence_threshold": r.confidence_threshold,
            "metadata": r.metadata,
        }
        for r in routes
    }


@pytest.fixture(
    params=[LocalIndex, HybridLocalIndex, PineconeIndex, PostgresIndex, QdrantIndex],
    ids=lambda cls: cls.__name__,
)
def index(request, make_index) -> BaseIndex:
    return make_index(request.param)


def test_get_routes_rebuilds_the_routes(index):
    index.add(**records(index, ROUTES))

    assert as_dicts(index.get_routes()) == as_dicts(ROUTES)
    politics = next(r for r in index.get_routes() if r.name == "politics")
    assert politics.description == "Talk about politics."
    assert politics.confidence_threshold == 0.8
    # reserved keys are stripped, so a route's metadata reads back as it was written
    assert politics.metadata == {"team": "a"}


@pytest.mark.asyncio
async def test_aget_routes_matches_get_routes(index):
    await index.aadd(**records(index, ROUTES))

    assert as_dicts(await index.aget_routes()) == as_dicts(ROUTES)
    assert as_dicts(index.get_routes()) == as_dicts(ROUTES)


def test_unset_route_fields_read_back_as_none(index):
    index.add(**records(index, ROUTES))

    extra = next(r for r in index.get_routes() if r.name == "politics_extra")
    assert extra.description is None
    assert extra.confidence_threshold is None
    assert extra.metadata == {}


def test_adding_the_same_routes_again_upserts(index):
    index.add(**records(index, ROUTES))
    index.add(**records(index, ROUTES))

    assert len(index.get_utterances()) == N_RECORDS
    assert as_dicts(index.get_routes()) == as_dicts(ROUTES)


@pytest.mark.asyncio
async def test_aadd_the_same_routes_again_upserts(index):
    await index.aadd(**records(index, ROUTES))
    await index.aadd(**records(index, ROUTES))

    assert len(await index.aget_utterances()) == N_RECORDS


def test_adding_again_updates_the_route_fields(index):
    index.add(**records(index, ROUTES))
    changed = ROUTES[0].model_copy(
        update={
            "description": "Politics, rewritten.",
            "confidence_threshold": 0.5,
            "metadata": {"team": "c"},
        }
    )
    index.add(**records(index, [changed]))

    politics = next(r for r in index.get_routes() if r.name == "politics")
    assert politics.description == "Politics, rewritten."
    assert politics.confidence_threshold == 0.5
    assert politics.metadata == {"team": "c"}
    assert len(index.get_utterances()) == N_RECORDS


def test_delete_removes_only_that_route(index):
    index.add(**records(index, ROUTES))
    index.delete("politics")

    assert {r.name for r in index.get_routes()} == {"politics_extra", "chitchat"}
    assert len(index.get_utterances()) == N_RECORDS - 2


@pytest.mark.asyncio
async def test_adelete_removes_only_that_route(index):
    await index.aadd(**records(index, ROUTES))
    await index.adelete("politics")

    assert {r.name for r in await index.aget_routes()} == {"politics_extra", "chitchat"}


@pytest.mark.asyncio
async def test_an_index_that_was_never_written_has_no_routes(index):
    assert index.get_routes() == []
    assert await index.aget_routes() == []


@pytest.mark.asyncio
async def test_aquery_finds_the_nearest_route(index):
    kwargs = records(index, ROUTES)
    index.add(**kwargs)
    i = kwargs["utterances"].index("how's the weather today?")
    query: dict = {"vector": np.array(kwargs["embeddings"][i]), "top_k": 1}
    if "sparse_embeddings" in kwargs:
        query["sparse_vector"] = kwargs["sparse_embeddings"][i]

    _, names = await index.aquery(**query)

    assert names == ["chitchat"]


def test_delete_index_then_add_recreates_it(index):
    # IndexedRouter rebuilds the index this way when a fitted sparse encoder refits
    index.add(**records(index, ROUTES))
    index.delete_index()
    index.add(**records(index, ROUTES[:1]))

    assert {r.name for r in index.get_routes()} == {"politics"}


@pytest.mark.parametrize(
    "index_cls", [LocalIndex, PostgresIndex, QdrantIndex], ids=lambda c: c.__name__
)
def test_dense_only_index_rejects_sparse_embeddings(index_cls, make_index):
    index = make_index(index_cls)
    kwargs = records(index, ROUTES)
    kwargs["sparse_embeddings"] = [SparseEmbedding.from_dict({0: 1.0})] * N_RECORDS

    with pytest.raises(ValueError, match="dense vectors only"):
        index.add(**kwargs)


def test_postgres_adds_the_metadata_column_to_an_older_table(make_index):
    import psycopg

    index = make_index(PostgresIndex)
    assert isinstance(index, PostgresIndex)
    # a table as 0.x created it, with no metadata column
    with psycopg.connect(index.connection_string) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"CREATE TABLE {index._get_table_name()} (id VARCHAR(255) PRIMARY KEY, "
            f"route VARCHAR(255), utterance TEXT, vector VECTOR({DIMS}))"
        )
        conn.execute(f"COMMENT ON COLUMN {index._get_table_name()}.vector IS '{DIMS}'")

    index.add(**records(index, ROUTES))

    assert as_dicts(index.get_routes()) == as_dicts(ROUTES)
