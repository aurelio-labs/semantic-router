"""Unit tests for QdrantIndex — uses in-memory Qdrant, no server needed."""

import uuid

import numpy as np
import pytest

pytest.importorskip("qdrant_client", reason="qdrant-client not installed")

from qdrant_client import models

from semantic_router.index.qdrant import (
    SR_ROUTE_PAYLOAD_KEY,
    QdrantIndex,
)

DIMS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_index(namespace=None, index_name=None):
    """Create a QdrantIndex backed by an in-memory Qdrant store."""
    return QdrantIndex(
        index_name=index_name or f"test-{uuid.uuid4().hex[:8]}",
        dimensions=DIMS,
        namespace=namespace,
    )


def rand_vecs(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, DIMS)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).tolist()


@pytest.fixture
def shared_pair():
    """Two QdrantIndex instances backed by the same in-memory QdrantClient.

    Path-based Qdrant uses an exclusive file lock so two clients can't open
    the same directory simultaneously. Sharing the client object is the clean
    way to give two QdrantIndex instances a truly shared collection without
    needing a live server.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    collection = f"shared-{uuid.uuid4().hex[:8]}"

    def _make(namespace):
        idx = make_index(namespace=namespace, index_name=collection)
        idx.client = client  # replace isolated client with shared one
        return idx

    return _make


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self):
        idx = make_index()
        assert idx.namespace is None
        assert not idx._collection_initialized

    def test_namespace_stored(self):
        idx = make_index(namespace="org-abc")
        assert idx.namespace == "org-abc"
        assert not idx._collection_initialized

    def test_clients_initialized(self):
        idx = make_index()
        assert idx.client is not None
        # aclient is None for in-memory (qdrant-client limitation)
        assert idx.aclient is None


# ---------------------------------------------------------------------------
# _build_filter
# ---------------------------------------------------------------------------


class TestBuildFilter:
    def test_no_namespace_no_base(self):
        idx = make_index()
        assert idx._build_filter() is None

    def test_no_namespace_with_base(self):
        idx = make_index()
        base = models.Filter(
            must=[models.FieldCondition(key="x", match=models.MatchValue(value="y"))]
        )
        assert idx._build_filter(base) is base  # returned unchanged

    def test_namespace_no_base(self):
        idx = make_index(namespace="org-1")
        f = idx._build_filter()
        assert f is not None
        assert len(f.must) == 1
        cond = f.must[0]
        assert cond.key == "namespace"
        assert cond.match.value == "org-1"

    def test_namespace_with_base(self):
        idx = make_index(namespace="org-1")
        base = models.Filter(
            must=[
                models.FieldCondition(
                    key=SR_ROUTE_PAYLOAD_KEY, match=models.MatchAny(any=["billing"])
                )
            ]
        )
        f = idx._build_filter(base)
        assert len(f.must) == 2
        # One element must be the namespace FieldCondition
        ns_items = [
            c
            for c in f.must
            if isinstance(c, models.FieldCondition) and c.key == "namespace"
        ]
        assert len(ns_items) == 1
        assert ns_items[0].match.value == "org-1"
        # The base Filter must be present unchanged
        assert base in f.must


# ---------------------------------------------------------------------------
# _init_collection
# ---------------------------------------------------------------------------


class TestInitCollection:
    def test_creates_collection(self):
        idx = make_index()
        idx._init_collection()
        assert idx._collection_initialized
        assert idx.client.collection_exists(idx.index_name)

    def test_idempotent(self):
        idx = make_index()
        idx._init_collection()
        idx._init_collection()  # must not raise
        assert idx._collection_initialized

    def test_deferred_until_first_write(self):
        idx = make_index()
        assert not idx._collection_initialized
        idx.add(rand_vecs(1), ["r"], ["u"])
        assert idx._collection_initialized

    def test_creates_payload_index_for_namespace(self):
        idx = make_index(namespace="org-x")
        idx._init_collection()
        # Collection should exist; payload index creation is a no-op in local mode
        assert idx.client.collection_exists(idx.index_name)


# ---------------------------------------------------------------------------
# add / payload / UUIDs
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_writes_points(self):
        idx = make_index()
        idx.add(
            embeddings=rand_vecs(2), routes=["r1", "r1"], utterances=["hello", "hi"]
        )
        assert len(idx.get_utterances()) == 2

    def test_add_injects_namespace_payload(self):
        idx = make_index(namespace="org-ns")
        idx.add(embeddings=rand_vecs(1), routes=["r"], utterances=["test"])
        records, _ = idx.client.scroll(idx.index_name, with_payload=True, limit=10)
        assert records[0].payload["namespace"] == "org-ns"

    def test_add_no_namespace_payload_key_absent(self):
        idx = make_index()
        idx.add(embeddings=rand_vecs(1), routes=["r"], utterances=["test"])
        records, _ = idx.client.scroll(idx.index_name, with_payload=True, limit=10)
        assert "namespace" not in records[0].payload

    def test_deterministic_uuid_without_namespace(self):
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, "r:utt"))
        idx = make_index()
        idx.add(embeddings=rand_vecs(1), routes=["r"], utterances=["utt"])
        records, _ = idx.client.scroll(idx.index_name, with_payload=True, limit=10)
        assert str(records[0].id) == expected

    def test_deterministic_uuid_with_namespace(self):
        ns = "org-abc"
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{ns}:r:utt"))
        idx = make_index(namespace=ns)
        idx.add(embeddings=rand_vecs(1), routes=["r"], utterances=["utt"])
        records, _ = idx.client.scroll(idx.index_name, with_payload=True, limit=10)
        assert str(records[0].id) == expected

    def test_namespace_uuid_differs_from_plain(self):
        assert str(uuid.uuid5(uuid.NAMESPACE_DNS, "r:utt")) != str(
            uuid.uuid5(uuid.NAMESPACE_DNS, "org-x:r:utt")
        )

    def test_different_namespaces_produce_different_uuids(self):
        uuid_a = str(uuid.uuid5(uuid.NAMESPACE_DNS, "org-a:r:utt"))
        uuid_b = str(uuid.uuid5(uuid.NAMESPACE_DNS, "org-b:r:utt"))
        assert uuid_a != uuid_b

    def test_upsert_idempotent(self):
        idx = make_index()
        idx.add(embeddings=rand_vecs(1), routes=["r"], utterances=["hello"])
        idx.add(embeddings=rand_vecs(1), routes=["r"], utterances=["hello"])
        assert len(idx.get_utterances()) == 1


# ---------------------------------------------------------------------------
# get_utterances — with and without namespace (shared in-memory client)
# ---------------------------------------------------------------------------


class TestGetUtterances:
    def test_returns_empty_for_missing_collection(self):
        idx = make_index()  # no namespace
        assert idx.get_utterances() == []

    def test_returns_all_points_without_namespace(self):
        idx = make_index()  # no namespace — no filter applied
        idx.add(rand_vecs(3), ["r1", "r1", "r2"], ["a", "b", "c"])
        assert len(idx.get_utterances()) == 3

    def test_namespace_filters_results(self, shared_pair):
        idx_a = shared_pair("org-a")
        idx_b = shared_pair("org-b")

        idx_a.add(rand_vecs(2), ["route-a", "route-a"], ["a1", "a2"])
        idx_b.add(rand_vecs(1), ["route-b"], ["b1"])

        a_utts = idx_a.get_utterances()
        b_utts = idx_b.get_utterances()

        assert len(a_utts) == 2
        assert all(u.route == "route-a" for u in a_utts)
        assert len(b_utts) == 1
        assert b_utts[0].route == "route-b"

    def test_decoy_route_not_visible_across_namespaces(self, shared_pair):
        idx_a = shared_pair("org-a")
        idx_b = shared_pair("org-b")

        idx_a.add(rand_vecs(3), ["billing"] * 3, ["u1", "u2", "u3"])
        idx_b.add(rand_vecs(2), ["decoy"] * 2, ["d1", "d2"])

        routes_seen = {u.route for u in idx_a.get_utterances()}
        assert "decoy" not in routes_seen
        assert "billing" in routes_seen


# ---------------------------------------------------------------------------
# query — with and without namespace (shared in-memory client)
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_returns_results(self):
        idx = make_index()
        idx.add(rand_vecs(3), ["r1", "r1", "r2"], ["a", "b", "c"])
        scores, routes = idx.query(vector=np.array(rand_vecs(1)[0]), top_k=2)
        assert len(scores) == 2
        assert len(routes) == 2

    def test_query_scoped_to_namespace(self, shared_pair):
        idx_a = shared_pair("org-a")
        idx_b = shared_pair("org-b")

        idx_a.add(rand_vecs(2), ["route-a", "route-a"], ["u1", "u2"])
        idx_b.add(rand_vecs(2), ["route-b", "route-b"], ["u3", "u4"])

        _, routes = idx_a.query(vector=np.array(rand_vecs(1)[0]), top_k=5)
        assert all(r == "route-a" for r in routes)

    def test_route_filter_with_namespace(self, shared_pair):
        idx = shared_pair("org-f")
        idx.add(
            rand_vecs(4),
            ["billing", "billing", "support", "support"],
            ["u1", "u2", "u3", "u4"],
        )
        _, routes = idx.query(
            vector=np.array(rand_vecs(1)[0]), top_k=5, route_filter=["billing"]
        )
        assert all(r == "billing" for r in routes)

    def test_query_does_not_return_other_namespace(self, shared_pair):
        idx_a = shared_pair("org-a")
        idx_b = shared_pair("org-b")

        idx_a.add(rand_vecs(2), ["route-a"] * 2, ["u1", "u2"])
        idx_b.add(rand_vecs(3), ["route-b"] * 3, ["u3", "u4", "u5"])

        _, routes = idx_a.query(vector=np.array(rand_vecs(1)[0]), top_k=10)
        assert "route-b" not in routes


# ---------------------------------------------------------------------------
# delete — with and without namespace (shared in-memory client)
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_route(self):
        idx = make_index()
        idx.add(rand_vecs(3), ["r1", "r1", "r2"], ["a", "b", "c"])
        idx.delete("r1")
        utterances = idx.get_utterances()
        assert all(u.route != "r1" for u in utterances)
        assert len(utterances) == 1

    def test_delete_scoped_to_namespace(self, shared_pair):
        idx_a = shared_pair("org-a")
        idx_b = shared_pair("org-b")

        idx_a.add(rand_vecs(2), ["r", "r"], ["a1", "a2"])
        idx_b.add(rand_vecs(2), ["r", "r"], ["b1", "b2"])

        idx_a.delete("r")

        assert idx_a.get_utterances() == []
        assert len(idx_b.get_utterances()) == 2  # unaffected


# ---------------------------------------------------------------------------
# async paths — with and without namespace (shared in-memory client)
# aclient is None for in-memory Qdrant; async operations fall back to sync.
# This still exercises every async code path and the namespace logic.
# ---------------------------------------------------------------------------


class TestAsync:
    @pytest.mark.asyncio
    async def test_aadd_writes_points(self):
        idx = make_index()  # no namespace
        await idx.aadd(rand_vecs(2), ["r", "r"], ["x", "y"])
        assert len(await idx.aget_utterances()) == 2

    @pytest.mark.asyncio
    async def test_adelete_removes_route(self):
        idx = make_index()  # no namespace
        await idx.aadd(rand_vecs(3), ["r1", "r1", "r2"], ["a", "b", "c"])
        await idx.adelete("r1")
        utts = await idx.aget_utterances()
        assert all(u.route != "r1" for u in utts)
        assert len(utts) == 1

    @pytest.mark.asyncio
    async def test_aquery_returns_results(self):
        idx = make_index()  # no namespace
        await idx.aadd(rand_vecs(3), ["r1", "r1", "r2"], ["a", "b", "c"])
        scores, routes = await idx.aquery(vector=np.array(rand_vecs(1)[0]), top_k=2)
        assert len(scores) == 2
        assert len(routes) == 2

    @pytest.mark.asyncio
    async def test_aadd_injects_namespace_payload(self):
        idx = make_index(namespace="org-ns")
        await idx.aadd(rand_vecs(1), ["r"], ["test"])
        records, _ = idx.client.scroll(idx.index_name, with_payload=True, limit=10)
        assert records[0].payload["namespace"] == "org-ns"

    @pytest.mark.asyncio
    async def test_aadd_namespace_isolation(self, shared_pair):
        idx_a = shared_pair("org-a")
        idx_b = shared_pair("org-b")

        await idx_a.aadd(rand_vecs(2), ["ra", "ra"], ["a1", "a2"])
        await idx_b.aadd(rand_vecs(1), ["rb"], ["b1"])

        assert len(await idx_a.aget_utterances()) == 2
        assert len(await idx_b.aget_utterances()) == 1

    @pytest.mark.asyncio
    async def test_adelete_scoped_to_namespace(self, shared_pair):
        idx_a = shared_pair("org-a")
        idx_b = shared_pair("org-b")

        await idx_a.aadd(rand_vecs(2), ["r", "r"], ["a1", "a2"])
        await idx_b.aadd(rand_vecs(2), ["r", "r"], ["b1", "b2"])

        await idx_a.adelete("r")

        assert await idx_a.aget_utterances() == []
        assert len(await idx_b.aget_utterances()) == 2  # unaffected

    @pytest.mark.asyncio
    async def test_aquery_scoped_to_namespace(self, shared_pair):
        idx_a = shared_pair("org-a")
        idx_b = shared_pair("org-b")

        await idx_a.aadd(rand_vecs(2), ["ra", "ra"], ["a1", "a2"])
        await idx_b.aadd(rand_vecs(2), ["rb", "rb"], ["b1", "b2"])

        _, routes = await idx_a.aquery(vector=np.array(rand_vecs(1)[0]), top_k=5)
        assert all(r == "ra" for r in routes)

    @pytest.mark.asyncio
    async def test_aquery_does_not_return_other_namespace(self, shared_pair):
        idx_a = shared_pair("org-a")
        idx_b = shared_pair("org-b")

        await idx_a.aadd(rand_vecs(2), ["route-a"] * 2, ["u1", "u2"])
        await idx_b.aadd(rand_vecs(3), ["route-b"] * 3, ["u3", "u4", "u5"])

        _, routes = await idx_a.aquery(vector=np.array(rand_vecs(1)[0]), top_k=10)
        assert "route-b" not in routes
