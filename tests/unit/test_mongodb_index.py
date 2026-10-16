"""Unit tests for MongoDbIndex — uses mocked pymongo/motor clients, no live server needed."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from semantic_router.index.base import SR_ROUTE, SR_UTTERANCE
from semantic_router.index.mongodb import (
    SR_METADATA_FIELD,
    SR_NAMESPACE_FIELD,
    SR_ROUTE_FIELD,
    SR_UTTERANCE_FIELD,
    SR_VECTOR_FIELD,
    MongoDbIndex,
)
from semantic_router.schema import Metric

DIMS = 4
SAMPLE_EMBEDDINGS = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
SAMPLE_ROUTES = ["route_a", "route_b"]
SAMPLE_UTTERANCES = ["hello", "world"]


def _make_sync_client_mock():
    sync_client = MagicMock()
    sync_client.admin.command.return_value = {"ok": 1}
    sync_db = MagicMock()
    sync_client.__getitem__.return_value = sync_db
    sync_collection = MagicMock()
    sync_db.__getitem__.return_value = sync_collection
    sync_db.list_collection_names.return_value = ["semantic-router-index"]
    sync_collection.find.return_value = iter([])
    return sync_client, sync_db, sync_collection


def _make_async_client_mock():
    async_client = MagicMock()
    async_client.admin.command = AsyncMock(return_value={"ok": 1})
    async_db = MagicMock()
    async_client.__getitem__.return_value = async_db
    async_collection = MagicMock()
    async_db.__getitem__.return_value = async_collection
    async_db.list_collection_names = AsyncMock(return_value=["semantic-router-index"])
    async_collection.bulk_write = AsyncMock()
    async_collection.count_documents = AsyncMock(return_value=0)
    async_collection.create_index = AsyncMock()
    return async_client, async_db, async_collection


@pytest.fixture
def mock_mongo_index():
    def _make(**overrides):
        sync_client, sync_db, sync_collection = _make_sync_client_mock()
        async_client, async_db, async_collection = _make_async_client_mock()

        with patch("pymongo.MongoClient", return_value=sync_client), patch(
            "motor.motor_asyncio.AsyncIOMotorClient", return_value=async_client
        ):
            kwargs = {
                "connection_uri": "mongodb://fake:27017/test",
                "dimensions": DIMS,
                **overrides,
            }
            idx = MongoDbIndex(**kwargs)
            idx._sync_mock = sync_collection
            idx._async_mock = async_collection
            idx._sync_db_mock = sync_db
            idx._async_db_mock = async_db
            return idx

    return _make


class TestConstruction:
    def test_defaults(self, mock_mongo_index):
        idx = mock_mongo_index()
        assert idx.type == "mongodb"
        assert idx.database_name == "semantic_router"
        assert idx.collection_name == "semantic-router-index"
        assert idx.vector_index_name == "vector_index"
        assert idx.metric == Metric.COSINE

    def test_env_uri_fallback(self):
        import os

        os.environ["MONGO_CONNECTION_URI"] = "mongodb://env-fake:27017/test"
        try:
            sc, _, _ = _make_sync_client_mock()
            ac, _, _ = _make_async_client_mock()
            with patch("pymongo.MongoClient", return_value=sc), patch(
                "motor.motor_asyncio.AsyncIOMotorClient", return_value=ac
            ):
                idx = MongoDbIndex(dimensions=DIMS)
                assert idx.connection_uri == "mongodb://env-fake:27017/test"
        finally:
            del os.environ["MONGO_CONNECTION_URI"]

    def test_missing_uri_raises(self):
        with pytest.raises(ValueError, match="MONGO_CONNECTION_URI"):
            MongoDbIndex(dimensions=DIMS)

    def test_unknown_kwargs_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MongoDbIndex(connection_uri="mongodb://x", collection_name_extra="oops")


class TestDocumentIds:
    def test_id_deterministic(self, mock_mongo_index):
        idx = mock_mongo_index()
        assert idx._document_id("route_a", "hello") == idx._document_id("route_a", "hello")

    def test_namespace_changes_id(self, mock_mongo_index):
        a = mock_mongo_index(namespace="ns1")._document_id("r", "u")
        b = mock_mongo_index(namespace="ns2")._document_id("r", "u")
        assert a != b

    def test_document_structure_with_namespace(self, mock_mongo_index):
        idx = mock_mongo_index(namespace="ns1")
        doc = idx._document("abc", [0.1, 0.2], "r", "u", {"x": 1})
        assert doc["_id"] == "abc"
        assert doc[SR_VECTOR_FIELD] == [0.1, 0.2]
        assert doc[SR_ROUTE_FIELD] == "r"
        assert doc[SR_UTTERANCE_FIELD] == "u"
        assert doc[SR_NAMESPACE_FIELD] == "ns1"
        assert doc[SR_METADATA_FIELD] == {"x": 1}

    def test_document_no_namespace(self, mock_mongo_index):
        idx = mock_mongo_index()
        doc = idx._document("abc", [0.1], "r", "u", {})
        assert SR_NAMESPACE_FIELD not in doc


class TestAdd:
    def test_add_upserts(self, mock_mongo_index):
        idx = mock_mongo_index()
        idx.add(SAMPLE_EMBEDDINGS, SAMPLE_ROUTES, SAMPLE_UTTERANCES)
        idx._sync_mock.bulk_write.assert_called_once()
        ops = idx._sync_mock.bulk_write.call_args[0][0]
        assert len(ops) == 2
        for op in ops:
            assert "replace_one" in op
            assert op["replace_one"]["upsert"] is True

    def test_add_sparse_rejected(self, mock_mongo_index):
        from semantic_router.schema import SparseEmbedding

        idx = mock_mongo_index()
        with pytest.raises(ValueError, match="dense vectors only"):
            idx.add(
                SAMPLE_EMBEDDINGS, SAMPLE_ROUTES, SAMPLE_UTTERANCES,
                sparse_embeddings=[SparseEmbedding(embedding=np.array([[0, 1.0]]))],
            )

    def test_add_empty_noop(self, mock_mongo_index):
        idx = mock_mongo_index()
        idx.add([], [], [])
        idx._sync_mock.bulk_write.assert_not_called()


class TestQuery:
    def test_query_pipeline_shape(self, mock_mongo_index):
        idx = mock_mongo_index()
        idx._sync_mock.aggregate.return_value = iter([
            {"score": 0.9, SR_ROUTE_FIELD: "route_a"},
            {"score": 0.7, SR_ROUTE_FIELD: "route_b"},
        ])
        scores, names = idx.query(np.array([0.1, 0.2, 0.3, 0.4]), top_k=5)
        pipeline = idx._sync_mock.aggregate.call_args[0][0]
        assert pipeline[0]["$vectorSearch"]["index"] == idx.vector_index_name
        assert pipeline[0]["$vectorSearch"]["limit"] == 5
        assert pipeline[1]["$project"]["score"]["$meta"] == "vectorSearchScore"
        assert list(scores) == [0.9, 0.7]
        assert names == ["route_a", "route_b"]

    def test_query_multi_route_filter(self, mock_mongo_index):
        idx = mock_mongo_index()
        idx._sync_mock.aggregate.return_value = iter([])
        idx.query(np.array([0.1] * DIMS), top_k=3, route_filter=["r1", "r2"])
        flt = idx._sync_mock.aggregate.call_args[0][0][0]["$vectorSearch"].get("filter", {})
        assert flt[SR_ROUTE_FIELD]["$in"] == ["r1", "r2"]

    def test_query_single_route_filter(self, mock_mongo_index):
        idx = mock_mongo_index()
        idx._sync_mock.aggregate.return_value = iter([])
        idx.query(np.array([0.1] * DIMS), top_k=3, route_filter=["only"])
        flt = idx._sync_mock.aggregate.call_args[0][0][0]["$vectorSearch"].get("filter", {})
        assert flt[SR_ROUTE_FIELD] == "only"

    def test_query_sparse_rejected(self, mock_mongo_index):
        idx = mock_mongo_index()
        with pytest.raises(ValueError, match="does not support sparse"):
            idx.query(np.array([0.1] * DIMS), sparse_vector={0: 1.0})


class TestDelete:
    def test_delete_filters_by_route_and_namespace(self, mock_mongo_index):
        idx = mock_mongo_index(namespace="ns1")
        idx.delete("route_a")
        idx._sync_mock.delete_many.assert_called_once_with(
            {SR_NAMESPACE_FIELD: "ns1", SR_ROUTE_FIELD: "route_a"}
        )


class TestGetAll:
    def test_get_all_ids_only(self, mock_mongo_index):
        idx = mock_mongo_index()
        idx._sync_mock.find.return_value = iter([
            {"_id": "abc", SR_ROUTE_FIELD: "r", SR_UTTERANCE_FIELD: "u"},
        ])
        ids, meta = idx._get_all()
        assert ids == ["abc"]
        assert meta == []

    def test_get_all_with_metadata(self, mock_mongo_index):
        idx = mock_mongo_index()
        idx._sync_mock.find.return_value = iter([
            {
                "_id": "abc",
                SR_ROUTE_FIELD: "route_a",
                SR_UTTERANCE_FIELD: "hello",
                SR_METADATA_FIELD: {"k": "v"},
            }
        ])
        ids, meta = idx._get_all(include_metadata=True)
        assert ids == ["abc"]
        assert meta[0][SR_ROUTE] == "route_a"
        assert meta[0][SR_UTTERANCE] == "hello"
        assert meta[0]["k"] == "v"


class TestDescribe:
    def test_describe_counts_docs(self, mock_mongo_index):
        idx = mock_mongo_index()
        idx._sync_mock.aggregate.return_value = iter([{"n": 42}])
        cfg = idx.describe()
        assert cfg.type == "mongodb"
        assert cfg.dimensions == DIMS
        assert cfg.vectors == 42

    def test_is_ready_true(self, mock_mongo_index):
        idx = mock_mongo_index()
        idx._sync_db_mock.list_collection_names.return_value = [idx.collection_name]
        assert idx.is_ready() is True

    def test_is_ready_false_when_collection_missing(self, mock_mongo_index):
        idx = mock_mongo_index()
        idx._sync_db_mock.list_collection_names.return_value = ["other"]
        assert idx.is_ready() is False

    def test_len_counts_docs(self, mock_mongo_index):
        idx = mock_mongo_index()
        idx._sync_mock.count_documents.return_value = 5
        assert len(idx) == 5
