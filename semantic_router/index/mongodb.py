"""MongoDB Atlas Vector Search index implementation.

Uses `$vectorSearch` aggregation pipeline backed by `pymongo` for sync and
`motor` for async operations. Requires a MongoDB Atlas cluster with vector
indexes enabled, or a local MongoDB 8.0+ instance that supports $vectorSearch.
"""

import uuid
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import Field, PrivateAttr

from semantic_router.index.base import (
    SR_ROUTE,
    SR_UTTERANCE,
    BaseIndex,
    IndexConfig,
    check_records,
)
from semantic_router.schema import Metric, SparseEmbedding
from semantic_router.utils.logger import logger

DEFAULT_COLLECTION_NAME = "semantic-router-index"
SR_UTTERANCE_FIELD = "sr_utterance"
SR_ROUTE_FIELD = "sr_route"
SR_METADATA_FIELD = "metadata"
SR_NAMESPACE_FIELD = "namespace"
SR_VECTOR_FIELD = "embedding"
DEFAULT_VECTOR_INDEX_NAME = "vector_index"


class MongoDbIndex(BaseIndex):
    """MongoDB Atlas Vector Search index for Semantic Router.

    Uses MongoDB's native ``$vectorSearch`` aggregation stage for approximate
    nearest-neighbor search over dense embeddings. Requires MongoDB Atlas
    or a self-hosted MongoDB 8.0+ cluster with vector indexes.

    Both sync (pymongo) and async (motor) clients are supported. The async
    client is created alongside the sync one when the connection URI is
    available, and each method has a corresponding ``a``-prefixed async
    variant.

    Example::

        index = MongoDbIndex(
            connection_uri="mongodb+srv://user:pass@cluster.mongodb.net/",
            database_name="semantic_router",
            collection_name="routes",
            dimensions=1536,
        )
        index.add(embeddings=[...], routes=[...], utterances=[...])
        scores, names = index.query(vector=np.array([...]), top_k=5)
    """

    type: str = Field(default="mongodb", description="Index type identifier.")
    connection_uri: Optional[str] = Field(
        default=None,
        description=(
            "MongoDB connection URI. Falls back to the ``MONGO_CONNECTION_URI`` "
            "environment variable when unset."
        ),
    )
    database_name: str = Field(
        default="semantic_router",
        description="Database name to use.",
    )
    collection_name: str = Field(
        default=DEFAULT_COLLECTION_NAME,
        description="Collection name inside the database.",
    )
    vector_index_name: str = Field(
        default=DEFAULT_VECTOR_INDEX_NAME,
        description="Name of the MongoDB Atlas vector search index on the collection.",
    )
    dimensions: Union[int, None] = Field(
        default=None,
        description="Embedding dimensions. Defaults to the encoder output.",
    )
    metric: Metric = Field(
        default=Metric.COSINE,
        description="Similarity metric used by the vector index.",
    )
    namespace: Optional[str] = Field(
        default=None,
        description=(
            "Optional namespace (tenant) scoping. When set, every document is "
            "tagged with this namespace, a compound index is created, and all "
            "queries/scrolls/deletes filter by it."
        ),
    )
    # Private fields excluded from Pydantic serialization.
    client: Any = Field(default=None, exclude=True)
    aclient: Any = Field(default=None, exclude=True)
    _collection_initialized: bool = PrivateAttr(default=False)

    # ------------------------------------------------------------------ init

    def __init__(self, **kwargs):
        # Resolve connection_uri from env if missing.
        import os

        if "connection_uri" not in kwargs:
            kwargs["connection_uri"] = os.environ.get("MONGO_CONNECTION_URI")
        super().__init__(**kwargs)
        self.client, self.aclient = self._initialize_clients()

    def _initialize_clients(self) -> Tuple[Any, Optional[Any]]:
        """Create pymongo sync and motor async clients.

        :return: (sync_client, async_client_or_None)
        """
        if not self.connection_uri:
            raise ValueError(
                "MONGO_CONNECTION_URI is not set. Provide ``connection_uri`` "
                "or export the environment variable."
            )
        try:
            import motor.motor_asyncio
            from pymongo import MongoClient
        except ImportError as e:
            raise ImportError(
                "Please install 'pymongo' and 'motor' to use MongoDbIndex. "
                "You can install them with: `pip install pymongo motor`"
            ) from e

        try:
            sync_client: Any = MongoClient(
                self.connection_uri, serverSelectionTimeoutMS=2000
            )
            # Ping to fail fast on bad URIs.
            sync_client.admin.command("ping")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to MongoDB at {self.connection_uri}: {e}"
            ) from e

        try:
            async_client: Any = motor.motor_asyncio.AsyncIOMotorClient(
                self.connection_uri, serverSelectionTimeoutMS=2000
            )
        except Exception as e:
            logger.warning(f"Could not create async MongoDB client: {e}")
            async_client = None

        return sync_client, async_client

    # ----------------------------------------------------------- collection

    @property
    def _db(self):
        return self.client[self.database_name]

    @property
    def _collection(self):
        return self._db[self.collection_name]

    @property
    def _async_db(self):
        return self.aclient[self.database_name]

    @property
    def _async_collection(self):
        return self._async_db[self.collection_name]

    def _init_collection(self) -> None:
        """Create the collection and helper indexes if they don't exist."""
        if self._collection_initialized:
            return
        collection = self._collection
        # Create collection (idempotent on most MongoDB versions).
        if self.collection_name not in self._db.list_collection_names():
            # MongoDB requires ``$vectorSearch`` to be backed by a vector
            # search index. We cannot create one here because vector indexes
            # on Atlas are managed via the Atlas UI or Atlas Search API; we
            # only create a unique compound index on the natural keys.
            try:
                self._db.create_collection(self.collection_name)
            except Exception as e:
                logger.debug(f"Collection creation skipped (may already exist): {e}")
        # Helpful indexes for namespace and route filters.
        if self.namespace is not None:
            collection.create_index([(SR_NAMESPACE_FIELD, 1), (SR_ROUTE_FIELD, 1)])
        else:
            collection.create_index([(SR_ROUTE_FIELD, 1)])
        self._collection_initialized = True

    async def _async_init_collection(self) -> None:
        """Async counterpart of :meth:`_init_collection`."""
        if self._collection_initialized or self.aclient is None:
            return
        collection = self._async_collection
        # Make sure the collection exists.
        existing = await self._async_db.list_collection_names()
        if self.collection_name not in existing:
            try:
                await self._async_db.create_collection(self.collection_name)
            except Exception as e:
                logger.debug(f"Async collection creation skipped: {e}")
        # Helper indexes.
        if self.namespace is not None:
            await collection.create_index([(SR_NAMESPACE_FIELD, 1), (SR_ROUTE_FIELD, 1)])
        else:
            await collection.create_index([(SR_ROUTE_FIELD, 1)])
        self._collection_initialized = True

    # --------------------------------------------------------------- metrics

    def _vector_search_path(self) -> str:
        """Return the MongoDB ``numCandidates`` path hint — distance vs cosine
        does not affect the pipeline shape, but metric is echoed for callers.

        MongoDB Atlas vector search indexes only support ``cosine`` and
        ``euclidean`` distance metrics natively. Dot product and Manhattan
        need a custom index. We'll map dotproduct → cosine with a note.
        """
        if self.metric not in (Metric.COSINE, Metric.EUCLIDEAN):
            logger.warning(
                f"MongoDB Atlas vector search natively supports COSINE and "
                f"EUCLIDEAN; {self.metric.value} will be used as-is but may "
                f"require a custom vector index definition."
            )
        return self.metric.value

    # -------------------------------------------------------------- id gen

    def _document_id(self, route: str, utterance: Any) -> str:
        """Deterministic document id for a (route, utterance) pair."""
        key = (
            f"{self.namespace}:{route}:{utterance}"
            if self.namespace is not None
            else f"{route}:{utterance}"
        )
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def _document(
        self,
        doc_id: str,
        embedding: List[float],
        route: str,
        utterance: Any,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the MongoDB document for one record."""
        doc: Dict[str, Any] = {
            "_id": doc_id,
            SR_VECTOR_FIELD: embedding,
            SR_ROUTE_FIELD: route,
            SR_UTTERANCE_FIELD: utterance,
            SR_METADATA_FIELD: metadata,
        }
        if self.namespace is not None:
            doc[SR_NAMESPACE_FIELD] = self.namespace
        return doc

    # --------------------------------------------------------------- CRUD

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        batch_size: int = 200,
        **kwargs,
    ):
        """Upsert records into the MongoDB collection.

        Uses a deterministic ``_id`` so inserting the same (route, utterance)
        pair twice is a single document replacement.
        """
        if sparse_embeddings is not None:
            raise ValueError(
                "MongoDbIndex stores dense vectors only. Use a hybrid index "
                "(e.g. HybridLocalIndex) if sparse embeddings are needed."
            )
        metadata_list = check_records(embeddings, routes, utterances, metadata_list)
        if not embeddings:
            return
        self.dimensions = self.dimensions or len(embeddings[0])
        self._init_collection()

        operations = []
        for emb, route, utt, meta in zip(embeddings, routes, utterances, metadata_list):
            doc_id = self._document_id(route, utt)
            doc = self._document(doc_id, emb, route, utt, meta)
            operations.append(
                {
                    "replace_one": {
                        "filter": {"_id": doc_id},
                        "replacement": doc,
                        "upsert": True,
                    }
                }
            )
        # Batch bulk writes for efficiency.
        for i in range(0, len(operations), batch_size):
            batch = operations[i : i + batch_size]
            self._collection.bulk_write(batch, ordered=False)

    async def aadd(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        batch_size: int = 200,
        **kwargs,
    ):
        """Async upsert into the MongoDB collection."""
        if self.aclient is None:
            logger.debug("Async client unavailable; falling back to sync add.")
            return self.add(
                embeddings=embeddings,
                routes=routes,
                utterances=utterances,
                metadata_list=metadata_list,
                sparse_embeddings=sparse_embeddings,
                batch_size=batch_size,
            )
        if sparse_embeddings is not None:
            raise ValueError(
                "MongoDbIndex stores dense vectors only. Use a hybrid index "
                "(e.g. HybridLocalIndex) if sparse embeddings are needed."
            )
        metadata_list = check_records(embeddings, routes, utterances, metadata_list)
        if not embeddings:
            return
        self.dimensions = self.dimensions or len(embeddings[0])
        await self._async_init_collection()

        operations = []
        for emb, route, utt, meta in zip(embeddings, routes, utterances, metadata_list):
            doc_id = self._document_id(route, utt)
            doc = self._document(doc_id, emb, route, utt, meta)
            operations.append(
                {
                    "replace_one": {
                        "filter": {"_id": doc_id},
                        "replacement": doc,
                        "upsert": True,
                    }
                }
            )
        for i in range(0, len(operations), batch_size):
            batch = operations[i : i + batch_size]
            await self._async_collection.bulk_write(batch, ordered=False)

    def _base_filter(self) -> Dict[str, Any]:
        """Document filter scoped to the current namespace."""
        if self.namespace is None:
            return {}
        return {SR_NAMESPACE_FIELD: self.namespace}

    def _query_filter(
        self, route_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Build a MQL filter combining namespace and optional route names."""
        filter_doc = self._base_filter()
        if route_filter is not None:
            if len(route_filter) == 1:
                filter_doc[SR_ROUTE_FIELD] = route_filter[0]
            else:
                filter_doc[SR_ROUTE_FIELD] = {"$in": route_filter}
        return filter_doc

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Vector search via MongoDB ``$vectorSearch`` aggregation."""
        if sparse_vector is not None:
            raise ValueError(
                "MongoDbIndex does not support sparse vector queries."
            )
        vector_list = [float(x) for x in vector.flatten().tolist()]
        filter_doc = self._query_filter(route_filter)
        # Build the pipeline:
        #   $vectorSearch → $match (extra filter) → $project → $limit
        pipeline: List[Dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": SR_VECTOR_FIELD,
                    "queryVector": vector_list,
                    "numCandidates": max(top_k * 10, 100),
                    "limit": top_k,
                    **(
                        {"filter": filter_doc} if filter_doc else {}
                    ),
                }
            },
            # Expose the search score alongside the document.
            {"$project": {"score": {"$meta": "vectorSearchScore"}, SR_ROUTE_FIELD: 1}},
        ]
        results = list(self._collection.aggregate(pipeline))
        scores = [float(r["score"]) for r in results]
        route_names = [r[SR_ROUTE_FIELD] for r in results]
        return np.array(scores), route_names

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Async vector search."""
        if self.aclient is None:
            logger.debug("Async client unavailable; falling back to sync query.")
            return self.query(vector, top_k, route_filter, sparse_vector)
        if sparse_vector is not None:
            raise ValueError(
                "MongoDbIndex does not support sparse vector queries."
            )
        vector_list = [float(x) for x in vector.flatten().tolist()]
        filter_doc = self._query_filter(route_filter)
        pipeline: List[Dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": SR_VECTOR_FIELD,
                    "queryVector": vector_list,
                    "numCandidates": max(top_k * 10, 100),
                    "limit": top_k,
                    **(
                        {"filter": filter_doc} if filter_doc else {}
                    ),
                }
            },
            {"$project": {"score": {"$meta": "vectorSearchScore"}, SR_ROUTE_FIELD: 1}},
        ]
        scores: List[float] = []
        route_names: List[str] = []
        async for doc in self._async_collection.aggregate(pipeline):
            scores.append(float(doc["score"]))
            route_names.append(doc[SR_ROUTE_FIELD])
        return np.array(scores), route_names

    def delete(self, route_name: str) -> None:
        """Delete every document whose ``sr_route`` equals *route_name*."""
        filter_doc = self._base_filter()
        filter_doc[SR_ROUTE_FIELD] = route_name
        self._collection.delete_many(filter_doc)

    async def adelete(self, route_name: str) -> list[str]:
        """Async version of :meth:`delete`. Returns deleted document ids."""
        if self.aclient is None:
            logger.debug("Async client unavailable; falling back to sync delete.")
            self.delete(route_name)
            return []
        filter_doc = self._base_filter()
        filter_doc[SR_ROUTE_FIELD] = route_name
        cursor = self._async_collection.find(filter_doc, {"_id": 1})
        ids: list[str] = []
        async for doc in cursor:
            ids.append(str(doc["_id"]))
        result = await self._async_collection.delete_many(filter_doc)
        logger.debug(f"Deleted {result.deleted_count} documents for route '{route_name}'")
        return ids

    def delete_all(self) -> None:
        """Delete every document in the collection."""
        filter_doc = self._base_filter()
        self._collection.delete_many(filter_doc)

    async def adelete_all(self) -> None:
        """Async :meth:`delete_all`."""
        if self.aclient is None:
            return self.delete_all()
        filter_doc = self._base_filter()
        await self._async_collection.delete_many(filter_doc)

    def delete_index(self) -> None:
        """Drop the entire collection."""
        self._collection.drop()
        self._collection_initialized = False

    async def adelete_index(self) -> None:
        """Async :meth:`delete_index`."""
        if self.aclient is None:
            return self.delete_index()
        await self._async_collection.drop()
        self._collection_initialized = False

    def describe(self) -> IndexConfig:
        """Return an :class:`IndexConfig` describing the index state."""
        try:
            cursor = self._collection.aggregate(
                [
                    {"$match": self._base_filter()},
                    {"$count": "n"},
                ]
            )
            count_doc = next(cursor, {})
            total = count_doc.get("n", 0)
        except Exception:
            total = 0
        return IndexConfig(
            type=self.type,
            dimensions=self.dimensions or 0,
            vectors=total,
        )

    def is_ready(self) -> bool:
        """Check that the collection exists and the client is reachable."""
        try:
            self.client.admin.command("ping")
            return self.collection_name in self._db.list_collection_names()
        except Exception as e:
            logger.warning(f"MongoDB readiness check failed: {e}")
            return False

    async def ais_ready(self) -> bool:
        """Async :meth:`is_ready`."""
        if self.aclient is None:
            return False
        try:
            await self.aclient.admin.command("ping")
            return self.collection_name in await self._async_db.list_collection_names()
        except Exception as e:
            logger.warning(f"Async MongoDB readiness check failed: {e}")
            return False

    # ------------------------------------------------------------- scan all

    def _get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Return every document's id and optionally its flat metadata.

        :param prefix: Ignored; filtering is done via namespace + optionally
            route name.
        :param include_metadata: If True, return the document metadata with
            ``sr_route`` and ``sr_utterance`` inlined as flat keys.
        """
        if self.collection_name not in self._db.list_collection_names():
            return [], []
        filter_doc = self._base_filter()
        cursor = self._collection.find(filter_doc)
        ids: list[str] = []
        metadata: list[dict] = []
        for doc in cursor:
            ids.append(str(doc["_id"]))
            if include_metadata:
                meta = dict(doc.get(SR_METADATA_FIELD, {}) or {})
                meta[SR_ROUTE] = doc.get(SR_ROUTE_FIELD, "")
                meta[SR_UTTERANCE] = doc.get(SR_UTTERANCE_FIELD, "")
                metadata.append(meta)
        return ids, metadata

    async def _async_get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Async :meth:`_get_all`."""
        if self.aclient is None:
            return self._get_all(prefix, include_metadata)
        existing = await self._async_db.list_collection_names()
        if self.collection_name not in existing:
            return [], []
        filter_doc = self._base_filter()
        ids: list[str] = []
        metadata: list[dict] = []
        cursor = self._async_collection.find(filter_doc)
        async for doc in cursor:
            ids.append(str(doc["_id"]))
            if include_metadata:
                meta = dict(doc.get(SR_METADATA_FIELD, {}) or {})
                meta[SR_ROUTE] = doc.get(SR_ROUTE_FIELD, "")
                meta[SR_UTTERANCE] = doc.get(SR_UTTERANCE_FIELD, "")
                metadata.append(meta)
        return ids, metadata

    # ---------------------------------------------------------------- len

    def __len__(self) -> int:
        """Total number of documents in the collection (optionally filtered by namespace)."""
        try:
            return self._collection.count_documents(self._base_filter())
        except Exception:
            return 0

    async def alen(self) -> int:
        """Async :meth:`__len__`."""
        if self.aclient is None:
            return len(self)
        try:
            return await self._async_collection.count_documents(self._base_filter())
        except Exception:
            return 0
