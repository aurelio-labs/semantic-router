"""MongoDB Atlas Vector Search index.

Records are MongoDB documents, searched with the ``$vectorSearch`` aggregation stage.
Works with MongoDB Atlas, and with the ``mongodb/mongodb-atlas-local`` image (MongoDB
with search) that the integration tests run against.
"""

import asyncio
import os
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, overload

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

ROUTE_FIELD = "sr_route"
UTTERANCE_FIELD = "sr_utterance"
METADATA_FIELD = "metadata"
NAMESPACE_FIELD = "namespace"
VECTOR_FIELD = "embedding"

# Atlas vector search similarity per metric; manhattan has no Atlas equivalent
_SIMILARITY = {
    Metric.COSINE: "cosine",
    Metric.DOTPRODUCT: "dotProduct",
    Metric.EUCLIDEAN: "euclidean",
}


class MongoDbIndex(BaseIndex):
    """MongoDB Atlas Vector Search index.

    The first ``add`` creates the collection and its vector search index, with
    ``sr_route``, ``namespace`` and ``_id`` as filter fields, and waits until the index
    is queryable. Atlas indexes new documents asynchronously, so ``add`` also waits
    (up to ``sync_timeout`` seconds) until its records are searchable, so a query
    straight after an add sees them.

    Scores are on the same scale as the other indexes: Atlas reports cosine and dot
    product scores as ``(1 + similarity) / 2``, which this index maps back to the raw
    similarity. Euclidean scores are Atlas's ``1 / (1 + distance)``.

    One pymongo client serves everything. The async methods run it in a worker
    thread, so the index works from any event loop and keeps a single connection pool.

    :param index_name: The collection that holds the records.
    :param database_name: The database the collection lives in.
    :param vector_index_name: The name of the collection's vector search index.
    :param connection_uri: The MongoDB connection string. Read from
        ``MONGO_CONNECTION_URI`` when not given, and never written to a file.
    :param namespace: Scopes every record, query and delete to one tenant.
    :param sync_timeout: Seconds ``add`` waits for new records to become searchable;
        0 returns as soon as they are written.
    """

    type: str = "mongodb"
    index_name: str = "semantic-router-index"
    database_name: str = "semantic_router"
    vector_index_name: str = "vector_index"
    connection_uri: Optional[str] = Field(default=None, exclude=True, repr=False)
    metric: Metric = Metric.COSINE
    namespace: Optional[str] = None
    sync_timeout: float = 30.0

    _client: Any = PrivateAttr(default=None)
    _ready: bool = PrivateAttr(default=False)

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.metric not in _SIMILARITY:
            raise ValueError(
                f"MongoDB vector search has no {self.metric.value} similarity; use "
                f"one of {', '.join(m.value for m in _SIMILARITY)}"
            )
        self.connection_uri = self.connection_uri or os.getenv("MONGO_CONNECTION_URI")
        if not self.connection_uri:
            raise ValueError("pass connection_uri or set MONGO_CONNECTION_URI")
        try:
            from pymongo import MongoClient
        except ImportError as e:
            raise ImportError(
                "Please install pymongo to use MongoDbIndex. You can install it with: "
                "`pip install 'semantic-router[mongodb]'`"
            ) from e
        self._client = MongoClient(self.connection_uri)

    @property
    def _collection(self) -> Any:
        return self._client[self.database_name][self.index_name]

    # ---------------------------------------------------------------- setup

    def _index_model(self, dimensions: int) -> Any:
        from pymongo.operations import SearchIndexModel

        fields = [
            {
                "type": "vector",
                "path": VECTOR_FIELD,
                "numDimensions": dimensions,
                "similarity": _SIMILARITY[self.metric],
            },
            {"type": "filter", "path": ROUTE_FIELD},
            {"type": "filter", "path": NAMESPACE_FIELD},
            # lets add() wait for its own records to become searchable
            {"type": "filter", "path": "_id"},
        ]
        return SearchIndexModel(
            definition={"fields": fields},
            name=self.vector_index_name,
            type="vectorSearch",
        )

    def _step(self, found: list, dimensions: int) -> str:
        """What to do next given the listed search indexes: "create", "wait" or
        "ready". A dropped collection's index lingers as DOES_NOT_EXIST or DELETING for
        a moment, and a new one can only be made once it's gone."""
        if not found:
            return "create"
        if found[0].get("status") in ("DOES_NOT_EXIST", "DELETING"):
            return "wait"
        fields = (found[0].get("latestDefinition") or {}).get("fields", [])
        dims = next(
            (f.get("numDimensions") for f in fields if f.get("type") == "vector"), None
        )
        if dims is not None and dims != dimensions:
            raise ValueError(
                f"the vector index {self.vector_index_name!r} holds {dims}-dimension "
                f"vectors, but these have {dimensions}"
            )
        return "ready" if found[0].get("queryable") else "wait"

    def _ensure_index(self, dimensions: int) -> None:
        """Create the collection and its vector search index if missing, then wait
        until the index is queryable."""
        if self._ready:
            return
        db = self._client[self.database_name]
        if self.index_name not in db.list_collection_names():
            db.create_collection(self.index_name)
        deadline = time.monotonic() + max(self.sync_timeout, 60.0)
        while True:
            found = list(self._collection.list_search_indexes(self.vector_index_name))
            step = self._step(found, dimensions)
            if step == "ready":
                break
            if step == "create":
                self._collection.create_search_index(self._index_model(dimensions))
            elif time.monotonic() > deadline:
                raise TimeoutError(
                    f"the vector index {self.vector_index_name!r} is not queryable yet"
                )
            time.sleep(0.2)
        self.dimensions = dimensions
        self._ready = True

    # ---------------------------------------------------------------- write

    def _document_id(self, route: str, utterance: Any) -> str:
        key = f"{route}:{utterance}"
        if self.namespace is not None:
            key = f"{self.namespace}:{key}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

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
        """Upsert one document per (route, utterance), then wait until they are
        searchable.

        :param embeddings: The dense vectors.
        :param routes: The route of each record.
        :param utterances: The utterance of each record.
        :param metadata_list: Each record's metadata, including the reserved ``sr_``
            route-level keys.
        :param sparse_embeddings: Not supported: MongoDbIndex stores dense vectors only.
        :param batch_size: Documents per bulk write.
        """
        from pymongo import ReplaceOne

        if sparse_embeddings is not None:
            raise ValueError("MongoDbIndex stores dense vectors only.")
        metadata_list = check_records(embeddings, routes, utterances, metadata_list)
        if not embeddings:
            return
        self._ensure_index(len(embeddings[0]))
        ops = []
        for emb, route, utt, meta in zip(embeddings, routes, utterances, metadata_list):
            doc: Dict[str, Any] = {
                "_id": self._document_id(route, utt),
                VECTOR_FIELD: [float(x) for x in emb],
                ROUTE_FIELD: route,
                UTTERANCE_FIELD: utt,
                METADATA_FIELD: meta,
            }
            if self.namespace is not None:
                doc[NAMESPACE_FIELD] = self.namespace
            ops.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))
        for i in range(0, len(ops), batch_size):
            self._collection.bulk_write(ops[i : i + batch_size], ordered=False)
        # the last record written, found by a search once the index has caught up
        last = self._document_id(routes[-1], utterances[-1])
        probe = self._pipeline(embeddings[-1], 1, {"_id": last})
        deadline = time.monotonic() + self.sync_timeout
        while not list(self._collection.aggregate(probe)):
            if time.monotonic() > deadline:
                logger.warning(
                    "new records are not searchable yet; queries may miss them"
                )
                return
            time.sleep(0.1)

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
        """Async :meth:`add`."""
        await asyncio.to_thread(
            self.add,
            embeddings=embeddings,
            routes=routes,
            utterances=utterances,
            metadata_list=metadata_list,
            sparse_embeddings=sparse_embeddings,
            batch_size=batch_size,
        )

    # ---------------------------------------------------------------- query

    def _scope(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """A filter scoped to this namespace, with any extra conditions."""
        scope: Dict[str, Any] = {}
        if self.namespace is not None:
            scope[NAMESPACE_FIELD] = self.namespace
        return {**scope, **(extra or {})}

    def _pipeline(
        self, vector: Any, top_k: int, extra: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        search: Dict[str, Any] = {
            "index": self.vector_index_name,
            "path": VECTOR_FIELD,
            "queryVector": [float(x) for x in np.asarray(vector).flatten()],
            "numCandidates": max(top_k * 10, 100),
            "limit": top_k,
        }
        scope = self._scope(extra)
        if scope:
            search["filter"] = scope
        return [
            {"$vectorSearch": search},
            {"$project": {"score": {"$meta": "vectorSearchScore"}, ROUTE_FIELD: 1}},
        ]

    @overload
    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: Literal[False] = False,
    ) -> Tuple[np.ndarray, List[str]]: ...

    @overload
    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: Literal[True] = ...,
    ) -> Tuple[np.ndarray, List[str], List[str]]: ...

    @overload
    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: bool = ...,
    ) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]: ...
    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: bool = False,
    ) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]:
        """Search the index with ``$vectorSearch``.

        :param vector: The query vector.
        :param top_k: The number of results to return.
        :param route_filter: Only search these routes.
        :param sparse_vector: Not used: MongoDbIndex is dense only.
        :param return_utterances: If True, also return matched utterance texts.
            MongoDB Atlas records do not store the original utterance text, so
            this is accepted for API compatibility but does not add utterances.
        :return: The scores and route names of the top_k results.
        """
        extra = {ROUTE_FIELD: {"$in": route_filter}} if route_filter is not None else {}
        docs = list(self._collection.aggregate(self._pipeline(vector, top_k, extra)))
        scores = np.array([float(d["score"]) for d in docs])
        if self.metric is not Metric.EUCLIDEAN:
            # atlas reports (1 + similarity) / 2; the other indexes report the similarity
            scores = scores * 2 - 1
        return scores, [d[ROUTE_FIELD] for d in docs]

    @overload
    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: Literal[False] = False,
    ) -> Tuple[np.ndarray, List[str]]: ...

    @overload
    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: Literal[True] = ...,
    ) -> Tuple[np.ndarray, List[str], List[str]]: ...

    @overload
    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: bool = ...,
    ) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]: ...
    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: bool = False,
    ) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]:
        """Async :meth:`query`."""
        return await asyncio.to_thread(
            self.query,
            vector=vector,
            top_k=top_k,
            route_filter=route_filter,
            sparse_vector=sparse_vector,
            return_utterances=return_utterances,
        )

    # --------------------------------------------------------------- delete

    def delete(self, route_name: str) -> list[str]:
        """Delete every record of one route. Returns the deleted record ids."""
        scope = self._scope({ROUTE_FIELD: route_name})
        ids = [str(d["_id"]) for d in self._collection.find(scope, {"_id": 1})]
        self._collection.delete_many(scope)
        return ids

    async def adelete(self, route_name: str) -> list[str]:
        """Async :meth:`delete`."""
        return await asyncio.to_thread(self.delete, route_name)

    def delete_all(self) -> None:
        """Delete every record in this namespace, keeping the collection."""
        self._collection.delete_many(self._scope())

    def delete_index(self) -> None:
        """Drop the collection, and its vector search index with it."""
        self._collection.drop()
        self._ready = False

    async def adelete_index(self) -> None:
        """Async :meth:`delete_index`."""
        await asyncio.to_thread(self.delete_index)

    # ----------------------------------------------------------------- read

    def _get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Every record's id and, optionally, its metadata with ``sr_route`` and
        ``sr_utterance`` inlined. ``prefix`` is not used."""
        projection = None if include_metadata else {"_id": 1}
        docs = list(self._collection.find(self._scope(), projection))
        records = []
        if include_metadata:
            for d in docs:
                meta = dict(d.get(METADATA_FIELD) or {})
                meta[SR_ROUTE] = d.get(ROUTE_FIELD, "")
                meta[SR_UTTERANCE] = d.get(UTTERANCE_FIELD, "")
                records.append(meta)
        return [str(d["_id"]) for d in docs], records

    async def _async_get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Async :meth:`_get_all`."""
        return await asyncio.to_thread(self._get_all, prefix, include_metadata)

    def describe(self) -> IndexConfig:
        """The index type, its dimensions and how many records it holds."""
        return IndexConfig(
            type=self.type, dimensions=self.dimensions or 0, vectors=len(self)
        )

    def is_ready(self) -> bool:
        """Whether the vector search index exists and is queryable."""
        found = list(self._collection.list_search_indexes(self.vector_index_name))
        return bool(found and found[0].get("queryable"))

    async def ais_ready(self) -> bool:
        """Async :meth:`is_ready`."""
        return await asyncio.to_thread(self.is_ready)

    def __len__(self) -> int:
        return self._collection.count_documents(self._scope())

    async def alen(self) -> int:
        return await asyncio.to_thread(len, self)
