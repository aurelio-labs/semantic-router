"""MongoDB Atlas Vector Search index backend for semantic-router.

Uses `pymongo` (sync) and `motor` (async) clients against a MongoDB Atlas
deployment with a pre-configured Atlas Vector Search index on the embedding
field. Users create the search index once in Atlas or via the CLI — the driver
only upserts documents and issues aggregation pipelines.
"""

import hashlib
import uuid
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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
DEFAULT_SEARCH_INDEX_NAME = "vector_index"
DEFAULT_NUM_CANDIDATES = 100

# Maps Metric enum values to MongoDB Atlas Vector Search similarity strings.
METRIC_TO_SIMILARITY = {
    Metric.COSINE: "cosine",
    Metric.DOTPRODUCT: "dotProduct",
    Metric.EUCLIDEAN: "euclidean",
    # Manhattan is not supported by Atlas Vector Search; fall back to euclidean.
    Metric.MANHATTAN: "euclidean",
}


def _doc_id(route: str, utterance: str, namespace: Optional[str] = None) -> str:
    """Deterministic document id for a (namespace, route, utterance) triple.

    Matches the upsert semantics of other backends: adding the same pair twice
    replaces the prior document instead of duplicating it.
    """
    route_clean = route.strip().replace(" ", "-")
    if namespace:
        key = f"{namespace}:{route_clean}:{utterance}"
    else:
        key = f"{route_clean}:{utterance}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"{route_clean}#{digest}"


class MongoDocument(BaseModel):
    """Shape of a single document stored in the Atlas collection."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    _id: str
    vector: List[float]
    metadata: Dict[str, Any]


class MongoDBAtlasIndex(BaseIndex):
    """Atlas Vector Search backed index.

    Requires ``pymongo`` for sync and ``motor`` for async operations. The
    embedding collection and its Atlas search index must exist before the
    first query — ``is_ready`` checks for both.
    """

    type: str = "mongodb"

    # ── Connection fields ──
    uri: Optional[str] = Field(
        default=None,
        description="MongoDB connection URI, e.g. "
        "'mongodb+srv://user:pass@cluster.mongodb.net/'. When omitted, the "
        "default ``mongodb://localhost:27017`` is used.",
    )
    database: str = Field(
        default="semantic_router",
        description="Database name.",
    )
    collection_name: str = Field(
        default=DEFAULT_COLLECTION_NAME,
        description="Collection name.",
    )
    search_index_name: str = Field(
        default=DEFAULT_SEARCH_INDEX_NAME,
        description="Name of the pre-configured Atlas Vector Search index "
        "on the collection.",
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="Embedding dimensions. Populated lazily from the first "
        "``add`` call when omitted.",
    )
    metric: Metric = Field(
        default=Metric.COSINE,
        description="Distance metric used at index creation time.",
    )
    num_candidates: int = Field(
        default=DEFAULT_NUM_CANDIDATES,
        description="Number of nearest-neighbours to consider before the "
        "``limit`` cutoff. Higher values recall more candidates at the cost "
        "of latency.",
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Optional tenant namespace. Injects a ``_namespace`` "
        "field into every document and scopes all queries/deletes to it.",
    )
    client: Any = Field(default=None, exclude=True)
    aclient: Any = Field(default=None, exclude=True)
    _initialized: bool = PrivateAttr(default=False)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._connect()

    def _connect(self) -> None:
        """Build the sync and async Mongo clients.

        Uses ``mongodb://localhost:27017`` when no ``uri`` is supplied. The
        async client shares the same connection pool settings as its sync
        counterpart so they behave consistently under typical hostnames.
        """
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            from pymongo import MongoClient
        except ImportError as exc:
            raise ImportError(
                "Please install 'pymongo' and 'motor' to use MongoDBAtlasIndex. "
                "You can install them with: "
                "`pip install 'semantic-router[mongodb]'`"
            ) from exc

        uri = self.uri or "mongodb://localhost:27017"
        self.client: MongoClient = MongoClient(uri)
        self.aclient: AsyncIOMotorClient = AsyncIOMotorClient(uri)

    # ── Upsert ───────────────────────────────────────────────────────────

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        **kwargs,
    ) -> None:
        if sparse_embeddings is not None:
            raise ValueError(
                "MongoDBAtlasIndex does not support sparse embeddings. "
                "Use a hybrid index backed by LocalIndex for sparse vectors."
            )
        self._init_index()
        self._upsert_many(embeddings, routes, utterances, metadata_list)

    async def aadd(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        **kwargs,
    ) -> None:
        if sparse_embeddings is not None:
            raise ValueError(
                "MongoDBAtlasIndex does not support sparse embeddings. "
                "Use a hybrid index backed by LocalIndex for sparse vectors."
            )
        await self._init_async_index()
        await self._aupsert_many(embeddings, routes, utterances, metadata_list)

    def _build_documents(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Build Mongo documents from user inputs with reserved ``sr_`` keys."""
        metadata_list = check_records(embeddings, routes, utterances, metadata_list)
        docs: List[Dict[str, Any]] = []
        for vector, route, utterance, meta in zip(
            embeddings, routes, utterances, metadata_list
        ):
            record_meta = {**meta, SR_ROUTE: route, SR_UTTERANCE: str(utterance)}
            if self.namespace is not None:
                record_meta["_namespace"] = self.namespace
            doc = {
                "_id": _doc_id(route, str(utterance), self.namespace),
                "vector": [float(v) for v in vector],
                "metadata": record_meta,
            }
            docs.append(doc)
        return docs

    def _upsert_many(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]],
    ) -> None:
        from pymongo import UpdateOne

        db = self.client[self.database]
        coll = db[self.collection_name]
        docs = self._build_documents(embeddings, routes, utterances, metadata_list)
        if not docs:
            return

        ops = [UpdateOne({"_id": d["_id"]}, {"$set": d}, upsert=True) for d in docs]
        result = coll.bulk_write(ops, ordered=False)
        logger.info(
            f"MongoDB upserted {result.upserted_count} new documents; "
            f"{result.modified_count} updated."
        )

    async def _aupsert_many(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]],
    ) -> None:
        from pymongo import UpdateOne

        db = self.aclient[self.database]
        coll = db[self.collection_name]
        docs = self._build_documents(embeddings, routes, utterances, metadata_list)
        if not docs:
            return

        ops = [UpdateOne({"_id": d["_id"]}, {"$set": d}, upsert=True) for d in docs]
        result = await coll.bulk_write(ops, ordered=False)
        logger.info(
            f"MongoDB async upserted {result.upserted_count} new documents; "
            f"{result.modified_count} updated."
        )

    # ── Delete ───────────────────────────────────────────────────────────

    def delete(self, route_name: str) -> None:
        self._init_index()
        coll = self.client[self.database][self.collection_name]
        filt: Dict[str, Any] = {"metadata." + SR_ROUTE: route_name}
        if self.namespace is not None:
            filt["metadata._namespace"] = self.namespace
        result = coll.delete_many(filt)
        logger.info(f"MongoDB deleted {result.deleted_count} documents for route '{route_name}'.")

    async def adelete(self, route_name: str) -> list[str]:
        await self._init_async_index()
        coll = self.aclient[self.database][self.collection_name]
        filt: Dict[str, Any] = {"metadata." + SR_ROUTE: route_name}
        if self.namespace is not None:
            filt["metadata._namespace"] = self.namespace
        # First collect the ids so we can return them, then delete.
        ids = []
        async for doc in coll.find(filt, {"_id": 1}):
            ids.append(str(doc["_id"]))
        if ids:
            await coll.delete_many(filt)
        return ids

    # ── Describe / readiness ─────────────────────────────────────────────

    def describe(self) -> IndexConfig:
        if self.dimensions:
            dims = self.dimensions
        else:
            coll = self.client[self.database][self.collection_name]
            sample = coll.find_one({"vector": {"$exists": True}}, {"vector": 1})
            dims = len(sample["vector"]) if sample else 0
        coll = self.client[self.database][self.collection_name]
        return IndexConfig(
            type=self.type,
            dimensions=dims,
            vectors=coll.count_documents(
                {"vector": {"$exists": True}, **(
                    {"metadata._namespace": self.namespace} if self.namespace else {}
                )}
            ),
        )

    def is_ready(self) -> bool:
        try:
            coll = self.client[self.database][self.collection_name]
            # A collection exists once it contains documents or an index.
            coll_name = coll.name
            self.client[self.database].validate_collection(coll_name)
            return True
        except Exception:
            return False

    async def ais_ready(self) -> bool:
        try:
            coll = self.aclient[self.database][self.collection_name]
            await coll.find_one({})
            return True
        except Exception:
            return False

    # ── Vector search ────────────────────────────────────────────────────

    def _vector_search_pipeline(
        self,
        vector: List[float],
        top_k: int,
        route_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build the ``$vectorSearch`` aggregation pipeline.

        Atlas Vector Search uses a single stage; we append ``$project`` to
        surface the score alongside the document metadata.
        """
        vs_stage: Dict[str, Any] = {
            "index": self.search_index_name,
            "path": "vector",
            "queryVector": [float(v) for v in vector],
            "numCandidates": self.num_candidates,
            "limit": top_k,
        }
        # Route filter is expressed as a pre-filter on reserved keys.
        pre_filter: Dict[str, Any] = {}
        if route_filter:
            pre_filter["metadata." + SR_ROUTE] = {"$in": list(route_filter)}
        if self.namespace is not None:
            pre_filter["metadata._namespace"] = self.namespace
        if pre_filter:
            vs_stage["filter"] = pre_filter

        pipeline: List[Dict[str, Any]] = [
            {"$vectorSearch": vs_stage},
            {
                "$project": {
                    "vector": 0,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return pipeline

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: Optional[dict | SparseEmbedding] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if sparse_vector is not None:
            raise ValueError(
                "MongoDBAtlasIndex does not support sparse vector queries."
            )
        self._init_index()
        coll = self.client[self.database][self.collection_name]
        pipeline = self._vector_search_pipeline(
            vector.tolist(), top_k, route_filter
        )
        scores: List[float] = []
        routes: List[str] = []
        for doc in coll.aggregate(pipeline):
            scores.append(float(doc["score"]))
            routes.append(doc.get("metadata", {}).get(SR_ROUTE, ""))
        return np.array(scores, dtype=np.float32), routes

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: Optional[dict | SparseEmbedding] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if sparse_vector is not None:
            raise ValueError(
                "MongoDBAtlasIndex does not support sparse vector queries."
            )
        await self._init_async_index()
        coll = self.aclient[self.database][self.collection_name]
        pipeline = self._vector_search_pipeline(
            vector.tolist(), top_k, route_filter
        )
        scores: List[float] = []
        routes: List[str] = []
        async for doc in coll.aggregate(pipeline):
            scores.append(float(doc["score"]))
            routes.append(doc.get("metadata", {}).get(SR_ROUTE, ""))
        return np.array(scores, dtype=np.float32), routes

    # ── Read all ─────────────────────────────────────────────────────────

    def _get_all(
        self,
        prefix: Optional[str] = None,
        include_metadata: bool = False,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        coll = self.client[self.database][self.collection_name]
        filt: Dict[str, Any] = {}
        if self.namespace is not None:
            filt["metadata._namespace"] = self.namespace
        projection: Dict[str, Any] = {"_id": 1, "metadata": 1}
        ids: List[str] = []
        records: List[Dict[str, Any]] = []
        for doc in coll.find(filt, projection):
            ids.append(str(doc["_id"]))
            if include_metadata:
                records.append(doc.get("metadata", {}))
        return ids, records

    async def _async_get_all(
        self,
        prefix: Optional[str] = None,
        include_metadata: bool = False,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        coll = self.aclient[self.database][self.collection_name]
        filt: Dict[str, Any] = {}
        if self.namespace is not None:
            filt["metadata._namespace"] = self.namespace
        projection: Dict[str, Any] = {"_id": 1, "metadata": 1}
        ids: List[str] = []
        records: List[Dict[str, Any]] = []
        async for doc in coll.find(filt, projection):
            ids.append(str(doc["_id"]))
            if include_metadata:
                records.append(doc.get("metadata", {}))
        return ids, records

    # ── Lifecycle helpers ────────────────────────────────────────────────

    def _init_index(self, force_create: bool = False) -> Union[Any, None]:
        """No eager collection creation — Atlas collections materialise on the
        first insert. We only cache dimensions here."""
        if self._initialized:
            return self.client
        self._initialized = True
        if not self.dimensions:
            coll = self.client[self.database][self.collection_name]
            sample = coll.find_one({"vector": {"$exists": True}}, {"vector": 1})
            if sample:
                self.dimensions = len(sample["vector"])
        return self.client

    async def _init_async_index(
        self, force_create: bool = False
    ) -> Union[Any, None]:
        if self._initialized:
            return self.aclient
        self._initialized = True
        if not self.dimensions:
            coll = self.aclient[self.database][self.collection_name]
            sample = await coll.find_one(
                {"vector": {"$exists": True}}, {"vector": 1}
            )
            if sample:
                self.dimensions = len(sample["vector"])
        return self.aclient

    def delete_index(self) -> None:
        """Drop the collection entirely (and any search index attached)."""
        db = self.client[self.database]
        if self.collection_name in db.list_collection_names():
            db.drop_collection(self.collection_name)
        self._initialized = False

    async def adelete_index(self) -> None:
        db = self.aclient[self.database]
        if self.collection_name in await db.list_collection_names():
            await db.drop_collection(self.collection_name)
        self._initialized = False

    def delete_all(self) -> None:
        """Delete every document without dropping the collection."""
        coll = self.client[self.database][self.collection_name]
        filt: Dict[str, Any] = {}
        if self.namespace is not None:
            filt["metadata._namespace"] = self.namespace
        coll.delete_many(filt)

    def __len__(self) -> int:
        if not self._initialized:
            return 0
        coll = self.client[self.database][self.collection_name]
        filt: Dict[str, Any] = {"vector": {"$exists": True}}
        if self.namespace is not None:
            filt["metadata._namespace"] = self.namespace
        return coll.count_documents(filt)
