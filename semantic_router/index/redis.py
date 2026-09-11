import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import Field, PrivateAttr

from semantic_router.index.base import BaseIndex, IndexConfig
from semantic_router.schema import ConfigParameter, Metric, SparseEmbedding
from semantic_router.utils.logger import logger

DEFAULT_INDEX_NAME = "semantic-router-index"
DEFAULT_UPLOAD_BATCH_SIZE = 100
PAGE_SIZE = 1000
VECTOR_TYPE = "FLOAT32"

# Hash field names used for every record.
SR_ROUTE_FIELD = "sr_route"
SR_RECORD_FIELD = "sr_record"
SR_NAMESPACE_FIELD = "namespace"
VECTOR_FIELD = "vector"

# Config values are stored as plain Redis hashes, independent of the search index.
CONFIG_KEY_PREFIX = "sr_config"

# RediSearch only supports these three distance metrics for VECTOR fields.
# See: https://redis.io/docs/latest/develop/ai/search-and-query/vectors/#distance-metrics
_METRIC_MAP = {
    Metric.COSINE: "COSINE",
    Metric.EUCLIDEAN: "L2",
    Metric.DOTPRODUCT: "IP",
}

# Characters that RediSearch's query parser treats as special and that must be
# backslash-escaped when they appear inside a TAG filter value (e.g. a route
# name containing a space or hyphen).
_TAG_ESCAPE_RE = re.compile(r"([,.<>{}\[\]\"':;!@#$%^&*()\-+=~\s\\|])")


def _escape_tag(value: str) -> str:
    """Escape a value for safe use inside a RediSearch TAG filter.

    :param value: The raw value (e.g. a route name) to escape.
    :type value: str
    :return: The escaped value, safe to embed in a `@field:{...}` filter.
    :rtype: str
    """
    return _TAG_ESCAPE_RE.sub(r"\\\1", str(value))


def _import_index_definition():
    """Import IndexDefinition/IndexType, tolerating the module rename between
    redis-py major versions (snake_case `index_definition` from 6.0 on;
    camelCase `indexDefinition` before that).

    :return: The (IndexDefinition, IndexType) classes.
    """
    try:
        from redis.commands.search.index_definition import IndexDefinition, IndexType
    except ImportError:
        from redis.commands.search.indexDefinition import (  # type: ignore[no-redef]
            IndexDefinition,
            IndexType,
        )
    return IndexDefinition, IndexType


class RedisIndex(BaseIndex):
    """Redis implementation of Index, backed by RediSearch vector search.

    Requires a Redis server with the RediSearch module (e.g. Redis Stack, or
    Redis 8+, which bundles it by default). Plain open-source Redis without
    the module cannot be used, since it has no vector index type.
    """

    index_name: str = Field(
        default=DEFAULT_INDEX_NAME,
        description="Name of the RediSearch index and prefix for the Redis keys "
        f"it indexes. Default: '{DEFAULT_INDEX_NAME}'",
    )
    connection_string: Optional[str] = Field(
        default=None,
        description="A redis:// or rediss:// connection URL. Overrides "
        "host/port/db/username/password when set. Falls back to the "
        "REDIS_URL environment variable, then to the individual host/port "
        "fields.",
    )
    host: Optional[str] = Field(
        default=None,
        description="Redis host. Defaults to the REDIS_HOST env var, or 'localhost'.",
    )
    port: Optional[int] = Field(
        default=None,
        description="Redis port. Defaults to the REDIS_PORT env var, or 6379.",
    )
    db: int = Field(
        default=0,
        description="Redis logical database number.",
    )
    username: Optional[str] = Field(
        default=None,
        description="Redis username, defaults to the REDIS_USERNAME env var.",
    )
    password: Optional[str] = Field(
        default=None,
        description="Redis password, defaults to the REDIS_PASSWORD env var.",
    )
    ssl: bool = Field(
        default=False,
        description="Whether to connect over TLS.",
    )
    dimensions: Union[int, None] = Field(
        default=None,
        description="Embedding dimensions. Defaults to the embedding length of "
        "the configured encoder.",
    )
    metric: Metric = Field(
        default=Metric.COSINE,
        description="Distance metric to use for similarity search. Redis "
        "supports cosine, dotproduct (inner product) and euclidean (L2).",
    )
    algorithm: str = Field(
        default="FLAT",
        description="RediSearch vector algorithm: 'FLAT' (exact) or 'HNSW' "
        "(approximate, scales better to large indexes).",
    )
    algorithm_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra schema attributes for the vector field, e.g. "
        "{'M': 40, 'EF_CONSTRUCTION': 250} when algorithm='HNSW'.",
    )
    namespace: Optional[str] = Field(
        default=None,
        description=(
            "Optional namespace string (e.g. an organization ID) used to scope "
            "all index operations to a single tenant within a shared index. When "
            "set, a `namespace` TAG field is stored on every record and all "
            "queries/scans/deletes are automatically filtered to this namespace. "
            "Record IDs are also namespaced via `uuid5('{namespace}:{route}:{utterance}')`, "
            "preventing cross-tenant ID collisions."
        ),
    )
    client: Any = Field(default=None, exclude=True)
    aclient: Any = Field(default=None, exclude=True)
    _index_initialized: bool = PrivateAttr(default=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "redis"
        if self.connection_string is None:
            self.connection_string = os.getenv("REDIS_URL")
        if self.host is None:
            self.host = os.getenv("REDIS_HOST", "localhost")
        if self.port is None:
            self.port = int(os.getenv("REDIS_PORT", "6379"))
        if self.password is None:
            self.password = os.getenv("REDIS_PASSWORD")
        if self.username is None:
            self.username = os.getenv("REDIS_USERNAME")
        self.client, self.aclient = self._initialize_clients()
        # Unlike the RediSearch schema (created lazily, on first `add`), nothing
        # about reading requires local setup -- `self.index` is BaseIndex's
        # "have I got a handle on the index" marker, and a fresh RedisIndex
        # object always has one, even onto an index some other process created.
        # Setting it eagerly (matching PostgresIndex) means a read-only handle
        # onto an already-populated remote index -- e.g. `auto_sync="remote"`
        # against a fresh process -- doesn't look uninitialized.
        self.index = self

    def _initialize_clients(self):
        """Initialize the clients for the Redis index.

        :return: A tuple of the sync and async clients.
        :rtype: Tuple[redis.Redis, redis.asyncio.Redis]
        """
        try:
            import redis
            from redis import asyncio as redis_asyncio
        except ImportError as e:
            raise ImportError(
                "Please install 'redis' to use RedisIndex. "
                "You can install it with: `pip install 'semantic-router[redis]'`"
            ) from e

        if self.connection_string:
            sync_client = redis.Redis.from_url(self.connection_string)
            async_client = redis_asyncio.Redis.from_url(self.connection_string)
        else:
            conn_kwargs: Dict[str, Any] = dict(
                host=self.host,
                port=self.port,
                db=self.db,
                username=self.username,
                password=self.password,
                ssl=self.ssl,
            )
            sync_client = redis.Redis(**conn_kwargs)
            async_client = redis_asyncio.Redis(**conn_kwargs)
        return sync_client, async_client

    # ___________________________ SCHEMA / INDEX LIFECYCLE ___________________________

    def _key_prefix(self) -> str:
        """The Redis key prefix used for every record hash, and the RediSearch
        index's PREFIX filter.

        :return: The key prefix.
        :rtype: str
        """
        return f"{self.index_name}:"

    def _redis_metric(self) -> str:
        """Convert the configured metric to a RediSearch DISTANCE_METRIC value.

        :return: One of 'COSINE', 'L2', 'IP'.
        :rtype: str
        :raises ValueError: If the metric has no Redis equivalent.
        """
        try:
            return _METRIC_MAP[self.metric]
        except KeyError:
            raise ValueError(
                f"Unsupported Redis distance metric: {self.metric}. Redis "
                "supports cosine, dotproduct (IP) and euclidean (L2)."
            )

    def _distance_to_score(self, distance: float) -> float:
        """Convert a RediSearch KNN distance into a higher-is-better score.

        Redis always returns a value where *smaller* means *closer*. For
        COSINE and IP, that value is defined as `1 - similarity`, so we invert
        it back into a similarity score. For L2 the raw (Euclidean) distance
        is returned as-is.

        :param distance: The `vector_score` value returned by Redis.
        :type distance: float
        :return: A similarity score.
        :rtype: float
        """
        if self.metric in (Metric.COSINE, Metric.DOTPRODUCT):
            return 1 - distance
        return distance

    def _index_exists(self) -> bool:
        """Check whether the RediSearch index already exists.

        :return: True if the index exists.
        :rtype: bool
        """
        try:
            self.client.ft(self.index_name).info()
            return True
        except Exception:
            return False

    async def _aindex_exists(self) -> bool:
        """Async version of :meth:`_index_exists`.

        :return: True if the index exists.
        :rtype: bool
        """
        try:
            await self.aclient.ft(self.index_name).info()
            return True
        except Exception:
            return False

    def _schema_fields(self):
        """Build the RediSearch schema field definitions.

        :return: A tuple of Field objects for FT.CREATE.
        """
        from redis.commands.search.field import TagField, VectorField

        vector_attrs = {
            "TYPE": VECTOR_TYPE,
            "DIM": self.dimensions,
            "DISTANCE_METRIC": self._redis_metric(),
            **self.algorithm_params,
        }
        return (
            TagField(SR_ROUTE_FIELD),
            TagField(SR_NAMESPACE_FIELD),
            VectorField(VECTOR_FIELD, self.algorithm, vector_attrs),
        )

    def _create_index(self) -> None:
        """Create the RediSearch index over hashes under this index's key prefix.

        :raises ValueError: If dimensions have not been set.
        """
        if not self.dimensions:
            raise ValueError(
                "Cannot create a Redis index without specifying `dimensions`."
            )
        IndexDefinition, IndexType = _import_index_definition()

        definition = IndexDefinition(
            prefix=[self._key_prefix()], index_type=IndexType.HASH
        )
        self.client.ft(self.index_name).create_index(
            fields=self._schema_fields(), definition=definition
        )

    async def _acreate_index(self) -> None:
        """Async version of :meth:`_create_index`."""
        if not self.dimensions:
            raise ValueError(
                "Cannot create a Redis index without specifying `dimensions`."
            )
        IndexDefinition, IndexType = _import_index_definition()

        definition = IndexDefinition(
            prefix=[self._key_prefix()], index_type=IndexType.HASH
        )
        await self.aclient.ft(self.index_name).create_index(
            fields=self._schema_fields(), definition=definition
        )

    def _ensure_index(self) -> None:
        """Create the index if it doesn't exist yet. Idempotent and cheap once
        created (checked via a local flag before falling back to a server round-trip).
        """
        if self._index_initialized:
            return
        if not self._index_exists():
            self._create_index()
        self._index_initialized = True
        self.index = self

    async def _aensure_index(self) -> None:
        """Async version of :meth:`_ensure_index`."""
        if self._index_initialized:
            return
        if not await self._aindex_exists():
            await self._acreate_index()
        self._index_initialized = True
        self.index = self

    # ___________________________ FILTERING / IDS ___________________________

    def _namespace_filter_terms(self) -> List[str]:
        """Build the namespace filter term(s), if a namespace is configured.

        :return: A list with zero or one filter terms.
        :rtype: List[str]
        """
        if self.namespace is None:
            return []
        return [f"@{SR_NAMESPACE_FIELD}:{{{_escape_tag(self.namespace)}}}"]

    def _route_filter_term(self, route_names: List[str]) -> str:
        """Build a TAG filter term matching any of the given route names.

        :param route_names: The route names to match.
        :type route_names: List[str]
        :return: A `@sr_route:{...}` filter term.
        :rtype: str
        """
        escaped = "|".join(_escape_tag(r) for r in route_names)
        return f"@{SR_ROUTE_FIELD}:{{{escaped}}}"

    def _filter_query(self, extra_terms: Optional[List[str]] = None) -> str:
        """Build a RediSearch filter query string from the namespace and any
        extra terms, defaulting to match-everything.

        :param extra_terms: Additional filter terms to AND together.
        :type extra_terms: Optional[List[str]]
        :return: The filter query string.
        :rtype: str
        """
        terms = self._namespace_filter_terms() + (extra_terms or [])
        if not terms:
            return "*"
        return "(" + " ".join(terms) + ")"

    def _id_for(self, route: str, utterance: str) -> str:
        """Compute a deterministic ID for a route/utterance pair, namespaced
        when a namespace is set, to avoid cross-tenant collisions.

        :param route: The route name.
        :type route: str
        :param utterance: The utterance text.
        :type utterance: str
        :return: A UUID5 string.
        :rtype: str
        """
        key = (
            f"{self.namespace}:{route}:{utterance}"
            if self.namespace is not None
            else f"{route}:{utterance}"
        )
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def _key_for(self, route: str, utterance: str) -> str:
        """The full Redis key for a route/utterance pair's hash record.

        :param route: The route name.
        :type route: str
        :param utterance: The utterance text.
        :type utterance: str
        :return: The Redis key.
        :rtype: str
        """
        return f"{self._key_prefix()}{self._id_for(route, utterance)}"

    def _build_mapping(
        self,
        route: str,
        utterance: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]],
        function_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the Redis hash mapping for a single record.

        The full parseable record (route, utterance, function schema, and any
        additional metadata) is stored as one JSON blob so arbitrary metadata
        shapes are supported; `sr_route` is duplicated as its own TAG field so
        it can be used in RediSearch filters.

        :param route: The route name.
        :param utterance: The utterance text.
        :param embedding: The dense embedding.
        :param metadata: Additional metadata to store alongside the record.
        :param function_schema: The function schema for this utterance, if any.
        :return: The Redis hash mapping.
        :rtype: Dict[str, Any]
        """
        record = {
            "sr_route": route,
            "sr_utterance": utterance,
            "sr_function_schema": json.dumps(function_schema or {}),
        }
        record.update(metadata or {})
        mapping: Dict[str, Any] = {
            SR_ROUTE_FIELD: route,
            SR_RECORD_FIELD: json.dumps(record),
            VECTOR_FIELD: np.asarray(embedding, dtype=np.float32).tobytes(),
        }
        if self.namespace is not None:
            mapping[SR_NAMESPACE_FIELD] = self.namespace
        return mapping

    # ___________________________ ADD ___________________________

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        batch_size: int = DEFAULT_UPLOAD_BATCH_SIZE,
        **kwargs,
    ):
        """Add records to the index.

        :param embeddings: The embeddings to add.
        :type embeddings: List[List[float]]
        :param routes: The routes to add.
        :type routes: List[str]
        :param utterances: The utterances to add.
        :type utterances: List[str]
        :param function_schemas: The function schemas to add.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: The metadata to add.
        :type metadata_list: List[Dict[str, Any]]
        :param batch_size: The batch size to use for the upload.
        :type batch_size: int
        """
        self.dimensions = self.dimensions or len(embeddings[0])
        self._ensure_index()

        if not metadata_list or len(metadata_list) != len(utterances):
            metadata_list = [{} for _ in utterances]
        if not function_schemas or len(function_schemas) != len(utterances):
            function_schemas = [{} for _ in utterances]

        for i in range(0, len(embeddings), batch_size):
            pipe = self.client.pipeline(transaction=False)
            for route, utterance, embedding, metadata, function_schema in zip(
                routes[i : i + batch_size],
                utterances[i : i + batch_size],
                embeddings[i : i + batch_size],
                metadata_list[i : i + batch_size],
                function_schemas[i : i + batch_size],
            ):
                mapping = self._build_mapping(
                    route, utterance, embedding, metadata, function_schema
                )
                pipe.hset(self._key_for(route, utterance), mapping=mapping)
            pipe.execute()

    async def aadd(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        batch_size: int = DEFAULT_UPLOAD_BATCH_SIZE,
        **kwargs,
    ):
        """Asynchronously add records to the index.

        :param embeddings: The embeddings to add.
        :type embeddings: List[List[float]]
        :param routes: The routes to add.
        :type routes: List[str]
        :param utterances: The utterances to add.
        :type utterances: List[str]
        :param function_schemas: The function schemas to add.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: The metadata to add.
        :type metadata_list: List[Dict[str, Any]]
        :param batch_size: The batch size to use for the upload.
        :type batch_size: int
        """
        self.dimensions = self.dimensions or len(embeddings[0])
        await self._aensure_index()

        if not metadata_list or len(metadata_list) != len(utterances):
            metadata_list = [{} for _ in utterances]
        if not function_schemas or len(function_schemas) != len(utterances):
            function_schemas = [{} for _ in utterances]

        for i in range(0, len(embeddings), batch_size):
            pipe = self.aclient.pipeline(transaction=False)
            for route, utterance, embedding, metadata, function_schema in zip(
                routes[i : i + batch_size],
                utterances[i : i + batch_size],
                embeddings[i : i + batch_size],
                metadata_list[i : i + batch_size],
                function_schemas[i : i + batch_size],
            ):
                mapping = self._build_mapping(
                    route, utterance, embedding, metadata, function_schema
                )
                pipe.hset(self._key_for(route, utterance), mapping=mapping)
            await pipe.execute()

    # ___________________________ QUERY ___________________________

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Query the index.

        :param vector: The vector to query.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The route filter to apply.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: Unused; RedisIndex does not support hybrid search.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple of the scores and route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        if not self._index_exists():
            return np.array([]), []
        from redis.commands.search.query import Query

        extra_terms = [self._route_filter_term(route_filter)] if route_filter else []
        filter_query = self._filter_query(extra_terms)
        q = (
            Query(f"{filter_query}=>[KNN {top_k} @{VECTOR_FIELD} $vec AS vector_score]")
            .sort_by("vector_score")
            .paging(0, top_k)
            .return_fields(SR_ROUTE_FIELD, "vector_score")
            .dialect(2)
        )
        vector_bytes = np.asarray(vector, dtype=np.float32).tobytes()
        results = self.client.ft(self.index_name).search(
            q, query_params={"vec": vector_bytes}
        )
        scores = [
            self._distance_to_score(float(doc.vector_score)) for doc in results.docs
        ]
        route_names = [getattr(doc, SR_ROUTE_FIELD) for doc in results.docs]
        return np.array(scores), route_names

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Asynchronously query the index.

        :param vector: The vector to query.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The route filter to apply.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: Unused; RedisIndex does not support hybrid search.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple of the scores and route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        if not await self._aindex_exists():
            return np.array([]), []
        from redis.commands.search.query import Query

        extra_terms = [self._route_filter_term(route_filter)] if route_filter else []
        filter_query = self._filter_query(extra_terms)
        q = (
            Query(f"{filter_query}=>[KNN {top_k} @{VECTOR_FIELD} $vec AS vector_score]")
            .sort_by("vector_score")
            .paging(0, top_k)
            .return_fields(SR_ROUTE_FIELD, "vector_score")
            .dialect(2)
        )
        vector_bytes = np.asarray(vector, dtype=np.float32).tobytes()
        results = await self.aclient.ft(self.index_name).search(
            q, query_params={"vec": vector_bytes}
        )
        scores = [
            self._distance_to_score(float(doc.vector_score)) for doc in results.docs
        ]
        route_names = [getattr(doc, SR_ROUTE_FIELD) for doc in results.docs]
        return np.array(scores), route_names

    # ___________________________ GET ALL ___________________________

    def _search_ids(self, filter_query: str) -> List[str]:
        """Collect every document ID matching a filter query, paginating
        through results.

        :param filter_query: The RediSearch filter query.
        :type filter_query: str
        :return: A list of matching document (Redis key) IDs.
        :rtype: List[str]
        """
        from redis.commands.search.query import Query

        ids: List[str] = []
        offset = 0
        while True:
            q = Query(filter_query).paging(offset, PAGE_SIZE).no_content()
            results = self.client.ft(self.index_name).search(q)
            docs = results.docs
            if not docs:
                break
            ids.extend(doc.id for doc in docs)
            offset += len(docs)
            if len(docs) < PAGE_SIZE:
                break
        return ids

    async def _asearch_ids(self, filter_query: str) -> List[str]:
        """Async version of :meth:`_search_ids`."""
        from redis.commands.search.query import Query

        ids: List[str] = []
        offset = 0
        while True:
            q = Query(filter_query).paging(offset, PAGE_SIZE).no_content()
            results = await self.aclient.ft(self.index_name).search(q)
            docs = results.docs
            if not docs:
                break
            ids.extend(doc.id for doc in docs)
            offset += len(docs)
            if len(docs) < PAGE_SIZE:
                break
        return ids

    def _get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Retrieves all vector IDs (and optionally parsed records) via FT.SEARCH.

        :param prefix: Unused for Redis; filtering is handled by namespace.
        :param include_metadata: Whether to include record dicts in the return value.
        :return: Tuple of (ids, metadata_list).
        """
        if not self._index_exists():
            return [], []
        from redis.commands.search.query import Query

        filter_query = self._filter_query()
        ids: List[str] = []
        metadata: List[dict] = []
        offset = 0
        while True:
            q = Query(filter_query).paging(offset, PAGE_SIZE)
            q = q.return_fields(SR_RECORD_FIELD) if include_metadata else q.no_content()
            results = self.client.ft(self.index_name).search(q)
            docs = results.docs
            if not docs:
                break
            for doc in docs:
                ids.append(doc.id)
                if include_metadata:
                    metadata.append(
                        json.loads(getattr(doc, SR_RECORD_FIELD, "{}") or "{}")
                    )
            offset += len(docs)
            if len(docs) < PAGE_SIZE:
                break
        return ids, metadata

    async def _async_get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Async version of :meth:`_get_all`."""
        if not await self._aindex_exists():
            return [], []
        from redis.commands.search.query import Query

        filter_query = self._filter_query()
        ids: List[str] = []
        metadata: List[dict] = []
        offset = 0
        while True:
            q = Query(filter_query).paging(offset, PAGE_SIZE)
            q = q.return_fields(SR_RECORD_FIELD) if include_metadata else q.no_content()
            results = await self.aclient.ft(self.index_name).search(q)
            docs = results.docs
            if not docs:
                break
            for doc in docs:
                ids.append(doc.id)
                if include_metadata:
                    metadata.append(
                        json.loads(getattr(doc, SR_RECORD_FIELD, "{}") or "{}")
                    )
            offset += len(docs)
            if len(docs) < PAGE_SIZE:
                break
        return ids, metadata

    async def aget_routes(self) -> list[tuple]:
        """Asynchronously get a list of route and utterance objects currently
        stored in the index.

        :return: A list of (route_name, utterance, function_schemas, metadata) tuples.
        :rtype: list[tuple]
        """
        return await self._async_get_routes()

    # ___________________________ DELETE ___________________________

    def delete(self, route_name: str):
        """Delete records from the index by route name.

        :param route_name: The name of the route to delete.
        :type route_name: str
        """
        if not self._index_exists():
            return
        filter_query = self._filter_query([self._route_filter_term([route_name])])
        ids = self._search_ids(filter_query)
        if ids:
            self.client.delete(*ids)

    async def adelete(self, route_name: str) -> list[str]:
        """Asynchronously delete records from the index by route name.

        :param route_name: The name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        if not await self._aindex_exists():
            return []
        filter_query = self._filter_query([self._route_filter_term([route_name])])
        ids = await self._asearch_ids(filter_query)
        if ids:
            await self.aclient.delete(*ids)
        return ids

    def _remove_and_sync(self, routes_to_delete: dict):
        """Remove specific utterances from the index.

        IDs are deterministic (see :meth:`_id_for`), so this deletes directly
        without a round-trip to look them up first; deleting a key that
        doesn't exist is a no-op in Redis.

        :param routes_to_delete: Dict mapping route name to list of utterances to remove.
        :type routes_to_delete: dict
        """
        keys = [
            self._key_for(route, utterance)
            for route, utterances in routes_to_delete.items()
            for utterance in utterances
        ]
        if keys:
            self.client.delete(*keys)

    async def _async_remove_and_sync(self, routes_to_delete: dict):
        """Asynchronously remove specific utterances from the index.

        :param routes_to_delete: Dict mapping route name to list of utterances to remove.
        :type routes_to_delete: dict
        """
        keys = [
            self._key_for(route, utterance)
            for route, utterances in routes_to_delete.items()
            for utterance in utterances
        ]
        if keys:
            await self.aclient.delete(*keys)

    def delete_all(self):
        """Deletes all records from the index (but keeps the index/schema)."""
        if not self._index_exists():
            return
        ids = self._search_ids(self._filter_query())
        if ids:
            self.client.delete(*ids)

    def delete_index(self):
        """Delete the RediSearch index and all of its documents."""
        if self._index_exists():
            self.client.ft(self.index_name).dropindex(delete_documents=True)
        self._index_initialized = False

    async def adelete_index(self):
        """Asynchronously delete the RediSearch index and all of its documents."""
        if await self._aindex_exists():
            await self.aclient.ft(self.index_name).dropindex(delete_documents=True)
        self._index_initialized = False

    # ___________________________ DESCRIBE / READY / LEN ___________________________

    def _num_docs(self, info: Any) -> int:
        """Extract `num_docs` from an FT.INFO reply, tolerating both the
        dict-like and flat list-of-pairs shapes redis-py may return.

        :param info: The raw FT.INFO reply.
        :return: The number of documents in the index.
        :rtype: int
        """
        if isinstance(info, dict):
            return int(info.get("num_docs", 0))
        pairs = dict(zip(info[::2], info[1::2]))
        return int(pairs.get("num_docs", 0))

    def describe(self) -> IndexConfig:
        """Describe the index.

        :return: The index configuration.
        :rtype: IndexConfig
        """
        try:
            info = self.client.ft(self.index_name).info()
            num_docs = self._num_docs(info)
        except Exception:
            num_docs = 0
        return IndexConfig(
            type=self.type,
            dimensions=self.dimensions or 0,
            vectors=num_docs,
        )

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self._index_exists()

    async def ais_ready(self) -> bool:
        """Checks if the index is ready to be used asynchronously.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return await self._aindex_exists()

    def __len__(self):
        """Returns the total number of vectors in the index. If the index is
        not initialized returns 0.

        :return: The total number of vectors.
        :rtype: int
        """
        return self.describe().vectors

    async def alen(self):
        """Async version of __len__. Returns the total number of vectors in the index.

        :return: The total number of vectors.
        :rtype: int
        """
        try:
            info = await self.aclient.ft(self.index_name).info()
            return self._num_docs(info)
        except Exception:
            return 0

    # ___________________________ CONFIG ___________________________
    # Config values (hash, lock, etc.) are plain Redis hashes -- no vector
    # index needed, unlike backends where everything must be a vector record.

    def _config_key(self, field: str, scope: Optional[str]) -> str:
        """Build the Redis key for a config parameter.

        :param field: The config field name.
        :param scope: The config scope, defaults to this index's name.
        :return: The Redis key.
        :rtype: str
        """
        return (
            f"{CONFIG_KEY_PREFIX}:{self.index_name}:{scope or self.index_name}:{field}"
        )

    @staticmethod
    def _decode_hash(data: Dict[Any, Any]) -> Dict[str, str]:
        """Decode a raw HGETALL reply (bytes keys/values) into a str->str dict.

        :param data: The raw hash reply.
        :return: The decoded hash.
        :rtype: Dict[str, str]
        """
        return {
            (k.decode() if isinstance(k, bytes) else k): (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in data.items()
        }

    def _write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Write a config parameter to Redis.

        :param config: The config parameter to write.
        :type config: ConfigParameter
        :return: The config parameter that was written.
        :rtype: ConfigParameter
        """
        key = self._config_key(config.field, config.scope)
        self.client.hset(
            key,
            mapping={
                "field": config.field,
                "value": config.value,
                "scope": config.scope or self.index_name,
                "created_at": config.created_at,
            },
        )
        return config

    def _read_config(self, field: str, scope: Optional[str] = None) -> ConfigParameter:
        """Read a config parameter from Redis.

        :param field: The field to read.
        :type field: str
        :param scope: The scope to read.
        :type scope: str | None
        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        key = self._config_key(field, scope)
        data = self.client.hgetall(key)
        if not data:
            logger.warning(f"Configuration for {field} parameter not found in Redis.")
            return ConfigParameter(
                field=field, value="", scope=scope or self.index_name
            )
        data = self._decode_hash(data)
        return ConfigParameter(
            field=data.get("field", field),
            value=data.get("value", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            scope=data.get("scope", scope or self.index_name),
        )

    async def _async_write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Asynchronously write a config parameter to Redis."""
        key = self._config_key(config.field, config.scope)
        await self.aclient.hset(
            key,
            mapping={
                "field": config.field,
                "value": config.value,
                "scope": config.scope or self.index_name,
                "created_at": config.created_at,
            },
        )
        return config

    async def _async_read_config(
        self, field: str, scope: Optional[str] = None
    ) -> ConfigParameter:
        """Asynchronously read a config parameter from Redis."""
        key = self._config_key(field, scope)
        data = await self.aclient.hgetall(key)
        if not data:
            logger.warning(f"Configuration for {field} parameter not found in Redis.")
            return ConfigParameter(
                field=field, value="", scope=scope or self.index_name
            )
        data = self._decode_hash(data)
        return ConfigParameter(
            field=data.get("field", field),
            value=data.get("value", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            scope=data.get("scope", scope or self.index_name),
        )

    def close(self):
        """Closes the Redis client connections if they exist."""
        if self.client is not None:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")

    def __del__(self):
        self.close()
