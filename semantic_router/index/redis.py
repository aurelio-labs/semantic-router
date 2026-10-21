import asyncio
import hashlib
import json
import math
import os
import time
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
# How often to poll FT.INFO while an index's initial background scan runs.
SCAN_POLL_SECONDS = 0.05
VECTOR_TYPE = "FLOAT32"

# Hash field names used for every record.
SR_ROUTE_FIELD = "sr_route"
SR_RECORD_FIELD = "sr_record"
SR_NAMESPACE_FIELD = "namespace"
VECTOR_FIELD = "vector"
# The indexed TAG fields hold a digest of the route name and namespace (see `_tag`).
SR_ROUTE_TAG_FIELD = "sr_route_tag"
SR_NAMESPACE_TAG_FIELD = "sr_namespace_tag"

# Config values (sr_hash, sr_lock, ...) live in one plain Redis hash per index,
# `sr_config:{index_name}`, outside the search index.
CONFIG_KEY_PREFIX = "sr_config"

# RediSearch only supports these three distance metrics for VECTOR fields.
# See: https://redis.io/docs/latest/develop/ai/search-and-query/vectors/#distance-metrics
_METRIC_MAP = {
    Metric.COSINE: "COSINE",
    Metric.EUCLIDEAN: "L2",
    Metric.DOTPRODUCT: "IP",
}


def _tag(value: str) -> str:
    """A filter-safe stand-in for a route name or namespace.

    RediSearch TAG fields fold case and split values on commas, and TAG queries
    need punctuation escaped, so filtering on the raw name lets `Chat` match
    `chat` and `a` match `a,b`. A hex digest matches only its own value.

    :param value: The route name or namespace.
    :type value: str
    :return: The SHA-256 hex digest of the value.
    :rtype: str
    """
    return hashlib.sha256(str(value).encode()).hexdigest()


def _missing_index(error: Exception) -> bool:
    """Whether a ResponseError from FT.INFO means the index doesn't exist.

    Only the replies Redis gives for a missing index count: "Unknown index name"
    (Redis Stack 7.x, "Unknown Index name" in older RediSearch), "SEARCH_INDEX_NOT_FOUND
    Index not found: <name>" (Redis 8.x) and "<name>: no such index". Anything
    else (no RediSearch, no permission, a server still loading) raises rather
    than reads as an empty index.

    :param error: The ResponseError raised by FT.INFO.
    :type error: Exception
    :return: True if the error means the index doesn't exist.
    :rtype: bool
    """
    message = str(error).strip().lower()
    return message.startswith(
        ("unknown index name", "search_index_not_found")
    ) or message.endswith(": no such index")


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
            "Record IDs are also namespaced, via uuid5 of "
            "`json.dumps([namespace, route, utterance])`, preventing cross-tenant "
            "ID collisions."
        ),
    )
    index_ready_timeout: float = Field(
        default=60.0,
        description="Seconds to wait for a new or reopened index to finish its "
        "initial background scan of the existing keys before using it; raises "
        "a TimeoutError after that.",
    )
    client: Any = Field(default=None, exclude=True)
    aclient: Any = Field(default=None, exclude=True)
    _index_initialized: bool = PrivateAttr(default=False)
    _standalone_checked: bool = PrivateAttr(default=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.index_name == CONFIG_KEY_PREFIX:
            raise ValueError(
                f"'{CONFIG_KEY_PREFIX}' can't be a RedisIndex index_name: its key "
                "prefix would take in every index's config hash."
            )
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

        RediSearch matches keys by plain prefix, so `routes:` would also cover
        the records of an index named `routes:v2`. Colons in the name are
        written as `%3A`, so no index's prefix can start another's.

        :return: The key prefix.
        :rtype: str
        """
        name = self.index_name.replace("%", "%25").replace(":", "%3A")
        return f"{name}:"

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
        it back into a similarity score. For L2, Redis returns the squared
        distance, so we take its root to match the Euclidean distance that
        QdrantIndex and PostgresIndex return.

        :param distance: The `vector_score` value returned by Redis.
        :type distance: float
        :return: A similarity score.
        :rtype: float
        """
        if self.metric in (Metric.COSINE, Metric.DOTPRODUCT):
            return 1 - distance
        return math.sqrt(max(distance, 0.0))

    def _cluster_error(self) -> ValueError:
        """The error for a server running in OSS cluster mode."""
        kwargs = self.client.connection_pool.connection_kwargs
        return ValueError(
            f"Redis at {kwargs.get('host')}:{kwargs.get('port')} runs in cluster "
            "mode, which RedisIndex doesn't support: it talks to a single Redis "
            "endpoint. Use a standalone Redis 8 or Redis Stack server, or a Redis "
            "Cloud or Enterprise database without the OSS Cluster API."
        )

    def _check_standalone(self) -> None:
        """Raise, once per handle and before anything is written, if the server
        runs in OSS cluster mode.

        This index uses a single-endpoint client, so on a cluster its writes
        and config reads fail with MOVED redirections, or (on Redis Stack 7.x,
        whose search runs per shard) searches see only one shard's records. An
        ACL user that may not run INFO skips the check.

        :raises ValueError: If the server runs in cluster mode.
        """
        if self._standalone_checked:
            return
        from redis.exceptions import NoPermissionError

        try:
            cluster = self.client.info("cluster").get("cluster_enabled")
        except NoPermissionError:
            cluster = 0
        if int(cluster or 0):
            raise self._cluster_error()
        self._standalone_checked = True

    async def _acheck_standalone(self) -> None:
        """Async version of :meth:`_check_standalone`."""
        if self._standalone_checked:
            return
        from redis.exceptions import NoPermissionError

        try:
            cluster = (await self.aclient.info("cluster")).get("cluster_enabled")
        except NoPermissionError:
            cluster = 0
        if int(cluster or 0):
            raise self._cluster_error()
        self._standalone_checked = True

    def _ft_info(self) -> Any:
        """FT.INFO for this index, on a server checked not to be a cluster."""
        self._check_standalone()
        return self.client.ft(self.index_name).info()

    async def _aft_info(self) -> Any:
        """Async version of :meth:`_ft_info`."""
        await self._acheck_standalone()
        return await self.aclient.ft(self.index_name).info()

    def _index_exists(self, wait: bool = False) -> bool:
        """Check whether the RediSearch index already exists.

        Only a missing index counts as "no": a connection error raises, so an
        outage is not mistaken for an empty index.

        :param wait: Whether to wait for the index's initial background scan
            (see :meth:`_wait_for_scan`), for callers that search it next.
        :type wait: bool
        :return: True if the index exists.
        :rtype: bool
        """
        from redis.exceptions import ResponseError

        try:
            info = self._ft_info()
        except ResponseError as e:
            if not _missing_index(e):
                raise
            return False
        if wait:
            self._wait_for_scan(info)
        return True

    async def _aindex_exists(self, wait: bool = False) -> bool:
        """Async version of :meth:`_index_exists`.

        :param wait: Whether to wait for the index's initial background scan.
        :type wait: bool
        :return: True if the index exists.
        :rtype: bool
        """
        from redis.exceptions import ResponseError

        try:
            info = await self._aft_info()
        except ResponseError as e:
            if not _missing_index(e):
                raise
            return False
        if wait:
            await self._await_scan(info)
        return True

    @staticmethod
    def _scanning(info: Any) -> bool:
        """Whether an FT.INFO reply shows the index's initial background scan
        still running.

        :param info: The raw FT.INFO reply.
        :return: True while FT.INFO's `indexing` is set.
        :rtype: bool
        """
        if not isinstance(info, dict):
            info = dict(zip(info[::2], info[1::2]))
        indexing = info.get("indexing", 0)
        if isinstance(indexing, bytes):
            indexing = indexing.decode()
        return float(indexing) != 0

    def _log_scan(self) -> None:
        """Say why a call is waiting, since the scan can take a while."""
        logger.info(
            f"Waiting for Redis to index the existing keys for '{self.index_name}' "
            f"(up to {self.index_ready_timeout} s)."
        )

    def _scan_timeout(self) -> TimeoutError:
        """The error for a scan that outlasts `index_ready_timeout`."""
        return TimeoutError(
            f"Redis index '{self.index_name}' was still indexing the existing keys "
            f"after {self.index_ready_timeout} s. Try again later, or raise "
            "index_ready_timeout."
        )

    def _wait_for_scan(self, info: Any) -> Any:
        """Wait, up to `index_ready_timeout` seconds, for the index's initial
        background scan to finish.

        After FT.CREATE, Redis indexes the keys already in the database in the
        background. On Redis 8, which runs queries on worker threads, a search
        during that scan can miss the records it is re-indexing, and writes
        made during it get re-indexed. So nothing writes to or searches the
        index until the scan is done.

        :param info: A fresh FT.INFO reply for the index.
        :return: The FT.INFO reply that shows the scan finished.
        :raises TimeoutError: If the scan outlasts `index_ready_timeout`.
        """
        deadline = time.monotonic() + self.index_ready_timeout
        if self._scanning(info):
            self._log_scan()
        while self._scanning(info):
            if time.monotonic() >= deadline:
                raise self._scan_timeout()
            time.sleep(SCAN_POLL_SECONDS)
            info = self._ft_info()
        return info

    async def _await_scan(self, info: Any) -> Any:
        """Async version of :meth:`_wait_for_scan`, yielding to the event loop
        between polls."""
        deadline = time.monotonic() + self.index_ready_timeout
        if self._scanning(info):
            self._log_scan()
        while self._scanning(info):
            if time.monotonic() >= deadline:
                raise self._scan_timeout()
            await asyncio.sleep(SCAN_POLL_SECONDS)
            info = await self._aft_info()
        return info

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
            TagField(SR_ROUTE_TAG_FIELD),
            TagField(SR_NAMESPACE_TAG_FIELD),
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

    @staticmethod
    def _vector_attributes(info: Any) -> Dict[str, str]:
        """Extract the vector field's attributes (DIM, DISTANCE_METRIC, ...)
        from an FT.INFO reply.

        :param info: The raw FT.INFO reply.
        :return: The vector field's attributes, lower-cased keys to str
            values, or an empty dict if the index has no vector field.
        :rtype: Dict[str, str]
        """
        if not isinstance(info, dict):
            info = dict(zip(info[::2], info[1::2]))
        for attribute in info.get("attributes", []):
            fields = {
                (k.decode() if isinstance(k, bytes) else str(k)).lower(): (
                    v.decode() if isinstance(v, bytes) else str(v)
                )
                for k, v in zip(attribute[::2], attribute[1::2])
            }
            if fields.get("type") == "VECTOR":
                return fields
        return {}

    def _check_index(self, info: Any) -> None:
        """Check an existing index against this handle: adopt its dimensions,
        or raise if they or its metric differ.

        Redis stores a vector of the wrong size without indexing it, and scores
        an index by the metric it was created with, so a mismatch would
        otherwise lose writes or return the wrong scores, without an error.

        :param info: The FT.INFO reply for the existing index.
        :raises ValueError: If the index's dimensions or metric differ from
            this handle's.
        """
        attributes = self._vector_attributes(info)
        if not attributes:
            return
        dimensions = int(attributes["dim"])
        if self.dimensions is None:
            self.dimensions = dimensions
        elif self.dimensions != dimensions:
            raise ValueError(
                f"Redis index '{self.index_name}' holds {dimensions}-dimensional "
                f"vectors, not {self.dimensions}-dimensional ones. Use another "
                "index_name, or delete_index() first."
            )
        metric = attributes.get("distance_metric", "").upper()
        if metric and metric != self._redis_metric():
            names = {v: k.value for k, v in _METRIC_MAP.items()}
            raise ValueError(
                f"Redis index '{self.index_name}' uses the "
                f"{names.get(metric, metric)} metric, not {self.metric.value}. "
                "Pass that metric, use another index_name, or delete_index() first."
            )

    def _check_vectors(self, embeddings: List[List[float]]) -> None:
        """Raise, before anything is written, if a vector isn't `dimensions`
        long.

        Redis would store it without indexing it: it would never be found, and
        its hash would outlive delete_index() and turn up in a later index.

        :param embeddings: The embeddings about to be written.
        :type embeddings: List[List[float]]
        :raises ValueError: If any embedding has the wrong length.
        """
        for i, embedding in enumerate(embeddings):
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"Embedding {i} has {len(embedding)} dimensions, but Redis "
                    f"index '{self.index_name}' holds {self.dimensions}-dimensional "
                    "vectors. Nothing was written."
                )

    def _ensure_index(self, create: bool = True) -> bool:
        """Create the index (when `create`) or check an existing one's
        dimensions and metric, then wait for its initial background scan, once
        per handle; later calls skip the server round-trip.

        :param create: Whether to create the index if it doesn't exist.
        :type create: bool
        :return: True if the index exists.
        :rtype: bool
        """
        if self._index_initialized:
            return True
        from redis.exceptions import ResponseError

        try:
            info = self._ft_info()
        except ResponseError as e:
            if not _missing_index(e):
                raise
            if not create:
                return False
            self._create_index()
            info = self._ft_info()
        else:
            self._check_index(info)
        self._wait_for_scan(info)
        self._index_initialized = True
        self.index = self
        return True

    async def _aensure_index(self, create: bool = True) -> bool:
        """Async version of :meth:`_ensure_index`."""
        if self._index_initialized:
            return True
        from redis.exceptions import ResponseError

        try:
            info = await self._aft_info()
        except ResponseError as e:
            if not _missing_index(e):
                raise
            if not create:
                return False
            await self._acreate_index()
            info = await self._aft_info()
        else:
            self._check_index(info)
        await self._await_scan(info)
        self._index_initialized = True
        self.index = self
        return True

    # ___________________________ FILTERING / IDS ___________________________

    def _namespace_filter_terms(self) -> List[str]:
        """Build the namespace filter term(s), if a namespace is configured.

        :return: A list with zero or one filter terms.
        :rtype: List[str]
        """
        if self.namespace is None:
            return []
        return [f"@{SR_NAMESPACE_TAG_FIELD}:{{{_tag(self.namespace)}}}"]

    def _route_filter_term(self, route_names: List[str]) -> str:
        """Build a TAG filter term matching any of the given route names.

        :param route_names: The route names to match.
        :type route_names: List[str]
        :return: A `@sr_route_tag:{...}` filter term.
        :rtype: str
        """
        tags = "|".join(_tag(r) for r in route_names)
        return f"@{SR_ROUTE_TAG_FIELD}:{{{tags}}}"

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
        when a namespace is set, to avoid cross-tenant collisions. The parts are
        JSON-encoded, so no split of the same characters (`a` + `b:c` against
        `a:b` + `c`) gives the same ID.

        :param route: The route name.
        :type route: str
        :param utterance: The utterance text.
        :type utterance: str
        :return: A UUID5 string.
        :rtype: str
        """
        key = json.dumps([self.namespace, route, utterance])
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
        shapes are supported; `sr_route` is duplicated as its own field so
        queries can return it, and its digest is the TAG that filters use.

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
            SR_ROUTE_TAG_FIELD: _tag(route),
            SR_RECORD_FIELD: json.dumps(record),
            VECTOR_FIELD: np.asarray(embedding, dtype=np.float32).tobytes(),
        }
        if self.namespace is not None:
            mapping[SR_NAMESPACE_FIELD] = self.namespace
            mapping[SR_NAMESPACE_TAG_FIELD] = _tag(self.namespace)
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
        self._check_vectors(embeddings)
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
        self._check_vectors(embeddings)
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
        if route_filter is not None and not route_filter:
            return np.array([]), []  # no routes allowed, as in the other indexes
        if not self._ensure_index(create=False):
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
        return self._scores_and_routes(results.docs)

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
        if route_filter is not None and not route_filter:
            return np.array([]), []  # no routes allowed, as in the other indexes
        if not await self._aensure_index(create=False):
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
        return self._scores_and_routes(results.docs)

    def _scores_and_routes(self, docs: List[Any]) -> Tuple[np.ndarray, List[str]]:
        """Read the scores and route names from KNN search results.

        :param docs: The documents of a KNN search.
        :return: A tuple of the scores and route names.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises RuntimeError: If a hit came back without its score or route,
            rather than drop it and return incomplete results.
        """
        scores, route_names = [], []
        for doc in docs:
            distance = getattr(doc, "vector_score", None)
            route = getattr(doc, SR_ROUTE_FIELD, None)
            if distance is None or route is None:
                raise RuntimeError(
                    f"Redis returned search result '{doc.id}' without its score or "
                    "route. Redis 8 can do this while records are rewritten if search "
                    "workers are on (search-workers > 0); see the RedisIndex docs."
                )
            scores.append(self._distance_to_score(float(distance)))
            route_names.append(route)
        return np.array(scores), route_names

    # ___________________________ GET ALL ___________________________

    @staticmethod
    def _rows(rows: List[Any]) -> List[Dict[str, Any]]:
        """FT.AGGREGATE rows, flat [name, value, ...] lists, as dicts with str
        names and the Redis key as a str under `__key`."""
        records = []
        for row in rows:
            record = {
                (k.decode() if isinstance(k, bytes) else k): v
                for k, v in zip(row[::2], row[1::2])
            }
            key = record.get("__key")
            record["__key"] = key.decode() if isinstance(key, bytes) else key
            records.append(record)
        return records

    def _aggregate_request(self, filter_query: str, fields: List[str]) -> Any:
        """An FT.AGGREGATE over every record matching a filter query, read
        through a cursor.

        FT.SEARCH can't page past MAXSEARCHRESULTS (10,000 in the Redis Stack
        images): past it, a page comes back empty without an error. A cursor
        has no such cap.

        :param filter_query: The RediSearch filter query.
        :param fields: The hash fields to load besides the key.
        :return: The aggregate request.
        """
        from redis.commands.search.aggregation import AggregateRequest

        return (
            AggregateRequest(filter_query)
            .load(*(f"@{f}" for f in ["__key", *fields]))
            .cursor(count=PAGE_SIZE)
        )

    def _read_all(
        self, filter_query: str, fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Every record matching a filter query, with its key and the given
        hash fields.

        :param filter_query: The RediSearch filter query.
        :param fields: The hash fields to load besides the key.
        :return: One dict per record.
        :rtype: List[Dict[str, Any]]
        """
        ft = self.client.ft(self.index_name)
        result = ft.aggregate(self._aggregate_request(filter_query, fields or []))
        rows = list(result.rows)
        try:
            while result.cursor and result.cursor.cid:
                result = ft.aggregate(result.cursor)
                rows.extend(result.rows)
        finally:
            # a read that fails midway leaves its cursor open (idle for 300 s,
            # 128 per index by default); a finished one has cid 0
            if result.cursor and result.cursor.cid:
                self._delete_cursor(result.cursor.cid)
        return self._rows(rows)

    async def _aread_all(
        self, filter_query: str, fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Async version of :meth:`_read_all`."""
        ft = self.aclient.ft(self.index_name)
        result = await ft.aggregate(self._aggregate_request(filter_query, fields or []))
        rows = list(result.rows)
        try:
            while result.cursor and result.cursor.cid:
                result = await ft.aggregate(result.cursor)
                rows.extend(result.rows)
        finally:
            if result.cursor and result.cursor.cid:
                await self._adelete_cursor(result.cursor.cid)
        return self._rows(rows)

    def _delete_cursor(self, cid: int) -> None:
        """Delete a cursor a failed or cancelled read left open, best effort,
        so the read's own error is the one that propagates.

        :param cid: The cursor id.
        :type cid: int
        """
        try:
            self.client.execute_command("FT.CURSOR", "DEL", self.index_name, cid)
        except Exception as e:
            logger.warning(f"Could not delete Redis cursor {cid}: {e}")

    async def _adelete_cursor(self, cid: int) -> None:
        """Async version of :meth:`_delete_cursor`."""
        try:
            await self.aclient.execute_command("FT.CURSOR", "DEL", self.index_name, cid)
        except Exception as e:
            logger.warning(f"Could not delete Redis cursor {cid}: {e}")

    def _search_ids(self, filter_query: str) -> List[str]:
        """Collect every document ID matching a filter query.

        :param filter_query: The RediSearch filter query.
        :type filter_query: str
        :return: A list of matching document (Redis key) IDs.
        :rtype: List[str]
        """
        return [record["__key"] for record in self._read_all(filter_query)]

    async def _asearch_ids(self, filter_query: str) -> List[str]:
        """Async version of :meth:`_search_ids`."""
        return [record["__key"] for record in await self._aread_all(filter_query)]

    @staticmethod
    def _parse_records(
        records: List[Dict[str, Any]], include_metadata: bool
    ) -> tuple[list[str], list[dict]]:
        """Split read records into their ids and, if asked, their parsed
        `sr_record` JSON."""
        ids = [record["__key"] for record in records]
        metadata = (
            [json.loads(record.get(SR_RECORD_FIELD) or "{}") for record in records]
            if include_metadata
            else []
        )
        return ids, metadata

    def _get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Retrieves all vector IDs (and optionally parsed records).

        :param prefix: Unused for Redis; filtering is handled by namespace.
        :param include_metadata: Whether to include record dicts in the return value.
        :return: Tuple of (ids, metadata_list).
        """
        if not self._index_exists(wait=True):
            return [], []
        fields = [SR_RECORD_FIELD] if include_metadata else []
        records = self._read_all(self._filter_query(), fields)
        return self._parse_records(records, include_metadata)

    async def _async_get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Async version of :meth:`_get_all`."""
        if not await self._aindex_exists(wait=True):
            return [], []
        fields = [SR_RECORD_FIELD] if include_metadata else []
        records = await self._aread_all(self._filter_query(), fields)
        return self._parse_records(records, include_metadata)

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
        if not self._index_exists(wait=True):
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
        if not await self._aindex_exists(wait=True):
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
        if not self._index_exists(wait=True):
            return
        ids = self._search_ids(self._filter_query())
        if ids:
            self.client.delete(*ids)

    def delete_index(self):
        """Delete the RediSearch index, all of its documents and its config
        (every namespace's sync hash and lock)."""
        if self._index_exists():
            self.client.ft(self.index_name).dropindex(delete_documents=True)
        self.client.delete(self._config_key())
        self._index_initialized = False

    async def adelete_index(self):
        """Asynchronously delete the RediSearch index, all of its documents and
        its config (every namespace's sync hash and lock)."""
        if await self._aindex_exists():
            await self.aclient.ft(self.index_name).dropindex(delete_documents=True)
        await self.aclient.delete(self._config_key())
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
        from redis.exceptions import ResponseError

        try:
            info = self._ft_info()
        except ResponseError as e:
            if not _missing_index(e):
                raise
            num_docs = 0
        else:
            num_docs = self._num_docs(self._wait_for_scan(info))
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
        from redis.exceptions import ResponseError

        try:
            info = await self._aft_info()
        except ResponseError as e:
            if not _missing_index(e):
                raise
            return 0
        return self._num_docs(await self._await_scan(info))

    # ___________________________ CONFIG ___________________________
    # Config values (hash, lock, etc.) are fields of one plain Redis hash per
    # index -- no vector index needed, unlike backends where everything must be
    # a vector record -- so delete_index() can drop them all with one DEL.

    def _config_key(self) -> str:
        """The Redis key of this index's config hash.

        :return: The Redis key.
        :rtype: str
        """
        return f"{CONFIG_KEY_PREFIX}:{self.index_name}"

    def _config_field(self, field: str, scope: Optional[str]) -> str:
        """The config hash field for a parameter, scoped to this handle's
        namespace so tenants sharing an index keep their own sync hash and lock.

        :param field: The config field name.
        :param scope: The config scope, defaults to this index's name.
        :return: The hash field name.
        :rtype: str
        """
        return json.dumps([self.namespace, scope or self.index_name, field])

    def _config_value(self, config: ConfigParameter) -> str:
        """Serialise a config parameter for the config hash.

        :param config: The config parameter.
        :return: Its JSON encoding.
        :rtype: str
        """
        return json.dumps(
            {
                "field": config.field,
                "value": config.value,
                "scope": config.scope or self.index_name,
                "created_at": config.created_at,
            }
        )

    def _parse_config(
        self, raw: Optional[Union[str, bytes]], field: str, scope: Optional[str]
    ) -> ConfigParameter:
        """Parse a config hash value, or return an empty parameter if unset.

        :param raw: The stored value, or None if the parameter isn't set.
        :param field: The config field name.
        :param scope: The config scope, defaults to this index's name.
        :return: The config parameter.
        :rtype: ConfigParameter
        """
        if raw is None:
            logger.warning(f"Configuration for {field} parameter not found in Redis.")
            return ConfigParameter(
                field=field, value="", scope=scope or self.index_name
            )
        data = json.loads(raw)
        return ConfigParameter(
            field=data.get("field", field),
            value=data.get("value", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            scope=data.get("scope", scope or self.index_name),
        )

    def _write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Write a config parameter to Redis.

        :param config: The config parameter to write.
        :type config: ConfigParameter
        :return: The config parameter that was written.
        :rtype: ConfigParameter
        """
        self._check_standalone()
        self.client.hset(
            self._config_key(),
            self._config_field(config.field, config.scope),
            self._config_value(config),
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
        self._check_standalone()
        raw = self.client.hget(self._config_key(), self._config_field(field, scope))
        return self._parse_config(raw, field, scope)

    async def _async_write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Asynchronously write a config parameter to Redis."""
        await self._acheck_standalone()
        await self.aclient.hset(
            self._config_key(),
            self._config_field(config.field, config.scope),
            self._config_value(config),
        )
        return config

    async def _async_read_config(
        self, field: str, scope: Optional[str] = None
    ) -> ConfigParameter:
        """Asynchronously read a config parameter from Redis."""
        await self._acheck_standalone()
        raw = await self.aclient.hget(
            self._config_key(), self._config_field(field, scope)
        )
        return self._parse_config(raw, field, scope)

    def close(self):
        """Closes the Redis client connections if they exist."""
        if self.client is not None:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")

    def __del__(self):
        self.close()
