import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

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
DEFAULT_UPLOAD_BATCH_SIZE = 100
SCROLL_SIZE = 1000
SR_UTTERANCE_PAYLOAD_KEY = "sr_utterance"
SR_ROUTE_PAYLOAD_KEY = "sr_route"


class QdrantIndex(BaseIndex):
    "The name of the collection to use"

    index_name: str = Field(
        default=DEFAULT_COLLECTION_NAME,
        description="Name of the Qdrant collection."
        f"Default: '{DEFAULT_COLLECTION_NAME}'",
    )
    location: Optional[str] = Field(
        default=":memory:",
        description="If ':memory:' - use an in-memory Qdrant instance."
        "Used as 'url' value otherwise",
    )
    url: Optional[str] = Field(
        default=None,
        description="Qualified URL of the Qdrant instance."
        "Optional[scheme], host, Optional[port], Optional[prefix]",
    )
    port: Optional[int] = Field(
        default=6333,
        description="Port of the REST API interface.",
    )
    grpc_port: int = Field(
        default=6334,
        description="Port of the gRPC interface.",
    )
    prefer_grpc: Optional[bool] = Field(
        default=None,
        description="Whether to use gPRC interface whenever possible in methods",
    )
    https: Optional[bool] = Field(
        default=None,
        description="Whether to use HTTPS(SSL) protocol.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication in Qdrant Cloud.",
    )
    prefix: Optional[str] = Field(
        default=None,
        description="Prefix to the REST URL path. Example: `http://localhost:6333/some/prefix/{qdrant-endpoint}`.",
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Timeout for REST and gRPC API requests.",
    )
    host: Optional[str] = Field(
        default=None,
        description="Host name of Qdrant service."
        "If url and host are None, set to 'localhost'.",
    )
    path: Optional[str] = Field(
        default=None,
        description="Persistence path for Qdrant local",
    )
    grpc_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Options to be passed to the low-level GRPC client, if used.",
    )
    dimensions: Union[int, None] = Field(
        default=None,
        description="Embedding dimensions."
        "Defaults to the embedding length of the configured encoder.",
    )
    metric: Metric = Field(
        default=Metric.COSINE,
        description="Distance metric to use for similarity search.",
    )
    config: Optional[Dict[str, Any]] = Field(
        default={},
        description="Collection options passed to `QdrantClient#create_collection`.",
    )
    namespace: Optional[str] = Field(
        default=None,
        description=(
            "Optional namespace string (e.g. an organization ID) used to scope all "
            "index operations to a single tenant within a shared collection. When set, "
            "a ``namespace`` field is injected into every point's payload, a Qdrant "
            "keyword payload index is created on that field, and all queries/scrolls/"
            "deletes are automatically filtered to this namespace. Point UUIDs are also "
            "namespaced via ``uuid5('{namespace}:{route}:{utterance}')``, preventing "
            "cross-tenant ID collisions."
        ),
    )
    client: Any = Field(default=None, exclude=True)
    aclient: Any = Field(default=None, exclude=True)
    _collection_initialized: bool = PrivateAttr(default=False)

    def __init__(self, **kwargs):
        if "location" not in kwargs and any(
            kwargs.get(k) for k in ("url", "host", "path")
        ):
            # location defaults to in-memory, and the client takes only one of these
            kwargs["location"] = None
        super().__init__(**kwargs)
        self.type = "qdrant"
        self.client, self.aclient = self._initialize_clients()

    def _initialize_clients(self):
        """Initialize the clients for the Qdrant index.

        :return: A tuple of the sync and async clients.
        :rtype: Tuple[QdrantClient, Optional[AsyncQdrantClient]]
        """
        try:
            from qdrant_client import AsyncQdrantClient, QdrantClient

            sync_client = QdrantClient(
                location=self.location,
                url=self.url,
                port=self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                https=self.https,
                api_key=self.api_key,
                prefix=self.prefix,
                timeout=self.timeout,
                host=self.host,
                path=self.path,
                grpc_options=self.grpc_options,
            )

            async_client: Optional[AsyncQdrantClient] = None

            if all([self.location != ":memory:", self.path is None]):
                # Local Qdrant cannot interoperate with sync and async clients
                # We fallback to sync operations in this case
                async_client = AsyncQdrantClient(
                    location=self.location,
                    url=self.url,
                    port=self.port,
                    grpc_port=self.grpc_port,
                    prefer_grpc=self.prefer_grpc,
                    https=self.https,
                    api_key=self.api_key,
                    prefix=self.prefix,
                    timeout=self.timeout,
                    host=self.host,
                    path=self.path,
                    grpc_options=self.grpc_options,
                )

            return sync_client, async_client
        except ImportError as e:
            raise ImportError(
                "Please install 'qdrant-client' to use QdrantIndex."
                "You can install it with: "
                "`pip install 'semantic-router[qdrant]'`"
            ) from e

    def _init_collection(self) -> None:
        """Initialize the collection for the Qdrant index.

        :return: None
        :rtype: None
        """
        if self._collection_initialized:
            return

        from qdrant_client import QdrantClient, models

        self.client: QdrantClient
        if not self.client.collection_exists(self.index_name):
            if not self.dimensions:
                raise ValueError(
                    "Cannot create a collection without specifying the dimensions."
                )

            self.client.create_collection(
                collection_name=self.index_name,
                vectors_config=models.VectorParams(
                    size=self.dimensions, distance=self.convert_metric(self.metric)
                ),
                **(self.config or {}),
            )

        if self.namespace is not None:
            self.client.create_payload_index(
                collection_name=self.index_name,
                field_name="namespace",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        self._collection_initialized = True

    def _build_filter(self, base_filter: Optional[Any] = None) -> Optional[Any]:
        """Return a filter scoped to this namespace, optionally AND-ed with *base_filter*."""
        if self.namespace is None:
            return base_filter

        from qdrant_client import models

        ns_condition = models.FieldCondition(
            key="namespace", match=models.MatchValue(value=self.namespace)
        )
        if base_filter is None:
            return models.Filter(must=[ns_condition])

        return models.Filter(must=[ns_condition, base_filter])

    def _point_id(self, route: str, utterance: Any) -> str:
        """Deterministic point id for a (route, utterance) pair, namespace-prefixed
        when set to avoid cross-tenant collisions.

        :param route: The route name.
        :type route: str
        :param utterance: The utterance.
        :type utterance: Any
        :return: The point id.
        :rtype: str
        """
        key = (
            f"{self.namespace}:{route}:{utterance}"
            if self.namespace is not None
            else f"{route}:{utterance}"
        )
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def _points(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]],
        sparse_embeddings: Optional[List[SparseEmbedding]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Builds the point ids and payloads for an upsert.

        :return: A tuple of the point ids and their payloads.
        :rtype: Tuple[List[str], List[Dict[str, Any]]]
        """
        if sparse_embeddings is not None:
            raise ValueError(
                "QdrantIndex stores dense vectors only, use HybridLocalIndex or "
                "PineconeIndex for sparse embeddings."
            )
        metadata_list = check_records(embeddings, routes, utterances, metadata_list)
        ids = [self._point_id(r, u) for r, u in zip(routes, utterances)]
        payloads = [
            {
                SR_ROUTE_PAYLOAD_KEY: route,
                SR_UTTERANCE_PAYLOAD_KEY: utterance,
                "metadata": metadata,
                **({"namespace": self.namespace} if self.namespace is not None else {}),
            }
            for route, utterance, metadata in zip(routes, utterances, metadata_list)
        ]
        return ids, payloads

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        batch_size: int = DEFAULT_UPLOAD_BATCH_SIZE,
        **kwargs,
    ):
        """Upsert records into the index, one point per (route, utterance) pair.

        :param embeddings: The embeddings to add.
        :type embeddings: List[List[float]]
        :param routes: The routes to add.
        :type routes: List[str]
        :param utterances: The utterances to add.
        :type utterances: List[str]
        :param metadata_list: The metadata to add.
        :type metadata_list: Optional[List[Dict[str, Any]]]
        :param sparse_embeddings: Not supported, QdrantIndex is dense only.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        :param batch_size: The batch size to use for the upload.
        :type batch_size: int
        """
        ids, payloads = self._points(
            embeddings, routes, utterances, metadata_list, sparse_embeddings
        )
        if not ids:
            return
        self.dimensions = self.dimensions or len(embeddings[0])
        self._init_collection()
        self.client.upload_collection(
            self.index_name,
            vectors=embeddings,
            payload=payloads,
            ids=ids,
            batch_size=batch_size,
        )

    def _records(self, results: list, include_metadata: bool) -> tuple[list, list]:
        """Splits scrolled points into ids and flat record metadata, with the
        ``sr_route`` and ``sr_utterance`` keys next to the stored metadata.
        """
        ids = [str(r.id) for r in results]
        if not include_metadata:
            return ids, []
        metadata = []
        for r in results:
            payload = r.payload or {}
            metadata.append(
                {
                    **(payload.get("metadata") or {}),
                    SR_ROUTE: payload.get(SR_ROUTE_PAYLOAD_KEY, ""),
                    SR_UTTERANCE: payload.get(SR_UTTERANCE_PAYLOAD_KEY, ""),
                }
            )
        return ids, metadata

    def _route_selector(self, route_name: str):
        """Filter that selects every point of one route, by exact name."""
        from qdrant_client import models

        return self._build_filter(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key=SR_ROUTE_PAYLOAD_KEY,
                        match=models.MatchValue(value=route_name),
                    )
                ]
            )
        )

    def _get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Retrieves all vector IDs (and optionally payloads) via scroll.

        :param prefix: Unused for Qdrant; filtering is handled by namespace.
        :param include_metadata: Whether to include payload dicts in the return value.
        :return: Tuple of (ids, metadata_list).
        """
        if not self.client.collection_exists(self.index_name):
            return [], []

        from qdrant_client import grpc

        results = []
        next_offset = None
        stop_scrolling = False
        while not stop_scrolling:
            records, next_offset = self.client.scroll(
                self.index_name,
                limit=SCROLL_SIZE,
                offset=next_offset,
                with_payload=True,
                scroll_filter=self._build_filter(),
            )
            stop_scrolling = next_offset is None or (
                isinstance(next_offset, grpc.PointId)
                and next_offset.num == 0
                and next_offset.uuid == ""
            )
            results.extend(records)

        return self._records(results, include_metadata)

    async def _async_get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Async version of _get_all.

        :param prefix: Unused for Qdrant; filtering is handled by namespace.
        :param include_metadata: Whether to include payload dicts in the return value.
        :return: Tuple of (ids, metadata_list).
        """
        if self.aclient is None:
            return self._get_all(prefix=prefix, include_metadata=include_metadata)

        if not await self.aclient.collection_exists(self.index_name):
            return [], []

        from qdrant_client import grpc

        results = []
        next_offset = None
        stop_scrolling = False
        while not stop_scrolling:
            records, next_offset = await self.aclient.scroll(
                self.index_name,
                limit=SCROLL_SIZE,
                offset=next_offset,
                with_payload=True,
                scroll_filter=self._build_filter(),
            )
            stop_scrolling = next_offset is None or (
                isinstance(next_offset, grpc.PointId)
                and next_offset.num == 0
                and next_offset.uuid == ""
            )
            results.extend(records)

        return self._records(results, include_metadata)

    def delete(self, route_name: str):
        """Delete records from the index.

        :param route_name: The name of the route to delete.
        :type route_name: str
        """
        self.client.delete(
            self.index_name, points_selector=self._route_selector(route_name)
        )

    def describe(self) -> IndexConfig:
        """Describe the index.

        :return: The index configuration.
        :rtype: IndexConfig
        """
        collection_info = self.client.get_collection(self.index_name)

        return IndexConfig(
            type=self.type,
            dimensions=collection_info.config.params.vectors.size,
            vectors=collection_info.points_count,
        )

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self.client.collection_exists(self.index_name)

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
        :param sparse_vector: The sparse vector to query.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple of the scores and route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        from qdrant_client import QdrantClient, models

        self.client: QdrantClient
        filter = None
        if route_filter is not None:
            filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=SR_ROUTE_PAYLOAD_KEY,
                        match=models.MatchAny(any=route_filter),
                    )
                ]
            )
        filter = self._build_filter(filter)

        results = self.client.query_points(
            self.index_name,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=filter,
        )
        scores = [result.score for result in results.points]
        route_names = [
            result.payload[SR_ROUTE_PAYLOAD_KEY] for result in results.points
        ]
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
        :param sparse_vector: The sparse vector to query.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple of the scores and route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        from qdrant_client import AsyncQdrantClient, models

        self.aclient: Optional[AsyncQdrantClient]
        if self.aclient is None:
            # in-memory and on-disk qdrant have no async client (see _initialize_clients),
            # so the async methods run the sync ones, like LocalIndex does
            logger.debug("Cannot use async query with an in-memory Qdrant instance")
            return self.query(vector, top_k, route_filter)

        filter = None
        if route_filter is not None:
            filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=SR_ROUTE_PAYLOAD_KEY,
                        match=models.MatchAny(any=route_filter),
                    )
                ]
            )
        filter = self._build_filter(filter)

        results = await self.aclient.query_points(
            self.index_name,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=filter,
        )
        scores = [result.score for result in results.points]
        route_names = [
            result.payload[SR_ROUTE_PAYLOAD_KEY] for result in results.points
        ]
        return np.array(scores), route_names

    def delete_index(self):
        """Delete the index.

        :return: None
        :rtype: None
        """
        self.client.delete_collection(self.index_name)
        self._collection_initialized = False

    def convert_metric(self, metric: Metric):
        """Convert the metric to a Qdrant distance metric.

        :param metric: The metric to convert.
        :type metric: Metric
        :return: The converted metric.
        :rtype: Distance
        """
        from qdrant_client.models import Distance

        mapping = {
            Metric.COSINE: Distance.COSINE,
            Metric.EUCLIDEAN: Distance.EUCLID,
            Metric.DOTPRODUCT: Distance.DOT,
            Metric.MANHATTAN: Distance.MANHATTAN,
        }

        if metric not in mapping:
            raise ValueError(f"Unsupported Qdrant similarity metric: {metric}")

        return mapping[metric]

    def __len__(self):
        """Returns the total number of vectors in the index. If the index is not initialized
        returns 0.

        :return: The total number of vectors.
        :rtype: int
        """
        if not self.client.collection_exists(self.index_name):
            return 0
        return self.client.get_collection(self.index_name).points_count or 0

    async def adelete(self, route_name: str) -> list[str]:
        """Asynchronously delete records from the index by route name.

        :param route_name: The name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted (empty list, as Qdrant does not return IDs).
        :rtype: list[str]
        """
        if self.aclient is None:
            logger.debug(
                "Cannot use async delete with an in-memory Qdrant instance; falling back to sync delete."
            )
            self.delete(route_name)
            return []

        await self.aclient.delete(
            self.index_name, points_selector=self._route_selector(route_name)
        )
        return []

    async def adelete_index(self):
        """Asynchronously delete the index (collection) from Qdrant.

        :return: None
        :rtype: None
        """

        if self.aclient is None:
            logger.debug(
                "Cannot use async delete_index with an in-memory Qdrant instance; falling back to sync delete_index."
            )
            self.delete_index()
            return

        await self.aclient.delete_collection(self.index_name)
        self._collection_initialized = False

    async def ais_ready(self) -> bool:
        """Checks if the index is ready to be used asynchronously."""
        if self.aclient is None:
            # In-memory or sync-only mode, treat as not ready for async
            return False
        # A failed check raises, as is_ready() does. Callers treat False as "not set up yet"
        # and initialise or sync the index, so it must not stand in for "could not check".
        return await self.aclient.collection_exists(self.index_name)

    async def aadd(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        batch_size: int = DEFAULT_UPLOAD_BATCH_SIZE,
        **kwargs,
    ):
        """Asynchronously upsert records into the index, one point per (route,
        utterance) pair."""
        if self.aclient is None:
            logger.debug(
                "Cannot use async add with an in-memory Qdrant instance; falling back to sync add."
            )
            return self.add(
                embeddings=embeddings,
                routes=routes,
                utterances=utterances,
                metadata_list=metadata_list,
                sparse_embeddings=sparse_embeddings,
                batch_size=batch_size,
            )
        ids, payloads = self._points(
            embeddings, routes, utterances, metadata_list, sparse_embeddings
        )
        if not ids:
            return
        self.dimensions = self.dimensions or len(embeddings[0])
        self._init_collection()
        # AsyncQdrantClient.upload_collection is not a coroutine; use batched upsert instead.
        from qdrant_client import models as qdrant_models

        points = [
            qdrant_models.PointStruct(id=pid, vector=emb, payload=payload)
            for pid, emb, payload in zip(ids, embeddings, payloads)
        ]
        for i in range(0, len(points), batch_size):
            await self.aclient.upsert(
                collection_name=self.index_name,
                points=points[i : i + batch_size],
            )
