import asyncio
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from pydantic import BaseModel, Field

from semantic_router.index.base import BaseIndex, IndexConfig
from semantic_router.schema import ConfigParameter, SparseEmbedding
from semantic_router.utils.logger import logger


def clean_route_name(route_name: str) -> str:
    # standardize route names
    return route_name.strip().replace(" ", "-")


class DeleteRequest(BaseModel):
    ids: list[str] | None = None
    delete_all: bool = False
    namespace: str | None = None
    filter: dict[str, Any] | None = None


def build_records(
    embeddings: List[List[float]],
    routes: List[str],
    utterances: List[str],
    function_schemas: Optional[Optional[List[Dict[str, Any]]]] = None,
    metadata_list: List[Dict[str, Any]] = [],
    sparse_embeddings: Optional[Optional[List[SparseEmbedding]]] = None,
) -> List[Dict]:
    """Build records for Pinecone upsert.

    :param embeddings: List of embeddings to upsert.
    :type embeddings: List[List[float]]
    :param routes: List of routes to upsert.
    :type routes: List[str]
    :param utterances: List of utterances to upsert.
    :type utterances: List[str]
    :param function_schemas: List of function schemas to upsert.
    :type function_schemas: Optional[List[Dict[str, Any]]]
    :param metadata_list: List of metadata to upsert.
    :type metadata_list: List[Dict[str, Any]]
    :param sparse_embeddings: List of sparse embeddings to upsert.
    :type sparse_embeddings: Optional[List[SparseEmbedding]]
    :return: List of records to upsert.
    :rtype: List[Dict]
    """
    if function_schemas is None:
        function_schemas = [{}] * len(embeddings)
    if sparse_embeddings is None:
        vectors_to_upsert = [
            PineconeRecord(
                values=vector,
                route=route,
                utterance=utterance,
                function_schema=json.dumps(function_schema),
                metadata=metadata,
            ).to_dict()
            for vector, route, utterance, function_schema, metadata in zip(
                embeddings,
                routes,
                utterances,
                function_schemas,
                metadata_list,
            )
        ]
    else:
        vectors_to_upsert = [
            PineconeRecord(
                values=vector,
                sparse_values=sparse_emb.to_pinecone(),
                route=route,
                utterance=utterance,
                function_schema=json.dumps(function_schema),
                metadata=metadata,
            ).to_dict()
            for vector, route, utterance, function_schema, metadata, sparse_emb in zip(
                embeddings,
                routes,
                utterances,
                function_schemas,
                metadata_list,
                sparse_embeddings,
            )
        ]
    return vectors_to_upsert


class PineconeRecord(BaseModel):
    id: str = ""
    values: List[float]
    sparse_values: Optional[dict[str, list]] = None
    route: str
    utterance: str
    function_schema: str = "{}"
    metadata: Dict[str, Any] = {}  # Additional metadata dictionary

    def __init__(self, **data):
        """Initialize PineconeRecord.

        :param **data: Keyword arguments to pass to the BaseModel constructor.
        :type **data: dict
        """
        super().__init__(**data)
        clean_route = clean_route_name(self.route)
        # Use SHA-256 for a more secure hash
        utterance_id = hashlib.sha256(self.utterance.encode()).hexdigest()
        self.id = f"{clean_route}#{utterance_id}"
        self.metadata.update(
            {
                "sr_route": self.route,
                "sr_utterance": self.utterance,
                "sr_function_schema": self.function_schema,
            }
        )

    def to_dict(self):
        """Convert PineconeRecord to a dictionary.

        :return: Dictionary representation of the PineconeRecord.
        :rtype: dict
        """
        d = {
            "id": self.id,
            "values": self.values,
            "metadata": self.metadata,
        }
        if self.sparse_values:
            d["sparse_values"] = self.sparse_values
        return d


class PineconeIndex(BaseIndex):
    index_prefix: str = "semantic-router--"
    api_key: Optional[str] = None
    index_name: str = "index"
    dimensions: Union[int, None] = None
    metric: str = "dotproduct"
    cloud: str = "aws"
    region: str = "us-east-1"
    host: str = ""
    client: Any = Field(default=None, exclude=True)
    index: Optional[Any] = Field(default=None, exclude=True)
    ServerlessSpec: Any = Field(default=None, exclude=True)
    namespace: Optional[str] = ""
    base_url: Optional[str] = None
    headers: dict[str, str] = {}
    index_host: Optional[str] = "http://localhost:5080"
    init_async_index: bool = False
    _using_local_emulator: bool = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: str = "index",
        dimensions: Optional[int] = None,
        metric: str = "dotproduct",
        cloud: str = "aws",
        region: str = "us-east-1",
        host: str = "",
        namespace: Optional[str] = "",
        base_url: Optional[str] = "https://api.pinecone.io",
        init_async_index: bool = False,
    ):
        """Initialize PineconeIndex.

        :param api_key: Pinecone API key.
        :type api_key: Optional[str]
        :param index_name: Name of the index.
        :type index_name: str
        :param dimensions: Dimensions of the index.
        :type dimensions: Optional[int]
        :param metric: Metric of the index.
        :type metric: str
        :param cloud: Cloud provider of the index.
        :type cloud: str
        :param region: Region of the index.
        :type region: str
        :param host: Host of the index.
        :type host: str
        :param namespace: Namespace of the index.
        :type namespace: Optional[str]
        :param base_url: Base URL of the Pinecone API.
        :type base_url: Optional[str]
        :param init_async_index: Whether to initialize the index asynchronously.
        :type init_async_index: bool
        """
        super().__init__()
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key is required.")

        self.headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "source_tag=semanticrouter",
        }

        # Set base_url from env or argument
        if os.getenv("PINECONE_API_BASE_URL"):
            self.base_url = os.getenv("PINECONE_API_BASE_URL")
        else:
            self.base_url = base_url

        # Determine if using local emulator or cloud
        if self.base_url and (
            "localhost" in self.base_url or "pinecone:5080" in self.base_url
        ):
            self.index_host = "http://pinecone:5080"
            self._using_local_emulator = True
        else:
            self.index_host = None  # Let Pinecone SDK handle host for cloud
            self._using_local_emulator = False

        if self.base_url and "api.pinecone.io" in self.base_url:
            self.headers["X-Pinecone-API-Version"] = "2024-07"

        # Preserve requested name for potential namespace use
        requested_index_name = index_name
        # Persist the originally requested index name for namespace isolation when reusing a shared index
        self._requested_index_name = requested_index_name
        self.index_name = index_name
        self.dimensions = dimensions
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.host = host
        if namespace == "sr_config":
            raise ValueError("Namespace 'sr_config' is reserved for internal use.")
        self.namespace = namespace
        self.type = "pinecone"

        logger.warning(
            "Default region changed from us-west-2 to us-east-1 in v0.1.0.dev6"
        )

        self.client = self._initialize_client(api_key=self.api_key)

        # If running against Pinecone Cloud and a shared index is provided via env,
        # reuse that index and push isolation to namespaces based on requested name
        if self.base_url and "api.pinecone.io" in self.base_url:
            shared_index = os.getenv("PINECONE_INDEX_NAME")
            if shared_index:
                shared_index = shared_index.strip()
                if shared_index:
                    self.index_name = shared_index
                    if not self.namespace:
                        # Use the originally requested index name to isolate data
                        self.namespace = requested_index_name

        # try initializing index
        if not init_async_index:
            self.index = self._init_index()

    def _initialize_client(self, api_key: Optional[str] = None):
        """Initialize the Pinecone client.

        :param api_key: Pinecone API key.
        :type api_key: Optional[str]
        :return: Pinecone client.
        :rtype: Pinecone
        """
        try:
            from pinecone import Pinecone, ServerlessSpec

            self.ServerlessSpec = ServerlessSpec
        except ImportError:
            raise ImportError(
                "Please install the Pinecone SDK v7+ to use PineconeIndex. "
                "You can install it with: `pip install 'semantic-router[pinecone]'`"
            )
        pinecone_args = {
            "api_key": api_key,
            "source_tag": "semanticrouter",
            "host": self.base_url,
        }
        if self.namespace:
            pinecone_args["namespace"] = self.namespace
        return Pinecone(**pinecone_args)  # type: ignore[arg-type]

    def _calculate_index_host(self):
        """Calculate the index host. Used to differentiate between Pinecone cloud and local emulator."""
        # Local emulator: base_url explicitly points to localhost or the pinecone service alias
        if self.base_url and (
            "localhost" in self.base_url or "pinecone:5080" in self.base_url
        ):
            self.index_host = "http://pinecone:5080"
            self._sdk_host_for_validation = "http://pinecone:5080"
        elif self.base_url and "localhost" in self.base_url:
            match = re.match(r"http://localhost:(\d+)", self.base_url)
            port = match.group(1) if match else "5080"
            self.index_host = f"http://localhost:{port}"
            self._sdk_host_for_validation = self.index_host
        elif self.index_host and self.base_url:
            # Cloud: keep the described host, ensure scheme if needed
            if not str(self.index_host).startswith("http"):
                self.index_host = f"https://{self.index_host}"
            self._sdk_host_for_validation = self.index_host
        else:
            self._sdk_host_for_validation = self.index_host

    def _init_index(self, force_create: bool = False) -> Union[Any, None]:
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        dimensions_given = self.dimensions is not None
        if self.index is None:
            index_exists = self.client.has_index(name=self.index_name)
            if dimensions_given and not index_exists:
                logger.debug(
                    f"[PineconeIndex] Creating index: {self.index_name} with dimensions={self.dimensions}, metric={self.metric}, cloud={self.cloud}, region={self.region}"
                )
                try:
                    self.client.create_index(
                        name=self.index_name,
                        dimension=self.dimensions,
                        metric=self.metric,
                        spec=self.ServerlessSpec(cloud=self.cloud, region=self.region),
                    )
                except Exception as e:
                    # If index creation is forbidden (likely quota), surface a clear
                    # instruction to reuse an existing index instead of adding fallback logic.
                    from pinecone.exceptions import ForbiddenException

                    if isinstance(e, ForbiddenException):
                        raise RuntimeError(
                            "Pinecone index creation forbidden (likely quota). "
                            "Set PINECONE_INDEX_NAME to an existing index and rerun."
                        ) from e
                    raise
                logger.debug(
                    f"[PineconeIndex] Index created; proceeding without readiness wait: {self.index_name}"
                )
                index = self.client.Index(self.index_name)
                self.index = index
                # Best-effort to populate dimensions; let errors surface if not ready
                self.dimensions = index.describe_index_stats()["dimension"]
            elif index_exists:
                # Let the SDK pick the correct host (cloud or local) based on client configuration
                index = self.client.Index(self.index_name)
                self.index = index
                self.dimensions = index.describe_index_stats()["dimension"]
            elif force_create and not dimensions_given:
                raise ValueError("Dimensions must be provided to create a new index.")
            else:
                index = self.index
                # Creation was not possible and index does not exist; give a clear error for cloud
                if (
                    self.base_url
                    and "api.pinecone.io" in self.base_url
                    and self.index is None
                ):
                    raise RuntimeError(
                        "Pinecone index unavailable and cannot be created due to quota. "
                        "Set PINECONE_INDEX_NAME to an existing index and rerun."
                    )
        else:
            index = self.index
        if self.index is not None and self.host == "":
            # Get the data-plane host from describe; normalize scheme for cloud
            self.index_host = self.client.describe_index(self.index_name).host
            if self.index_host:
                if str(self.index_host).startswith("http"):
                    self.host = str(self.index_host)
                else:
                    self.host = f"https://{self.index_host}"
        logger.debug(
            f"[PineconeIndex] _init_index returning index: {self.index_name}, index={self.index}"
        )
        return index

    async def _init_async_index(self, force_create: bool = False):
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        This method is used to initialize the index asynchronously.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        index_stats = None
        # first try getting dimensions if needed
        if self.dimensions is None:
            # check if the index exists
            indexes = await self._async_list_indexes()
            index_names = [i["name"] for i in indexes["indexes"]]
            index_exists = self.index_name in index_names
            if index_exists:
                # we can get the dimensions from the index
                index_stats = await self._async_describe_index(self.index_name)
                self.dimensions = index_stats["dimension"]
            elif index_exists and not force_create:
                # if the index doesn't exist and we don't have the dimensions
                # we raise warning
                logger.warning(
                    "Index could not be initialized. Init parameters: "
                    f"{self.index_name=}, {self.dimensions=}, {self.metric=}, "
                    f"{self.cloud=}, {self.region=}, {self.host=}, {self.namespace=}, "
                    f"{force_create=}"
                )
            elif force_create:
                raise ValueError(
                    "Index could not be initialized. Init parameters: "
                    f"{self.index_name=}, {self.dimensions=}, {self.metric=}, "
                )
            else:
                raise NotImplementedError(
                    "Unexpected init conditions. Please report this issue in GitHub."
                )
        # now check if we have dimensions
        if self.dimensions:
            # check if the index exists
            indexes = await self._async_list_indexes()
            index_names = [i["name"] for i in indexes["indexes"]]
            index_exists = self.index_name in index_names
            # if the index doesn't exist, we create it
            if not index_exists:
                # confirm if the index exists
                index_stats = await self._async_describe_index(self.index_name)
                # index_stats["status"] can be either a dict or an int (e.g. 404)
                index_status = index_stats.get("status", {})
                index_ready = (
                    index_status.get("ready", False)
                    if isinstance(index_status, dict)
                    else False
                )
                if (
                    index_ready == "true"
                    or isinstance(index_ready, bool)
                    and index_ready
                ):
                    # if the index is ready, we return it
                    self.index_host = index_stats["host"]
                    self._calculate_index_host()
                    self.host = self.index_host
                    return index_stats
                else:
                    # if the index is not ready, we create it
                    await self._async_create_index(
                        name=self.index_name,
                        dimension=self.dimensions,
                        metric=self.metric,
                        cloud=self.cloud,
                        region=self.region,
                    )
                    # Proceed without readiness loop; caller will handle transient errors if any
                    return await self._async_describe_index(self.index_name)
            else:
                # if the index exists, we return it
                index_stats = await self._async_describe_index(self.index_name)
                self.index_host = index_stats["host"]
                self._calculate_index_host()
                self.host = self.index_host
                return index_stats
        if index_stats:
            self.index_host = index_stats["host"]
            self._calculate_index_host()
            self.host = self.index_host
        else:
            self.host = ""

    def _batch_upsert(self, batch: List[Dict]):
        """Helper method for upserting a single batch of records.

        :param batch: The batch of records to upsert.
        :type batch: List[Dict]
        """
        if self.index is not None:
            logger.debug(
                f"[PineconeIndex] Upserting to index: {self.index_name}, batch size: {len(batch)}"
            )
            self.index.upsert(vectors=batch, namespace=self.namespace)
            logger.debug(
                f"[PineconeIndex] Upsert succeeded for index: {self.index_name}"
            )
        else:
            raise ValueError("Index is None, could not upsert.")

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[Optional[List[Dict[str, Any]]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        batch_size: int = 100,
        sparse_embeddings: Optional[Optional[List[SparseEmbedding]]] = None,
        **kwargs,
    ):
        """Add vectors to Pinecone in batches.

        :param embeddings: List of embeddings to upsert.
        :type embeddings: List[List[float]]
        :param routes: List of routes to upsert.
        :type routes: List[str]
        :param utterances: List of utterances to upsert.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to upsert.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to upsert.
        :type metadata_list: List[Dict[str, Any]]
        :param batch_size: Number of vectors to upsert in a single batch.
        :type batch_size: int, optional
        :param sparse_embeddings: List of sparse embeddings to upsert.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        if self.index is None:
            self.dimensions = self.dimensions or len(embeddings[0])
            self.index = self._init_index(force_create=True)
        vectors_to_upsert = build_records(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances,
            function_schemas=function_schemas,
            metadata_list=metadata_list,
            sparse_embeddings=sparse_embeddings,
        )
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            self._batch_upsert(batch)

    async def aadd(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[Optional[List[Dict[str, Any]]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        batch_size: int = 100,
        sparse_embeddings: Optional[Optional[List[SparseEmbedding]]] = None,
        **kwargs,
    ):
        """Add vectors to Pinecone in batches.

        :param embeddings: List of embeddings to upsert.
        :type embeddings: List[List[float]]
        :param routes: List of routes to upsert.
        :type routes: List[str]
        :param utterances: List of utterances to upsert.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to upsert.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to upsert.
        :type metadata_list: List[Dict[str, Any]]
        :param batch_size: Number of vectors to upsert in a single batch.
        :type batch_size: int, optional
        :param sparse_embeddings: List of sparse embeddings to upsert.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        vectors_to_upsert = build_records(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances,
            function_schemas=function_schemas,
            metadata_list=metadata_list,
            sparse_embeddings=sparse_embeddings,
        )

        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            await self._async_upsert(
                vectors=batch,
                namespace=self.namespace or "",
            )

    def _remove_and_sync(self, routes_to_delete: dict):
        """Remove specified routes from index if they exist.

        :param routes_to_delete: Routes to delete.
        :type routes_to_delete: dict
        """
        for route, utterances in routes_to_delete.items():
            remote_routes = self._get_routes_with_ids(route_name=route)
            ids_to_delete = [
                r["id"]
                for r in remote_routes
                if (r["route"], r["utterance"])
                in zip([route] * len(utterances), utterances)
            ]
            if ids_to_delete and self.index:
                self.index.delete(ids=ids_to_delete, namespace=self.namespace)

    async def _async_remove_and_sync(self, routes_to_delete: dict):
        """Remove specified routes from index if they exist.

        This method is asyncronous.

        :param routes_to_delete: Routes to delete.
        :type routes_to_delete: dict
        """
        for route, utterances in routes_to_delete.items():
            remote_routes = await self._async_get_routes_with_ids(route_name=route)
            ids_to_delete = [
                r["id"]
                for r in remote_routes
                if (r["route"], r["utterance"])
                in zip([route] * len(utterances), utterances)
            ]
            if ids_to_delete and self.index:
                await self._async_delete(
                    ids=ids_to_delete, namespace=self.namespace or ""
                )

    def _get_route_ids(self, route_name: str):
        """Get the IDs of the routes in the index.

        :param route_name: Name of the route to get the IDs for.
        :type route_name: str
        :return: List of IDs of the routes.
        :rtype: list[str]
        """
        clean_route = clean_route_name(route_name)
        ids, _ = self._get_all(prefix=f"{clean_route}#")
        return ids

    async def _async_get_route_ids(self, route_name: str):
        """Get the IDs of the routes in the index.

        :param route_name: Name of the route to get the IDs for.
        :type route_name: str
        :return: List of IDs of the routes.
        :rtype: list[str]
        """
        clean_route = clean_route_name(route_name)
        ids, _ = await self._async_get_all(prefix=f"{clean_route}#")
        return ids

    def _get_routes_with_ids(self, route_name: str):
        """Get the routes with their IDs from the index.

        :param route_name: Name of the route to get the routes with their IDs for.
        :type route_name: str
        :return: List of routes with their IDs.
        :rtype: list[dict]
        """
        clean_route = clean_route_name(route_name)
        ids, metadata = self._get_all(prefix=f"{clean_route}#", include_metadata=True)
        route_tuples = []
        for id, data in zip(ids, metadata):
            route_tuples.append(
                {
                    "id": id,
                    "route": data["sr_route"],
                    "utterance": data["sr_utterance"],
                }
            )
        return route_tuples

    async def _async_get_routes_with_ids(self, route_name: str):
        """Get the routes with their IDs from the index.

        :param route_name: Name of the route to get the routes with their IDs for.
        :type route_name: str
        :return: List of routes with their IDs.
        :rtype: list[dict]
        """
        clean_route = clean_route_name(route_name)
        ids, metadata = await self._async_get_all(
            prefix=f"{clean_route}#", include_metadata=True
        )
        route_tuples = []
        for id, data in zip(ids, metadata):
            route_tuples.append(
                {"id": id, "route": data["sr_route"], "utterance": data["sr_utterance"]}
            )
        return route_tuples

    def _get_all(self, prefix: Optional[str] = None, include_metadata: bool = False):
        """Retrieves all vector IDs from the Pinecone index using pagination.

        :param prefix: The prefix to filter the vectors by.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the response.
        :type include_metadata: bool
        :return: A tuple containing a list of vector IDs and a list of metadata dictionaries.
        :rtype: tuple[list[str], list[dict]]
        """
        if self.index is None:
            self._init_index()
        if self.index is None:
            raise ValueError("Index is None, could not retrieve vector IDs.")
        all_vector_ids = []
        metadata = []
        try:
            for ids in self.index.list(prefix=prefix, namespace=self.namespace):
                all_vector_ids.extend(ids)
                if include_metadata:
                    for id in ids:
                        res_meta = (
                            self.index.fetch(ids=[id], namespace=self.namespace)
                            if self.index
                            else None
                        )
                        if res_meta is not None and hasattr(res_meta, "vectors"):
                            for vec in res_meta.vectors.values():
                                md = getattr(vec, "metadata", None) or {}
                                metadata.append(md)
        except Exception as e:
            from pinecone.exceptions import NotFoundException

            if isinstance(e, NotFoundException):
                # Index exists but is empty, treat as no vectors
                return [], []
            else:
                raise
        return all_vector_ids, metadata

    def delete(self, route_name: str) -> list[str]:
        """Delete specified route from index if it exists. Returns the IDs of the vectors
        deleted.

        :param route_name: Name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        route_vec_ids = self._get_route_ids(route_name=route_name)
        if self.index is not None:
            logger.info("index is not None, deleting...")
            if self.base_url and "api.pinecone.io" in self.base_url:
                self.index.delete(ids=route_vec_ids, namespace=self.namespace)
            else:
                response = requests.post(
                    f"{self.index_host}/vectors/delete",
                    json=DeleteRequest(
                        ids=route_vec_ids,
                        delete_all=True,
                        namespace=self.namespace,
                    ).model_dump(exclude_none=True),
                    timeout=10,
                )
                if response.status_code == 200:
                    logger.info(
                        f"Deleted {len(route_vec_ids)} vectors from index {self.index_name}."
                    )
                else:
                    error_message = response.text
                    raise Exception(
                        f"Failed to delete vectors: {response.status_code} : {error_message}"
                    )
            return route_vec_ids
        else:
            raise ValueError("Index is None, could not delete.")

    async def adelete(self, route_name: str) -> list[str]:
        """Asynchronously delete specified route from index if it exists. Returns the IDs
        of the vectors deleted.

        :param route_name: Name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        if not (await self.ais_ready()):
            raise ValueError("Async index is not initialized.")
        route_vec_ids = await self._async_get_route_ids(route_name=route_name)
        await self._async_delete(ids=route_vec_ids, namespace=self.namespace or "")
        return route_vec_ids

    def delete_all(self):
        """Delete all routes from index if it exists.

        :return: None
        :rtype: None
        """
        if self.index is not None:
            self.index.delete(delete_all=True, namespace=self.namespace)
        else:
            raise ValueError("Index is None, could not delete.")

    def describe(self) -> IndexConfig:
        """Describe the index.

        :return: IndexConfig
        :rtype: IndexConfig
        """
        if self.index is not None:
            stats = self.index.describe_index_stats()
            return IndexConfig(
                type=self.type,
                dimensions=stats["dimension"],
                vectors=stats["namespaces"][self.namespace]["vector_count"],
            )
        else:
            return IndexConfig(
                type=self.type,
                dimensions=self.dimensions or 0,
                vectors=0,
            )

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self.index is not None

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Search the index for the query vector and return the top_k results.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param sparse_vector: An optional sparse vector to include in the query.
        :type sparse_vector: Optional[SparseEmbedding]
        :param kwargs: Additional keyword arguments for the query, including sparse_vector.
        :type kwargs: Any
        :return: A tuple containing an array of scores and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises ValueError: If the index is not populated.
        """
        if self.index is None:
            raise ValueError("Index is not populated.")
        query_vector_list = vector.tolist()
        if route_filter is not None:
            filter_query = {"sr_route": {"$in": route_filter}}
        else:
            filter_query = None
        if sparse_vector is not None:
            logger.error(f"sparse_vector exists:{sparse_vector}")
            if isinstance(sparse_vector, dict):
                sparse_vector = SparseEmbedding.from_dict(sparse_vector)
            if isinstance(sparse_vector, SparseEmbedding):
                # unnecessary if-statement but mypy didn't like this otherwise
                sparse_vector = sparse_vector.to_pinecone()
        try:
            results = self.index.query(
                vector=[query_vector_list],
                sparse_vector=sparse_vector,
                top_k=top_k,
                filter=filter_query,
                include_metadata=True,
                namespace=self.namespace,
            )
        except Exception:
            logger.error("retrying query with vector as str")
            results = self.index.query(
                vector=query_vector_list,
                sparse_vector=sparse_vector,
                top_k=top_k,
                filter=filter_query,
                include_metadata=True,
                namespace=self.namespace,
            )
        scores = [result["score"] for result in results["matches"]]
        route_names = [result["metadata"]["sr_route"] for result in results["matches"]]
        return np.array(scores), route_names

    def _read_config(self, field: str, scope: str | None = None) -> ConfigParameter:
        """Read a config parameter from the index.

        :param field: The field to read.
        :type field: str
        :param scope: The scope to read.
        :type scope: str | None
        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        scope = scope or self.namespace
        if self.index is None:
            return ConfigParameter(
                field=field,
                value="",
                scope=scope,
            )
        config_id = f"{field}#{scope}"
        config_record = self.index.fetch(
            ids=[config_id],
            namespace="sr_config",
        )
        # Pinecone v7: FetchResponse with .vectors mapping id -> Vector
        if hasattr(config_record, "vectors") and config_id in config_record.vectors:
            vec = config_record.vectors[config_id]
            metadata = getattr(vec, "metadata", {}) or {}
            value = metadata.get("value", "")
            created_raw = metadata.get("created_at")
            if not isinstance(created_raw, str):
                raise TypeError(
                    f"Invalid created_at type: {type(created_raw)} for config {field}. Expected str."
                )
            created_at: str = created_raw
            return ConfigParameter(
                field=field,
                value=value,
                created_at=created_at,
                scope=scope,
            )
        logger.warning(f"Configuration for {field} parameter not found in index.")
        return ConfigParameter(
            field=field,
            value="",
            scope=scope,
        )

    async def _async_read_config(
        self, field: str, scope: str | None = None
    ) -> ConfigParameter:
        """Read a config parameter from the index asynchronously.

        :param field: The field to read.
        :type field: str
        :param scope: The scope to read.
        :type scope: str | None
        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        scope = scope or self.namespace
        if self.index is None:
            return ConfigParameter(
                field=field,
                value="",
                scope=scope,
            )
        config_id = f"{field}#{scope}"
        config_record = await self._async_fetch_metadata(
            vector_id=config_id, namespace="sr_config"
        )
        if config_record:
            try:
                created_raw = config_record.get("created_at")
                if not isinstance(created_raw, str):
                    raise TypeError(
                        f"Invalid created_at type: {type(created_raw)} for config {field}. Expected str."
                    )
                return ConfigParameter(
                    field=field,
                    value=config_record["value"],
                    created_at=created_raw,
                    scope=scope,
                )
            except KeyError:
                raise ValueError(
                    f"Found invalid config record during sync: {config_record}"
                )
        else:
            logger.warning(f"Configuration for {field} parameter not found in index.")
            return ConfigParameter(
                field=field,
                value="",
                scope=scope,
            )

    def _write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Method to write a config parameter to the remote Pinecone index.

        :param config: The config parameter to write to the index.
        :type config: ConfigParameter
        """
        config.scope = config.scope or self.namespace
        if self.index is None:
            raise ValueError("Index has not been initialized.")
        if self.dimensions is None:
            raise ValueError("Must set PineconeIndex.dimensions before writing config.")
        self.index.upsert(
            vectors=[config.to_pinecone(dimensions=self.dimensions)],
            namespace="sr_config",
        )
        return config

    async def _async_write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Method to write a config parameter to the remote Pinecone index.

        :param config: The config parameter to write to the index.
        :type config: ConfigParameter
        """
        config.scope = config.scope or self.namespace
        if self.dimensions is None:
            raise ValueError("Must set PineconeIndex.dimensions before writing config.")
        pinecone_config = config.to_pinecone(dimensions=self.dimensions)
        await self._async_upsert(
            vectors=[pinecone_config],
            namespace="sr_config",
        )
        return config

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Asynchronously search the index for the query vector and return the top_k results.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param kwargs: Additional keyword arguments for the query, including sparse_vector.
        :type kwargs: Any
        :keyword sparse_vector: An optional sparse vector to include in the query.
        :type sparse_vector: Optional[dict]
        :return: A tuple containing an array of scores and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises ValueError: If the index is not populated.
        """
        if not (await self.ais_ready()):
            raise ValueError("Async index is not initialized.")
        query_vector_list = vector.tolist()
        if route_filter is not None:
            filter_query = {"sr_route": {"$in": route_filter}}
        else:
            filter_query = None
        # set sparse_vector_obj
        sparse_vector_obj: dict[str, Any] | None = None
        if sparse_vector is not None:
            if isinstance(sparse_vector, dict):
                sparse_vector_obj = SparseEmbedding.from_dict(sparse_vector)
            if isinstance(sparse_vector, SparseEmbedding):
                # unnecessary if-statement but mypy didn't like this otherwise
                sparse_vector_obj = sparse_vector.to_pinecone()
        results = await self._async_query(
            vector=query_vector_list,
            sparse_vector=sparse_vector_obj,
            namespace=self.namespace or "",
            filter=filter_query,
            top_k=top_k,
            include_metadata=True,
        )
        scores = [result["score"] for result in results["matches"]]
        route_names = [result["metadata"]["sr_route"] for result in results["matches"]]
        return np.array(scores), route_names

    async def aget_routes(self) -> list[tuple]:
        """Asynchronously get a list of route and utterance objects currently
        stored in the index.

        :return: A list of (route_name, utterance) objects.
        :rtype: List[Tuple]
        """
        if not (await self.ais_ready()):
            raise ValueError("Async index is not initialized.")

        return await self._async_get_routes()

    def delete_index(self):
        """Delete the index.

        :return: None
        :rtype: None
        """
        self.client.delete_index(self.index_name)
        self.index = None

    # __ASYNC CLIENT METHODS__
    async def adelete_index(self):
        """Asynchronously delete the index."""
        if not (await self.ais_ready()):
            raise ValueError("Async index is not initialized.")
        if not self.base_url:
            raise ValueError("base_url is not set for PineconeIndex.")

        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.base_url}/indexes/{self.index_name}",
                headers=self.headers,
            ) as response:
                res = await response.json(content_type=None)
                if response.status != 202:
                    raise Exception(f"Failed to delete index: {response.status}", res)
        self.host = ""
        return res

    async def _async_query(
        self,
        vector: list[float],
        sparse_vector: dict[str, Any] | None = None,
        namespace: str = "",
        filter: Optional[dict] = None,
        top_k: int = 5,
        include_metadata: bool = False,
    ):
        """Asynchronously query the index for the query vector and return the top_k results.

        :param vector: The query vector to search for.
        :type vector: list[float]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[str, Any] | None
        :param namespace: The namespace to search for.
        :type namespace: str
        :param filter: The filter to search for.
        :type filter: Optional[dict]
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param include_metadata: Whether to include metadata in the results, defaults to False.
        :type include_metadata: bool, optional
        """
        # Params now passed directly via SDK below
        if not (await self.ais_ready()):
            raise ValueError("Async index is not initialized.")
        # Use Pinecone async SDK instead of manual HTTP
        try:
            from pinecone import PineconeAsyncio
        except ImportError as e:
            raise ImportError(
                'Pinecone asyncio support not installed. Install with `pip install "pinecone[asyncio]"`.'
            ) from e

        async with PineconeAsyncio(api_key=self.api_key) as apc:
            # Resolve host via describe if not already known
            index_host: Optional[str] = self.host or None
            if not index_host:
                desc = await apc.describe_index(self.index_name)
                candidate = (
                    desc.get("host")
                    if isinstance(desc, dict)
                    else getattr(desc, "host", None)
                )
                if isinstance(candidate, str):
                    index_host = candidate
                else:
                    index_host = None
            if self._using_local_emulator and not index_host:
                index_host = "http://pinecone:5080"
            if not index_host:
                raise ValueError(
                    "Could not resolve Pinecone index host for async query"
                )
            if not index_host.startswith("http"):
                index_host = f"https://{index_host}"

            async with apc.IndexAsyncio(host=index_host) as aindex:
                return await aindex.query(
                    vector=vector,
                    sparse_vector=sparse_vector,
                    namespace=namespace,
                    filter=filter,
                    top_k=top_k,
                    include_metadata=include_metadata,
                )

    async def ais_ready(self, client_only: bool = False) -> bool:
        """Checks if class attributes exist to be used for async operations.

        :param client_only: Whether to check only the client attributes. If False
            attributes will be checked for both client and index operations. If True
            only attributes for client operations will be checked. Defaults to False.
        :type client_only: bool, optional
        :return: True if the class attributes exist, False otherwise.
        :rtype: bool
        """
        # first check client only attributes
        if not (self.cloud or self.region or self.base_url):
            return False
        if not client_only:
            # now check index attributes
            if not (
                self.index_name
                and self.dimensions
                and self.metric
                and self.host
                and self.host != ""
            ):
                # try to create index
                await self._init_async_index()
                if not (
                    self.index_name
                    and self.dimensions
                    and self.metric
                    and self.host
                    and self.host != ""
                ):
                    return False
        return True

    async def _async_list_indexes(self):
        """Asynchronously lists all indexes within the current Pinecone project.

        :return: List of indexes.
        :rtype: list[dict]
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/indexes",
                headers=self.headers,
            ) as response:
                return await response.json(content_type=None)

    async def _async_upsert(
        self,
        vectors: list[dict],
        namespace: str = "",
    ):
        """Asynchronously upserts vectors into the index.

        :param vectors: The vectors to upsert.
        :type vectors: list[dict]
        :param namespace: The namespace to upsert the vectors into.
        :type namespace: str
        """
        if not (await self.ais_ready()):
            raise ValueError("Async index is not initialized.")

        # Params now passed directly via SDK below

        # Use Pinecone async SDK for upsert
        try:
            from pinecone import PineconeAsyncio
        except ImportError as e:
            raise ImportError(
                'Pinecone asyncio support not installed. Install with `pip install "pinecone[asyncio]"`.'
            ) from e

        async with PineconeAsyncio(api_key=self.api_key) as apc:
            index_host: Optional[str] = self.host or None
            if not index_host:
                desc = await apc.describe_index(self.index_name)
                candidate = (
                    desc.get("host")
                    if isinstance(desc, dict)
                    else getattr(desc, "host", None)
                )
                index_host = candidate if isinstance(candidate, str) else None
            if self._using_local_emulator and not index_host:
                index_host = "http://pinecone:5080"
            if not index_host:
                raise ValueError(
                    "Could not resolve Pinecone index host for async upsert"
                )
            if not index_host.startswith("http"):
                index_host = f"https://{index_host}"
            async with apc.IndexAsyncio(host=index_host) as aindex:
                return await aindex.upsert(vectors=vectors, namespace=namespace)

    async def _async_create_index(
        self,
        name: str,
        dimension: int,
        cloud: str,
        region: str,
        metric: str = "dotproduct",
    ):
        """Asynchronously creates a new index in Pinecone.

        :param name: The name of the index to create.
        :type name: str
        :param dimension: The dimension of the index.
        :type dimension: int
        :param cloud: The cloud provider to create the index on.
        :type cloud: str
        :param region: The region to create the index in.
        :type region: str
        :param metric: The metric to use for the index, defaults to "dotproduct".
        :type metric: str, optional
        """
        params = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "spec": {"serverless": {"cloud": cloud, "region": region}},
            "deletion_protection": "disabled",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/indexes",
                json=params,
                headers=self.headers,
            ) as response:
                return await response.json(content_type=None)

    async def _async_delete(self, ids: list[str], namespace: str = ""):
        """Asynchronously deletes vectors from the index.

        :param ids: The IDs of the vectors to delete.
        :type ids: list[str]
        :param namespace: The namespace to delete the vectors from.
        :type namespace: str
        """
        params = {
            "ids": ids,
            "namespace": namespace,
        }
        if self.base_url and "api.pinecone.io" in self.base_url:
            if not self.host.startswith("http"):
                logger.error(f"host exists:{self.host}")
                self.host = f"https://{self.host}"
        elif self.host.startswith("localhost") and self.base_url:
            self.host = f"http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.host}/vectors/delete",
                json=params,
                headers=self.headers,
            ) as response:
                return await response.json(content_type=None)

    async def _async_describe_index(self, name: str):
        """Asynchronously describes the index.

        :param name: The name of the index to describe.
        :type name: str
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/indexes/{name}",
                headers=self.headers,
            ) as response:
                return await response.json(content_type=None)

    async def _async_get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Retrieves all vector IDs from the Pinecone index using pagination
        asynchronously.

        :param prefix: The prefix to filter the vectors by.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the response.
        :type include_metadata: bool
        :return: A tuple containing a list of vector IDs and a list of metadata dictionaries.
        :rtype: tuple[list[str], list[dict]]
        """
        if not (await self.ais_ready()):
            raise ValueError("Async index is not initialized.")

        all_vector_ids = []
        next_page_token = None

        if prefix:
            prefix_str = f"?prefix={prefix}"
        else:
            prefix_str = ""

        if self.base_url and "api.pinecone.io" in self.base_url:
            if not self.host.startswith("http"):
                logger.error(f"host exists:{self.host}")
                self.host = f"https://{self.host}"

        elif self.host.startswith("localhost") and self.base_url:
            self.host = f"http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}"

        list_url = f"{self.host}/vectors/list{prefix_str}"
        params: dict = {}
        if self.namespace:
            params["namespace"] = self.namespace
        metadata = []

        async with aiohttp.ClientSession() as session:
            while True:
                if next_page_token:
                    params["paginationToken"] = next_page_token

                async with session.get(
                    list_url,
                    params=params,
                    headers=self.headers,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error fetching vectors: {error_text}")
                        break

                    response_data = await response.json(content_type=None)

                vector_ids = [vec["id"] for vec in response_data.get("vectors", [])]
                if not vector_ids:
                    break
                all_vector_ids.extend(vector_ids)

                if include_metadata:
                    metadata_tasks = [
                        self._async_fetch_metadata(id) for id in vector_ids
                    ]
                    metadata_results = await asyncio.gather(*metadata_tasks)
                    metadata.extend(metadata_results)

                next_page_token = response_data.get("pagination", {}).get("next")
                if not next_page_token:
                    break

        return all_vector_ids, metadata

    async def _async_fetch_metadata(
        self,
        vector_id: str,
        namespace: str | None = None,
    ) -> dict:
        """Fetch metadata for a single vector ID asynchronously using the
        ClientSession.

        :param vector_id: The ID of the vector to fetch metadata for.
        :type vector_id: str
        :param namespace: The namespace to fetch metadata for.
        :type namespace: str | None
        :return: A dictionary containing the metadata for the vector.
        :rtype: dict
        """
        if not (await self.ais_ready()):
            raise ValueError("Async index is not initialized.")
        if self.base_url and "api.pinecone.io" in self.base_url:
            if not self.host.startswith("http"):
                logger.error(f"host exists:{self.host}")
                self.host = f"https://{self.host}"
        elif self.host.startswith("localhost") and self.base_url:
            self.host = f"http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}"

        # Use Pinecone async SDK to fetch metadata
        try:
            from pinecone import PineconeAsyncio
        except ImportError as e:
            raise ImportError(
                'Pinecone asyncio support not installed. Install with `pip install "pinecone[asyncio]"`.'
            ) from e
        async with PineconeAsyncio(api_key=self.api_key) as apc:
            index_host: Optional[str] = self.host or None
            if not index_host:
                desc = await apc.describe_index(self.index_name)
                candidate = (
                    desc.get("host")
                    if isinstance(desc, dict)
                    else getattr(desc, "host", None)
                )
                index_host = candidate if isinstance(candidate, str) else None
                if index_host and not str(index_host).startswith("http"):
                    index_host = f"https://{index_host}"
            if self._using_local_emulator and not index_host:
                index_host = "http://pinecone:5080"
            if not index_host:
                raise ValueError(
                    "Could not resolve Pinecone index host for async fetch"
                )
            async with apc.IndexAsyncio(host=index_host) as aindex:
                data = await aindex.fetch(
                    ids=[vector_id], namespace=namespace or self.namespace
                )
                try:
                    if hasattr(data, "vectors"):
                        vectors = data.vectors
                    else:
                        vectors = (
                            data.get("vectors", []) if isinstance(data, dict) else []
                        )
                    if vectors:
                        first = (
                            vectors[0]
                            if isinstance(vectors, list)
                            else vectors.get(vector_id)
                        )
                        metadata = getattr(first, "metadata", None) or (
                            first.get("metadata") if isinstance(first, dict) else {}
                        )
                        return metadata or {}
                except Exception as e:
                    logger.error(f"Error parsing metadata response: {e}")
                return {}

    def __len__(self):
        """Returns the total number of vectors in the index. If the index is not initialized
        returns 0.

        :return: The total number of vectors.
        :rtype: int
        """
        if self.index is None:
            logger.warning("Index is not initialized, returning 0")
            return 0
        namespace_stats = self.index.describe_index_stats()["namespaces"].get(
            self.namespace
        )
        if namespace_stats:
            return namespace_stats["vector_count"]
        else:
            return 0

    async def alen(self):
        """Async version of __len__. Returns the total number of vectors in the index.
        If the index is not initialized, initializes it first or returns 0.

        :return: The total number of vectors.
        :rtype: int
        """
        if not await self.ais_ready():
            logger.warning("Index is not ready, returning 0")
            return 0

        namespace_stats = await self._async_describe_index_stats()
        if namespace_stats and "namespaces" in namespace_stats:
            ns_stats = namespace_stats["namespaces"].get(self.namespace)
            if ns_stats:
                return ns_stats["vectorCount"]
        return 0

    async def _async_describe_index_stats(self):
        """Async version of describe_index_stats.

        :return: Index statistics.
        :rtype: dict
        """
        # Use Pinecone async SDK to describe index stats
        try:
            from pinecone import PineconeAsyncio
        except ImportError as e:
            raise ImportError(
                'Pinecone asyncio support not installed. Install with `pip install "pinecone[asyncio]"`.'
            ) from e
        async with PineconeAsyncio(api_key=self.api_key) as apc:
            index_host = self.host
            if not index_host:
                desc = await apc.describe_index(self.index_name)
                index_host = (
                    desc.get("host")
                    if isinstance(desc, dict)
                    else getattr(desc, "host", None)
                )
                if index_host and not str(index_host).startswith("http"):
                    index_host = f"https://{index_host}"
            if self._using_local_emulator and not index_host:
                index_host = "http://pinecone:5080"
            async with apc.IndexAsyncio(host=index_host) as aindex:
                return await aindex.describe_index_stats(namespace=self.namespace)
