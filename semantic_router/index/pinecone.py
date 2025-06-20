import asyncio
import hashlib
import json
import os
import time
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from pydantic import BaseModel, Field

from semantic_router.index.base import BaseIndex, IndexConfig
from semantic_router.schema import ConfigParameter, SparseEmbedding
from semantic_router.utils.logger import logger


def clean_route_name(route_name: str) -> str:
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

        if base_url is not None or os.getenv("PINECONE_API_BASE_URL"):
            logger.info("Using pinecone remote API.")
            if os.getenv("PINECONE_API_BASE_URL"):
                self.base_url = os.getenv("PINECONE_API_BASE_URL")
            else:
                self.base_url = base_url

        if self.base_url and "api.pinecone.io" in self.base_url:
            self.headers["X-Pinecone-API-Version"] = "2024-07"

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

        # try initializing index
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
                "Please install pinecone-client to use PineconeIndex. "
                "You can install it with: "
                "`pip install 'semantic-router[pinecone]'`"
            )
        pinecone_args = {
            "api_key": api_key,
            "source_tag": "semanticrouter",
            "host": self.base_url,
        }
        if self.namespace:
            pinecone_args["namespace"] = self.namespace

        return Pinecone(**pinecone_args)

    def _calculate_index_host(self):
        """Calculate the index host. Used to differentiate between normal
        Pinecone and Pinecone Local instance.

        :return: None
        :rtype: None
        """
        if self.index_host and self.base_url:
            if "api.pinecone.io" in self.base_url:
                if not self.index_host.startswith("http"):
                    self.index_host = f"https://{self.index_host}"
            else:
                if "http" not in self.index_host:
                    self.index_host = f"http://{self.base_url.split(':')[-2].strip('/')}:{self.index_host.split(':')[-1]}"
                elif not self.index_host.startswith("http://"):
                    if "localhost" in self.index_host:
                        self.index_host = f"http://{self.base_url.split(':')[-2].strip('/')}:{self.index_host.split(':')[-1]}"
                    else:
                        self.index_host = f"http://{self.index_host}"

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
            index_exists = self.index_name in self.client.list_indexes().names()
            if dimensions_given and not index_exists:
                # if the index doesn't exist and we have dimension value
                # we create the index
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimensions,
                    metric=self.metric,
                    spec=self.ServerlessSpec(cloud=self.cloud, region=self.region),
                )
                # wait for index to be created
                while not self.client.describe_index(self.index_name).status["ready"]:
                    time.sleep(0.2)
                index = self.client.Index(self.index_name)
                self.index = index
                time.sleep(0.2)
            elif index_exists:
                # if the index exists we just return it
                # index = self.client.Index(self.index_name)

                self.index_host = self.client.describe_index(self.index_name).host
                self._calculate_index_host()
                index = self.client.Index(self.index_name, host=self.index_host)
                self.index = index

                # grab the dimensions from the index
                self.dimensions = index.describe_index_stats()["dimension"]
            elif force_create and not dimensions_given:
                raise ValueError(
                    "Cannot create an index without specifying the dimensions."
                )
            else:
                # if the index doesn't exist and we don't have the dimensions
                # we return None
                logger.warning(
                    "Index could not be initialized. Init parameters: "
                    f"{self.index_name=}, {self.dimensions=}, {self.metric=}, "
                    f"{self.cloud=}, {self.region=}, {self.host=}, {self.namespace=}, "
                    f"{force_create=}"
                )
                index = None
        else:
            index = self.index
        if self.index is not None and self.host == "":
            # if the index exists we just return it
            self.index_host = self.client.describe_index(self.index_name).host

            if self.index_host and self.base_url:
                self._calculate_index_host()
                index = self.client.Index(self.index_name, host=self.index_host)
                self.host = self.index_host
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
                index_ready = index_stats["status"]["ready"]
                if index_ready == "true":
                    # if the index is ready, we return it
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
                    index_ready = "false"
                    while index_ready != "true":
                        index_stats = await self._async_describe_index(self.index_name)
                        index_ready = index_stats["status"]["ready"]
                        await asyncio.sleep(0.1)
                    return index_stats
            else:
                # if the index exists, we return it
                return index_stats
        self.host = index_stats["host"] if index_stats else ""

    def _batch_upsert(self, batch: List[Dict]):
        """Helper method for upserting a single batch of records.

        :param batch: The batch of records to upsert.
        :type batch: List[Dict]
        """
        if self.index is not None:
            self.index.upsert(vectors=batch, namespace=self.namespace)
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
        if self.index is None:
            self.dimensions = self.dimensions or len(embeddings[0])
            self.index = await self._init_async_index(force_create=True)
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
            raise ValueError("Index is None, could not retrieve vector IDs.")
        all_vector_ids = []
        metadata = []

        for ids in self.index.list(prefix=prefix, namespace=self.namespace):
            all_vector_ids.extend(ids)

            if include_metadata:
                for id in ids:
                    res_meta = (
                        self.index.fetch(ids=[id], namespace=self.namespace)
                        if self.index
                        else {}
                    )
                    metadata.extend(
                        [x["metadata"] for x in res_meta["vectors"].values()]
                    )

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
        route_vec_ids = await self._async_get_route_ids(route_name=route_name)
        if self.index is not None:
            await self._async_delete(ids=route_vec_ids, namespace=self.namespace or "")
            return route_vec_ids
        else:
            raise ValueError("Index is None, could not delete.")

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
        if config_record.get("vectors"):
            return ConfigParameter(
                field=field,
                value=config_record["vectors"][config_id]["metadata"]["value"],
                created_at=config_record["vectors"][config_id]["metadata"][
                    "created_at"
                ],
                scope=scope,
            )
        else:
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
                return ConfigParameter(
                    field=field,
                    value=config_record["value"],
                    created_at=config_record["created_at"],
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
        if self.index is None:
            raise ValueError("Index has not been initialized.")
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
        if self.host == "":
            raise ValueError("Host is not initialized.")
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
        if self.host == "":
            raise ValueError("Host is not initialized.")

        return await self._async_get_routes()

    def delete_index(self):
        """Delete the index.

        :return: None
        :rtype: None
        """
        self.client.delete_index(self.index_name)
        self.index = None

    # __ASYNC CLIENT METHODS__
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
        params = {
            "vector": vector,
            "sparse_vector": sparse_vector,
            "namespace": namespace,
            "filter": filter,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "topK": top_k,
            "includeMetadata": include_metadata,
        }
        if self.host == "":
            raise ValueError("self.host is not initialized.")
        elif self.base_url and "api.pinecone.io" in self.base_url:
            if not self.host.startswith("http"):
                logger.error(f"host exists:{self.host}")

                self.host = f"https://{self.host}"
        elif self.host.startswith("localhost") and self.base_url:
            self.host = f"http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.host}/query",
                json=params,
                headers=self.headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error in query response: {error_text}")
                    return {}  # or handle the error as needed

                try:
                    return await response.json(content_type=None)
                except JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    return {}

    async def _is_async_ready(self, client_only: bool = False) -> bool:
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
            if not (self.index_name or self.dimensions or self.metric or self.host):
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
        params = {
            "vectors": vectors,
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
                f"{self.host}/vectors/upsert",
                json=params,
                headers=self.headers,
            ) as response:
                res = await response.json(content_type=None)
                return res

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
        if self.index is None:
            raise ValueError("Index is None, could not retrieve vector IDs.")
        if self.host == "":
            raise ValueError("self.host is not initialized.")

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
        if self.host == "":
            raise ValueError("self.host is not initialized.")
        if self.base_url and "api.pinecone.io" in self.base_url:
            if not self.host.startswith("http"):
                logger.error(f"host exists:{self.host}")
                self.host = f"https://{self.host}"
        elif self.host.startswith("localhost") and self.base_url:
            self.host = f"http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}"

        url = f"{self.host}/vectors/fetch"

        params = {
            "ids": [vector_id],
        }

        if namespace:
            params["namespace"] = [namespace]
        elif self.namespace:
            params["namespace"] = [self.namespace]

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, headers=self.headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error fetching metadata: {error_text}")
                    return {}

                try:
                    response_data = await response.json(content_type=None)
                except Exception as e:
                    logger.warning(f"No metadata found for vector {vector_id}: {e}")
                    return {}

                return (
                    response_data.get("vectors", {})
                    .get(vector_id, {})
                    .get("metadata", {})
                )

    def __len__(self):
        namespace_stats = self.index.describe_index_stats()["namespaces"].get(
            self.namespace
        )
        if namespace_stats:
            return namespace_stats["vector_count"]
        else:
            return 0
