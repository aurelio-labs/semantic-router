import aiohttp
import asyncio
import hashlib
import os
import time
import json

from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from pydantic import BaseModel, Field

from semantic_router.index.base import BaseIndex, IndexConfig
from semantic_router.schema import ConfigParameter, SparseEmbedding
from semantic_router.utils.logger import logger


def clean_route_name(route_name: str) -> str:
    return route_name.strip().replace(" ", "-")


def build_records(
    embeddings: List[List[float]],
    routes: List[str],
    utterances: List[str],
    function_schemas: Optional[Optional[List[Dict[str, Any]]]] = None,
    metadata_list: List[Dict[str, Any]] = [],
    sparse_embeddings: Optional[Optional[List[SparseEmbedding]]] = None,
) -> List[Dict]:
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
    async_client: Any = Field(default=None, exclude=True)
    index: Optional[Any] = Field(default=None, exclude=True)
    ServerlessSpec: Any = Field(default=None, exclude=True)
    namespace: Optional[str] = ""
    base_url: Optional[str] = "https://api.pinecone.io"

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
        super().__init__()
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
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.base_url = base_url

        logger.warning(
            "Default region changed from us-west-2 to us-east-1 in v0.1.0.dev6"
        )

        if self.api_key is None:
            raise ValueError("Pinecone API key is required.")

        self.client = self._initialize_client(api_key=self.api_key)
        if init_async_index:
            self.async_client = self._initialize_async_client(api_key=self.api_key)
        else:
            self.async_client = None
        # try initializing index
        self.index = self._init_index()

    def _initialize_client(self, api_key: Optional[str] = None):
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

    def _initialize_async_client(self, api_key: Optional[str] = None):
        api_key = api_key or self.api_key
        if api_key is None:
            raise ValueError("Pinecone API key is required.")
        async_client = aiohttp.ClientSession(
            headers={
                "Api-Key": api_key,
                "Content-Type": "application/json",
                "X-Pinecone-API-Version": "2024-07",
                "User-Agent": "source_tag=semanticrouter",
            }
        )
        return async_client

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
        index_exists = self.index_name in self.client.list_indexes().names()
        dimensions_given = self.dimensions is not None
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
                time.sleep(1)
            index = self.client.Index(self.index_name)
            time.sleep(0.5)
        elif index_exists:
            # if the index exists we just return it
            index = self.client.Index(self.index_name)
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
        if index is not None:
            self.host = self.client.describe_index(self.index_name)["host"]
        return index

    async def _init_async_index(self, force_create: bool = False):
        index_stats = None
        indexes = await self._async_list_indexes()
        index_names = [i["name"] for i in indexes["indexes"]]
        index_exists = self.index_name in index_names
        if self.dimensions is not None and not index_exists:
            await self._async_create_index(
                name=self.index_name,
                dimension=self.dimensions,
                metric=self.metric,
                cloud=self.cloud,
                region=self.region,
            )
            # TODO describe index and async sleep
            index_ready = "false"
            while index_ready != "true":
                index_stats = await self._async_describe_index(self.index_name)
                index_ready = index_stats["status"]["ready"]
                await asyncio.sleep(1)
        elif index_exists:
            index_stats = await self._async_describe_index(self.index_name)
            # grab dimensions for the index
            self.dimensions = index_stats["dimension"]
        elif force_create and self.dimensions is None:
            raise ValueError(
                "Cannot create an index without specifying the dimensions."
            )
        else:
            # if the index doesn't exist and we don't have the dimensions
            # we raise warning
            logger.warning(
                "Index could not be initialized. Init parameters: "
                f"{self.index_name=}, {self.dimensions=}, {self.metric=}, "
                f"{self.cloud=}, {self.region=}, {self.host=}, {self.namespace=}, "
                f"{force_create=}"
            )
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
        """Add vectors to Pinecone in batches."""
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
        """Add vectors to Pinecone in batches."""
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
        clean_route = clean_route_name(route_name)
        ids, _ = self._get_all(prefix=f"{clean_route}#")
        return ids

    async def _async_get_route_ids(self, route_name: str):
        clean_route = clean_route_name(route_name)
        ids, _ = await self._async_get_all(prefix=f"{clean_route}#")
        return ids

    def _get_routes_with_ids(self, route_name: str):
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
        """
        Retrieves all vector IDs from the Pinecone index using pagination.

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

    def delete(self, route_name: str):
        route_vec_ids = self._get_route_ids(route_name=route_name)
        if self.index is not None:
            self.index.delete(ids=route_vec_ids, namespace=self.namespace)
        else:
            raise ValueError("Index is None, could not delete.")

    def delete_all(self):
        self.index.delete(delete_all=True, namespace=self.namespace)

    def describe(self) -> IndexConfig:
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
        """
        Checks if the index is ready to be used.
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
            if isinstance(sparse_vector, dict):
                sparse_vector = SparseEmbedding.from_dict(sparse_vector)
            if isinstance(sparse_vector, SparseEmbedding):
                # unnecessary if-statement but mypy didn't like this otherwise
                sparse_vector = sparse_vector.to_pinecone()
        results = self.index.query(
            vector=[query_vector_list],
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
        """
        Asynchronously search the index for the query vector and return the top_k results.

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
        if self.async_client is None or self.host == "":
            raise ValueError("Async client or host are not initialized.")
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
        if self.async_client is None or self.host == "":
            raise ValueError("Async client or host are not initialized.")

        return await self._async_get_routes()

    def delete_index(self):
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
        params = {
            "vector": vector,
            "sparse_vector": sparse_vector,
            "namespace": namespace,
            "filter": filter,
            "top_k": top_k,
            "include_metadata": include_metadata,
        }
        if self.host == "":
            raise ValueError("self.host is not initialized.")
        async with self.async_client.post(
            f"https://{self.host}/query",
            json=params,
        ) as response:
            return await response.json(content_type=None)

    async def _async_list_indexes(self):
        async with self.async_client.get(f"{self.base_url}/indexes") as response:
            return await response.json(content_type=None)

    async def _async_upsert(
        self,
        vectors: list[dict],
        namespace: str = "",
    ):
        params = {
            "vectors": vectors,
            "namespace": namespace,
        }
        async with self.async_client.post(
            f"https://{self.host}/vectors/upsert",
            json=params,
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
        params = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "spec": {"serverless": {"cloud": cloud, "region": region}},
        }
        async with self.async_client.post(
            f"{self.base_url}/indexes",
            json=params,
        ) as response:
            return await response.json(content_type=None)

    async def _async_delete(self, ids: list[str], namespace: str = ""):
        params = {
            "ids": ids,
            "namespace": namespace,
        }
        async with self.async_client.post(
            f"https://{self.host}/vectors/delete",
            json=params,
        ) as response:
            return await response.json(content_type=None)

    async def _async_describe_index(self, name: str):
        async with self.async_client.get(f"{self.base_url}/indexes/{name}") as response:
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

        list_url = f"https://{self.host}/vectors/list{prefix_str}"
        params: dict = {}
        if self.namespace:
            params["namespace"] = self.namespace
        metadata = []

        while True:
            if next_page_token:
                params["paginationToken"] = next_page_token

            async with self.async_client.get(
                list_url, params=params, headers={"Api-Key": self.api_key}
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
                metadata_tasks = [self._async_fetch_metadata(id) for id in vector_ids]
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
        async_client.

        :param vector_id: The ID of the vector to fetch metadata for.
        :type vector_id: str
        :param namespace: The namespace to fetch metadata for.
        :type namespace: str | None
        :return: A dictionary containing the metadata for the vector.
        :rtype: dict
        """
        if self.host == "":
            raise ValueError("self.host is not initialized.")
        url = f"https://{self.host}/vectors/fetch"

        params = {
            "ids": [vector_id],
        }

        if namespace:
            params["namespace"] = [namespace]
        elif self.namespace:
            params["namespace"] = [self.namespace]

        headers = {
            "Api-Key": self.api_key,
        }

        async with self.async_client.get(
            url, params=params, headers=headers
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
                response_data.get("vectors", {}).get(vector_id, {}).get("metadata", {})
            )

    def __len__(self):
        namespace_stats = self.index.describe_index_stats()["namespaces"].get(
            self.namespace
        )
        if namespace_stats:
            return namespace_stats["vector_count"]
        else:
            return 0
