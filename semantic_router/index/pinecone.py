import aiohttp
import asyncio
import hashlib
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from pydantic.v1 import BaseModel, Field

from semantic_router.index.base import BaseIndex
from semantic_router.utils.logger import logger


def clean_route_name(route_name: str) -> str:
    return route_name.strip().replace(" ", "-")


class PineconeRecord(BaseModel):
    id: str = ""
    values: List[float]
    route: str
    utterance: str

    def __init__(self, **data):
        super().__init__(**data)
        clean_route = clean_route_name(self.route)
        # Use SHA-256 for a more secure hash
        utterance_id = hashlib.sha256(self.utterance.encode()).hexdigest()
        self.id = f"{clean_route}#{utterance_id}"

    def to_dict(self):
        return {
            "id": self.id,
            "values": self.values,
            "metadata": {"sr_route": self.route, "sr_utterance": self.utterance},
        }


class PineconeIndex(BaseIndex):
    index_prefix: str = "semantic-router--"
    api_key: Optional[str] = None
    index_name: str = "index"
    dimensions: Union[int, None] = None
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-west-2"
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
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-west-2",
        host: str = "",
        namespace: Optional[str] = "",
        base_url: Optional[str] = "https://api.pinecone.io",
    ):
        super().__init__()
        self.index_name = index_name
        self.dimensions = dimensions
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.host = host
        self.namespace = namespace
        self.type = "pinecone"
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.base_url = base_url

        if self.api_key is None:
            raise ValueError("Pinecone API key is required.")

        self.client = self._initialize_client(api_key=self.api_key)
        self.async_client = self._initialize_async_client(api_key=self.api_key)

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
        pinecone_args = {"api_key": api_key, "source_tag": "semanticrouter"}
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
            logger.warning("Index could not be initialized.")
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
            logger.warning("Index could not be initialized.")
        self.host = index_stats["host"] if index_stats else None

    def _batch_upsert(self, batch: List[Dict]):
        """Helper method for upserting a single batch of records."""
        if self.index is not None:
            self.index.upsert(vectors=batch, namespace=self.namespace)
        else:
            raise ValueError("Index is None, could not upsert.")

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        batch_size: int = 100,
    ):
        """Add vectors to Pinecone in batches."""
        if self.index is None:
            self.dimensions = self.dimensions or len(embeddings[0])
            self.index = self._init_index(force_create=True)

        vectors_to_upsert = [
            PineconeRecord(values=vector, route=route, utterance=utterance).to_dict()
            for vector, route, utterance in zip(embeddings, routes, utterances)
        ]

        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            self._batch_upsert(batch)

    def _get_route_ids(self, route_name: str):
        clean_route = clean_route_name(route_name)
        ids, _ = self._get_all(prefix=f"{clean_route}#")
        return ids

    def _get_all(self, prefix: Optional[str] = None, include_metadata: bool = False):
        """
        Retrieves all vector IDs from the Pinecone index using pagination.
        """
        if self.index is None:
            raise ValueError("Index is None, could not retrieve vector IDs.")
        all_vector_ids = []
        next_page_token = None

        if prefix:
            prefix_str = f"?prefix={prefix}"
        else:
            prefix_str = ""

        # Construct the request URL for listing vectors. Adjust parameters as needed.
        list_url = f"https://{self.host}/vectors/list{prefix_str}"
        params: Dict = {}
        if self.namespace:
            params["namespace"] = self.namespace
        headers = {"Api-Key": self.api_key}
        metadata = []

        while True:
            if next_page_token:
                params["paginationToken"] = next_page_token

            # Make the request to list vectors. Adjust headers and parameters as needed.
            response = requests.get(list_url, params=params, headers=headers)
            response_data = response.json()

            # Extract vector IDs from the response and add them to the list
            vector_ids = [vec["id"] for vec in response_data.get("vectors", [])]
            # check that there are vector IDs, otherwise break the loop
            if not vector_ids:
                break
            all_vector_ids.extend(vector_ids)

            # if we need metadata, we fetch it
            if include_metadata:
                res_meta = self.index.fetch(ids=vector_ids, namespace=self.namespace)
                # extract metadata only
                metadata.extend([x["metadata"] for x in res_meta["vectors"].values()])

            # Check if there's a next page token; if not, break the loop
            next_page_token = response_data.get("pagination", {}).get("next")
            if not next_page_token:
                break

        return all_vector_ids, metadata

    def get_routes(self) -> List[Tuple]:
        """
        Gets a list of route and utterance objects currently stored in the index.

        Returns:
            List[Tuple]: A list of (route_name, utterance) objects.
        """
        # Get all records
        _, metadata = self._get_all(include_metadata=True)
        route_tuples = [(x["sr_route"], x["sr_utterance"]) for x in metadata]
        return route_tuples

    def delete(self, route_name: str):
        route_vec_ids = self._get_route_ids(route_name=route_name)
        if self.index is not None:
            self.index.delete(ids=route_vec_ids, namespace=self.namespace)
        else:
            raise ValueError("Index is None, could not delete.")

    def delete_all(self):
        self.index.delete(delete_all=True, namespace=self.namespace)

    def describe(self) -> Dict:
        if self.index is not None:
            stats = self.index.describe_index_stats()
            return {
                "type": self.type,
                "dimensions": stats["dimension"],
                "vectors": stats["total_vector_count"],
            }
        else:
            raise ValueError("Index is None, cannot describe index stats.")

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if self.index is None:
            raise ValueError("Index is not populated.")
        query_vector_list = vector.tolist()
        if route_filter is not None:
            filter_query = {"sr_route": {"$in": route_filter}}
        else:
            filter_query = None
        results = self.index.query(
            vector=[query_vector_list],
            top_k=top_k,
            filter=filter_query,
            include_metadata=True,
            namespace=self.namespace,
        )
        scores = [result["score"] for result in results["matches"]]
        route_names = [result["metadata"]["sr_route"] for result in results["matches"]]
        return np.array(scores), route_names

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if self.async_client is None or self.host is None:
            raise ValueError("Async client or host are not initialized.")
        query_vector_list = vector.tolist()
        if route_filter is not None:
            filter_query = {"sr_route": {"$in": route_filter}}
        else:
            filter_query = None
        results = await self._async_query(
            vector=query_vector_list,
            namespace=self.namespace or "",
            filter=filter_query,
            top_k=top_k,
            include_metadata=True,
        )
        scores = [result["score"] for result in results["matches"]]
        route_names = [result["metadata"]["sr_route"] for result in results["matches"]]
        return np.array(scores), route_names

    def delete_index(self):
        self.client.delete_index(self.index_name)

    # __ASYNC CLIENT METHODS__
    async def _async_query(
        self,
        vector: list[float],
        namespace: str = "",
        filter: Optional[dict] = None,
        top_k: int = 5,
        include_metadata: bool = False,
    ):
        params = {
            "vector": vector,
            "namespace": namespace,
            "filter": filter,
            "top_k": top_k,
            "include_metadata": include_metadata,
        }
        async with self.async_client.post(
            f"https://{self.host}/query",
            json=params,
        ) as response:
            return await response.json(content_type=None)

    async def _async_list_indexes(self):
        async with self.async_client.get(f"{self.base_url}/indexes") as response:
            return await response.json(content_type=None)

    async def _async_create_index(
        self,
        name: str,
        dimension: int,
        cloud: str,
        region: str,
        metric: str = "cosine",
    ):
        params = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "spec": {"serverless": {"cloud": cloud, "region": region}},
        }
        async with self.async_client.post(
            f"{self.base_url}/indexes",
            headers={"Api-Key": self.api_key},
            json=params,
        ) as response:
            return await response.json(content_type=None)

    async def _async_describe_index(self, name: str):
        async with self.async_client.get(f"{self.base_url}/indexes/{name}") as response:
            return await response.json(content_type=None)

    def __len__(self):
        return self.index.describe_index_stats()["total_vector_count"]
