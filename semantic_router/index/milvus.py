from semantic_router.index.base import BaseIndex
from typing import Union, Optional, Any, List, Dict, Tuple

import numpy as np
from semantic_router.schema import Metric
from pydantic.v1 import Field, BaseModel
from semantic_router.utils.logger import logger


DEFAULT_COLLECTION_NAME = "semantic_router_index"
ID = "id"
EMBEDDINGS = "embeddings"
ROUTE = "route"
UTTERANCE = "utterance"
OUTPUT_LIMIT = 1000


class MilvusIndex(BaseIndex):
    uri: str = Field(
        default="milvus_demo.db",
        description="URI of Milvus server",
    )

    index_name: str = Field(
        default=DEFAULT_COLLECTION_NAME,
        description="Collection name of Milvus index",
    )

    token: Optional[str] = Field(
        default=None,
        description="Token for Milvus server",
    )

    utterance_max_length: Optional[int] = Field(
        default=100,
        description="Max length of each utterance",
    )

    route_max_length: Optional[int] = Field(
        default=50,
        description="Max length of each route",
    )

    routes: Optional[np.ndarray] = None
    utterances: Optional[np.ndarray] = None
    dimensions: Union[int, None] = None
    client: Any = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = self._initialize_client()
        self._clear_collections()
        self.type = "milvus"

    def _clear_collections(self):
        """Clear existing collections in Milvus."""
        if self.client.has_collection(collection_name=DEFAULT_COLLECTION_NAME):
            self.client.drop_collection(collection_name=DEFAULT_COLLECTION_NAME)

    def _initialize_client(self):
        """Initialize the Milvus client."""
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError("Please install pymilvus to use MilvusIndex.")

        milvus_args = {"uri": self.uri}
        if self.token:
            milvus_args["token"] = self.token

        return MilvusClient(**milvus_args)

    def _initialize_collection(self):
        """Initialize the collection schema and index."""
        try:
            from pymilvus import MilvusClient, DataType
        except ImportError:
            raise ImportError("Please check if pymilvus is properly installed.")

        self.client: MilvusClient
        if self.client.has_collection(collection_name=self.index_name):
            self.client.drop_collection(collection_name=self.index_name)

        if self.dimensions is None:
            raise ValueError("Cannot initiate index without dimensions.")

        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name=ID, datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name=EMBEDDINGS, datatype=DataType.FLOAT_VECTOR, dim=self.dimensions
        )
        schema.add_field(field_name=ROUTE, datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(
            field_name=UTTERANCE, datatype=DataType.VARCHAR, max_length=100
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=ID,
        )
        index_params.add_index(
            field_name=EMBEDDINGS,
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=self.index_name, schema=schema, index_params=index_params
        )

    def _remove_and_sync(self, routes_to_delete: dict):
        """Remove and sync is not implemented for MilvusIndex."""
        if self.sync is not None:
            logger.error("Sync remove is not implemented for MilvusIndex.")

    def _sync_index(
        self,
        local_route_names: List[str],
        local_utterances_list: List[str],
        local_function_schemas: List[Dict[str, Any]],
        local_metadata_list: List[Dict[str, Any]],
        dimensions: int,
    ):
        """Sync index is not implemented for MilvusIndex."""
        if self.sync is not None:
            logger.error("Sync index is not implemented for MilvusIndex.")

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
    ):
        """
        Add embeddings to the Milvus index.

        :param embeddings: List of embedding vectors.
        :param routes: List of route names.
        :param utterances: List of utterances.
        :param function_schemas: Optional function schemas.
        :param metadata_list: List of metadata dictionaries.
        """
        self.dimensions = self.dimensions or len(embeddings[0])
        if not self.client.has_collection(collection_name=self.index_name):
            self._initialize_collection()

        data = [
            {EMBEDDINGS: embedding, ROUTE: route, UTTERANCE: utterance}
            for embedding, route, utterance in zip(embeddings, routes, utterances)
        ]

        self.client.insert(collection_name=self.index_name, data=data)

    def get_routes(self) -> List[Tuple]:
        """
        Retrieves a list of routes and their associated utterances from the index.

        :returns: A list of tuples, each containing a route name and an associated utterance.
        """
        res = self.client.query(
            collection_name=self.index_name,
            filter='route not in ["***"]',
            output_fields=[ROUTE, UTTERANCE],
            limit=OUTPUT_LIMIT,
        )

        res = list(res)
        route_utterance = [(group["route"], group["utterance"]) for group in res]
        return route_utterance

    def delete(self, route_name: str):
        """Delete a specific route from the index."""
        filter_route = f"route == '{route_name}'"
        self.client.delete(collection_name=self.index_name, filter=filter_route)

    def describe(self):
        """Describe the index with statistics."""
        info = self.client.describe_collection(collection_name=self.index_name)
        stats = self.client.get_collection_stats(collection_name=self.index_name)
        params = {
            "vectors": stats["row_count"],
            "dimensions": self.dimensions,
            "type": self.type,
        }
        return params

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Search the index for the query vector and return top_k results.

        :param vector: The query vector.
        :param top_k: Number of top results to return.
        :param route_filter: Optional filter to restrict routes.
        :returns: A tuple of distances and route names.
        """
        if not self.client.has_collection(collection_name=self.index_name):
            raise ValueError("Index not found.")

        vector = [vector.tolist()]

        if route_filter:
            filter_rule = f"route in {str(route_filter)}"
            res = self.client.search(
                collection_name=self.index_name,
                limit=top_k,
                data=vector,
                filter=filter_rule,
                output_fields=[EMBEDDINGS, ROUTE],
            )
        else:
            res = self.client.search(
                collection_name=self.index_name,
                limit=top_k,
                data=vector,
                output_fields=[EMBEDDINGS, ROUTE],
            )

        res = list(res)
        distances = [gp["distance"] for gp in res[0]]
        routes = [gp["entity"][ROUTE] for gp in res[0]]
        return np.array(distances), routes

    def delete_index(self):
        """Delete or reset the index."""
        self.client.drop_collection(collection_name=self.index_name)

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Async query is not implemented for MilvusIndex."""
        if self.sync is not None:
            logger.error("Async query is not implemented for Milvus.")

    def aget_routes(self):
        """Async get_routes is not implemented for MilvusIndex."""
        logger.error("Async get_routes is not implemented for MilvusIndex.")

    def convert_metric(self, metric: Metric):
        """
        Convert metric to a Milvus-compatible format.

        :param metric: The metric to convert.
        :returns: The converted metric string.
        """
        metric_hash = {
            Metric.COSINE: "COSINE",
            Metric.DOTPRODUCT: "IP",
            Metric.EUCLIDEAN: "L2",
        }

        if metric not in metric_hash:
            raise ValueError(f"Unsupported Milvus similarity metric: {metric}")

        return metric_hash[metric]

    def __len__(self):
        """Return the number of vectors in the index."""
        return self.client.get_collection_stats(self.index_name)["row_count"]
