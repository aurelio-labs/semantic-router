from semantic_router.index.base import BaseIndex
from typing import Union, Optional, Any, List, Dict, Tuple

import numpy as np

from pydantic.v1 import Field, BaseModel


DEFAULT_COLLECTION_NAME = "semantic_router_index"
ID = "id"
EMBEDDINGS = "embeddings"
ROUTE = "route"
UTTERANCE = "utterance"
OUTPUT_LIMIT = 1000



class MilvusIndex(BaseIndex):
    uri: str = Field(
        default="http://10.100.30.11:19530",
        description="uri of Milvus server",
    )
    
    index_name: str = Field(
        default=DEFAULT_COLLECTION_NAME,
        description="Collection name of "
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
        self.client = self._initiate_client()
        
    def _initiate_client(self):
        try:
            from pymilvus import MilvusClient
            
        except ImportError:
            raise ImportError(
                "Please install pymilvus to use MilvusIndex."
            )
        
        milvus_args = {'uri': self.uri}
        if self.token:
            milvus_args['token'] = self.token
        
        return MilvusClient(**milvus_args)
    
    def _initiate_collection(self):
        try:
            from pymilvus import MilvusClient, DataType
        except ImportError:
            raise ImportError(
                "Please check if pymilvus is properly installed"
            )
            
        self.client: MilvusClient
        if not self.client.has_collection(collection_name=self.index_name):
            # need to create a new collection
            if self.dimensions == None:
                raise ValueError(
                    "Cannot initiate index without dimensions"
                )
            
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name=ID, datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name=EMBEDDINGS, datatype=DataType.FLOAT_VECTOR, dim=self.dimensions)
            schema.add_field(field_name=ROUTE, datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name=UTTERANCE, datatype=DataType.VARCHAR, max_length=100)
            
            index_params = self.client.prepare_index_params()
            
            index_params.add_index(
                field_name=ID,
                index_type="STL_SORT",
            )
            index_params.add_index(
                field_name=EMBEDDINGS,
                metric_type="COSINE",
            )
            
            self.client.create_collection(
                collection_name=self.index_name,
                schema=schema,
                index_params=index_params
            )
    
    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
    ):
        """
        Add embeddings to the index.
        This method should be implemented by subclasses.
        """
        self.dimensions = self.dimensions or len(embeddings[0])
        self._initiate_collection()
        
        data = [
            {EMBEDDINGS: embedding, ROUTE: route, UTTERANCE: utterance}
            for embedding, route, utterance in zip(embeddings, routes, utterances)
        ]
        
        self.client.insert(collection_name=self.index_name, data=data)
        
    def get_routes(self) -> List[Tuple]:
        """
        Retrieves a list of routes and their associated utterances from the index.

        :returns: A list of tuples, each containing a route name and an associated utterance.
        :rtype: list[tuple]
        """
        
        res = self.client.query(
            collection_name=self.index_name,
            filter='route not in ["***"]',
            output_fields=[ROUTE, UTTERANCE],
            limit=OUTPUT_LIMIT,
        )
        
        res = list(res)
        route_utterance = [(group['route'], group['utterance']) for group in res]
        return route_utterance
        
        
    def delete(self, route_name: str):
        filter_route = f"route == {route_name}"
        self.client.delete(
            collection_name=self.index_name,
            filter=filter_route
        )
    
    def describe(self):
        return self.client.describe()

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Search the index for the query_vector and return top_k results.
        """
        
        vector = vector.tolist()
        if route_filter:
            filter_rule = "route in [" + ",".join(route_filter) + "]"
            res = self.client.search(
                collection_name=self.index_name,
                limit=top_k,
                data=vector,
                filter=filter_rule,
                search_params={"metric_type": "COSINE"},
                output_fields=[EMBEDDINGS, ROUTE]
            )        
            res = list(res)
            distances = [gp['distance'] for gp in res]
            routes = [gp[ROUTE] for gp in res]
            return distances, routes   
        else:
            res = self.client.search(
                collection_name=self.index_name,
                limit=top_k,
                data=vector,
                search_params={"metric_type": "COSINE"},
                output_fields=[EMBEDDINGS, ROUTE]
            )
            res = list(res)
            distances = [gp['distance'] for gp in res[0]]
            routes = [gp['entity'][ROUTE] for gp in res[0]]
            return distances, routes
        
    
      