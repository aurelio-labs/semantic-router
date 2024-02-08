from pydantic import BaseModel, Field
import os
import pinecone
from typing import Any, List, Tuple
from semantic_router.indices.base import BaseIndex
import numpy as np

class PineconeIndex(BaseIndex):
    index_name: str
    dimension: int = 768
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-west-2" 
    pinecone: Any = Field(default=None, exclude=True)
    vector_id_counter: int = -1

    def __init__(self, **data):
        super().__init__(**data) 
        # Initialize Pinecone environment with the new API
        self.pinecone = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Create or connect to an existing Pinecone index
        if self.index_name not in self.pinecone.list_indexes().names():
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pinecone.create_index(
                name=self.index_name, 
                dimension=self.dimension, 
                metric=self.metric,
                spec=pinecone.ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )
        self.index = self.pinecone.Index(self.index_name)
        
    def add(self, embeds: List[List[float]]):
        # Format embeds as a list of dictionaries for Pinecone's upsert method
        vectors_to_upsert = []
        for vector in embeds:
            self.vector_id_counter += 1  # Increment the counter for each new vector
            vector_id = str(self.vector_id_counter)  # Convert counter to string ID

            # Prepare for upsert
            vectors_to_upsert.append({"id": vector_id, "values": vector})

        # Perform the upsert operation
        self.index.upsert(vectors=vectors_to_upsert)

    def remove(self, ids_to_remove: List[str]):
        self.index.delete(ids=ids_to_remove)

    def remove_all(self):
        self.index.delete(delete_all=True)

    def is_index_populated(self) -> bool:
        stats = self.index.describe_index_stats()
        return stats["dimension"] > 0 and stats["total_vector_count"] > 0
    
    def query(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        query_vector_list = query_vector.tolist()
        results = self.index.query(vector=[query_vector_list], top_k=top_k)
        ids = [int(result["id"]) for result in results["matches"]]
        scores = [result["score"] for result in results["matches"]]
        # DEBUGGING: Start.
        print('#'*50)
        print('ids')
        print(ids)
        print('#'*50)
        # DEBUGGING: End.
        # DEBUGGING: Start.
        print('#'*50)
        print('scores')
        print(scores)
        print('#'*50)
        # DEBUGGING: End.
        return np.array(scores), np.array(ids)

    def delete_index(self):
        pinecone.delete_index(self.index_name)