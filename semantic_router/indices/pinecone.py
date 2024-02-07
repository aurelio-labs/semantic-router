import os
import pinecone
import numpy as np
from typing import List, Tuple
from semantic_router.indices.base import BaseIndex


class PineconeIndex(BaseIndex):
    def __init__(self, index_name: str, environment: str = 'us-west1-gcp', metric: str = 'cosine', dimension: int = 768):
        super().__init__()
        
        # Initialize Pinecone environment
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=environment)
        
        # Create or connect to an existing Pinecone index
        if index_name not in pinecone.list_indexes():
            print(f"Creating new Pinecone index: {index_name}")
            pinecone.create_index(name=index_name, metric=metric, dimension=dimension)
        self.index = pinecone.Index(index_name)
        
        # Store the index name for potential deletion
        self.index_name = index_name

    def add(self, embeds: List[np.ndarray]):
        # Assuming embeds is a list of tuples (id, vector)
        self.index.upsert(vectors=embeds)

    def remove(self, ids_to_remove: List[str]):
        self.index.delete(ids=ids_to_remove)

    def is_index_populated(self) -> bool:
        stats = self.index.describe_index_stats()
        return stats["dimension"] > 0 and stats["index_size"] > 0

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        results = self.index.query(queries=[query_vector], top_k=top_k)
        ids = [result["id"] for result in results["matches"]]
        scores = [result["score"] for result in results["matches"]]
        return np.array(ids), np.array(scores)

    def delete_index(self):
        """
        Deletes the Pinecone index.
        """
        pinecone.delete_index(self.index_name)