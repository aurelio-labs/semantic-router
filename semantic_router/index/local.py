import numpy as np
from typing import List, Any, Tuple, Optional
from semantic_router.linear import similarity_matrix, top_scores
from semantic_router.index.base import BaseIndex


class LocalIndex(BaseIndex):

    def __init__(self):
        super().__init__() 
        self.type = "local"

    class Config:  # Stop pydantic from complaining about  Optional[np.ndarray] type hints.
        arbitrary_types_allowed = True

    def add(self, embeddings: List[List[float]], routes: List[str], utterances: List[str]):
        embeds = np.array(embeddings)  # type: ignore
        if self.index is None:
            self.index = embeds  # type: ignore
        else:
            self.index = np.concatenate([self.index, embeds])

    def delete(self, indices_to_remove: List[int]):
        """
        Remove all items of a specific category from the index.
        """
        if self.index is not None:
            self.index = np.delete(self.index, indices_to_remove, axis=0)

    def describe(self):
        return {
            "type": self.type,
            "dimensions": self.index.shape[1] if self.index is not None else 0,
            "vectors": self.index.shape[0] if self.index is not None else 0
        }

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[str]]:
        """
        Search the index for the query and return top_k results.
        """
        if self.index is None:
            raise ValueError("Index is not populated.")
        sim = similarity_matrix(query_vector, self.index)
        return top_scores(sim, top_k)
    
    def delete_index(self):
        """
        Deletes the index, effectively clearing it and setting it to None.
        """
        self.index = None
