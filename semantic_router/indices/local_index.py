import numpy as np
from typing import List, Any
from .base import BaseIndex
from semantic_router.linear import similarity_matrix, top_scores
from typing import Tuple

class LocalIndex(BaseIndex):
    """
    Local index implementation using numpy arrays.
    """

    def __init__(self):
        self.index = None

    def add(self, embeds: List[Any]):
        """
        Add items to the index.
        """
        embeds = np.array(embeds)
        if self.index is None:
            self.index = embeds
        else:
            self.index = np.concatenate([self.index, embeds])

    def remove(self, indices_to_remove: List[int]):
        """
        Remove all items of a specific category from the index.
        """
        self.index = np.delete(self.index, indices_to_remove, axis=0)

    def is_index_populated(self):
        return self.index is not None and len(self.index) > 0

    def search(self, query_vector: Any, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the query and return top_k results.
        """
        if self.index is None:
            raise ValueError("Index is not populated.")
        sim = similarity_matrix(query_vector, self.index)
        return top_scores(sim, top_k)
                          
