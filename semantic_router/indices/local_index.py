import numpy as np
from typing import List, Any, Tuple, Optional
from semantic_router.linear import similarity_matrix, top_scores
from semantic_router.indices.base import BaseIndex


class LocalIndex(BaseIndex):

    class Config:  # Stop pydantic from complaining about  Optional[np.ndarray] type hints.
        arbitrary_types_allowed = True

    def add(self, embeds: List[Any]):
        embeds = np.array(embeds)  # type: ignore
        if self.index is None:
            self.index = embeds  # type: ignore
        else:
            self.index = np.concatenate([self.index, embeds])

    def remove(self, indices_to_remove: List[int]):
        """
        Remove all items of a specific category from the index.
        """
        if self.index is not None:
            self.index = np.delete(self.index, indices_to_remove, axis=0)

    def is_index_populated(self):
        return self.index is not None and len(self.index) > 0

    def query(self, query_vector: Any, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
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
