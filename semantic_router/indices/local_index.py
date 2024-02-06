import numpy as np
from typing import List, Any
from .base import BaseIndex

class LocalIndex(BaseIndex):
    """
    Local index implementation using numpy arrays.
    """

    def __init__(self):
        self.index = None
        self.categories = None

    def add(self, items: List[Any], categories: List[str]):
        """
        Add items to the index with their corresponding categories.
        """
        embeds = np.array(items)
        if self.index is None:
            self.index = embeds
            self.categories = np.array(categories)
        else:
            self.index = np.concatenate([self.index, embeds])
            self.categories = np.concatenate([self.categories, np.array(categories)])

    def remove(self, category: str):
        """
        Remove all items of a specific category from the index.
        """
        if self.categories is not None:
            indices_to_remove = np.where(self.categories == category)[0]
            self.index = np.delete(self.index, indices_to_remove, axis=0)
            self.categories = np.delete(self.categories, indices_to_remove, axis=0)

    def search(self, query: Any, top_k: int = 5) -> List[Any]:
        """
        Search the index for the query and return top_k results.
        """
        if self.index is None:
            return []
        sim = np.dot(self.index, query) / (np.linalg.norm(self.index, axis=1) * np.linalg.norm(query))
        idx = np.argsort(sim)[-top_k:]
        return [(self.categories[i], sim[i]) for i in idx[::-1]]