from typing import List, Optional, Tuple, Dict

import numpy as np

from semantic_router.index.base import BaseIndex
from semantic_router.linear import similarity_matrix, top_scores


class LocalIndex(BaseIndex):
    def __init__(
        self,
        index: Optional[np.ndarray] = None,
        routes: Optional[np.ndarray] = None,
        utterances: Optional[np.ndarray] = None,
    ):
        super().__init__(index=index, routes=routes, utterances=utterances)
        self.type = "local"

    class Config:
        # Stop pydantic from complaining about Optional[np.ndarray]type hints.
        arbitrary_types_allowed = True

    def add(
        self, embeddings: List[List[float]], routes: List[str], utterances: List[str]
    ):
        embeds = np.array(embeddings)  # type: ignore
        routes_arr = np.array(routes)
        if isinstance(utterances[0], str):
            utterances_arr = np.array(utterances)
        else:
            utterances_arr = np.array(utterances, dtype=object)
        if self.index is None:
            self.index = embeds  # type: ignore
            self.routes = routes_arr
            self.utterances = utterances_arr
        else:
            self.index = np.concatenate([self.index, embeds])
            self.routes = np.concatenate([self.routes, routes_arr])
            self.utterances = np.concatenate([self.utterances, utterances_arr])

    def get_routes(self) -> List[Tuple]:
        """
        Gets a list of route and utterance objects currently stored in the index.

        Returns:
            List[Tuple]: A list of (route_name, utterance) objects.
        """
        if self.routes is None or self.utterances is None:
            raise ValueError("No routes have been added to the index.")
        return list(zip(self.routes, self.utterances))

    def describe(self) -> Dict:
        return {
            "type": self.type,
            "dimensions": self.index.shape[1] if self.index is not None else 0,
            "vectors": self.index.shape[0] if self.index is not None else 0,
        }

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Search the index for the query and return top_k results.
        """
        if self.index is None or self.routes is None:
            raise ValueError("Index or routes are not populated.")
        if route_filter is not None:
            filtered_index = []
            filtered_routes = []
            for route, vec in zip(self.routes, self.index):
                if route in route_filter:
                    filtered_index.append(vec)
                    filtered_routes.append(route)
            if not filtered_routes:
                raise ValueError("No routes found matching the filter criteria.")
            sim = similarity_matrix(vector, np.array(filtered_index))
            scores, idx = top_scores(sim, top_k)
            route_names = [filtered_routes[i] for i in idx]
        else:
            sim = similarity_matrix(vector, self.index)
            scores, idx = top_scores(sim, top_k)
            route_names = [self.routes[i] for i in idx]
        return scores, route_names

    def delete(self, route_name: str):
        """
        Delete all records of a specific route from the index.
        """
        if (
            self.index is not None
            and self.routes is not None
            and self.utterances is not None
        ):
            delete_idx = self._get_indices_for_route(route_name=route_name)
            self.index = np.delete(self.index, delete_idx, axis=0)
            self.routes = np.delete(self.routes, delete_idx, axis=0)
            self.utterances = np.delete(self.utterances, delete_idx, axis=0)
        else:
            raise ValueError(
                "Attempted to delete route records but either index, routes or "
                "utterances is None."
            )

    def delete_index(self):
        """
        Deletes the index, effectively clearing it and setting it to None.
        """
        self.index = None

    def _get_indices_for_route(self, route_name: str):
        """Gets an array of indices for a specific route."""
        if self.routes is None:
            raise ValueError("Routes are not populated.")
        idx = [i for i, route in enumerate(self.routes) if route == route_name]
        return idx

    def __len__(self):
        if self.index is not None:
            return self.index.shape[0]
        else:
            return 0
