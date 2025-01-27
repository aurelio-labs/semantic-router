from typing import List, Optional, Tuple, Dict

import numpy as np

from semantic_router.schema import ConfigParameter, SparseEmbedding, Utterance
from semantic_router.index.base import BaseIndex, IndexConfig
from semantic_router.linear import similarity_matrix, top_scores
from semantic_router.utils.logger import logger
from typing import Any


class LocalIndex(BaseIndex):
    type: str = "local"

    def __init__(self):
        super().__init__()

    class Config:
        # Stop pydantic from complaining about Optional[np.ndarray]type hints.
        arbitrary_types_allowed = True

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        **kwargs,
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

    def _remove_and_sync(self, routes_to_delete: dict) -> np.ndarray:
        if self.index is None or self.routes is None or self.utterances is None:
            raise ValueError("Index, routes, or utterances are not populated.")
        # TODO JB: implement routes and utterances as a numpy array
        route_utterances = np.array([self.routes, self.utterances]).T
        # initialize our mask with all true values (ie keep all)
        mask = np.ones(len(route_utterances), dtype=bool)
        for route, utterances in routes_to_delete.items():
            # TODO JB: we should be able to vectorize this?
            for utterance in utterances:
                mask &= ~(
                    (route_utterances[:, 0] == route)
                    & (route_utterances[:, 1] == utterance)
                )
        # apply the mask to index, routes, and utterances
        self.index = self.index[mask]
        self.routes = self.routes[mask]
        self.utterances = self.utterances[mask]
        # return what was removed
        return route_utterances[~mask]

    def get_utterances(self, include_metadata: bool = False) -> List[Utterance]:
        """Gets a list of route and utterance objects currently stored in the index.

        :param include_metadata: Whether to include function schemas and metadata in
        the returned Utterance objects - LocalIndex doesn't include metadata so this
        parameter is ignored.
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if self.routes is None or self.utterances is None:
            return []
        return [Utterance.from_tuple(x) for x in zip(self.routes, self.utterances)]

    def describe(self) -> IndexConfig:
        return IndexConfig(
            type=self.type,
            dimensions=self.index.shape[1] if self.index is not None else 0,
            vectors=self.index.shape[0] if self.index is not None else 0,
        )

    def is_ready(self) -> bool:
        """
        Checks if the index is ready to be used.
        """
        return self.index is not None and self.routes is not None

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
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

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
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

    def aget_routes(self):
        logger.error("Sync remove is not implemented for LocalIndex.")

    def _write_config(self, config: ConfigParameter):
        logger.warning("No config is written for LocalIndex.")

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
        self.routes = None
        self.utterances = None

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
