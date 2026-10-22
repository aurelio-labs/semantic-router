from typing import Any, Dict, List, Literal, Optional, Tuple, overload

import numpy as np
from numpy.linalg import norm

from semantic_router.index.local import LocalIndex
from semantic_router.schema import SparseEmbedding
from semantic_router.utils.logger import logger


class HybridLocalIndex(LocalIndex):
    type: str = "hybrid_local"
    sparse_index: Optional[list[dict]] = None
    route_names: Optional[np.ndarray] = None

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        **kwargs,
    ):
        """Upsert records into the index, one per (route, utterance) pair. A pair
        that is already stored has its embeddings and metadata replaced.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of routes to add to the index.
        :type routes: List[str]
        :param utterances: List of utterances to add to the index.
        :type utterances: List[str]
        :param metadata_list: List of metadata to add to the index.
        :type metadata_list: Optional[List[Dict[str, Any]]]
        :param sparse_embeddings: List of sparse embeddings to add to the index.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        if sparse_embeddings is None:
            raise ValueError("Sparse embeddings are required for HybridLocalIndex.")
        if len(sparse_embeddings) != len(embeddings):
            raise ValueError(
                f"sparse_embeddings length ({len(sparse_embeddings)}) must match "
                f"embeddings length ({len(embeddings)})."
            )
        positions = self._upsert(embeddings, routes, utterances, metadata_list)
        # TODO: switch back to using SparseEmbedding later
        sparse_index = list(self.sparse_index or [])
        sparse_index.extend({} for _ in range(len(self) - len(sparse_index)))
        for i, sparse in zip(positions, sparse_embeddings):
            sparse_index[i] = sparse.to_dict()
        self.sparse_index = sparse_index

    def _sparse_dot_product(
        self, vec_a: dict[int, float], vec_b: dict[int, float]
    ) -> float:
        """Calculate the dot product of two sparse vectors.

        :param vec_a: The first sparse vector.
        :type vec_a: dict[int, float]
        :param vec_b: The second sparse vector.
        :type vec_b: dict[int, float]
        :return: The dot product of the two sparse vectors.
        :rtype: float
        """
        # switch vecs to ensure first is smallest for more efficiency
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        return sum(vec_a[i] * vec_b.get(i, 0) for i in vec_a)

    def _sparse_index_dot_product(self, vec_a: dict[int, float]) -> list[float]:
        """Calculate the dot product of a sparse vector and a list of sparse vectors.

        :param vec_a: The sparse vector.
        :type vec_a: dict[int, float]
        :return: A list of dot products.
        :rtype: list[float]
        """
        if self.sparse_index is None:
            raise ValueError("self.sparse_index is not populated.")
        dot_products = [
            self._sparse_dot_product(vec_a, vec_b) for vec_b in self.sparse_index
        ]
        return dot_products

    @overload
    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: Literal[False] = False,
    ) -> Tuple[np.ndarray, List[str]]: ...

    @overload
    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: Literal[True] = ...,
    ) -> Tuple[np.ndarray, List[str], List[str]]: ...

    @overload
    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: bool = ...,
    ) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]: ...
    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: bool = False,
    ) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]:
        """Search the index for the query and return top_k results.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param sparse_vector: The sparse vector to search for, must be provided.
        :type sparse_vector: dict[int, float]
        :param return_utterances: If True, also return the matched utterance texts.
        :type return_utterances: bool
        """
        if route_filter:
            raise ValueError("Route filter is not supported for HybridLocalIndex.")

        xq_d = vector.copy()
        # align sparse vector type
        if isinstance(sparse_vector, SparseEmbedding):
            xq_s = sparse_vector.to_dict()
        elif isinstance(sparse_vector, dict):
            xq_s = sparse_vector
        else:
            raise ValueError("Sparse vector must be a SparseEmbedding or dict.")

        if self.index is not None and self.sparse_index is not None:
            # calculate dense vec similarity. only the stored vectors are normalised:
            # the caller scales the query by its dense weight (alpha), and dividing by
            # the query's norm would cancel that weight
            index_norm = norm(self.index, axis=1)
            sim_d = np.squeeze(np.dot(self.index, xq_d.T)) / index_norm
            # calculate sparse vec similarity
            sim_s = np.array(self._sparse_index_dot_product(xq_s))
            total_sim = sim_d + sim_s
            # get indices of top_k records
            top_k = min(top_k, total_sim.shape[0])
            idx = np.argpartition(total_sim, -top_k)[-top_k:]
            scores = total_sim[idx]
            # get the utterance categories (route names)
            route_names = (
                [self.routes[i] for i in idx] if self.routes is not None else []
            )
            utterances = (
                [str(self.utterances[i]) for i in idx]
                if self.utterances is not None
                else []
            )
            if return_utterances:
                return scores, route_names, utterances
            return scores, route_names
        else:
            logger.warning("Index or sparse index is not populated.")
            empty_scores = np.array([])
            empty_routes: list[str] = []
            if return_utterances:
                return empty_scores, empty_routes, []
            return empty_scores, empty_routes

    @overload
    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: Literal[False] = False,
    ) -> Tuple[np.ndarray, List[str]]: ...

    @overload
    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: Literal[True] = ...,
    ) -> Tuple[np.ndarray, List[str], List[str]]: ...

    @overload
    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: bool = ...,
    ) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]: ...
    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: bool = False,
    ) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]:
        """Search the index for the query and return top_k results. This method calls the
        sync `query` method as everything uses numpy computations which is CPU-bound
        and so no benefit can be gained from making this async.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param sparse_vector: The sparse vector to search for, must be provided.
        :type sparse_vector: dict[int, float]
        :param return_utterances: If True, also return the matched utterance texts.
        :type return_utterances: bool
        """
        return self.query(
            vector=vector,
            top_k=top_k,
            route_filter=route_filter,
            sparse_vector=sparse_vector,
            return_utterances=return_utterances,
        )

    def _keep(self, mask: np.ndarray):
        """Keeps only the rows where mask is True, sparse rows included.

        :param mask: A boolean mask, one entry per row.
        :type mask: np.ndarray
        """
        super()._keep(mask)
        if self.sparse_index is not None:
            self.sparse_index = [s for s, keep in zip(self.sparse_index, mask) if keep]

    def delete_index(self):
        """Deletes the index, effectively clearing it and setting it to None.

        :return: None
        :rtype: None
        """
        super().delete_index()
        self.sparse_index = None

    def __len__(self):
        if self.index is not None:
            return self.index.shape[0]
        else:
            return 0
