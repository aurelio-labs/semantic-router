from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, overload

import numpy as np
from pydantic import ConfigDict, Field

from semantic_router.index.base import BaseIndex, IndexConfig, check_records
from semantic_router.linear import similarity_matrix, top_scores
from semantic_router.schema import SparseEmbedding, Utterance


class LocalIndex(BaseIndex):
    type: str = "local"
    metadata: Optional[np.ndarray] = Field(default=None, exclude=True)

    # Stop pydantic from complaining about Optional[np.ndarray]type hints.
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

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
        that is already stored has its embedding and metadata replaced.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of routes to add to the index.
        :type routes: List[str]
        :param utterances: List of utterances to add to the index.
        :type utterances: List[str]
        :param metadata_list: List of metadata to add to the index.
        :type metadata_list: Optional[List[Dict[str, Any]]]
        :param sparse_embeddings: Not supported, use HybridLocalIndex.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        if sparse_embeddings is not None:
            raise ValueError(
                f"{self.__class__.__name__} stores dense vectors only, use "
                "HybridLocalIndex for sparse embeddings."
            )
        self._upsert(embeddings, routes, utterances, metadata_list)

    async def aadd(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        **kwargs,
    ):
        """Upsert records into the index. Note that this just points to the sync add
        method as async makes no difference for the local computations of the
        LocalIndex.
        """
        self.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances,
            metadata_list=metadata_list,
            sparse_embeddings=sparse_embeddings,
        )

    def _upsert(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Writes records in place by (route, utterance), appending pairs that are not
        stored yet.

        :return: The row each record was written to.
        :rtype: List[int]
        """
        metadata_list = check_records(embeddings, routes, utterances, metadata_list)
        if not embeddings:
            return []
        vectors = list(self.index) if self.index is not None else []
        names = list(self.routes) if self.routes is not None else []
        texts = list(self.utterances) if self.utterances is not None else []
        metas = list(self.metadata) if self.metadata is not None else []
        # utterances may be non-string (e.g. images), so key on their string form
        rows = {(r, str(u)): i for i, (r, u) in enumerate(zip(names, texts))}
        positions = []
        for vector, route, utterance, metadata in zip(
            embeddings, routes, utterances, metadata_list
        ):
            key = (route, str(utterance))
            if key in rows:
                i = rows[key]
                vectors[i], texts[i], metas[i] = vector, utterance, metadata
            else:
                i = rows[key] = len(vectors)
                vectors.append(vector)
                names.append(route)
                texts.append(utterance)
                metas.append(metadata)
            positions.append(i)
        self.index = np.array(vectors)
        self.routes = np.array(names)
        self.utterances = (
            np.array(texts)
            if all(isinstance(t, str) for t in texts)
            else np.array(texts, dtype=object)
        )
        self.metadata = np.array(metas, dtype=object)
        return positions

    def get_utterances(self) -> List[Utterance]:
        """Gets every record stored in the index as an Utterance.

        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if self.routes is None or self.utterances is None or self.metadata is None:
            return []
        return [
            Utterance(route=route, utterance=utterance, metadata=meta)
            for route, utterance, meta in zip(
                self.routes, self.utterances, self.metadata
            )
        ]

    async def aget_utterances(self) -> List[Utterance]:
        """Gets every record stored in the index as an Utterance. Note that this just
        points to the sync get_utterances method.

        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        return self.get_utterances()

    def describe(self) -> IndexConfig:
        """Describe the index.

        :return: An IndexConfig object.
        :rtype: IndexConfig
        """
        return IndexConfig(
            type=self.type,
            dimensions=self.index.shape[1] if self.index is not None else 0,
            vectors=self.index.shape[0] if self.index is not None else 0,
        )

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self.index is not None and self.routes is not None

    async def ais_ready(self) -> bool:
        """Checks if the index is ready to be used asynchronously.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self.index is not None and self.routes is not None

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

        :param vector: The vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The routes to filter the search by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :param return_utterances: If True, also return the matched utterance texts.
        :type return_utterances: bool
        :return: ``(scores, route_names)`` by default, or ``(scores, route_names, utterances)``
            when ``return_utterances=True``.
        :rtype: Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]
        """
        if self.index is None or self.routes is None or self.utterances is None:
            raise ValueError("Index or routes are not populated.")
        if route_filter is not None:
            filtered_index = []
            filtered_routes = []
            filtered_utterances = []
            for route, vec, utt in zip(self.routes, self.index, self.utterances):
                if route in route_filter:
                    filtered_index.append(vec)
                    filtered_routes.append(route)
                    filtered_utterances.append(utt)
            if not filtered_routes:
                raise ValueError("No routes found matching the filter criteria.")
            sim = similarity_matrix(vector, np.array(filtered_index))
            scores, idx = top_scores(sim, top_k)
            route_names = [filtered_routes[i] for i in idx]
            utterances = [str(filtered_utterances[i]) for i in idx]
        else:
            sim = similarity_matrix(vector, self.index)
            scores, idx = top_scores(sim, top_k)
            route_names = [self.routes[i] for i in idx]
            utterances = [str(self.utterances[i]) for i in idx]
        if return_utterances:
            return scores, route_names, utterances
        return scores, route_names

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
        """Search the index for the query and return top_k results. Note that this
        just points to the sync query method, the computation is local and CPU-bound.

        :param vector: The vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The routes to filter the search by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :param return_utterances: If True, also return the matched utterance texts.
        :type return_utterances: bool
        :return: ``(scores, route_names)`` by default, or ``(scores, route_names, utterances)``
            when ``return_utterances=True``.
        :rtype: Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]
        """
        return self.query(
            vector=vector,
            top_k=top_k,
            route_filter=route_filter,
            sparse_vector=sparse_vector,
            return_utterances=return_utterances,
        )

    def delete(self, route_name: str):
        """Delete all records of a specific route from the index.

        :param route_name: The name of the route to delete.
        :type route_name: str
        """
        if self.routes is not None:
            self._keep(self.routes != route_name)

    async def adelete(self, route_name: str):
        """Delete all records of a specific route from the index. Note that this just points
        to the sync delete method as async makes no difference for the local computations
        of the LocalIndex.

        :param route_name: The name of the route to delete.
        :type route_name: str
        """
        self.delete(route_name)

    def _keep(self, mask: np.ndarray):
        """Keeps only the rows where mask is True.

        :param mask: A boolean mask, one entry per row.
        :type mask: np.ndarray
        """
        if self.index is not None:
            self.index = self.index[mask]
        if self.routes is not None:
            self.routes = self.routes[mask]
        if self.utterances is not None:
            self.utterances = self.utterances[mask]
        if self.metadata is not None:
            self.metadata = self.metadata[mask]

    def delete_index(self):
        """Deletes the index, effectively clearing it and setting it to None.

        :return: None
        :rtype: None
        """
        self.index = None
        self.routes = None
        self.utterances = None
        self.metadata = None

    def delete_all(self):
        """Deletes all records from the index.

        :return: None
        :rtype: None
        """
        self.delete_index()

    async def adelete_index(self):
        """Deletes the index, effectively clearing it and setting it to None. Note that this just points
        to the sync delete_index method as async makes no difference for the local computations
        of the LocalIndex.

        :return: None
        :rtype: None
        """
        self.delete_index()

    def __len__(self):
        if self.index is not None:
            return self.index.shape[0]
        else:
            return 0
