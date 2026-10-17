from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from semantic_router.route import Route
from semantic_router.schema import SparseEmbedding, Utterance
from semantic_router.utils.logger import logger

# Reserved record metadata keys. Every key starting with ``sr_`` belongs to semantic
# router: ``sr_route`` and ``sr_utterance`` identify a record, and the route-level
# fields ride on every record of a route so the index alone can rebuild each Route.
SR_PREFIX = "sr_"
SR_ROUTE = "sr_route"
SR_UTTERANCE = "sr_utterance"
SR_DESCRIPTION = "sr_description"
SR_CONFIDENCE_THRESHOLD = "sr_confidence_threshold"


class IndexConfig(BaseModel):
    type: str
    dimensions: int
    vectors: int


class BaseIndex(BaseModel):
    """
    Base class for indices using Pydantic's BaseModel.
    This class outlines the expected interface for index classes.
    Actual method implementations should be provided in subclasses.

    The index is the single source of truth for routes. Each record is one
    (route, utterance) pair with a deterministic id, so adding the same pair again
    replaces the record instead of duplicating it.
    """

    # You can define common attributes here if there are any.
    # For example, a placeholder for the index attribute:
    routes: Optional[np.ndarray] = None
    utterances: Optional[np.ndarray] = None
    dimensions: Union[int, None] = None
    type: str = "base"
    init_async_index: bool = False
    index: Optional[Any] = None

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        **kwargs,
    ):
        """Upsert records into the index, one per (route, utterance) pair.
        This method should be implemented by subclasses.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of route names, one per embedding.
        :type routes: List[str]
        :param utterances: List of utterances, one per embedding.
        :type utterances: List[str]
        :param metadata_list: List of metadata dicts, one per embedding. Holds the
            route's own metadata and the reserved ``sr_description`` and
            ``sr_confidence_threshold`` keys.
        :type metadata_list: Optional[List[Dict[str, Any]]]
        :param sparse_embeddings: List of sparse embeddings, one per embedding. Only
            hybrid indexes store them.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def aadd(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        **kwargs,
    ):
        """Upsert records into the index asynchronously.
        This method should be implemented by subclasses.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of route names, one per embedding.
        :type routes: List[str]
        :param utterances: List of utterances, one per embedding.
        :type utterances: List[str]
        :param metadata_list: List of metadata dicts, one per embedding.
        :type metadata_list: Optional[List[Dict[str, Any]]]
        :param sparse_embeddings: List of sparse embeddings, one per embedding.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        logger.warning("Async method not implemented.")
        return self.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances,
            metadata_list=metadata_list,
            sparse_embeddings=sparse_embeddings,
            **kwargs,
        )

    def get_utterances(self) -> List[Utterance]:
        """Gets every record stored in the index as an Utterance.

        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        _, metadata = self._get_all(include_metadata=True)
        return [to_utterance(record) for record in metadata]

    async def aget_utterances(self) -> List[Utterance]:
        """Gets every record stored in the index as an Utterance, asynchronously.

        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        _, metadata = await self._async_get_all(include_metadata=True)
        return [to_utterance(record) for record in metadata]

    def get_routes(self) -> List[Route]:
        """Rebuilds the routes stored in the index from its records.

        :return: A list of Route objects.
        :rtype: List[Route]
        """
        return to_routes(self.get_utterances())

    async def aget_routes(self) -> List[Route]:
        """Rebuilds the routes stored in the index from its records, asynchronously.

        :return: A list of Route objects.
        :rtype: List[Route]
        """
        return to_routes(await self.aget_utterances())

    def delete(self, route_name: str):
        """Deletes route by route name.
        This method should be implemented by subclasses.

        :param route_name: Name of the route to delete.
        :type route_name: str
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def adelete(self, route_name: str) -> list[str]:
        """Asynchronously delete specified route from index if it exists. Returns the IDs
        of the vectors deleted.
        This method should be implemented by subclasses.

        :param route_name: Name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def describe(self) -> IndexConfig:
        """Returns an IndexConfig object with index details such as type, dimensions,
        and total vector count.
        This method should be implemented by subclasses.

        :return: An IndexConfig object.
        :rtype: IndexConfig
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.
        This method should be implemented by subclasses.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def ais_ready(self) -> bool:
        """Checks if the index is ready to be used asynchronously.
        This method should be implemented by subclasses.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: bool = False,
    ) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]:
        """Search the index for the query_vector and return top_k results.
        This method should be implemented by subclasses.

        :param vector: The vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The routes to filter the search by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :param return_utterances: If True, also return the matched utterance texts
            alongside scores and route names.
        :type return_utterances: bool
        :return: ``(scores, route_names)`` by default, or ``(scores, route_names, utterances)``
            when ``return_utterances=True``.
        :rtype: Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
        return_utterances: bool = False,
    ) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]:
        """Search the index for the query_vector and return top_k results.
        This method should be implemented by subclasses.

        :param vector: The vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The routes to filter the search by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :param return_utterances: If True, also return the matched utterance texts
            alongside scores and route names.
        :type return_utterances: bool
        :return: ``(scores, route_names)`` by default, or ``(scores, route_names, utterances)``
            when ``return_utterances=True``.
        :rtype: Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], List[str]]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def delete_all(self):
        """Deletes all records from the index.
        This method should be implemented by subclasses.

        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        logger.warning("This method should be implemented by subclasses.")
        self.index = None
        self.routes = None
        self.utterances = None

    def delete_index(self):
        """Deletes or resets the index.
        This method should be implemented by subclasses.

        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        logger.warning("This method should be implemented by subclasses.")
        self.index = None

    async def adelete_index(self):
        """Deletes or resets the index asynchronously.
        This method should be implemented by subclasses.
        """
        logger.warning("This method should be implemented by subclasses.")
        self.index = None

    def _get_all(self, prefix: Optional[str] = None, include_metadata: bool = False):
        """Retrieves all vector IDs from the index.
        This method should be implemented by subclasses.

        :param prefix: The prefix to filter the vectors by.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the response.
        :type include_metadata: bool
        :return: A tuple containing a list of vector IDs and a list of metadata
            dictionaries, each with the ``sr_route`` and ``sr_utterance`` keys.
        :rtype: tuple[list[str], list[dict]]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def _async_get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> tuple[list[str], list[dict]]:
        """Retrieves all vector IDs from the index asynchronously.
        This method should be implemented by subclasses.

        :param prefix: The prefix to filter the vectors by.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the response.
        :type include_metadata: bool
        :return: A tuple containing a list of vector IDs and a list of metadata
            dictionaries, each with the ``sr_route`` and ``sr_utterance`` keys.
        :rtype: tuple[list[str], list[dict]]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    # an unknown keyword (e.g. collection_name for index_name) is an error, not ignored
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )

    def _init_index(self, force_create: bool = False) -> Union[Any, None]:
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        This method must be implemented by subclasses.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def _init_async_index(self, force_create: bool = False):
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        This method is used to initialize the index asynchronously.

        This method must be implemented by subclasses.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __len__(self):
        """Returns the total number of vectors in the index. If the index is not initialized
        returns 0.

        :return: The total number of vectors.
        :rtype: int
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def alen(self):
        """Async version of __len__. Returns the total number of vectors in the index.
        Default implementation just calls the sync version.

        :return: The total number of vectors.
        :rtype: int
        """
        return len(self)


def check_records(
    embeddings: List[List[float]],
    routes: List[str],
    utterances: List[Any],
    metadata_list: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Checks the lists passed to ``add`` line up, and returns one metadata dict per
    record (a copy, so an index never mutates the caller's dicts).

    :param embeddings: List of embeddings.
    :type embeddings: List[List[float]]
    :param routes: List of route names.
    :type routes: List[str]
    :param utterances: List of utterances.
    :type utterances: List[Any]
    :param metadata_list: List of metadata dicts, or None for empty metadata.
    :type metadata_list: Optional[List[Dict[str, Any]]]
    :return: One metadata dict per record.
    :rtype: List[Dict[str, Any]]
    """
    if not len(embeddings) == len(routes) == len(utterances):
        raise ValueError(
            f"embeddings ({len(embeddings)}), routes ({len(routes)}) and utterances "
            f"({len(utterances)}) must be the same length."
        )
    if metadata_list is None:
        return [{} for _ in embeddings]
    if len(metadata_list) != len(embeddings):
        raise ValueError(
            f"metadata_list length ({len(metadata_list)}) must match embeddings "
            f"length ({len(embeddings)})."
        )
    return [dict(metadata or {}) for metadata in metadata_list]


def to_utterance(record: Dict[str, Any]) -> Utterance:
    """Builds an Utterance from a record's metadata, which holds the ``sr_route`` and
    ``sr_utterance`` keys next to the rest of the record's metadata.

    :param record: The record's metadata.
    :type record: Dict[str, Any]
    :return: An Utterance object.
    :rtype: Utterance
    """
    metadata = {k: v for k, v in record.items() if k not in (SR_ROUTE, SR_UTTERANCE)}
    return Utterance(
        route=record.get(SR_ROUTE, ""),
        utterance=record.get(SR_UTTERANCE, ""),
        metadata=metadata,
    )


def to_routes(utterances: List[Utterance]) -> List[Route]:
    """Rebuilds routes from index records, in the order each route is first seen.
    The route-level fields come from the reserved ``sr_`` keys of the route's first
    record, and every ``sr_`` key is stripped from the route's metadata.

    :param utterances: The records, as returned by ``get_utterances``.
    :type utterances: List[Utterance]
    :return: A list of Route objects.
    :rtype: List[Route]
    """
    routes: Dict[str, Route] = {}
    for utt in utterances:
        if utt.route in routes:
            routes[utt.route].utterances.append(utt.utterance)
            continue
        routes[utt.route] = Route(
            name=utt.route,
            utterances=[utt.utterance],
            description=utt.metadata.get(SR_DESCRIPTION),
            confidence_threshold=utt.metadata.get(SR_CONFIDENCE_THRESHOLD),
            metadata={
                k: v for k, v in utt.metadata.items() if not k.startswith(SR_PREFIX)
            },
        )
    return list(routes.values())
