import asyncio
import json
import time
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from semantic_router.route import Route
from semantic_router.schema import ConfigParameter, SparseEmbedding, Utterance
from semantic_router.utils.logger import logger

RETRY_WAIT_TIME = 2.5


class IndexConfig(BaseModel):
    type: str
    dimensions: int
    vectors: int


class BaseIndex(BaseModel):
    """
    Base class for indices using Pydantic's BaseModel.
    This class outlines the expected interface for index classes.
    Actual method implementations should be provided in subclasses.
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
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        **kwargs,
    ):
        """Add embeddings to the index.
        This method should be implemented by subclasses.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of routes to add to the index.
        :type routes: List[str]
        :param utterances: List of utterances to add to the index.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to add to the index.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to add to the index.
        :type metadata_list: List[Dict[str, Any]]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def aadd(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[Optional[List[Dict[str, Any]]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        **kwargs,
    ):
        """Add vectors to the index asynchronously.
        This method should be implemented by subclasses.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of routes to add to the index.
        :type routes: List[str]
        :param utterances: List of utterances to add to the index.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to add to the index.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to add to the index.
        :type metadata_list: List[Dict[str, Any]]
        """
        logger.warning("Async method not implemented.")
        return self.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances,
            function_schemas=function_schemas,
            metadata_list=metadata_list,
            **kwargs,
        )

    def get_utterances(self, include_metadata: bool = False) -> List[Utterance]:
        """Gets a list of route and utterance objects currently stored in the
        index, including additional metadata.

        :param include_metadata: Whether to include function schemas and metadata in
        the returned Utterance objects.
        :type include_metadata: bool
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if self.index is None:
            logger.warning("Index is None, could not retrieve utterances.")
            return []
        _, metadata = self._get_all(include_metadata=True)  # include_metadata required
        route_tuples = parse_route_info(metadata=metadata)
        if not include_metadata:
            # we remove the metadata from the tuples (ie only keep 0, 1 items)
            route_tuples = [x[:2] for x in route_tuples]
        return [Utterance.from_tuple(x) for x in route_tuples]

    async def aget_utterances(self, include_metadata: bool = False) -> List[Utterance]:
        """Gets a list of route and utterance objects currently stored in the
        index, including additional metadata.

        :param include_metadata: Whether to include function schemas and metadata in
        the returned Utterance objects.
        :type include_metadata: bool
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if self.index is None:
            logger.warning("Index is None, could not retrieve utterances.")
            return []
        _, metadata = await self._async_get_all(include_metadata=True)
        route_tuples = parse_route_info(metadata=metadata)
        if not include_metadata:
            # we remove the metadata from the tuples (ie only keep 0, 1 items)
            route_tuples = [x[:2] for x in route_tuples]
        return [Utterance.from_tuple(x) for x in route_tuples]

    def get_routes(self) -> List[Route]:
        """Gets a list of route objects currently stored in the index.

        :return: A list of Route objects.
        :rtype: List[Route]
        """
        utterances = self.get_utterances(include_metadata=True)
        routes_dict: Dict[str, Route] = {}
        # first create a dictionary of route names to Route objects
        for utt in utterances:
            # if the route is not in the dictionary, add it
            if utt.route not in routes_dict:
                routes_dict[utt.route] = Route(
                    name=utt.route,
                    utterances=[utt.utterance],
                    function_schemas=utt.function_schemas,
                    metadata=utt.metadata,
                )
            else:
                # otherwise, add the utterance to the route
                routes_dict[utt.route].utterances.append(utt.utterance)
        # then create a list of routes from the dictionary
        routes: List[Route] = []
        for route_name, route in routes_dict.items():
            routes.append(route)
        return routes

    def _remove_and_sync(self, routes_to_delete: dict):
        """
        Remove embeddings in a routes syncing process from the index.
        This method should be implemented by subclasses.

        :param routes_to_delete: Dictionary of routes to delete.
        :type routes_to_delete: dict
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def _async_remove_and_sync(self, routes_to_delete: dict):
        """
        Remove embeddings in a routes syncing process from the index asynchronously.
        This method should be implemented by subclasses.

        :param routes_to_delete: Dictionary of routes to delete.
        :type routes_to_delete: dict
        """
        logger.warning("Async method not implemented.")
        return self._remove_and_sync(routes_to_delete=routes_to_delete)

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
    ) -> Tuple[np.ndarray, List[str]]:
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
        :return: A tuple containing the query vector and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
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
        :return: A tuple containing the query vector and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def aget_routes(self):
        """
        Asynchronously get a list of route and utterance objects currently stored in the index.
        This method should be implemented by subclasses.

        :returns: A list of tuples, each containing a route name and an associated utterance.
        :rtype: list[tuple]
        :raises NotImplementedError: If the method is not implemented by the subclass.
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

    # ___________________________ CONFIG ___________________________
    # When implementing a new index, the following methods should be implemented
    # to enable synchronization of remote indexes.

    def _read_config(self, field: str, scope: str | None = None) -> ConfigParameter:
        """Read a config parameter from the index.

        :param field: The field to read.
        :type field: str
        :param scope: The scope to read.
        :type scope: str | None
        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        logger.warning("This method should be implemented by subclasses.")
        return ConfigParameter(
            field=field,
            value="",
            scope=scope,
        )

    async def _async_read_config(
        self, field: str, scope: str | None = None
    ) -> ConfigParameter:
        """Read a config parameter from the index asynchronously.

        :param field: The field to read.
        :type field: str
        :param scope: The scope to read.
        :type scope: str | None
        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        logger.warning("_async_read_config method not implemented.")
        return self._read_config(field=field, scope=scope)

    def _write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Write a config parameter to the index.

        :param config: The config parameter to write.
        :type config: ConfigParameter
        :return: The config parameter that was written.
        :rtype: ConfigParameter
        """
        logger.warning("This method should be implemented by subclasses.")
        return config

    async def _async_write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Write a config parameter to the index asynchronously.

        :param config: The config parameter to write.
        :type config: ConfigParameter
        :return: The config parameter that was written.
        :rtype: ConfigParameter
        """
        logger.warning("Async method not implemented.")
        return self._write_config(config=config)

    # _________________________ END CONFIG _________________________

    def _read_hash(self) -> ConfigParameter:
        """Read the hash of the previously written index.

        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        return self._read_config(field="sr_hash")

    async def _async_read_hash(self) -> ConfigParameter:
        """Read the hash of the previously written index asynchronously.

        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        return await self._async_read_config(field="sr_hash")

    def _is_locked(self, scope: str | None = None) -> bool:
        """Check if the index is locked for a given scope (if applicable).

        :param scope: The scope to check.
        :type scope: str | None
        :return: True if the index is locked, False otherwise.
        :rtype: bool
        """
        lock_config = self._read_config(field="sr_lock", scope=scope)
        if lock_config.value == "True":
            return True
        elif lock_config.value == "False" or not lock_config.value:
            return False
        else:
            raise ValueError(f"Invalid lock value: {lock_config.value}")

    async def _ais_locked(self, scope: str | None = None) -> bool:
        """Check if the index is locked for a given scope (if applicable).

        :param scope: The scope to check.
        :type scope: str | None
        :return: True if the index is locked, False otherwise.
        :rtype: bool
        """
        lock_config = await self._async_read_config(field="sr_lock", scope=scope)
        if lock_config.value == "True":
            return True
        elif lock_config.value == "False" or not lock_config.value:
            return False
        else:
            raise ValueError(f"Invalid lock value: {lock_config.value}")

    def lock(
        self, value: bool, wait: int = 0, scope: str | None = None
    ) -> ConfigParameter:
        """Lock/unlock the index for a given scope (if applicable). If index
        already locked/unlocked, raises ValueError.

        :param scope: The scope to lock.
        :type scope: str | None
        :param wait: The number of seconds to wait for the index to be unlocked, if
        set to 0, will raise an error if index is already locked/unlocked.
        :type wait: int
        :return: The config parameter that was locked.
        :rtype: ConfigParameter
        """
        start_time = datetime.now()
        while True:
            if self._is_locked(scope=scope) != value:
                # in this case, we can set the lock value
                break
            elif not value:
                # if unlocking, we can break immediately — often with Pinecone the
                # lock/unlocked state takes a few seconds to update, so locking then
                # unlocking quickly will fail without this check
                break
            if (datetime.now() - start_time).total_seconds() < wait:
                # wait for a few seconds before checking again
                time.sleep(RETRY_WAIT_TIME)
            else:
                raise ValueError(
                    f"Index is already {'locked' if value else 'unlocked'}."
                )
        lock_param = ConfigParameter(
            field="sr_lock",
            value=str(value),
            scope=scope,
        )
        self._write_config(lock_param)
        return lock_param

    async def alock(
        self, value: bool, wait: int = 0, scope: str | None = None
    ) -> ConfigParameter:
        """Lock/unlock the index for a given scope (if applicable). If index
        already locked/unlocked, raises ValueError.
        """
        start_time = datetime.now()
        while True:
            if await self._ais_locked(scope=scope) != value:
                # in this case, we can set the lock value
                break
            if (datetime.now() - start_time).total_seconds() < wait:
                # wait for a few seconds before checking again
                await asyncio.sleep(RETRY_WAIT_TIME)
            else:
                raise ValueError(
                    f"Index is already {'locked' if value else 'unlocked'}."
                )
        lock_param = ConfigParameter(
            field="sr_lock",
            value=str(value),
            scope=scope,
        )
        await self._async_write_config(lock_param)
        return lock_param

    def _get_all(self, prefix: Optional[str] = None, include_metadata: bool = False):
        """Retrieves all vector IDs from the index.
        This method should be implemented by subclasses.

        :param prefix: The prefix to filter the vectors by.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the response.
        :type include_metadata: bool
        :return: A tuple containing a list of vector IDs and a list of metadata dictionaries.
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
        :return: A tuple containing a list of vector IDs and a list of metadata dictionaries.
        :rtype: tuple[list[str], list[dict]]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def _async_get_routes(self) -> List[Tuple]:
        """Asynchronously gets a list of route and utterance objects currently
        stored in the index, including additional metadata.

        :return: A list of tuples, each containing route, utterance, function
        schema and additional metadata.
        :rtype: List[Tuple]
        """
        if self.index is None:
            logger.warning("Index is None, could not retrieve route info.")
            return []
        _, metadata = await self._async_get_all(include_metadata=True)
        route_info = parse_route_info(metadata=metadata)
        return route_info  # type: ignore

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

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


def parse_route_info(metadata: List[Dict[str, Any]]) -> List[Tuple]:
    """Parses metadata from index to extract route, utterance, function
    schema and additional metadata.

    :param metadata: List of metadata dictionaries.
    :type metadata: List[Dict[str, Any]]
    :return: A list of tuples, each containing route, utterance, function schema and additional metadata.
    :rtype: List[Tuple]
    """
    route_info = []
    for record in metadata:
        sr_route = record.get("sr_route", "")
        sr_utterance = record.get("sr_utterance", "")
        sr_function_schema = json.loads(record.get("sr_function_schema", "{}"))
        if sr_function_schema == {} or sr_function_schema == "null":
            sr_function_schema = None

        additional_metadata = {
            key: value
            for key, value in record.items()
            if key not in ["sr_route", "sr_utterance", "sr_function_schema"]
        }
        if additional_metadata is None:
            additional_metadata = {}
        # TODO: Not a fan of tuple packing here
        route_info.append(
            (sr_route, sr_utterance, sr_function_schema, additional_metadata)
        )
    return route_info
