from typing import Any, List, Optional, Tuple, Union, Dict
import json

import numpy as np
from pydantic.v1 import BaseModel

from semantic_router.schema import ConfigParameter, Utterance
from semantic_router.route import Route
from semantic_router.utils.logger import logger


class BaseIndex(BaseModel):
    """
    Base class for indices using Pydantic's BaseModel.
    This class outlines the expected interface for index classes.
    Actual method implementations should be provided in subclasses.
    """

    # You can define common attributes here if there are any.
    # For example, a placeholder for the index attribute:
    index: Optional[Any] = None
    routes: Optional[np.ndarray] = None
    utterances: Optional[np.ndarray] = None
    dimensions: Union[int, None] = None
    type: str = "base"
    init_async_index: bool = False

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
    ):
        """
        Add embeddings to the index.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_utterances(self) -> List[Utterance]:
        """Gets a list of route and utterance objects currently stored in the
        index, including additional metadata.

        :return: A list of tuples, each containing route, utterance, function
        schema and additional metadata.
        :rtype: List[Tuple]
        """
        if self.index is None:
            logger.warning("Index is None, could not retrieve utterances.")
            return []
        _, metadata = self._get_all(include_metadata=True)
        route_tuples = parse_route_info(metadata=metadata)
        return [Utterance.from_tuple(x) for x in route_tuples]

    def get_routes(self) -> List[Route]:
        """Gets a list of route objects currently stored in the index.

        :return: A list of Route objects.
        :rtype: List[Route]
        """
        utterances = self.get_utterances()
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
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def delete(self, route_name: str):
        """
        Deletes route by route name.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def describe(self) -> Dict:
        """
        Returns a dictionary with index details such as type, dimensions, and total
        vector count.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Search the index for the query_vector and return top_k results.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Search the index for the query_vector and return top_k results.
        This method should be implemented by subclasses.
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
        """
        Deletes all records from the index.
        """
        logger.warning("This method should be implemented by subclasses.")
        self.index = None
        self.routes = None
        self.utterances = None

    def delete_index(self):
        """
        Deletes or resets the index.
        This method should be implemented by subclasses.
        """
        logger.warning("This method should be implemented by subclasses.")
        self.index = None

    def _read_hash(self) -> ConfigParameter:
        """
        Read the hash of the previously written index.

        This method should be implemented by subclasses.
        """
        logger.warning("This method should be implemented by subclasses.")
        return ConfigParameter(
            field="sr_hash",
            value="",
            namespace="",
        )

    def _write_config(self, config: ConfigParameter):
        """
        Write a config parameter to the index.

        This method should be implemented by subclasses.
        """
        logger.warning("This method should be implemented by subclasses.")

    def _get_all(self, prefix: Optional[str] = None, include_metadata: bool = False):
        """
        Retrieves all vector IDs from the index.

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

    class Config:
        arbitrary_types_allowed = True


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
