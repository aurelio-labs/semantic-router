from typing import Any, List, Optional, Tuple, Union, Dict
import json

import numpy as np
from pydantic.v1 import BaseModel

from semantic_router.schema import ConfigParameter
from semantic_router.route import Route


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
    sync: Union[str, None] = None

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

    def get_utterances(self) -> List[Tuple]:
        """Gets a list of route and utterance objects currently stored in the
        index, including additional metadata.

        :return: A list of tuples, each containing route, utterance, function
        schema and additional metadata.
        :rtype: List[Tuple]
        """
        _, metadata = self._get_all(include_metadata=True)
        route_tuples: List[
            Tuple[str, str, Optional[Dict[str, Any]], Dict[str, Any]]
        ] = [
            (
                x["sr_route"],
                x["sr_utterance"],
                x.get("sr_function_schema", None),
                x.get("sr_metadata", {}),
            )
            for x in metadata
        ]
        return route_tuples

    def get_routes(self) -> List[Route]:
        """Gets a list of route objects currently stored in the index.

        :return: A list of Route objects.
        :rtype: List[Route]
        """
        route_tuples = self.get_utterances()
        routes_dict: Dict[str, Route] = {}
        # first create a dictionary of route names to Route objects
        for route_name, utterance, function_schema, metadata in route_tuples:
            # if the route is not in the dictionary, add it
            if route_name not in routes_dict:
                routes_dict[route_name] = Route(
                    name=route_name,
                    utterances=[utterance],
                    function_schemas=function_schema,
                    metadata=metadata,
                )
            else:
                # otherwise, add the utterance to the route
                routes_dict[route_name].utterances.append(utterance)
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

    def delete_index(self):
        """
        Deletes or resets the index.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def is_synced(
        self,
        local_route_names: List[str],
        local_utterances_list: List[str],
        local_function_schemas_list: List[Dict[str, Any]],
        local_metadata_list: List[Dict[str, Any]],
    ) -> bool:
        """
        Checks whether local and remote index are synchronized.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _sync_index(
        self,
        local_route_names: List[str],
        local_utterances: List[str],
        local_function_schemas: List[Dict[str, Any]],
        local_metadata: List[Dict[str, Any]],
        dimensions: int,
    ):
        """
        Synchronize the local index with the remote index based on the specified mode.
        Modes:
        - "error": Raise an error if local and remote are not synchronized.
        - "remote": Take remote as the source of truth and update local to align.
        - "local": Take local as the source of truth and update remote to align.
        - "merge-force-remote": Merge both local and remote taking only remote routes features when a route with same route name is present both locally and remotely.
        - "merge-force-local": Merge both local and remote taking only local routes features when a route with same route name is present both locally and remotely.
        - "merge": Merge both local and remote, merging also local and remote features when a route with same route name is present both locally and remotely.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _read_hash(self) -> ConfigParameter:
        """
        Read the hash of the previously written index.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _write_config(self, config: ConfigParameter):
        """
        Write a config parameter to the index.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

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
        if sr_function_schema == {}:
            sr_function_schema = None

        additional_metadata = {
            key: value
            for key, value in record.items()
            if key not in ["sr_route", "sr_utterance", "sr_function_schema"]
        }
        # TODO: Not a fan of tuple packing here
        route_info.append(
            (sr_route, sr_utterance, sr_function_schema, additional_metadata)
        )
    return route_info
