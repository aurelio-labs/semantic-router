from typing import Any, List, Optional, Tuple, Union, Dict

import numpy as np
from pydantic.v1 import BaseModel


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

    def get_routes(self):
        """
        Retrieves a list of routes and their associated utterances from the index.
        This method should be implemented by subclasses.

        :returns: A list of tuples, each containing a route name and an associated utterance.
        :rtype: list[tuple]
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

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

    class Config:
        arbitrary_types_allowed = True
