from pydantic.v1 import BaseModel
from typing import Any, List, Tuple, Optional, Union
import numpy as np

class BaseIndex(BaseModel):
    """
    Base class for indices using Pydantic's BaseModel.
    This class outlines the expected interface for index classes.
    Actual method implementations should be provided in subclasses.
    """

    # You can define common attributes here if there are any.
    # For example, a placeholder for the index attribute:
    index: Optional[Any] = None
    routes: Optional[List[str]] = None
    utterances: Optional[List[str]] = None
    dimensions: Union[int, None] = None
    type: str = "base"

    def add(self, route_names: List[str], utterances: List[str]):
        """
        Add embeddings to the index.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def delete(self, route_name: str):
        """
        Deletes route by route name.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def describe(self) -> dict:
        """
        Returns a dictionary with index details such as type, dimensions, and total vector count.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def query(self, vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[str]]:
        """
        Search the index for the query_vector and return top_k results.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def delete_index(self):
        """
        Deletes or resets the index.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_routes(self) -> List[Tuple]:
        """
        Returns a list of all routes stored in the index.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    class Config:
        arbitrary_types_allowed = True
