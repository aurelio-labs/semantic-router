from typing import Any, List

from pydantic.v1 import BaseModel, Field


class BaseEncoder(BaseModel):
    name: str
    score_threshold: float
    type: str = Field(default="base")

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, docs: List[Any]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement this method")
