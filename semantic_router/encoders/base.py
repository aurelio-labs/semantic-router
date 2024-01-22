from typing import List

from pydantic.v1 import BaseModel, Field


class BaseEncoder(BaseModel):
    name: str
    score_threshold: float
    type: str = Field(default="base")

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement this method")
