from typing import Any, Coroutine, List, Optional

from pydantic.v1 import BaseModel, Field, validator

from semantic_router.schema import SparseEmbedding


class DenseEncoder(BaseModel):
    name: str
    score_threshold: Optional[float] = None
    type: str = Field(default="base")

    class Config:
        arbitrary_types_allowed = True

    @validator("score_threshold", pre=True, always=True)
    def set_score_threshold(cls, v):
        return float(v) if v is not None else None

    def __call__(self, docs: List[Any]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement this method")

    def acall(self, docs: List[Any]) -> Coroutine[Any, Any, List[List[float]]]:
        raise NotImplementedError("Subclasses must implement this method")


class SparseEncoder(BaseModel):
    name: str
    type: str = Field(default="base")

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, docs: List[str]) -> List[SparseEmbedding]:
        raise NotImplementedError("Subclasses must implement this method")

    def acall(self, docs: List[str]) -> Coroutine[Any, Any, List[SparseEmbedding]]:
        raise NotImplementedError("Subclasses must implement this method")