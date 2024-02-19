from typing import Any, List

from pydantic.v1 import BaseModel

from semantic_router.encoders import BaseEncoder


class BaseSplitter(BaseModel):
    name: str
    encoder: BaseEncoder
    score_threshold: float

    def __call__(self, docs: List[Any]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement this method")
