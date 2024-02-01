from typing import List

from pydantic.v1 import BaseModel
from semantic_router.encoders import BaseEncoder


class BaseSplitter(BaseModel):
    name: str
    encoder: BaseEncoder
    similarity_threshold: float

    def __call__(self, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement this method")
