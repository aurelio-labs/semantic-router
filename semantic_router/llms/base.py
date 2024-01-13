from typing import Optional, List

from pydantic import BaseModel

from semantic_router.schema import Message


class BaseLLM(BaseModel):
    name: str

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, messages: List[Message]) -> Optional[str]:
        raise NotImplementedError("Subclasses must implement this method")
