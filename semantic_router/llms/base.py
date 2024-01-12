from typing import Optional

from pydantic import BaseModel

from semantic_router.schema import Message


class BaseLLM(BaseModel):
    name: str

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, messages: list[Message]) -> Optional[str]:
        raise NotImplementedError("Subclasses must implement this method")
