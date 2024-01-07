from pydantic import BaseModel

from semantic_router.schema import Message


class BaseLLM(BaseModel):
    name: str

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, messages: list[Message]) -> str | None:
        raise NotImplementedError("Subclasses must implement this method")
