from pydantic import BaseModel


class BaseRanker(BaseModel):
    name: str
    top_n: int = 5

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, query: str, docs: list[str]) -> list[str]:
        raise NotImplementedError("Subclasses must implement this method")
