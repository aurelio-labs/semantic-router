from pydantic import BaseModel


class BaseEncoder(BaseModel):
    name: str

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, texts: list[str]) -> list[float]:
        raise NotImplementedError("Subclasses must implement this method")
