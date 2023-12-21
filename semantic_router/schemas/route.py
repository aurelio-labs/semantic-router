from pydantic import BaseModel


class Route(BaseModel):
    name: str
    utterances: list[str]
    description: str | None = None
