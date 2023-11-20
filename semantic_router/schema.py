from enum import Enum

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from semantic_router.retrievers import (
    BaseRetriever,
    CohereRetriever,
    OpenAIRetriever,
)


class Decision(BaseModel):
    name: str
    utterances: list[str]
    description: str | None = None


class RetrieverType(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    COHERE = "cohere"


@dataclass
class Retriever:
    type: RetrieverType
    name: str
    model: BaseRetriever

    def __init__(self, type: str, name: str):
        self.type = RetrieverType(type)
        self.name = name
        if self.type == RetrieverType.HUGGINGFACE:
            raise NotImplementedError
        elif self.type == RetrieverType.OPENAI:
            self.model = OpenAIRetriever(name)
        elif self.type == RetrieverType.COHERE:
            self.model = CohereRetriever(name)

    def __call__(self, texts: list[str]) -> list[float]:
        return self.model(texts)


@dataclass
class SemanticSpace:
    id: str
    decisions: list[Decision]
    encoder: str = ""

    def __init__(self, decisions: list[Decision] = []):
        self.id = ""
        self.decisions = decisions

    def add(self, decision: Decision):
        self.decisions.append(decision)
