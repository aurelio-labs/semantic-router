from enum import Enum

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from semantic_router.encoders import (
    BaseEncoder,
    CohereEncoder,
    OpenAIEncoder,
)


class Decision(BaseModel):
    name: str
    utterances: list[str]
    description: str | None = None


class EncoderType(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    COHERE = "cohere"


@dataclass
class Encoder:
    type: EncoderType
    name: str
    model: BaseEncoder

    def __init__(self, type: str, name: str):
        self.type = EncoderType(type)
        self.name = name
        if self.type == EncoderType.HUGGINGFACE:
            raise NotImplementedError
        elif self.type == EncoderType.OPENAI:
            self.model = OpenAIEncoder(name)
        elif self.type == EncoderType.COHERE:
            self.model = CohereEncoder(name)

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
