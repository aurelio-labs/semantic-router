from enum import Enum
from typing import List, Optional

from pydantic.v1 import BaseModel
from pydantic.v1.dataclasses import dataclass

from semantic_router.encoders import (
    BaseEncoder,
    CohereEncoder,
    FastEmbedEncoder,
    MistralEncoder,
    OpenAIEncoder,
)


class EncoderType(Enum):
    HUGGINGFACE = "huggingface"
    FASTEMBED = "fastembed"
    OPENAI = "openai"
    COHERE = "cohere"
    MISTRAL = "mistral"


class RouteChoice(BaseModel):
    name: Optional[str] = None
    function_call: Optional[dict] = None
    similarity_score: Optional[float] = None


@dataclass
class Encoder:
    type: EncoderType
    name: Optional[str]
    model: BaseEncoder

    def __init__(self, type: str, name: Optional[str]):
        self.type = EncoderType(type)
        self.name = name
        if self.type == EncoderType.HUGGINGFACE:
            raise NotImplementedError
        elif self.type == EncoderType.FASTEMBED:
            self.model = FastEmbedEncoder(name=name)
        elif self.type == EncoderType.OPENAI:
            self.model = OpenAIEncoder(name=name)
        elif self.type == EncoderType.COHERE:
            self.model = CohereEncoder(name=name)
        elif self.type == EncoderType.MISTRAL:
            self.model = MistralEncoder(name=name)
        else:
            raise ValueError

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model(texts)


class Message(BaseModel):
    role: str
    content: str

    def to_openai(self):
        if self.role.lower() not in ["user", "assistant", "system"]:
            raise ValueError("Role must be either 'user', 'assistant' or 'system'")
        return {"role": self.role, "content": self.content}

    def to_cohere(self):
        return {"role": self.role, "message": self.content}

    def to_llamacpp(self):
        return {"role": self.role, "content": self.content}

    def to_mistral(self):
        return {"role": self.role, "content": self.content}

    def __str__(self):
        return f"{self.role}: {self.content}"


class DocumentSplit(BaseModel):
    docs: List[str]
    is_triggered: bool = False
    triggered_score: Optional[float] = None
    token_count: Optional[int] = None
    metadata: Optional[dict] = None

    @property
    def content(self) -> str:
        return " ".join(self.docs)


class Metric(Enum):
    COSINE = "cosine"
    DOTPRODUCT = "dotproduct"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
