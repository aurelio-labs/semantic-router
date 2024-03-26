from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

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
    model: Optional[BaseEncoder] = None

    def __post_init__(self):
        if self.type == EncoderType.HUGGINGFACE:
            raise NotImplementedError
        elif self.type == EncoderType.FASTEMBED:
            self.model = FastEmbedEncoder(name=self.name)
        elif self.type == EncoderType.OPENAI:
            self.model = OpenAIEncoder(name=self.name)
        elif self.type == EncoderType.COHERE:
            self.model = CohereEncoder(name=self.name)
        elif self.type == EncoderType.MISTRAL:
            self.model = MistralEncoder(name=self.name)
        else:
            raise ValueError(f"Unsupported encoder type: {self.type}")

    def __call__(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            raise ValueError("Encoder model is not initialized.")
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
