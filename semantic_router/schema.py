from enum import Enum
from typing import List, Literal, Optional

from pydantic.v1 import BaseModel
from pydantic.v1.dataclasses import dataclass

from semantic_router.encoders import (
    BaseEncoder,
    CohereEncoder,
    FastEmbedEncoder,
    OpenAIEncoder,
)
from semantic_router.utils.splitters import DocumentSplit, semantic_splitter


class EncoderType(Enum):
    HUGGINGFACE = "huggingface"
    FASTEMBED = "fastembed"
    OPENAI = "openai"
    COHERE = "cohere"


class RouteChoice(BaseModel):
    name: Optional[str] = None
    function_call: Optional[dict] = None
    similarity_score: Optional[float] = None
    trigger: Optional[bool] = None


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


class Conversation(BaseModel):
    messages: List[Message]

    def split_by_topic(
        self,
        encoder: BaseEncoder,
        threshold: float = 0.5,
        split_method: Literal[
            "consecutive_similarity_drop", "cumulative_similarity_drop"
        ] = "consecutive_similarity_drop",
    ) -> list[DocumentSplit]:
        docs = [f"{m.role}: {m.content}" for m in self.messages]
        return semantic_splitter(
            encoder=encoder, docs=docs, threshold=threshold, split_method=split_method
        )
