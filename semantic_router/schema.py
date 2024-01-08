from enum import Enum

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from semantic_router.encoders import (
    BaseEncoder,
    CohereEncoder,
    FastEmbedEncoder,
    OpenAIEncoder,
)
from semantic_router.utils.splitters import semantic_splitter


class EncoderType(Enum):
    HUGGINGFACE = "huggingface"
    FASTEMBED = "fastembed"
    OPENAI = "openai"
    COHERE = "cohere"


class RouteChoice(BaseModel):
    name: str | None = None
    function_call: dict | None = None


@dataclass
class Encoder:
    type: EncoderType
    name: str | None
    model: BaseEncoder

    def __init__(self, type: str, name: str | None):
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

    def __call__(self, texts: list[str]) -> list[list[float]]:
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


class Conversation(BaseModel):
    messages: list[Message]

    def split_by_topic(
        self,
        encoder: BaseEncoder,
        threshold: float = 0.5,
        split_method: str = "consecutive_similarity_drop",
    ):
        docs = [f"{m.role}: {m.content}" for m in self.messages]
        return semantic_splitter(
            encoder=encoder, docs=docs, threshold=threshold, split_method=split_method
        )
