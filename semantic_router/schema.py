from enum import Enum
from typing import List, Optional

from pydantic.v1 import BaseModel


class EncoderType(Enum):
    AZURE = "azure"
    COHERE = "cohere"
    OPENAI = "openai"
    BM25 = "bm25"
    TFIDF = "tfidf"
    FASTEMBED = "fastembed"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"
    VIT = "vit"
    CLIP = "clip"
    GOOGLE = "google"


class EncoderInfo(BaseModel):
    name: str
    token_limit: int


class RouteChoice(BaseModel):
    name: Optional[str] = None
    function_call: Optional[dict] = None
    similarity_score: Optional[float] = None


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
