from enum import Enum
from typing import List, Optional, Union, Any, Dict
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
    BEDROCK = "bedrock"


class EncoderInfo(BaseModel):
    name: str
    token_limit: int
    threshold: Optional[float] = None


class RouteChoice(BaseModel):
    name: Optional[str] = None
    function_call: Optional[List[Dict]] = None
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
    docs: List[Union[str, Any]]
    is_triggered: bool = False
    triggered_score: Optional[float] = None
    token_count: Optional[int] = None
    metadata: Optional[Dict] = None

    @property
    def content(self) -> str:
        return " ".join([doc if isinstance(doc, str) else "" for doc in self.docs])


class Metric(Enum):
    COSINE = "cosine"
    DOTPRODUCT = "dotproduct"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
