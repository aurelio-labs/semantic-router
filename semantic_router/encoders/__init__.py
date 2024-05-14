from typing import List, Optional

from semantic_router.encoders.base import BaseEncoder
from semantic_router.encoders.bedrock import BedrockEncoder
from semantic_router.encoders.bm25 import BM25Encoder
from semantic_router.encoders.clip import CLIPEncoder
from semantic_router.encoders.cohere import CohereEncoder
from semantic_router.encoders.fastembed import FastEmbedEncoder
from semantic_router.encoders.google import GoogleEncoder
from semantic_router.encoders.huggingface import HuggingFaceEncoder
from semantic_router.encoders.huggingface import HFEndpointEncoder
from semantic_router.encoders.mistral import MistralEncoder
from semantic_router.encoders.openai import OpenAIEncoder
from semantic_router.encoders.tfidf import TfidfEncoder
from semantic_router.encoders.vit import VitEncoder
from semantic_router.encoders.zure import AzureOpenAIEncoder
from semantic_router.schema import EncoderType

__all__ = [
    "BaseEncoder",
    "AzureOpenAIEncoder",
    "CohereEncoder",
    "OpenAIEncoder",
    "BM25Encoder",
    "TfidfEncoder",
    "FastEmbedEncoder",
    "HuggingFaceEncoder",
    "HFEndpointEncoder",
    "MistralEncoder",
    "VitEncoder",
    "CLIPEncoder",
    "GoogleEncoder",
    "BedrockEncoder",
]


class AutoEncoder:
    type: EncoderType
    name: Optional[str]
    model: BaseEncoder

    def __init__(self, type: str, name: Optional[str]):
        self.type = EncoderType(type)
        self.name = name
        if self.type == EncoderType.AZURE:
            # TODO should change `model` to `name` JB
            self.model = AzureOpenAIEncoder(model=name)
        elif self.type == EncoderType.COHERE:
            self.model = CohereEncoder(name=name)
        elif self.type == EncoderType.OPENAI:
            self.model = OpenAIEncoder(name=name)
        elif self.type == EncoderType.BM25:
            if name is None:
                name = "bm25"
            self.model = BM25Encoder(name=name)
        elif self.type == EncoderType.TFIDF:
            if name is None:
                name = "tfidf"
            self.model = TfidfEncoder(name=name)
        elif self.type == EncoderType.FASTEMBED:
            self.model = FastEmbedEncoder(name=name)
        elif self.type == EncoderType.HUGGINGFACE:
            self.model = HuggingFaceEncoder(name=name)
        elif self.type == EncoderType.MISTRAL:
            self.model = MistralEncoder(name=name)
        elif self.type == EncoderType.VIT:
            self.model = VitEncoder(name=name)
        elif self.type == EncoderType.CLIP:
            self.model = CLIPEncoder(name=name)
        elif self.type == EncoderType.GOOGLE:
            self.model = GoogleEncoder(name=name)
        elif self.type == EncoderType.BEDROCK:
            self.model = BedrockEncoder(name=name)  # type: ignore
        else:
            raise ValueError(f"Encoder type '{type}' not supported")

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model(texts)
