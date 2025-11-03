from typing import List, Optional

from semantic_router.encoders.base import DenseEncoder, SparseEncoder  # isort: skip
from semantic_router.encoders.aurelio import AurelioSparseEncoder
from semantic_router.encoders.azure_openai import AzureOpenAIEncoder
from semantic_router.encoders.bedrock import BedrockEncoder
from semantic_router.encoders.bm25 import BM25Encoder
from semantic_router.encoders.clip import CLIPEncoder
from semantic_router.encoders.cohere import CohereEncoder
from semantic_router.encoders.fastembed import FastEmbedEncoder
from semantic_router.encoders.google import GoogleEncoder
from semantic_router.encoders.huggingface import HFEndpointEncoder, HuggingFaceEncoder
from semantic_router.encoders.jina import JinaEncoder
from semantic_router.encoders.litellm import LiteLLMEncoder
from semantic_router.encoders.local import LocalEncoder, LocalSparseEncoder
from semantic_router.encoders.mistral import MistralEncoder
from semantic_router.encoders.nvidia_nim import NimEncoder
from semantic_router.encoders.ollama import OllamaEncoder
from semantic_router.encoders.openai import OpenAIEncoder
from semantic_router.encoders.tfidf import TfidfEncoder
from semantic_router.encoders.vit import VitEncoder
from semantic_router.encoders.voyage import VoyageEncoder
from semantic_router.schema import EncoderType, SparseEmbedding

__all__ = [
    "AurelioSparseEncoder",
    "DenseEncoder",
    "SparseEncoder",
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
    "LiteLLMEncoder",
    "VoyageEncoder",
    "JinaEncoder",
    "NimEncoder",
    "OllamaEncoder",
    "LocalEncoder",
    "LocalSparseEncoder",
]


class AutoEncoder:
    type: EncoderType
    name: Optional[str]
    model: DenseEncoder | SparseEncoder

    def __init__(self, type: str, name: Optional[str]):
        self.type = EncoderType(type)
        self.name = name
        if self.type == EncoderType.AZURE:
            self.model = AzureOpenAIEncoder(name=name)
        elif self.type == EncoderType.COHERE:
            self.model = CohereEncoder(name=name)
        elif self.type == EncoderType.OPENAI:
            self.model = OpenAIEncoder(name=name)
        elif self.type == EncoderType.AURELIO:
            self.model = AurelioSparseEncoder(name=name)
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
        elif self.type == EncoderType.VOYAGE:
            self.model = VoyageEncoder(name=name)
        elif self.type == EncoderType.JINA:
            self.model = JinaEncoder(name=name)
        elif self.type == EncoderType.NIM:
            self.model = NimEncoder(name=name)
        elif self.type == EncoderType.VIT:
            self.model = VitEncoder(name=name)
        elif self.type == EncoderType.CLIP:
            self.model = CLIPEncoder(name=name)
        elif self.type == EncoderType.GOOGLE:
            self.model = GoogleEncoder(name=name)
        elif self.type == EncoderType.BEDROCK:
            self.model = BedrockEncoder(name=name)  # type: ignore
        elif self.type == EncoderType.LITELLM:
            self.model = LiteLLMEncoder(name=name)
        elif self.type == EncoderType.OLLAMA:
            self.model = OllamaEncoder(name=name)
        elif self.type == EncoderType.LOCAL:
            self.model = LocalEncoder(name=name)
        else:
            raise ValueError(f"Encoder type '{type}' not supported")

    def __call__(self, texts: List[str]) -> List[List[float]] | List[SparseEmbedding]:
        return self.model(texts)
