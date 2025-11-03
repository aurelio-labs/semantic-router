import os
from enum import Enum


class EncoderDefault(Enum):
    """Default model names for each encoder type."""

    FASTEMBED = {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "language_model": "BAAI/bge-small-en-v1.5",
    }
    OPENAI = {
        "embedding_model": os.getenv("OPENAI_MODEL_NAME", "text-embedding-3-small"),
        "language_model": os.getenv("OPENAI_CHAT_MODEL_NAME", "gpt-4o"),
    }
    COHERE = {
        "embedding_model": os.getenv("COHERE_MODEL_NAME", "embed-english-v3.0"),
        "language_model": os.getenv("COHERE_CHAT_MODEL_NAME", "command"),
    }
    MISTRAL = {
        "embedding_model": os.getenv("MISTRAL_MODEL_NAME", "mistral-embed"),
        "language_model": os.getenv("MISTRALAI_CHAT_MODEL_NAME", "mistral-tiny"),
    }
    VOYAGE = {
        "embedding_model": os.getenv("VOYAGE_MODEL_NAME", "voyage-3-lite"),
        "language_model": os.getenv("VOYAGE_CHAT_MODEL_NAME", "voyage-3-lite"),
    }
    JINA = {
        "embedding_model": os.getenv("JINA_MODEL_NAME", "jina-embeddings-v3"),
        "language_model": os.getenv("JINA_CHAT_MODEL_NAME", "ReaderLM-v2"),
    }
    NVIDIA_NIM = {
        "embedding_model": os.getenv(
            "NVIDIA_NIM_MODEL_NAME", "nvidia/nv-embedqa-e5-v5"
        ),
        "language_model": os.getenv(
            "NVIDIA_NIM_CHAT_MODEL_NAME", "meta/llama3-70b-instruct"
        ),
    }
    AZURE = {
        "embedding_model": os.getenv("AZURE_OPENAI_MODEL", "text-embedding-3-small"),
        "language_model": os.getenv("OPENAI_CHAT_MODEL_NAME", "gpt-4o"),
        "deployment_name": os.getenv(
            "AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-3-small"
        ),
    }
    GOOGLE = {
        "embedding_model": os.getenv(
            "GOOGLE_EMBEDDING_MODEL", "textembedding-gecko@003"
        ),
    }
    OLLAMA = {
        "embedding_model": os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:F16"
        )
    }
    BEDROCK = {
        "embedding_model": os.environ.get(
            "BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-image-v1"
        )
    }
