import os
from enum import Enum


class EncoderDefault(Enum):
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
    BEDROCK = {
        "embedding_model": os.environ.get(
            "BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-image-v1"
        ),
    }
    UNIFY = {
        "language_model": os.environ.get(
            "UNIFY_CHAT_MODEL_NAME", "llama-3-8b-chat"
        ),
        "language_provider": os.environ.get(
            "UNIFY_CHAT_MODEL_PROVIDER", "together-ai"
        ),
    }
