import os

import pytest

from semantic_router.encoders import (
    AurelioSparseEncoder,
    AzureOpenAIEncoder,
    BedrockEncoder,
    BM25Encoder,
    CLIPEncoder,
    CohereEncoder,
    FastEmbedEncoder,
    GoogleEncoder,
    HFEndpointEncoder,
    HuggingFaceEncoder,
    JinaEncoder,
    LiteLLMEncoder,
    LocalEncoder,
    MistralEncoder,
    NimEncoder,
    OpenAIEncoder,
    TfidfEncoder,
    VitEncoder,
    VoyageEncoder,
)

ENCODER_MATRIX = [
    (LocalEncoder, {}, "sentence_transformers"),
    (HuggingFaceEncoder, {}, "transformers"),
    (FastEmbedEncoder, {}, "fastembed"),
    (BM25Encoder, {"name": "bm25"}, None),
    (TfidfEncoder, {"name": "tfidf"}, None),
    # The following require API keys or cloud dependencies, so we skip if not set
    (OpenAIEncoder, {}, "openai"),
    (CohereEncoder, {}, "cohere"),
    (AzureOpenAIEncoder, {}, "openai"),
    (
        BedrockEncoder,
        {"access_key_id": "fake", "secret_access_key": "fake", "region": "us-west-2"},
        "boto3",
    ),
    (GoogleEncoder, {"project_id": "fake"}, "google.cloud.aiplatform"),
    (
        HFEndpointEncoder,
        {
            "huggingface_url": "https://api-inference.huggingface.co/models/bert-base-uncased",
            "huggingface_api_key": "fake",
        },
        "requests",
    ),
    (
        LiteLLMEncoder,
        {"name": "openai/text-embedding-3-small", "api_key": "fake"},
        "litellm",
    ),
    (MistralEncoder, {"mistralai_api_key": "fake"}, "litellm"),
    (VoyageEncoder, {"api_key": "fake"}, "litellm"),
    (JinaEncoder, {"api_key": "fake"}, "litellm"),
    (NimEncoder, {"api_key": "fake"}, "litellm"),
    (CLIPEncoder, {}, "transformers"),
    (VitEncoder, {}, "transformers"),
    (AurelioSparseEncoder, {"api_key": "fake"}, "aurelio_sdk"),
]


@pytest.mark.parametrize("encoder_cls, kwargs, dep", ENCODER_MATRIX)
def test_encoder_basic(encoder_cls, kwargs, dep):
    if dep:
        try:
            __import__(dep)
        except ImportError:
            pytest.skip(f"Dependency {dep} not installed")
    # Skip API-key based encoders if not set (except for those we pass fake keys)
    if (
        encoder_cls in [OpenAIEncoder, CohereEncoder, AzureOpenAIEncoder]
        and not os.getenv("OPENAI_API_KEY")
        and not kwargs.get("api_key")
    ):
        pytest.skip("OPENAI_API_KEY not set")
    if (
        encoder_cls is GoogleEncoder
        and not os.getenv("GOOGLE_PROJECT_ID")
        and not kwargs.get("project_id")
    ):
        pytest.skip("GOOGLE_PROJECT_ID not set")
    if encoder_cls is HFEndpointEncoder and not kwargs.get("huggingface_api_key"):
        pytest.skip("HF_API_KEY not set")
    if encoder_cls is AurelioSparseEncoder and not kwargs.get("api_key"):
        pytest.skip("AURELIO_API_KEY not set")
    encoder = encoder_cls(**kwargs)
    test_docs = ["This is a test", "This is another test"]
    try:
        embeddings = encoder(test_docs)
    except NotImplementedError:
        pytest.skip("Encoder does not implement __call__")
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(test_docs)
    assert all(isinstance(embedding, (list, object)) for embedding in embeddings)
