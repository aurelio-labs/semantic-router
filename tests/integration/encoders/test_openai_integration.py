import os

import pytest
from openai import OpenAIError

from semantic_router.encoders.base import DenseEncoder
from semantic_router.encoders.openai import OpenAIEncoder

with open("tests/integration/57640.4032.txt", "r") as fp:
    long_doc = fp.read()


def has_valid_openai_api_key():
    """Check if a valid OpenAI API key is available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    return api_key is not None and api_key.strip() != ""


@pytest.fixture
def openai_encoder():
    if not has_valid_openai_api_key():
        return DenseEncoder()
    else:
        return OpenAIEncoder()


class TestOpenAIEncoder:
    @pytest.mark.skipif(
        not has_valid_openai_api_key(), reason="OpenAI API key required"
    )
    def test_openai_encoder_init_success(self, openai_encoder):
        assert openai_encoder._client is not None

    @pytest.mark.skipif(
        not has_valid_openai_api_key(), reason="OpenAI API key required"
    )
    def test_openai_encoder_dims(self, openai_encoder):
        embeddings = openai_encoder(["test document"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    @pytest.mark.skipif(
        not has_valid_openai_api_key(), reason="OpenAI API key required"
    )
    def test_openai_encoder_call_truncation(self, openai_encoder):
        openai_encoder([long_doc])

    @pytest.mark.skipif(
        not has_valid_openai_api_key(), reason="OpenAI API key required"
    )
    def test_openai_encoder_call_no_truncation(self, openai_encoder):
        with pytest.raises(OpenAIError) as _:
            # default truncation is True
            openai_encoder([long_doc], truncate=False)

    @pytest.mark.skipif(
        not has_valid_openai_api_key(), reason="OpenAI API key required"
    )
    def test_openai_encoder_call_uninitialized_client(self, openai_encoder):
        # Set the client to None to simulate an uninitialized client
        openai_encoder._client = None
        with pytest.raises(ValueError) as e:
            openai_encoder(["test document"])
        assert "OpenAI client is not initialized." in str(e.value)
