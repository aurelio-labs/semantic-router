import os
import pytest
from semantic_router.encoders.base import BaseEncoder
from semantic_router.encoders.openai import OpenAIEncoder

with open("tests/integration/57640.4032.txt", "r") as fp:
    long_doc = fp.read()


@pytest.fixture
def openai_encoder():
    if os.environ.get("OPENAI_API_KEY") is None:
        return BaseEncoder()
    else:
        return OpenAIEncoder()


class TestOpenAIEncoder:
    @pytest.mark.skipif(
        os.environ.get("OPENAI_API_KEY") is None, reason="OpenAI API key required"
    )
    def test_openai_encoder_init_success(self, openai_encoder):
        assert openai_encoder.client is not None

    @pytest.mark.skipif(
        os.environ.get("OPENAI_API_KEY") is None, reason="OpenAI API key required"
    )
    def test_openai_encoder_dims(self, openai_encoder):
        embeddings = openai_encoder(["test document"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    @pytest.mark.skipif(
        os.environ.get("OPENAI_API_KEY") is None, reason="OpenAI API key required"
    )
    def test_openai_encoder_call_truncation(self, openai_encoder):
        openai_encoder([long_doc])

    @pytest.mark.skipif(
        os.environ.get("OPENAI_API_KEY") is None, reason="OpenAI API key required"
    )
    def test_openai_encoder_call_no_truncation(self, openai_encoder):
        with pytest.raises(ValueError) as _:
            # default truncation is True
            openai_encoder([long_doc], truncate=False)

    @pytest.mark.skipif(
        os.environ.get("OPENAI_API_KEY") is None, reason="OpenAI API key required"
    )
    def test_openai_encoder_call_uninitialized_client(self, openai_encoder):
        # Set the client to None to simulate an uninitialized client
        openai_encoder.client = None
        with pytest.raises(ValueError) as e:
            openai_encoder(["test document"])
        assert "OpenAI client is not initialized." in str(e.value)
