import os
import pytest
import openai
from semantic_router.encoders import OpenAIEncoder
from openai.error import RateLimitError


@pytest.fixture
def openai_encoder(mocker):
    mocker.patch("openai.Embedding.create")
    return OpenAIEncoder(name="test-engine", openai_api_key="test_api_key")


class TestOpenAIEncoder:
    def test_initialization_with_api_key(self, openai_encoder):
        assert openai.api_key == "test_api_key", "API key should be set correctly"
        assert openai_encoder.name == "test-engine", "Engine name not set correctly"

    def test_initialization_without_api_key(self, mocker, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mocker.patch("openai.Embedding.create")
        with pytest.raises(ValueError):
            OpenAIEncoder(name="test-engine")

    def test_call_method_success(self, openai_encoder, mocker):
        mocker.patch("openai.Embedding.create", return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]})

        result = openai_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1 and len(result[0]) == 3, "Result list size is incorrect"

    @pytest.mark.skip(reason="Currently quite a slow test")
    def test_call_method_rate_limit_error(self, openai_encoder, mocker):
        mocker.patch(
            "openai.Embedding.create", side_effect=RateLimitError(message="rate limit exceeded", http_status=429)
        )

        with pytest.raises(ValueError):
            openai_encoder(["test"])

    def test_call_method_failure(self, openai_encoder, mocker):
        mocker.patch("openai.Embedding.create", return_value={})

        with pytest.raises(ValueError):
            openai_encoder(["test"])
