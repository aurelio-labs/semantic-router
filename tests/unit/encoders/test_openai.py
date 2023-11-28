import openai
import pytest
from openai.error import RateLimitError

from semantic_router.encoders import OpenAIEncoder


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
        mocker.patch(
            "openai.Embedding.create",
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
        )

        result = openai_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1 and len(result[0]) == 3, "Result list size is incorrect"

    def test_call_method_rate_limit_error__raises_value_error_after_max_retries(
        self, openai_encoder, mocker
    ):
        mocker.patch("semantic_router.encoders.openai.sleep")
        mocker.patch(
            "openai.Embedding.create",
            side_effect=RateLimitError(message="rate limit exceeded", http_status=429),
        )

        with pytest.raises(ValueError):
            openai_encoder(["test"])

    def test_call_method_failure(self, openai_encoder, mocker):
        mocker.patch("openai.Embedding.create", return_value={})

        with pytest.raises(ValueError):
            openai_encoder(["test"])

    def test_call_method_rate_limit_error__exponential_backoff_single_retry(
        self, openai_encoder, mocker
    ):
        mock_sleep = mocker.patch("semantic_router.encoders.openai.sleep")
        mocker.patch(
            "openai.Embedding.create",
            side_effect=[
                RateLimitError("rate limit exceeded"),
                {"data": [{"embedding": [1, 2, 3]}]},
            ],
        )

        openai_encoder(["sample text"])

        mock_sleep.assert_called_once_with(1)  # 2**0

    def test_call_method_rate_limit_error__exponential_backoff_multiple_retries(
        self, openai_encoder, mocker
    ):
        mock_sleep = mocker.patch("semantic_router.encoders.openai.sleep")
        mocker.patch(
            "openai.Embedding.create",
            side_effect=[
                RateLimitError("rate limit exceeded"),
                RateLimitError("rate limit exceeded"),
                {"data": [{"embedding": [1, 2, 3]}]},
            ],
        )

        openai_encoder(["sample text"])

        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)  # 2**0
        mock_sleep.assert_any_call(2)  # 2**1

    def test_call_method_rate_limit_error__exponential_backoff_max_retries_exceeded(
        self, openai_encoder, mocker
    ):
        mock_sleep = mocker.patch("semantic_router.encoders.openai.sleep")
        mocker.patch(
            "openai.Embedding.create", side_effect=RateLimitError("rate limit exceeded")
        )

        with pytest.raises(ValueError):
            openai_encoder(["sample text"])

        assert mock_sleep.call_count == 5  # Assuming 5 retries
        mock_sleep.assert_any_call(1)  # 2**0
        mock_sleep.assert_any_call(2)  # 2**1
        mock_sleep.assert_any_call(4)  # 2**2
        mock_sleep.assert_any_call(8)  # 2**3
        mock_sleep.assert_any_call(16)  # 2**4

    def test_call_method_rate_limit_error__exponential_backoff_successful(
        self, openai_encoder, mocker
    ):
        mock_sleep = mocker.patch("semantic_router.encoders.openai.sleep")
        mocker.patch(
            "openai.Embedding.create",
            side_effect=[
                RateLimitError("rate limit exceeded"),
                RateLimitError("rate limit exceeded"),
                {"data": [{"embedding": [1, 2, 3]}]},
            ],
        )

        embeddings = openai_encoder(["sample text"])

        assert mock_sleep.call_count == 2
        assert embeddings == [[1, 2, 3]]
