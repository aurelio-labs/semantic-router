from unittest.mock import patch

import pytest
from mistralai.exceptions import MistralException
from mistralai.models.embeddings import EmbeddingObject, EmbeddingResponse, UsageInfo

from semantic_router.encoders import MistralEncoder


@pytest.fixture
def mistralai_encoder(mocker):
    mocker.patch("mistralai.client.MistralClient")
    return MistralEncoder(mistralai_api_key="test_api_key")


class TestMistralEncoder:
    def test_mistral_encoder_import_errors(self):
        with patch.dict("sys.modules", {"mistralai": None}):
            with pytest.raises(ImportError) as error:
                MistralEncoder()

        assert (
            "Please install MistralAI to use MistralEncoder. "
            "You can install it with: "
            "`pip install 'semantic-router[mistralai]'`" in str(error.value)
        )

    def test_mistralai_encoder_init_success(self, mocker):
        encoder = MistralEncoder(mistralai_api_key="test_api_key")
        assert encoder._client is not None
        assert encoder._mistralai is not None

    def test_mistralai_encoder_init_no_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            MistralEncoder()

    def test_mistralai_encoder_call_uninitialized_client(self, mistralai_encoder):
        # Set the client to None to simulate an uninitialized client
        mistralai_encoder._client = None
        with pytest.raises(ValueError) as e:
            mistralai_encoder(["test document"])
        assert "Mistral client not initialized" in str(e.value)

    def test_mistralai_encoder_init_exception(self, mocker):
        mocker.patch(
            "mistralai.client.MistralClient",
            side_effect=Exception("Initialization error"),
        )
        with pytest.raises(ValueError) as e:
            MistralEncoder()
        assert "Mistral API key not provided" in str(e.value)

    def test_mistralai_encoder_call_success(self, mistralai_encoder, mocker):
        mock_embeddings = mocker.Mock()
        mock_embeddings.data = [
            EmbeddingObject(embedding=[0.1, 0.2], index=0, object="embedding")
        ]

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)  # To speed up the test

        mock_embedding = EmbeddingObject(
            index=0, object="embedding", embedding=[0.1, 0.2]
        )
        # Mock the CreateEmbeddingResponse object
        mock_response = EmbeddingResponse(
            id="test-id",
            model="mistral-embed",
            object="list",
            usage=UsageInfo(prompt_tokens=1, total_tokens=20, completion_tokens=None),
            data=[mock_embedding],
        )

        responses = [MistralException("mistralai error"), mock_response]
        mocker.patch.object(
            mistralai_encoder._client, "embeddings", side_effect=responses
        )
        embeddings = mistralai_encoder(["test document"])
        assert embeddings == [[0.1, 0.2]]

    def test_mistralai_encoder_call_with_retries(self, mistralai_encoder, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)  # To speed up the test
        mocker.patch.object(
            mistralai_encoder._client,
            "embeddings",
            side_effect=MistralException("Test error"),
        )
        with pytest.raises(ValueError) as e:
            mistralai_encoder(["test document"])
        assert "No embeddings returned from MistralAI: Test error" in str(e.value)

    def test_mistralai_encoder_call_failure_non_mistralai_error(
        self, mistralai_encoder, mocker
    ):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)  # To speed up the test
        mocker.patch.object(
            mistralai_encoder._client,
            "embeddings",
            side_effect=Exception("Non-MistralException"),
        )
        with pytest.raises(ValueError) as e:
            mistralai_encoder(["test document"])

        assert (
            "Unable to connect to MistralAI ('Non-MistralException',): Non-MistralException"
            in str(e.value)
        )

    def test_mistralai_encoder_call_successful_retry(self, mistralai_encoder, mocker):
        mock_embeddings = mocker.Mock()
        mock_embeddings.data = [
            EmbeddingObject(embedding=[0.1, 0.2], index=0, object="embedding")
        ]

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)  # To speed up the test

        mock_embedding = EmbeddingObject(
            index=0, object="embedding", embedding=[0.1, 0.2]
        )
        # Mock the CreateEmbeddingResponse object
        mock_response = EmbeddingResponse(
            id="test-id",
            model="mistral-embed",
            object="list",
            usage=UsageInfo(prompt_tokens=1, total_tokens=20, completion_tokens=None),
            data=[mock_embedding],
        )

        responses = [MistralException("mistralai error"), mock_response]
        mocker.patch.object(
            mistralai_encoder._client, "embeddings", side_effect=responses
        )
        embeddings = mistralai_encoder(["test document"])
        assert embeddings == [[0.1, 0.2]]
