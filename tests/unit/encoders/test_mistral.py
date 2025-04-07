import litellm
import pytest

from semantic_router.encoders import MistralEncoder


@pytest.fixture
def mistralai_encoder(mocker):
    return MistralEncoder(mistralai_api_key="test_api_key")


class TestMistralEncoder:
    def test_mistralai_encoder_init_success(self, mocker):
        _ = MistralEncoder(mistralai_api_key="test_api_key")

    def test_mistralai_encoder_init_no_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            MistralEncoder()

    def test_mistralai_encoder_call_success(self, mistralai_encoder, mocker):
        mock_embeddings = mocker.Mock()
        mock_embeddings.data = [
            litellm.EmbeddingResponse(embedding=[0.1, 0.2], index=0, object="embedding")
        ]

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)  # To speed up the test

        mock_embedding = litellm.EmbeddingResponse(
            index=0, object="embedding", embedding=[0.1, 0.2]
        )
        # Mock the litellm.EmbeddingResponse object
        mock_response = litellm.EmbeddingResponse(
            id="test-id",
            model="mistral-embed",
            object="list",
            usage=litellm.Usage(
                prompt_tokens=1, total_tokens=20, completion_tokens=None
            ),
            data=[mock_embedding],
        )

        responses = [litellm.LitellmException("mistralai error"), mock_response]
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
            side_effect=litellm.LitellmException("Test error"),
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
            litellm.EmbeddingResponse(embedding=[0.1, 0.2], index=0, object="embedding")
        ]

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)  # To speed up the test

        mock_embedding = litellm.EmbeddingResponse(
            index=0, object="embedding", embedding=[0.1, 0.2]
        )
        # Mock the CreateEmbeddingResponse object
        mock_response = litellm.EmbeddingResponse(
            id="test-id",
            model="mistral-embed",
            object="list",
            usage=litellm.Usage(
                prompt_tokens=1, total_tokens=20, completion_tokens=None
            ),
            data=[mock_embedding],
        )

        responses = [litellm.LitellmException("mistralai error"), mock_response]
        mocker.patch.object(
            mistralai_encoder._client, "embeddings", side_effect=responses
        )
        embeddings = mistralai_encoder(["test document"])
        assert embeddings == [[0.1, 0.2]]
