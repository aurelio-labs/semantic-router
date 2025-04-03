import pytest
from google.api_core.exceptions import GoogleAPICallError
from vertexai.language_models import TextEmbedding
from vertexai.language_models._language_models import TextEmbeddingStatistics

from semantic_router.encoders import GoogleEncoder


@pytest.fixture
def google_encoder(mocker):
    mocker.patch("google.cloud.aiplatform.init")
    mocker.patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
    return GoogleEncoder(project_id="test_project_id")


class TestGoogleEncoder:
    def test_initialization_with_project_id(self, google_encoder):
        assert google_encoder.client is not None, "Client should be initialized"
        assert google_encoder.name == "textembedding-gecko@003", (
            "Default name not set correctly"
        )

    def test_initialization_without_project_id(self, mocker, monkeypatch):
        monkeypatch.delenv("GOOGLE_PROJECT_ID", raising=False)
        mocker.patch("google.cloud.aiplatform.init")
        mocker.patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
        with pytest.raises(ValueError):
            GoogleEncoder()

    def test_call_method(self, google_encoder, mocker):
        mock_embeddings = [
            TextEmbedding(
                values=[0.1, 0.2, 0.3],
                statistics=TextEmbeddingStatistics(token_count=5, truncated=False),
            )
        ]
        mocker.patch.object(
            google_encoder.client, "get_embeddings", return_value=mock_embeddings
        )

        result = google_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(sublist, list) for sublist in result), (
            "Each item in result should be a list"
        )
        google_encoder.client.get_embeddings.assert_called_once()

    def test_returns_list_of_embeddings_for_valid_input(self, google_encoder, mocker):
        mock_embeddings = [
            TextEmbedding(
                values=[0.1, 0.2, 0.3],
                statistics=TextEmbeddingStatistics(token_count=5, truncated=False),
            )
        ]
        mocker.patch.object(
            google_encoder.client, "get_embeddings", return_value=mock_embeddings
        )

        result = google_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(sublist, list) for sublist in result), (
            "Each item in result should be a list"
        )
        google_encoder.client.get_embeddings.assert_called_once()

    def test_handles_multiple_inputs_correctly(self, google_encoder, mocker):
        mock_embeddings = [
            TextEmbedding(
                values=[0.1, 0.2, 0.3],
                statistics=TextEmbeddingStatistics(token_count=5, truncated=False),
            ),
            TextEmbedding(
                values=[0.4, 0.5, 0.6],
                statistics=TextEmbeddingStatistics(token_count=6, truncated=False),
            ),
        ]
        mocker.patch.object(
            google_encoder.client, "get_embeddings", return_value=mock_embeddings
        )

        result = google_encoder(["test1", "test2"])
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(sublist, list) for sublist in result), (
            "Each item in result should be a list"
        )
        google_encoder.client.get_embeddings.assert_called_once()

    def test_raises_value_error_if_project_id_is_none(self, mocker, monkeypatch):
        monkeypatch.delenv("GOOGLE_PROJECT_ID", raising=False)
        mocker.patch("google.cloud.aiplatform.init")
        mocker.patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
        with pytest.raises(ValueError):
            GoogleEncoder()

    def test_raises_value_error_if_google_client_fails_to_initialize(self, mocker):
        mocker.patch(
            "google.cloud.aiplatform.init",
            side_effect=Exception("Failed to initialize client"),
        )
        with pytest.raises(ValueError):
            GoogleEncoder(project_id="test_project_id")

    def test_raises_value_error_if_google_client_is_not_initialized(self, mocker):
        mocker.patch("google.cloud.aiplatform.init")
        mocker.patch(
            "vertexai.language_models.TextEmbeddingModel.from_pretrained",
            return_value=None,
        )
        encoder = GoogleEncoder(project_id="test_project_id")
        with pytest.raises(ValueError):
            encoder(["test"])

    def test_call_method_raises_error_on_api_failure(self, google_encoder, mocker):
        mocker.patch.object(
            google_encoder.client,
            "get_embeddings",
            side_effect=GoogleAPICallError("API call failed"),
        )
        with pytest.raises(ValueError):
            google_encoder(["test"])
