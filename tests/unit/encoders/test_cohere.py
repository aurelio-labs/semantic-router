import pytest

from semantic_router.encoders import CohereEncoder


@pytest.fixture
def cohere_encoder(mocker):
    mocker.patch("cohere.Client")
    return CohereEncoder(cohere_api_key="test_api_key")


class TestCohereEncoder:
    def test_initialization_with_api_key(self, cohere_encoder):
        assert cohere_encoder._client is not None, "Client should be initialized"
        assert (
            cohere_encoder.name == "embed-english-v3.0"
        ), "Default name not set correctly"

    def test_initialization_without_api_key(self, mocker, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        mocker.patch("cohere.Client")
        with pytest.raises(ValueError):
            CohereEncoder()

    def test_call_method(self, cohere_encoder, mocker):
        mock_embed = mocker.MagicMock()
        mock_embed.embeddings = [[0.1, 0.2, 0.3]]
        cohere_encoder._client.embed.return_value = mock_embed

        result = cohere_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sublist, list) for sublist in result
        ), "Each item in result should be a list"
        cohere_encoder._client.embed.assert_called_once()

    def test_returns_list_of_embeddings_for_valid_input(self, cohere_encoder, mocker):
        mock_embed = mocker.MagicMock()
        mock_embed.embeddings = [[0.1, 0.2, 0.3]]
        cohere_encoder._client.embed.return_value = mock_embed

        result = cohere_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sublist, list) for sublist in result
        ), "Each item in result should be a list"
        cohere_encoder._client.embed.assert_called_once()

    def test_handles_multiple_inputs_correctly(self, cohere_encoder, mocker):
        mock_embed = mocker.MagicMock()
        mock_embed.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        cohere_encoder._client.embed.return_value = mock_embed

        result = cohere_encoder(["test1", "test2"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sublist, list) for sublist in result
        ), "Each item in result should be a list"
        cohere_encoder._client.embed.assert_called_once()

    def test_raises_value_error_if_api_key_is_none(self, mocker, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        mocker.patch("cohere.Client")
        with pytest.raises(ValueError):
            CohereEncoder()

    def test_raises_value_error_if_cohere_client_fails_to_initialize(self, mocker):
        mocker.patch(
            "cohere.Client", side_effect=Exception("Failed to initialize client")
        )
        with pytest.raises(ValueError):
            CohereEncoder(cohere_api_key="test_api_key")

    def test_raises_value_error_if_cohere_client_is_not_initialized(self, mocker):
        mocker.patch("cohere.Client", return_value=None)
        encoder = CohereEncoder(cohere_api_key="test_api_key")
        with pytest.raises(ValueError):
            encoder(["test"])

    def test_call_method_raises_error_on_api_failure(self, cohere_encoder, mocker):
        mocker.patch.object(
            cohere_encoder._client, "embed", side_effect=Exception("API call failed")
        )
        with pytest.raises(ValueError):
            cohere_encoder(["test"])
