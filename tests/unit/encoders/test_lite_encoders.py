import os

import litellm
import pytest
from litellm.types.utils import Embedding

from semantic_router.encoders import CohereEncoder, MistralEncoder

matrix = [
    ["cohere", "embed-english-v3.0", "COHERE_API_KEY", CohereEncoder],
    ["mistral", "mistral-embed", "MISTRAL_API_KEY", MistralEncoder],
]

@pytest.fixture
def mock_litellm(mocker):
    mock_embed = litellm.EmbeddingResponse(
        data=[
            Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding"),
        ]
    )
    mocker.patch.object(litellm, "embedding", return_value=mock_embed)
    return mock_embed

@pytest.mark.parametrize("name, model, api_key_env_var, encoder", matrix)
class TestEncoders:
    def test_initialization_with_api_key(self, name, model, api_key_env_var, encoder):
        os.environ[api_key_env_var] = "test_api_key"
        enc = encoder()
        assert enc.name == model, (
            "Default name not set correctly"
        )

    def test_initialization_without_api_key(self, monkeypatch, name, model, api_key_env_var, encoder):
        monkeypatch.delenv(api_key_env_var, raising=False)
        with pytest.raises(ValueError):
            encoder()

    def test_call_method(self, mock_litellm, name, model, api_key_env_var, encoder):
        os.environ[api_key_env_var] = "test_api_key"
        result = encoder()(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(sublist, list) for sublist in result), (
            "Each item in result should be a list"
        )
        litellm.embedding.assert_called_once()

    def test_returns_list_of_embeddings_for_valid_input(self, mock_litellm, name, model, api_key_env_var, encoder):
        os.environ[api_key_env_var] = "test_api_key"
        result = encoder()(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(sublist, list) for sublist in result), (
            "Each item in result should be a list"
        )
        litellm.embedding.assert_called_once()

    def test_handles_multiple_inputs_correctly(self, mocker, name, model, api_key_env_var, encoder):
        os.environ[api_key_env_var] = "test_api_key"
        mock_embed = litellm.EmbeddingResponse(
            data=[
                Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding"),
                Embedding(embedding=[0.4, 0.5, 0.6], index=1, object="embedding"),
            ]
        )
        mocker.patch.object(litellm, "embedding", return_value=mock_embed)

        result = encoder()(["test1", "test2"])
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(sublist, list) for sublist in result), (
            "Each item in result should be a list"
        )
        litellm.embedding.assert_called_once()

    def test_call_method_raises_error_on_api_failure(self, mocker, name, model, api_key_env_var, encoder):
        os.environ[api_key_env_var] = "test_api_key"
        mocker.patch.object(
            litellm, "embedding", side_effect=Exception("API call failed")
        )
        with pytest.raises(ValueError):
            encoder()(["test"])
