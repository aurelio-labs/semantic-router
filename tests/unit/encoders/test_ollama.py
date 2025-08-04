import os
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("ollama")

from semantic_router.encoders.ollama import OllamaEncoder


@pytest.fixture(autouse=True, scope="session")
def set_pinecone_api_key():
    os.environ["PINECONE_API_KEY"] = "test"


@pytest.fixture
def mock_ollama_client():
    with patch("ollama.Client") as mock_client:
        yield mock_client


class TestOllamaEncoder:
    def test_ollama_encoder_init_success(self, mocker):
        mocker.patch("ollama.Client", return_value=Mock())
        encoder = OllamaEncoder(base_url="http://localhost:11434")
        assert encoder.client is not None
        assert encoder.type == "ollama"

    def test_ollama_encoder_init_import_error(self, mocker):
        mocker.patch.dict("sys.modules", {"ollama": None})
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'ollama'")
        ):
            with pytest.raises(ImportError):
                OllamaEncoder(base_url="http://localhost:11434")

    def test_ollama_encoder_call_success(self, mocker):
        mock_client = Mock()
        mock_embed_result = Mock()
        mock_embed_result.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.return_value = mock_embed_result
        mocker.patch("ollama.Client", return_value=mock_client)
        encoder = OllamaEncoder(base_url="http://localhost:11434")
        encoder.client = mock_client
        docs = ["doc1", "doc2"]
        result = encoder(docs)
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.assert_called_once_with(model=encoder.name, input=docs)

    def test_ollama_encoder_call_client_not_initialized(self, mocker):
        encoder = OllamaEncoder(base_url="http://localhost:11434")
        encoder.client = None
        with pytest.raises(ValueError) as e:
            encoder(["doc1"])
        assert "OLLAMA Platform client is not initialized." in str(e.value)

    def test_ollama_encoder_call_api_error(self, mocker):
        mock_client = Mock()
        mock_client.embed.side_effect = Exception("API error")
        mocker.patch("ollama.Client", return_value=mock_client)
        encoder = OllamaEncoder(base_url="http://localhost:11434")
        encoder.client = mock_client
        with pytest.raises(ValueError) as e:
            encoder(["doc1"])
        assert "OLLAMA API call failed. Error: API error" in str(e.value)

    def test_ollama_encoder_uses_env_base_url(self, mocker):
        test_url = "http://env-ollama:1234"
        mock_client = Mock()
        mock_client.host = test_url  # Set the host attribute on the mock
        mocker.patch("ollama.Client", return_value=mock_client)
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": test_url}):
            encoder = OllamaEncoder()
            assert encoder.client is not None
            assert encoder.client.host == test_url
