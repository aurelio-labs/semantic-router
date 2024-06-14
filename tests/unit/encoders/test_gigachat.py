import pytest
from unittest.mock import patch, Mock, MagicMock
from semantic_router.encoders import GigaChatEncoder

@pytest.fixture
def gigachat_encoder(mocker):
    mocker.patch("gigachat.GigaChat")
    return GigaChatEncoder(auth_data="test_auth_data", scope="GIGACHAT_API_PERS")

class TestGigaChatEncoder:
    def test_gigachat_encoder_init_success(self):
        encoder = GigaChatEncoder(auth_data="test_auth_data", scope="GIGACHAT_API_PERS")
        assert encoder.client is not None
        assert encoder.type == "gigachat"

    @patch('os.getenv', return_value=None)
    def test_gigachat_encoder_init_no_auth_data(self, mock_getenv):
        with pytest.raises(ValueError) as e:
            GigaChatEncoder(scope="GIGACHAT_API_PERS")
        assert str(e.value) == "GigaChat authorization data cannot be 'None'."

    def test_gigachat_encoder_init_no_scope(self):
        with pytest.raises(ValueError) as e:
            GigaChatEncoder(auth_data="test_auth_data")
        assert str(
            e.value) == "GigaChat scope cannot be 'None'. Set 'GIGACHAT_API_PERS' for personal use or 'GIGACHAT_API_CORP' for corporate use."

    def test_gigachat_encoder_call_uninitialized_client(self, gigachat_encoder):
        gigachat_encoder.client = None
        with pytest.raises(ValueError) as e:
            gigachat_encoder(["test document"])
        assert "GigaChat client is not initialized." in str(e.value)

    def test_gigachat_encoder_call_success(self, gigachat_encoder, mocker):
        mock_embeddings = Mock()
        mock_embeddings.data = [Mock(embedding=[0.1, 0.2])]

        mocker.patch("time.sleep", return_value=None)

        mocker.patch.object(gigachat_encoder.client, 'embeddings', return_value=mock_embeddings)

        embeddings = gigachat_encoder(["test document"])

        assert embeddings == [[0.1, 0.2]]

        gigachat_encoder.client.embeddings.assert_called_with(["test document"])

    def test_call_method_api_failure(self, gigachat_encoder):
        gigachat_encoder.client.embeddings = MagicMock(side_effect=Exception("API failure"))
        docs = ["document1", "document2"]
        with pytest.raises(ValueError, match="GigaChat call failed. Error: API failure"):
            gigachat_encoder(docs)

    def test_init_failure_no_env_vars(self):
        with pytest.raises(ValueError) as excinfo:
            GigaChatEncoder()
        assert "GigaChat authorization data cannot be 'None'" in str(excinfo.value)