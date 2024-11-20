import pytest
from unittest.mock import patch, MagicMock, Mock
from semantic_router.encoders import BaseEncoder
from semantic_router.encoders import YandexGPTEncoder

@pytest.fixture
def yandexgpt_encoder(mocker):
    mocker.patch("requests.post")
    return YandexGPTEncoder(api_key="test_api_key", catalog_id="test_catalog_id")

class TestYandexGPTEncoder:

    def test_yandex_encoder_init_with_all_params(self):
        encoder = YandexGPTEncoder(api_key="api-key", catalog_id="catalog-id")
        assert encoder.client is not None
        assert encoder.client["api_key"] == "api-key"
        assert encoder.client["model_Uri"] == "emb://catalog-id/text-search-doc/latest"

    def test_yandex_encoder_init_no_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            YandexGPTEncoder(catalog_id="test_catalog_id")

    def test_yandex_encoder_init_missing_catalog_id(self):
        with pytest.raises(ValueError, match="YandexGPT catalog ID cannot be 'None'."):
            YandexGPTEncoder(api_key="api-key", catalog_id=None)

    def test_yandex_encoder_call_uninitialized_client(self, yandexgpt_encoder):
        yandexgpt_encoder.client = None
        with pytest.raises(ValueError) as e:
            yandexgpt_encoder(["test document"])
        assert "YandexGPT client is not initialized." in str(e.value)

    def test_yandex_encoder_call_success(self, yandexgpt_encoder, mocker):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2]}
        mocker.patch("requests.post", return_value=mock_response)
        embeddings = yandexgpt_encoder(["test document"])
        assert embeddings == [[0.1, 0.2]]

    def test_yandex_encoder_call_failure(self, yandexgpt_encoder, mocker):
        mock_response = Mock()
        mock_response.status_code = 500
        mocker.patch("requests.post", return_value=mock_response)
        with pytest.raises(ValueError) as e:
            yandexgpt_encoder(["test document"])
        assert "Failed to get embedding for document: test document" in str(e.value)

    def test_yandex_encoder_call_exception(self, yandexgpt_encoder, mocker):
        mocker.patch("requests.post", side_effect=Exception("API call error"))
        with pytest.raises(ValueError) as e:
            yandexgpt_encoder(["test document"])
        assert "YandexGPT API call failed. Error: API call error" in str(e.value)


    @patch('requests.post')
    def test_embedding_generation_success(self, mock_post, yandexgpt_encoder):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response

        docs = ["document1", "document2"]
        embeddings = yandexgpt_encoder(docs)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.1, 0.2, 0.3]
