import pytest

from semantic_router.encoders import MiniMaxEncoder


@pytest.fixture
def minimax_encoder(mocker):
    return MiniMaxEncoder(minimax_api_key="test_api_key")


class TestMiniMaxEncoder:
    def test_initialization_with_api_key(self, minimax_encoder):
        assert minimax_encoder._api_key == "test_api_key"
        assert minimax_encoder.name == "embo-01", "Default name not set correctly"
        assert minimax_encoder.type == "minimax", "Type not set correctly"

    def test_initialization_with_custom_name(self):
        enc = MiniMaxEncoder(name="custom-model", minimax_api_key="test_api_key")
        assert enc.name == "custom-model"

    def test_initialization_without_api_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(ValueError):
            MiniMaxEncoder()

    def test_initialization_with_env_api_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env_test_key")
        enc = MiniMaxEncoder()
        assert enc._api_key == "env_test_key"

    def test_build_request(self, minimax_encoder):
        payload = minimax_encoder._build_request(["hello", "world"])
        assert payload["model"] == "embo-01"
        assert payload["texts"] == ["hello", "world"]
        assert payload["type"] == "db"

    def test_build_headers(self, minimax_encoder):
        headers = minimax_encoder._build_headers()
        assert headers["Authorization"] == "Bearer test_api_key"
        assert headers["Content-Type"] == "application/json"

    def test_parse_response_success(self):
        data = {
            "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        result = MiniMaxEncoder._parse_response(data)
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_parse_response_error(self):
        data = {
            "base_resp": {"status_code": 1000, "status_msg": "invalid api key"},
        }
        with pytest.raises(ValueError, match="MiniMax API error"):
            MiniMaxEncoder._parse_response(data)

    def test_parse_response_no_vectors(self):
        data = {
            "vectors": [],
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        with pytest.raises(ValueError, match="No embeddings returned"):
            MiniMaxEncoder._parse_response(data)

    def test_call_method(self, minimax_encoder, mocker):
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "vectors": [[0.1, 0.2, 0.3]],
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        mock_response.raise_for_status = mocker.MagicMock()
        mocker.patch("requests.post", return_value=mock_response)

        result = minimax_encoder(["test"])
        assert isinstance(result, list)
        assert result == [[0.1, 0.2, 0.3]]

    def test_call_api_failure(self, minimax_encoder, mocker):
        mocker.patch("requests.post", side_effect=Exception("API call failed"))
        with pytest.raises(ValueError, match="MiniMax API call failed"):
            minimax_encoder(["test"])

    def test_call_multiple_inputs(self, minimax_encoder, mocker):
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        mock_response.raise_for_status = mocker.MagicMock()
        mocker.patch("requests.post", return_value=mock_response)

        result = minimax_encoder(["test1", "test2"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_custom_score_threshold(self):
        enc = MiniMaxEncoder(minimax_api_key="test_api_key", score_threshold=0.5)
        assert enc.score_threshold == 0.5

    def test_custom_base_url(self):
        enc = MiniMaxEncoder(
            minimax_api_key="test_api_key",
            base_url="https://custom.api.com/v1/",
        )
        assert enc._base_url == "https://custom.api.com/v1"

    @pytest.mark.asyncio
    async def test_acall_success(self, minimax_encoder, mocker):
        mock_resp = mocker.AsyncMock()
        mock_resp.json = mocker.AsyncMock(
            return_value={
                "vectors": [[0.1, 0.2, 0.3]],
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        )
        mock_resp.raise_for_status = mocker.MagicMock()

        mock_session = mocker.AsyncMock()
        mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)
        mock_session.post = mocker.MagicMock(return_value=mock_resp)
        mock_resp.__aenter__ = mocker.AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch("aiohttp.ClientSession", return_value=mock_session)

        result = await minimax_encoder.acall(["test"])
        assert isinstance(result, list)
        assert result == [[0.1, 0.2, 0.3]]

    @pytest.mark.asyncio
    async def test_acall_api_failure(self, minimax_encoder, mocker):
        mock_session = mocker.AsyncMock()
        mock_session.__aenter__ = mocker.AsyncMock(side_effect=Exception("API failed"))
        mock_session.__aexit__ = mocker.AsyncMock(return_value=None)
        mocker.patch("aiohttp.ClientSession", return_value=mock_session)

        with pytest.raises(ValueError, match="MiniMax API call failed"):
            await minimax_encoder.acall(["test"])
