import pytest

from semantic_router.llms import CohereLLM
from semantic_router.schema import Message


@pytest.fixture
def cohere_llm(mocker):
    mocker.patch("cohere.Client")
    return CohereLLM(cohere_api_key="test_api_key")


class TestCohereLLM:
    def test_initialization_with_api_key(self, cohere_llm):
        assert cohere_llm._client is not None, "Client should be initialized"
        assert cohere_llm.name == "command", "Default name not set correctly"

    def test_initialization_without_api_key(self, mocker, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        mocker.patch("cohere.Client")
        with pytest.raises(ValueError):
            CohereLLM()

    def test_call_method(self, cohere_llm, mocker):
        mock_llm = mocker.MagicMock()
        mock_llm.text = "test"
        cohere_llm._client.chat.return_value = mock_llm

        llm_input = [Message(role="user", content="test")]
        result = cohere_llm(llm_input)
        assert isinstance(result, str), "Result should be a str"
        cohere_llm._client.chat.assert_called_once()

    def test_raises_value_error_if_cohere_client_fails_to_initialize(self, mocker):
        mocker.patch(
            "cohere.Client", side_effect=Exception("Failed to initialize client")
        )
        with pytest.raises(ValueError):
            CohereLLM(cohere_api_key="test_api_key")

    def test_raises_value_error_if_cohere_client_is_not_initialized(self, mocker):
        mocker.patch("cohere.Client", return_value=None)
        llm = CohereLLM(cohere_api_key="test_api_key")
        with pytest.raises(ValueError):
            llm("test")

    def test_call_method_raises_error_on_api_failure(self, cohere_llm, mocker):
        mocker.patch.object(
            cohere_llm._client, "__call__", side_effect=Exception("API call failed")
        )
        with pytest.raises(ValueError):
            cohere_llm("test")
