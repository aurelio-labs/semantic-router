import json

import pytest
from chat_http import capture_chat_requests

from semantic_router.llms import OpenAICompatibleLLM, RequestyLLM
from semantic_router.schema import Message


@pytest.fixture
def requesty_llm(mocker, monkeypatch):
    monkeypatch.delenv("REQUESTY_CHAT_MODEL_NAME", raising=False)
    mocker.patch("openai.Client")
    return RequestyLLM(requesty_api_key="test_api_key")


class TestRequestyLLM:
    def test_requesty_llm_is_openai_compatible(self, requesty_llm):
        assert isinstance(requesty_llm, OpenAICompatibleLLM)
        assert requesty_llm._base_url == "https://router.requesty.ai/v1", (
            "Default base URL not set correctly"
        )

    def test_requesty_llm_init_with_api_key(self, requesty_llm):
        assert requesty_llm._client is not None, "Client should be initialized"
        assert requesty_llm.name == "openai/gpt-4o-mini", (
            "Default name not set correctly"
        )

    def test_requesty_llm_model_from_env_var(self, monkeypatch, mocker):
        mocker.patch("openai.Client")
        monkeypatch.setenv("REQUESTY_CHAT_MODEL_NAME", "anthropic/claude-sonnet-4")
        llm = RequestyLLM(requesty_api_key="test_api_key")
        assert llm.name == "anthropic/claude-sonnet-4"
        llm = RequestyLLM(name="openai/gpt-4o", requesty_api_key="test_api_key")
        assert llm.name == "openai/gpt-4o", "An explicit name should win"

    def test_requesty_llm_init_success(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = RequestyLLM()
        assert llm._client is not None

    def test_requesty_llm_init_from_env_var(self, monkeypatch, mocker):
        openai_mock = mocker.patch("openai.OpenAI")
        monkeypatch.setenv("REQUESTY_API_KEY", "env-api-key")
        monkeypatch.delenv("REQUESTY_CHAT_MODEL_NAME", raising=False)
        RequestyLLM()
        openai_mock.assert_called_once_with(
            api_key="env-api-key", base_url="https://router.requesty.ai/v1"
        )

    def test_requesty_llm_init_without_api_key(self, monkeypatch):
        monkeypatch.delenv("REQUESTY_API_KEY", raising=False)
        with pytest.raises(ValueError) as e:
            RequestyLLM()
        assert "REQUESTY_API_KEY" in str(e.value)

    def test_requesty_llm_call_uninitialized_client(self, requesty_llm):
        # Set the client to None to simulate an uninitialized client
        requesty_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            requesty_llm(llm_input)
        assert "OpenAI compatible client is not initialized." in str(e.value)

    def test_requesty_llm_init_exception(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("openai.OpenAI", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            RequestyLLM()
        assert (
            "OpenAI compatible API client failed to initialize. "
            "Error: Initialization error" in str(e.value)
        )

    def test_requesty_llm_call_success(self, requesty_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch.object(
            requesty_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = requesty_llm(llm_input)
        assert output == "test"

    def test_requesty_llm_call_goes_to_requesty(self, monkeypatch, mocker):
        monkeypatch.setenv("REQUESTY_API_KEY", "env-api-key")
        monkeypatch.delenv("REQUESTY_CHAT_MODEL_NAME", raising=False)
        chat_requests = capture_chat_requests(mocker)
        llm = RequestyLLM()
        assert llm([Message(role="user", content="test")]) == "test"
        (request,) = chat_requests
        assert str(request.url) == "https://router.requesty.ai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer env-api-key"
        assert json.loads(request.content)["model"] == "openai/gpt-4o-mini"
