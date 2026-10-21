import json

import pytest
from chat_http import capture_chat_requests

from semantic_router.llms import OpenAICompatibleLLM, OrcaRouterLLM
from semantic_router.schema import Message


@pytest.fixture
def orcarouter_llm(mocker, monkeypatch):
    monkeypatch.delenv("ORCAROUTER_CHAT_MODEL_NAME", raising=False)
    mocker.patch("openai.Client")
    return OrcaRouterLLM(orcarouter_api_key="test_api_key")


class TestOrcaRouterLLM:
    def test_orcarouter_llm_is_openai_compatible(self, orcarouter_llm):
        assert isinstance(orcarouter_llm, OpenAICompatibleLLM)
        assert orcarouter_llm._base_url == "https://api.orcarouter.ai/v1", (
            "Default base URL not set correctly"
        )

    def test_orcarouter_llm_init_with_api_key(self, orcarouter_llm):
        assert orcarouter_llm._client is not None, "Client should be initialized"
        assert orcarouter_llm.name == "orcarouter/auto", (
            "Default name not set correctly"
        )

    def test_orcarouter_llm_model_from_env_var(self, monkeypatch, mocker):
        mocker.patch("openai.Client")
        monkeypatch.setenv("ORCAROUTER_CHAT_MODEL_NAME", "openai/gpt-4o-mini")
        llm = OrcaRouterLLM(orcarouter_api_key="test_api_key")
        assert llm.name == "openai/gpt-4o-mini"
        llm = OrcaRouterLLM(name="openai/gpt-4o", orcarouter_api_key="test_api_key")
        assert llm.name == "openai/gpt-4o", "An explicit name should win"

    def test_orcarouter_llm_init_from_env_var(self, monkeypatch, mocker):
        openai_mock = mocker.patch("openai.OpenAI")
        monkeypatch.setenv("ORCAROUTER_API_KEY", "env-api-key")
        OrcaRouterLLM()
        openai_mock.assert_called_once_with(
            api_key="env-api-key", base_url="https://api.orcarouter.ai/v1"
        )

    def test_orcarouter_llm_init_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError) as e:
            OrcaRouterLLM()
        assert "ORCAROUTER_API_KEY" in str(e.value)

    def test_orcarouter_llm_call_uninitialized_client(self, orcarouter_llm):
        # Set the client to None to simulate an uninitialized client
        orcarouter_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            orcarouter_llm(llm_input)
        assert "OpenAI compatible client is not initialized." in str(e.value)

    def test_orcarouter_llm_init_exception(self, monkeypatch, mocker):
        monkeypatch.setenv("ORCAROUTER_API_KEY", "fake-api-key")
        mocker.patch("openai.OpenAI", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            OrcaRouterLLM()
        assert (
            "OpenAI compatible API client failed to initialize. "
            "Error: Initialization error" in str(e.value)
        )

    def test_orcarouter_llm_call_success(self, orcarouter_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch.object(
            orcarouter_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = orcarouter_llm(llm_input)
        assert output == "test"

    def test_orcarouter_llm_call_goes_to_orcarouter(self, monkeypatch, mocker):
        monkeypatch.setenv("ORCAROUTER_API_KEY", "env-api-key")
        monkeypatch.delenv("ORCAROUTER_CHAT_MODEL_NAME", raising=False)
        chat_requests = capture_chat_requests(mocker)
        llm = OrcaRouterLLM()
        assert llm([Message(role="user", content="test")]) == "test"
        (request,) = chat_requests
        assert str(request.url) == "https://api.orcarouter.ai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer env-api-key"
        assert json.loads(request.content)["model"] == "orcarouter/auto"
