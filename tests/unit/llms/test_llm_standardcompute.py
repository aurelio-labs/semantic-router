import json

import pytest
from chat_http import capture_chat_requests

from semantic_router.llms import OpenAICompatibleLLM, StandardComputeLLM
from semantic_router.schema import Message

DEFAULT_MODEL = "StandardCompute"


@pytest.fixture
def standardcompute_llm(mocker, monkeypatch):
    monkeypatch.delenv("STANDARDCOMPUTE_CHAT_MODEL_NAME", raising=False)
    mocker.patch("openai.Client")
    return StandardComputeLLM(standardcompute_api_key="test_api_key")


class TestStandardComputeLLM:
    def test_standardcompute_llm_is_openai_compatible(self, standardcompute_llm):
        assert isinstance(standardcompute_llm, OpenAICompatibleLLM)
        assert standardcompute_llm._base_url == "https://api.stdcmpt.com/v1", (
            "Default base URL not set correctly"
        )

    def test_standardcompute_llm_init_with_api_key(self, standardcompute_llm):
        assert standardcompute_llm._client is not None, "Client should be initialized"
        assert standardcompute_llm.name == DEFAULT_MODEL, (
            "Default name not set correctly"
        )

    def test_standardcompute_llm_model_from_env_var(self, monkeypatch, mocker):
        mocker.patch("openai.Client")
        monkeypatch.setenv(
            "STANDARDCOMPUTE_CHAT_MODEL_NAME", "deepseek/deepseek-v4-flash"
        )
        llm = StandardComputeLLM(standardcompute_api_key="test_api_key")
        assert llm.name == "deepseek/deepseek-v4-flash"
        llm = StandardComputeLLM(
            name="z-ai/glm-5.3", standardcompute_api_key="test_api_key"
        )
        assert llm.name == "z-ai/glm-5.3", "An explicit name should win"

    def test_standardcompute_llm_init_from_env_var(self, monkeypatch, mocker):
        openai_mock = mocker.patch("openai.OpenAI")
        monkeypatch.setenv("STANDARDCOMPUTE_API_KEY", "env-api-key")
        StandardComputeLLM()
        openai_mock.assert_called_once_with(
            api_key="env-api-key", base_url="https://api.stdcmpt.com/v1"
        )

    def test_standardcompute_llm_init_without_api_key(self, monkeypatch):
        monkeypatch.delenv("STANDARDCOMPUTE_API_KEY", raising=False)
        with pytest.raises(ValueError) as e:
            StandardComputeLLM()
        assert "STANDARDCOMPUTE_API_KEY" in str(e.value)

    def test_standardcompute_llm_call_uninitialized_client(self, standardcompute_llm):
        # Set the client to None to simulate an uninitialized client
        standardcompute_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            standardcompute_llm(llm_input)
        assert "OpenAI compatible client is not initialized." in str(e.value)

    def test_standardcompute_llm_init_exception(self, monkeypatch, mocker):
        monkeypatch.setenv("STANDARDCOMPUTE_API_KEY", "fake-api-key")
        mocker.patch("openai.OpenAI", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            StandardComputeLLM()
        assert (
            "OpenAI compatible API client failed to initialize. "
            "Error: Initialization error" in str(e.value)
        )

    def test_standardcompute_llm_call_success(self, standardcompute_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch.object(
            standardcompute_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = standardcompute_llm(llm_input)
        assert output == "test"

    def test_standardcompute_llm_call_goes_to_standardcompute(
        self, monkeypatch, mocker
    ):
        monkeypatch.setenv("STANDARDCOMPUTE_API_KEY", "env-api-key")
        monkeypatch.delenv("STANDARDCOMPUTE_CHAT_MODEL_NAME", raising=False)
        chat_requests = capture_chat_requests(mocker)
        llm = StandardComputeLLM()
        assert llm([Message(role="user", content="test")]) == "test"
        (request,) = chat_requests
        assert str(request.url) == "https://api.stdcmpt.com/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer env-api-key"
        assert json.loads(request.content)["model"] == DEFAULT_MODEL
