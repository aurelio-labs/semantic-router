import json

import pytest
from chat_http import capture_chat_requests

from semantic_router.llms import AtlasCloudLLM, OpenAICompatibleLLM
from semantic_router.schema import Message

DEFAULT_MODEL = "deepseek-ai/deepseek-v4-flash"


@pytest.fixture
def atlascloud_llm(mocker, monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_CHAT_MODEL_NAME", raising=False)
    mocker.patch("openai.Client")
    return AtlasCloudLLM(atlascloud_api_key="test_api_key")


class TestAtlasCloudLLM:
    def test_atlascloud_llm_is_openai_compatible(self, atlascloud_llm):
        assert isinstance(atlascloud_llm, OpenAICompatibleLLM)
        assert atlascloud_llm._base_url == "https://api.atlascloud.ai/v1", (
            "Default base URL not set correctly"
        )

    def test_atlascloud_llm_init_with_api_key(self, atlascloud_llm):
        assert atlascloud_llm._client is not None, "Client should be initialized"
        assert atlascloud_llm.name == DEFAULT_MODEL, "Default name not set correctly"

    def test_atlascloud_llm_model_from_env_var(self, monkeypatch, mocker):
        mocker.patch("openai.Client")
        monkeypatch.setenv("ATLASCLOUD_CHAT_MODEL_NAME", "qwen/qwen3.5-35b-a3b")
        llm = AtlasCloudLLM(atlascloud_api_key="test_api_key")
        assert llm.name == "qwen/qwen3.5-35b-a3b"
        llm = AtlasCloudLLM(name="zai-org/glm-5.1", atlascloud_api_key="test_api_key")
        assert llm.name == "zai-org/glm-5.1", "An explicit name should win"

    def test_atlascloud_llm_init_from_env_var(self, monkeypatch, mocker):
        openai_mock = mocker.patch("openai.OpenAI")
        monkeypatch.setenv("ATLASCLOUD_API_KEY", "env-api-key")
        AtlasCloudLLM()
        openai_mock.assert_called_once_with(
            api_key="env-api-key", base_url="https://api.atlascloud.ai/v1"
        )

    def test_atlascloud_llm_init_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
        with pytest.raises(ValueError) as e:
            AtlasCloudLLM()
        assert "ATLASCLOUD_API_KEY" in str(e.value)

    def test_atlascloud_llm_call_uninitialized_client(self, atlascloud_llm):
        # Set the client to None to simulate an uninitialized client
        atlascloud_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            atlascloud_llm(llm_input)
        assert "OpenAI compatible client is not initialized." in str(e.value)

    def test_atlascloud_llm_init_exception(self, monkeypatch, mocker):
        monkeypatch.setenv("ATLASCLOUD_API_KEY", "fake-api-key")
        mocker.patch("openai.OpenAI", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            AtlasCloudLLM()
        assert (
            "OpenAI compatible API client failed to initialize. "
            "Error: Initialization error" in str(e.value)
        )

    def test_atlascloud_llm_call_success(self, atlascloud_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch.object(
            atlascloud_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = atlascloud_llm(llm_input)
        assert output == "test"

    def test_atlascloud_llm_call_goes_to_atlascloud(self, monkeypatch, mocker):
        monkeypatch.setenv("ATLASCLOUD_API_KEY", "env-api-key")
        monkeypatch.delenv("ATLASCLOUD_CHAT_MODEL_NAME", raising=False)
        chat_requests = capture_chat_requests(mocker)
        llm = AtlasCloudLLM()
        assert llm([Message(role="user", content="test")]) == "test"
        (request,) = chat_requests
        assert str(request.url) == "https://api.atlascloud.ai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer env-api-key"
        assert json.loads(request.content)["model"] == DEFAULT_MODEL
