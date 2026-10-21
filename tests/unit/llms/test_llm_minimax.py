import json

import httpx
import pytest
from chat_http import CHAT_COMPLETION, capture_chat_requests

from semantic_router.llms import MiniMaxLLM, OpenAICompatibleLLM
from semantic_router.schema import Message


@pytest.fixture
def minimax_llm(mocker, monkeypatch):
    monkeypatch.delenv("MINIMAX_CHAT_MODEL_NAME", raising=False)
    mocker.patch("openai.OpenAI")
    mocker.patch("openai.AsyncOpenAI")
    return MiniMaxLLM(minimax_api_key="test_api_key")


def _mock_reply(mocker, llm, content):
    mock_completion = mocker.MagicMock()
    mock_completion.choices[0].message.content = content
    mocker.patch.object(
        llm._client.chat.completions, "create", return_value=mock_completion
    )


class TestMiniMaxLLM:
    def test_is_openai_compatible(self, minimax_llm):
        assert isinstance(minimax_llm, OpenAICompatibleLLM)
        assert minimax_llm._base_url == "https://api.minimax.io/v1", (
            "Default base URL not set correctly"
        )

    def test_initialization_with_api_key(self, minimax_llm):
        assert minimax_llm._client is not None, "Client should be initialized"
        assert minimax_llm._async_client is not None, (
            "Async client should be initialized"
        )
        assert minimax_llm.name == "MiniMax-M3.1-Flash", (
            "Default name not set correctly"
        )
        assert minimax_llm.max_tokens == 4096, "Thinking needs room in max_tokens"

    def test_initialization_with_custom_name(self, mocker):
        mocker.patch("openai.OpenAI")
        mocker.patch("openai.AsyncOpenAI")
        llm = MiniMaxLLM(name="MiniMax-M3", minimax_api_key="test_api_key")
        assert llm.name == "MiniMax-M3"

    def test_model_from_env_var(self, mocker, monkeypatch):
        mocker.patch("openai.OpenAI")
        mocker.patch("openai.AsyncOpenAI")
        monkeypatch.setenv("MINIMAX_CHAT_MODEL_NAME", "MiniMax-M3")
        llm = MiniMaxLLM(minimax_api_key="test_api_key")
        assert llm.name == "MiniMax-M3"

    def test_initialization_without_api_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(ValueError) as e:
            MiniMaxLLM()
        assert "MINIMAX_API_KEY" in str(e.value)

    def test_initialization_with_env_api_key(self, mocker, monkeypatch):
        openai_mock = mocker.patch("openai.OpenAI")
        async_openai_mock = mocker.patch("openai.AsyncOpenAI")
        monkeypatch.setenv("MINIMAX_API_KEY", "env_test_key")
        MiniMaxLLM()
        for client_mock in (openai_mock, async_openai_mock):
            client_mock.assert_called_once_with(
                api_key="env_test_key", base_url="https://api.minimax.io/v1"
            )

    def test_initialization_exception(self, mocker):
        mocker.patch("openai.OpenAI", side_effect=Exception("Init error"))
        with pytest.raises(ValueError) as exc_info:
            MiniMaxLLM(minimax_api_key="test_api_key")
        assert "OpenAI compatible API client failed to initialize" in str(
            exc_info.value
        )

    def test_async_initialization_exception(self, mocker):
        mocker.patch("openai.OpenAI")
        mocker.patch("openai.AsyncOpenAI", side_effect=Exception("Init error"))
        with pytest.raises(ValueError) as exc_info:
            MiniMaxLLM(minimax_api_key="test_api_key")
        assert "MiniMax async client failed to initialize" in str(exc_info.value)

    def test_temperature_clamping_low(self, mocker):
        mocker.patch("openai.OpenAI")
        mocker.patch("openai.AsyncOpenAI")
        llm = MiniMaxLLM(minimax_api_key="test_api_key", temperature=0.0)
        assert llm.temperature == 0.01

    def test_temperature_clamping_high(self, mocker):
        mocker.patch("openai.OpenAI")
        mocker.patch("openai.AsyncOpenAI")
        llm = MiniMaxLLM(minimax_api_key="test_api_key", temperature=2.0)
        assert llm.temperature == 1.0

    def test_temperature_normal(self, mocker):
        mocker.patch("openai.OpenAI")
        mocker.patch("openai.AsyncOpenAI")
        llm = MiniMaxLLM(minimax_api_key="test_api_key", temperature=0.5)
        assert llm.temperature == 0.5

    def test_call_uninitialized_client(self, minimax_llm):
        minimax_llm._client = None
        with pytest.raises(ValueError) as exc_info:
            llm_input = [Message(role="user", content="test")]
            minimax_llm(llm_input)
        assert "OpenAI compatible client is not initialized." in str(exc_info.value)

    def test_call_success(self, minimax_llm, mocker):
        _mock_reply(mocker, minimax_llm, "test response")
        llm_input = [Message(role="user", content="test")]
        output = minimax_llm(llm_input)
        assert output == "test response"

    def test_call_strips_think_tags(self, minimax_llm, mocker):
        _mock_reply(
            mocker, minimax_llm, "<think>internal reasoning</think>actual response"
        )
        llm_input = [Message(role="user", content="test")]
        output = minimax_llm(llm_input)
        assert output == "actual response"

    def test_call_only_thinking_raises(self, minimax_llm, mocker):
        # The model ran out of max_tokens while still thinking
        _mock_reply(mocker, minimax_llm, "<think>still reasoning about the")
        llm_input = [Message(role="user", content="test")]
        with pytest.raises(Exception, match="max_tokens"):
            minimax_llm(llm_input)

    def test_call_no_output(self, minimax_llm, mocker):
        _mock_reply(mocker, minimax_llm, None)
        llm_input = [Message(role="user", content="test")]
        with pytest.raises(Exception, match="LLM error"):
            minimax_llm(llm_input)

    def test_call_api_error(self, minimax_llm, mocker):
        mocker.patch.object(
            minimax_llm._client.chat.completions,
            "create",
            side_effect=Exception("API call failed"),
        )
        llm_input = [Message(role="user", content="test")]
        with pytest.raises(Exception, match="LLM error"):
            minimax_llm(llm_input)

    def test_call_goes_to_minimax(self, mocker, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-api-key")
        monkeypatch.delenv("MINIMAX_CHAT_MODEL_NAME", raising=False)
        chat_requests = capture_chat_requests(mocker)
        llm = MiniMaxLLM(temperature=0.0)
        assert llm([Message(role="user", content="test")]) == "test"
        (request,) = chat_requests
        assert str(request.url) == "https://api.minimax.io/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer env-api-key"
        body = json.loads(request.content)
        assert body["model"] == "MiniMax-M3.1-Flash"
        assert body["max_tokens"] == 4096
        assert body["temperature"] == 0.01, "Temperature should be clamped"

    def test_call_ignores_separate_reasoning(self, mocker, monkeypatch):
        # MiniMax-M3.1-Flash returns its thinking in reasoning_content
        monkeypatch.setenv("MINIMAX_API_KEY", "env-api-key")
        completion = {
            **CHAT_COMPLETION,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "the answer",
                        "reasoning_content": "first, think it through",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        def send(self, request, **kwargs):
            return httpx.Response(200, json=completion, request=request)

        mocker.patch.object(httpx.Client, "send", send)
        llm = MiniMaxLLM()
        assert llm([Message(role="user", content="test")]) == "the answer"

    def test_strip_think_tags_multiline(self):
        text = "<think>\nsome\nmultiline\nthinking\n</think>Hello world"
        result = MiniMaxLLM._strip_think_tags(text)
        assert result == "Hello world"

    def test_strip_think_tags_unclosed(self):
        text = "<think>\nthinking that hit max_tokens"
        assert MiniMaxLLM._strip_think_tags(text) == ""

    def test_strip_think_tags_no_tags(self):
        text = "Hello world"
        result = MiniMaxLLM._strip_think_tags(text)
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_acall_success(self, minimax_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[
            0
        ].message.content = "<think>reasoning</think>async response"
        mocker.patch.object(
            minimax_llm._async_client.chat.completions,
            "create",
            return_value=mocker.AsyncMock(return_value=mock_completion)(),
        )
        llm_input = [Message(role="user", content="test")]
        output = await minimax_llm.acall(llm_input)
        assert output == "async response"

    @pytest.mark.asyncio
    async def test_acall_uninitialized_client(self, minimax_llm):
        minimax_llm._async_client = None
        with pytest.raises(ValueError, match="MiniMax async client is not initialized"):
            llm_input = [Message(role="user", content="test")]
            await minimax_llm.acall(llm_input)

    @pytest.mark.asyncio
    async def test_acall_goes_to_minimax(self, mocker, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-api-key")
        requests = []

        async def send(self, request, **kwargs):
            requests.append(request)
            return httpx.Response(200, json=CHAT_COMPLETION, request=request)

        mocker.patch.object(httpx.AsyncClient, "send", send)
        llm = MiniMaxLLM()
        assert await llm.acall([Message(role="user", content="test")]) == "test"
        (request,) = requests
        assert str(request.url) == "https://api.minimax.io/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer env-api-key"
