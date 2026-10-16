import pytest

from semantic_router.llms import MiniMaxLLM
from semantic_router.schema import Message


@pytest.fixture
def minimax_llm(mocker):
    mocker.patch("openai.OpenAI")
    mocker.patch("openai.AsyncOpenAI")
    return MiniMaxLLM(minimax_api_key="test_api_key")


class TestMiniMaxLLM:
    def test_initialization_with_api_key(self, minimax_llm):
        assert minimax_llm._client is not None, "Client should be initialized"
        assert minimax_llm._async_client is not None, (
            "Async client should be initialized"
        )
        assert minimax_llm.name == "MiniMax-M2.5", "Default name not set correctly"

    def test_initialization_with_custom_name(self, mocker):
        mocker.patch("openai.OpenAI")
        mocker.patch("openai.AsyncOpenAI")
        llm = MiniMaxLLM(
            name="MiniMax-M2.5-highspeed", minimax_api_key="test_api_key"
        )
        assert llm.name == "MiniMax-M2.5-highspeed"

    def test_initialization_without_api_key(self, mocker, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(ValueError):
            MiniMaxLLM()

    def test_initialization_with_env_api_key(self, mocker, monkeypatch):
        mocker.patch("openai.OpenAI")
        mocker.patch("openai.AsyncOpenAI")
        monkeypatch.setenv("MINIMAX_API_KEY", "env_test_key")
        llm = MiniMaxLLM()
        assert llm._client is not None

    def test_initialization_exception(self, mocker):
        mocker.patch("openai.OpenAI", side_effect=Exception("Init error"))
        with pytest.raises(ValueError) as exc_info:
            MiniMaxLLM(minimax_api_key="test_api_key")
        assert "MiniMax API client failed to initialize" in str(exc_info.value)

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
        assert "MiniMax client is not initialized." in str(exc_info.value)

    def test_call_success(self, minimax_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test response"
        mocker.patch.object(
            minimax_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = minimax_llm(llm_input)
        assert output == "test response"

    def test_call_strips_think_tags(self, minimax_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = (
            "<think>internal reasoning</think>actual response"
        )
        mocker.patch.object(
            minimax_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = minimax_llm(llm_input)
        assert output == "actual response"

    def test_call_no_output(self, minimax_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = None
        mocker.patch.object(
            minimax_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
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

    def test_strip_think_tags_multiline(self):
        text = "<think>\nsome\nmultiline\nthinking\n</think>Hello world"
        result = MiniMaxLLM._strip_think_tags(text)
        assert result == "Hello world"

    def test_strip_think_tags_no_tags(self):
        text = "Hello world"
        result = MiniMaxLLM._strip_think_tags(text)
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_acall_success(self, minimax_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "async response"
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
