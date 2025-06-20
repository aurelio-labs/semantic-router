import pytest

from semantic_router.llms import OpenRouterLLM
from semantic_router.schema import Message


@pytest.fixture
def openrouter_llm(mocker):
    mocker.patch("openai.Client")
    return OpenRouterLLM(openrouter_api_key="test_api_key")


class TestOpenRouterLLM:
    def test_openrouter_llm_init_with_api_key(self, openrouter_llm):
        assert openrouter_llm._client is not None, "Client should be initialized"
        assert openrouter_llm.name == "mistralai/mistral-7b-instruct", (
            "Default name not set correctly"
        )

    def test_openrouter_llm_init_success(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = OpenRouterLLM()
        assert llm._client is not None

    def test_openrouter_llm_init_without_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            OpenRouterLLM()

    def test_openrouter_llm_call_uninitialized_client(self, openrouter_llm):
        # Set the client to None to simulate an uninitialized client
        openrouter_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            openrouter_llm(llm_input)
        assert "OpenRouter client is not initialized." in str(e.value)

    def test_openrouter_llm_init_exception(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("openai.OpenAI", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            OpenRouterLLM()
        assert (
            "OpenRouter API client failed to initialize. Error: Initialization error"
            in str(e.value)
        )

    def test_openrouter_llm_call_success(self, openrouter_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch.object(
            openrouter_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = openrouter_llm(llm_input)
        assert output == "test"
