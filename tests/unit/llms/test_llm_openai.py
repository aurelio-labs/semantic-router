import pytest

from semantic_router.llms import OpenAILLM
from semantic_router.schema import Message


@pytest.fixture
def openai_llm(mocker):
    mocker.patch("openai.Client")
    return OpenAILLM(openai_api_key="test_api_key")


class TestOpenAILLM:
    def test_openai_llm_init_with_api_key(self, openai_llm):
        assert openai_llm.client is not None, "Client should be initialized"
        assert openai_llm.name == "gpt-3.5-turbo", "Default name not set correctly"

    def test_openai_llm_init_success(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = OpenAILLM()
        assert llm.client is not None

    def test_openai_llm_init_without_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            OpenAILLM()

    def test_openai_llm_call_uninitialized_client(self, openai_llm):
        # Set the client to None to simulate an uninitialized client
        openai_llm.client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            openai_llm(llm_input)
        assert "OpenAI client is not initialized." in str(e.value)

    def test_openai_llm_init_exception(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("openai.OpenAI", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            OpenAILLM()
        assert (
            "OpenAI API client failed to initialize. Error: Initialization error"
            in str(e.value)
        )

    def test_openai_llm_call_success(self, openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch.object(
            openai_llm.client.chat.completions, "create", return_value=mock_completion
        )
        llm_input = [Message(role="user", content="test")]
        output = openai_llm(llm_input)
        assert output == "test"
