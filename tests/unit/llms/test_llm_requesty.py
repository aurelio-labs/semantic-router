import pytest

from semantic_router.llms import RequestyLLM
from semantic_router.schema import Message


@pytest.fixture
def requesty_llm(mocker):
    mocker.patch("openai.Client")
    return RequestyLLM(requesty_api_key="test_api_key")


class TestRequestyLLM:
    def test_requesty_llm_init_with_api_key(self, requesty_llm):
        assert requesty_llm._client is not None, "Client should be initialized"
        assert (
            requesty_llm.name == "openai/gpt-4o-mini"
        ), "Default name not set correctly"

    def test_requesty_llm_init_success(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = RequestyLLM()
        assert llm._client is not None

    def test_requesty_llm_init_without_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            RequestyLLM()

    def test_requesty_llm_call_uninitialized_client(self, requesty_llm):
        # Set the client to None to simulate an uninitialized client
        requesty_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            requesty_llm(llm_input)
        assert "Requesty client is not initialized." in str(e.value)

    def test_requesty_llm_init_exception(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("openai.OpenAI", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            RequestyLLM()
        assert (
            "Requesty API client failed to initialize. Error: Initialization error"
            in str(e.value)
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
