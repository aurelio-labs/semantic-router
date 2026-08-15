import pytest

from semantic_router.llms import OrcaRouterLLM
from semantic_router.schema import Message


@pytest.fixture
def orcarouter_llm(mocker):
    mocker.patch("openai.Client")
    return OrcaRouterLLM(orcarouter_api_key="test_api_key")


class TestOrcaRouterLLM:
    def test_orcarouter_llm_init_with_api_key(self, orcarouter_llm):
        assert orcarouter_llm._client is not None, "Client should be initialized"
        assert orcarouter_llm.name == "orcarouter/auto", "Default name not set correctly"

    def test_orcarouter_llm_init_success(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = OrcaRouterLLM()
        assert llm._client is not None

    def test_orcarouter_llm_init_without_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            OrcaRouterLLM()

    def test_orcarouter_llm_call_uninitialized_client(self, orcarouter_llm):
        # Set the client to None to simulate an uninitialized client
        orcarouter_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            orcarouter_llm(llm_input)
        assert "OrcaRouter client is not initialized." in str(e.value)

    def test_orcarouter_llm_init_exception(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("openai.OpenAI", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            OrcaRouterLLM()
        assert (
            "OrcaRouter API client failed to initialize. Error: Initialization error"
            in str(e.value)
        )

    def test_orcarouter_llm_call_success(self, orcarouter_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch.object(
            orcarouter_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = orcarouter_llm(llm_input)
        assert output == "test"
