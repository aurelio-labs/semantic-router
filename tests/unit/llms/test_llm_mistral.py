from unittest.mock import patch

import pytest

from semantic_router.llms import MistralAILLM
from semantic_router.schema import Message


@pytest.fixture
def mistralai_llm(mocker):
    mocker.patch("mistralai.client.MistralClient")
    return MistralAILLM(mistralai_api_key="test_api_key")


class TestMistralAILLM:
    def test_mistral_llm_import_errors(self):
        with patch.dict("sys.modules", {"mistralai": None}):
            with pytest.raises(ImportError) as error:
                MistralAILLM()

        assert (
            "Please install MistralAI to use MistralAI LLM. "
            "You can install it with: "
            "`pip install 'semantic-router[mistralai]'`" in str(error.value)
        )

    def test_mistralai_llm_init_with_api_key(self, mistralai_llm):
        assert mistralai_llm._client is not None, "Client should be initialized"
        assert mistralai_llm.name == "mistral-tiny", "Default name not set correctly"

    def test_mistralai_llm_init_success(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = MistralAILLM()
        assert llm._client is not None

    def test_mistralai_llm_init_without_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            MistralAILLM()

    def test_mistralai_llm_call_uninitialized_client(self, mistralai_llm):
        # Set the client to None to simulate an uninitialized client
        mistralai_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            mistralai_llm(llm_input)
        assert "MistralAI client is not initialized." in str(e.value)

    def test_mistralai_llm_init_exception(self, mocker):
        mocker.patch(
            "mistralai.client.MistralClient",
            side_effect=Exception("Initialization error"),
        )
        with pytest.raises(ValueError) as e:
            MistralAILLM()
        assert "MistralAI API key cannot be 'None'." in str(e.value)

    def test_mistralai_llm_call_success(self, mistralai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch.object(
            mistralai_llm._client,
            "chat",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = mistralai_llm(llm_input)
        assert output == "test"
