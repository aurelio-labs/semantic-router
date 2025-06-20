import pytest

from semantic_router.llms import AzureOpenAILLM
from semantic_router.schema import Message


@pytest.fixture
def azure_openai_llm(mocker):
    mocker.patch("openai.Client")
    return AzureOpenAILLM(openai_api_key="test_api_key", azure_endpoint="test_endpoint")


class TestOpenAILLM:
    def test_azure_openai_llm_init_with_api_key(self, azure_openai_llm):
        assert azure_openai_llm._client is not None, "Client should be initialized"
        assert azure_openai_llm.name == "gpt-4o", "Default name not set correctly"

    def test_azure_openai_llm_init_success(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = AzureOpenAILLM()
        assert llm._client is not None

    def test_azure_openai_llm_init_without_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            AzureOpenAILLM()

    # def test_azure_openai_llm_init_without_azure_endpoint(self, mocker):
    #     mocker.patch("os.getenv", side_effect=[None, "fake-api-key"])
    #     with pytest.raises(ValueError) as e:
    #         AzureOpenAILLM(openai_api_key="test_api_key")
    #     assert "Azure endpoint API key cannot be 'None'." in str(e.value)

    def test_azure_openai_llm_init_without_azure_endpoint(self, mocker):
        mocker.patch(
            "os.getenv",
            side_effect=lambda key, default=None: {
                "OPENAI_CHAT_MODEL_NAME": "test-model-name"
            }.get(key, default),
        )
        with pytest.raises(ValueError) as e:
            AzureOpenAILLM(openai_api_key="test_api_key")
        assert "Azure endpoint API key cannot be 'None'" in str(e.value)

    def test_azure_openai_llm_call_uninitialized_client(self, azure_openai_llm):
        # Set the client to None to simulate an uninitialized client
        azure_openai_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            azure_openai_llm(llm_input)
        assert "AzureOpenAI client is not initialized." in str(e.value)

    def test_azure_openai_llm_init_exception(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch(
            "openai.AzureOpenAI", side_effect=Exception("Initialization error")
        )
        with pytest.raises(ValueError) as e:
            AzureOpenAILLM()
        assert (
            "AzureOpenAI API client failed to initialize. Error: Initialization error"
            in str(e.value)
        )

    def test_azure_openai_llm_temperature_max_tokens_initialization(self):
        test_temperature = 0.5
        test_max_tokens = 100
        azure_llm = AzureOpenAILLM(
            openai_api_key="test_api_key",
            azure_endpoint="test_endpoint",
            temperature=test_temperature,
            max_tokens=test_max_tokens,
        )

        assert azure_llm.temperature == test_temperature, (
            "Temperature not set correctly"
        )
        assert azure_llm.max_tokens == test_max_tokens, "Max tokens not set correctly"

    def test_azure_openai_llm_call_success(self, azure_openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch.object(
            azure_openai_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = azure_openai_llm(llm_input)
        assert output == "test"
