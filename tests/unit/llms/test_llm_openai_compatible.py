import pytest

from semantic_router.llms import OpenAICompatibleLLM
from semantic_router.schema import Message

BASE_URL = "https://api.example.com/v1"


@pytest.fixture
def openai_compatible_llm(mocker):
    mocker.patch("openai.Client")
    return OpenAICompatibleLLM(
        name="provider/model", base_url=BASE_URL, api_key="test_api_key"
    )


class TestOpenAICompatibleLLM:
    def test_openai_compatible_llm_init_with_api_key(self, openai_compatible_llm):
        assert openai_compatible_llm._client is not None, "Client should be initialized"
        assert openai_compatible_llm.name == "provider/model", "Name not set correctly"
        assert openai_compatible_llm._base_url == BASE_URL, "Base URL not set correctly"

    def test_openai_compatible_llm_init_from_env_var(self, monkeypatch, mocker):
        mocker.patch("openai.Client")
        monkeypatch.setenv("EXAMPLE_API_KEY", "fake-api-key")
        llm = OpenAICompatibleLLM(
            name="provider/model",
            base_url=BASE_URL,
            api_key_var_name="EXAMPLE_API_KEY",
        )
        assert llm._client is not None

    def test_openai_compatible_llm_api_key_takes_precedence(self, monkeypatch, mocker):
        openai_mock = mocker.patch("openai.OpenAI")
        monkeypatch.setenv("EXAMPLE_API_KEY", "env-api-key")
        OpenAICompatibleLLM(
            name="provider/model",
            base_url=BASE_URL,
            api_key="explicit-api-key",
            api_key_var_name="EXAMPLE_API_KEY",
        )
        openai_mock.assert_called_once_with(
            api_key="explicit-api-key", base_url=BASE_URL
        )

    def test_openai_compatible_llm_init_without_api_key(self, monkeypatch):
        monkeypatch.delenv("EXAMPLE_API_KEY", raising=False)
        with pytest.raises(ValueError) as e:
            OpenAICompatibleLLM(
                name="provider/model",
                base_url=BASE_URL,
                api_key_var_name="EXAMPLE_API_KEY",
            )
        assert "EXAMPLE_API_KEY" in str(e.value)

    def test_openai_compatible_llm_init_without_api_key_or_var_name(self):
        with pytest.raises(ValueError) as _:
            OpenAICompatibleLLM(name="provider/model", base_url=BASE_URL)

    def test_openai_compatible_llm_init_without_base_url(self):
        with pytest.raises(ValueError) as e:
            OpenAICompatibleLLM(name="provider/model", api_key="test_api_key")
        assert "Base URL cannot be 'None'." in str(e.value)

    def test_openai_compatible_llm_init_without_name(self):
        with pytest.raises(ValueError) as e:
            OpenAICompatibleLLM(base_url=BASE_URL, api_key="test_api_key")
        assert "Model name cannot be 'None'." in str(e.value)

    def test_openai_compatible_llm_call_uninitialized_client(
        self, openai_compatible_llm
    ):
        # Set the client to None to simulate an uninitialized client
        openai_compatible_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            openai_compatible_llm(llm_input)
        assert "OpenAI compatible client is not initialized." in str(e.value)

    def test_openai_compatible_llm_init_exception(self, mocker):
        mocker.patch("openai.OpenAI", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            OpenAICompatibleLLM(
                name="provider/model", base_url=BASE_URL, api_key="test_api_key"
            )
        assert (
            "OpenAI compatible API client failed to initialize. "
            "Error: Initialization error" in str(e.value)
        )

    def test_openai_compatible_llm_call_success(self, openai_compatible_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch.object(
            openai_compatible_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = openai_compatible_llm(llm_input)
        assert output == "test"

    def test_openai_compatible_llm_call_failure(self, openai_compatible_llm, mocker):
        mocker.patch.object(
            openai_compatible_llm._client.chat.completions,
            "create",
            side_effect=Exception("API call failed"),
        )
        llm_input = [Message(role="user", content="test")]
        with pytest.raises(Exception) as e:
            openai_compatible_llm(llm_input)
        assert "LLM error: API call failed" in str(e.value)

    def test_openai_compatible_llm_call_no_output(self, openai_compatible_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = None

        mocker.patch.object(
            openai_compatible_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        with pytest.raises(Exception) as e:
            openai_compatible_llm(llm_input)
        assert "No output generated" in str(e.value)
