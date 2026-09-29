import pytest

from semantic_router.llms import AtlasCloudLLM
from semantic_router.schema import Message


@pytest.fixture
def atlascloud_llm(mocker):
    mocker.patch("openai.Client")
    return AtlasCloudLLM(atlascloud_api_key="test_api_key")


class TestAtlasCloudLLM:
    def test_atlascloud_llm_init_with_api_key(self, atlascloud_llm):
        assert atlascloud_llm._client is not None, "Client should be initialized"
        assert atlascloud_llm.name == "Qwen/Qwen3-235B-A22B-Instruct-2507", (
            "Default name not set correctly"
        )

    def test_atlascloud_llm_init_success(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = AtlasCloudLLM()
        assert llm._client is not None

    def test_atlascloud_llm_init_without_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            AtlasCloudLLM()

    def test_atlascloud_llm_call_uninitialized_client(self, atlascloud_llm):
        # Set the client to None to simulate an uninitialized client
        atlascloud_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            atlascloud_llm(llm_input)
        assert "Atlas Cloud client is not initialized." in str(e.value)

    def test_atlascloud_llm_init_exception(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("openai.OpenAI", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            AtlasCloudLLM()
        assert (
            "Atlas Cloud API client failed to initialize. Error: Initialization error"
            in str(e.value)
        )

    def test_atlascloud_llm_call_success(self, atlascloud_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch.object(
            atlascloud_llm._client.chat.completions,
            "create",
            return_value=mock_completion,
        )
        llm_input = [Message(role="user", content="test")]
        output = atlascloud_llm(llm_input)
        assert output == "test"
