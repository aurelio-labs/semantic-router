import pytest

from semantic_router.llms.unify import UnifyLLM
from semantic_router.schema import Message
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def unify_llm(mocker):
    mocker.patch("unify.clients.Unify")
    return UnifyLLM(unify_api_key="fake-api-key")


class TestUnifyLLM:

    def test_unify_llm_init_success_1(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = unify_llm
        assert llm.client is not None  

    def test_unify_llm_init_success_2(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = unify_llm
        assert llm.name == "llama-3-8b-chat@together-ai"
        assert llm.temperature == 0.01
        assert llm.max_tokens == 200
        assert llm.stream is False

    def test_unify_llm_call_success(self, mocker):
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"message": {"content": "test response"}}
        mocker.patch("unify.clients.Unify.generate", return_value=mock_response)

        output = unify_llm([Message(role="user", content="test")])
        assert output == "test response"

    def test_unify_llm_error_handling(self, mocker):
        mocker.patch("requests.post", side_effect=Exception("LLM error"))
        with pytest.raises(Exception) as exc_info:
            unify_llm([Message(role="user", content="test")])
        assert "LLM error" in str(exc_info.value)
