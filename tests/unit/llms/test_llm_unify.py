import pytest

from semantic_router.llms.unify import UnifyLLM
from semantic_router.schema import Message
from unify.clients import Unify, AsyncUnify
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def unify_llm(mocker):
    mocker.patch("unify.clients.Unify")
    mocker.patch.object(Unify, "set_endpoint", return_value=None)
    mocker.patch.object(AsyncUnify, "set_endpoint", return_value=None)

    return UnifyLLM(unify_api_key="fake-api-key")


class TestUnifyLLM:

    def test_unify_llm_init_success_1(self, unify_llm, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch.object(unify_llm.client, "set_endpoint", return_value=None)

        assert unify_llm.client is not None

    def test_unify_llm_init_success_2(self, unify_llm, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")

        assert unify_llm.name == "llama-3-8b-chat@together-ai"
        assert unify_llm.temperature == 0.01
        assert unify_llm.max_tokens == 200
        assert unify_llm.stream is False

    def test_unify_llm_call_success(self, unify_llm, mocker):

        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"message": {"content": "test response"}}
        mocker.patch.object(unify_llm.client, "generate", return_value=mock_response)

        output = unify_llm([Message(role="user", content="test")])
        assert output == "test response"

    def test_unify_llm_error_handling(self, unify_llm, mocker):

        mocker.patch("requests.post", side_effect=Exception("LLM error"))
        with pytest.raises(Exception) as exc_info:
            unify_llm([Message(role="user", content="test")])
        assert "LLM error" in str(exc_info.value)
