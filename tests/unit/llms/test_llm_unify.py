import pytest

from semantic_router.llms import UnifyLLM
from semantic_router.schema import Message


@pytest.fixture
def unify_llm():
    return UnifyLLM()


class TestUnifyLLM:
    def test_unify_llm_init_success(self, unify_llm):
        assert unify_llm.name == "gpt-4o@openai"
        assert unify_llm.temperature == 0.01
        assert unify_llm.max_tokens == 200
        assert unify_llm.stream is False

    def test_unify_llm_call_success(self, unify_llm, mocker):
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"message": {"content": "test response"}}
        mocker.patch("requests.post", return_value=mock_response)

        output = unify_llm([Message(role="user", content="test")])
        assert output == "test response"

    def test_ollama_llm_error_handling(self, unify_llm, mocker):
        mocker.patch("requests.post", side_effect=Exception("LLM error"))
        with pytest.raises(Exception) as exc_info:
            unify_llm([Message(role="user", content="test")])
        assert "LLM error" in str(exc_info.value)
