import pytest

from semantic_router.llms.ollama import OllamaLLM
from semantic_router.schema import Message


@pytest.fixture
def ollama_llm():
    return OllamaLLM()


class TestOllamaLLM:
    def test_ollama_llm_init_success(self, ollama_llm):
        assert ollama_llm.temperature == 0.2
        assert ollama_llm.name == "openhermes"
        assert ollama_llm.max_tokens == 200
        assert ollama_llm.stream is False

    def test_ollama_llm_call_success(self, ollama_llm, mocker):
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"message": {"content": "test response"}}
        mocker.patch("requests.post", return_value=mock_response)

        output = ollama_llm([Message(role="user", content="test")])
        assert output == "test response"

    def test_ollama_llm_error_handling(self, ollama_llm, mocker):
        mocker.patch("requests.post", side_effect=Exception("LLM error"))
        with pytest.raises(Exception) as exc_info:
            ollama_llm([Message(role="user", content="test")])
        assert "LLM error" in str(exc_info.value)
