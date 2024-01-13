import pytest
from llama_cpp import Llama

from semantic_router.llms import LlamaCppLLM
from semantic_router.schema import Message


@pytest.fixture
def llamacpp_llm(mocker):
    mock_llama = mocker.patch("llama_cpp.Llama", spec=Llama)
    llm = mock_llama.return_value
    return LlamaCppLLM(llm=llm)


class TestLlamaCppLLM:
    def test_llamacpp_llm_init_success(self, llamacpp_llm):
        assert llamacpp_llm.name == "llama.cpp"
        assert llamacpp_llm.temperature == 0.2
        assert llamacpp_llm.max_tokens == 200
        assert llamacpp_llm.llm is not None

    def test_llamacpp_llm_call_success(self, llamacpp_llm, mocker):
        llamacpp_llm.llm.create_chat_completion = mocker.Mock(
            return_value={"choices": [{"message": {"content": "test"}}]}
        )

        llm_input = [Message(role="user", content="test")]
        output = llamacpp_llm(llm_input)
        assert output == "test"

    def test_llamacpp_llm_grammar(self, llamacpp_llm):
        llamacpp_llm._grammar()
