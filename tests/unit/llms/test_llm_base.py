import pytest

from semantic_router.llms import BaseLLM


class TestBaseLLM:
    @pytest.fixture
    def base_llm(self):
        return BaseLLM(name="TestLLM")

    def test_base_llm_initialization(self, base_llm):
        assert base_llm.name == "TestLLM", "Initialization of name failed"

    def test_base_llm_call_method_not_implemented(self, base_llm):
        with pytest.raises(NotImplementedError):
            base_llm("test")
