from unittest.mock import patch

import pytest

_ = pytest.importorskip("llama_cpp")

from llama_cpp import Llama  # noqa: E402

from semantic_router.llms.llamacpp import LlamaCppLLM  # noqa: E402
from semantic_router.schema import Message  # noqa: E402


@pytest.fixture
def llamacpp_llm(mocker):
    mock_llama = mocker.patch("llama_cpp.Llama", spec=Llama)
    llm = mock_llama.return_value
    return LlamaCppLLM(llm=llm)


class TestLlamaCppLLM:
    def test_llama_cpp_import_errors(self, llamacpp_llm):
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError) as error:
                LlamaCppLLM(llamacpp_llm.llm)

        assert (
            "Please install LlamaCPP to use Llama CPP llm. "
            "You can install it with: "
            "`pip install 'semantic-router[local]'`" in str(error.value)
        )

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

    def test_llamacpp_extract_function_inputs(self, llamacpp_llm, mocker):
        llamacpp_llm.llm.create_chat_completion = mocker.Mock(
            return_value={
                "choices": [
                    {"message": {"content": "{'timezone': 'America/New_York'}"}}
                ]
            }
        )
        test_schema = {
            "name": "get_time",
            "description": 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
            "signature": "(timezone: str) -> str",
            "output": "<class 'str'>",
        }
        test_query = "What time is it in America/New_York?"

        llamacpp_llm.extract_function_inputs(
            query=test_query, function_schemas=[test_schema]
        )

    def test_llamacpp_extract_function_inputs_invalid(self, llamacpp_llm, mocker):
        with pytest.raises(ValueError):
            llamacpp_llm.llm.create_chat_completion = mocker.Mock(
                return_value={
                    "choices": [
                        {"message": {"content": "{'time': 'America/New_York'}"}}
                    ]
                }
            )
            test_schema = {
                "name": "get_time",
                "description": 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
                "signature": "(timezone: str) -> str",
                "output": "<class 'str'>",
            }
            test_query = "What time is it in America/New_York?"

            llamacpp_llm.extract_function_inputs(
                query=test_query, function_schemas=[test_schema]
            )
