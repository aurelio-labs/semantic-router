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

    def test_base_llm_is_valid_inputs_valid_input_pass(self, base_llm):
        test_schema = {
            "name": "get_time",
            "description": 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
            "signature": "(timezone: str) -> str",
            "output": "<class 'str'>",
        }
        test_inputs = {"timezone": "America/New_York"}

        assert base_llm._is_valid_inputs(test_inputs, test_schema) is True

    @pytest.mark.skip(reason="TODO: bug in is_valid_inputs")
    def test_base_llm_is_valid_inputs_valid_input_fail(self, base_llm):
        test_schema = {
            "name": "get_time",
            "description": 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
            "signature": "(timezone: str) -> str",
            "output": "<class 'str'>",
        }
        test_inputs = {"timezone": None}

        assert base_llm._is_valid_inputs(test_inputs, test_schema) is False

    def test_base_llm_is_valid_inputs_invalid_false(self, base_llm):
        test_schema = {
            "name": "get_time",
            "description": 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
        }
        test_inputs = {"timezone": "America/New_York"}

        assert base_llm._is_valid_inputs(test_inputs, test_schema) is False

    def test_base_llm_extract_function_inputs(self, base_llm):
        with pytest.raises(NotImplementedError):
            test_schema = {
                "name": "get_time",
                "description": 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
                "signature": "(timezone: str) -> str",
                "output": "<class 'str'>",
            }
            test_query = "What time is it in America/New_York?"
            base_llm.extract_function_inputs(test_schema, test_query)

    def test_base_llm_extract_function_inputs_no_output(self, base_llm, mocker):
        with pytest.raises(Exception):
            base_llm.output = mocker.Mock(return_value=None)
            test_schema = {
                "name": "get_time",
                "description": 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
                "signature": "(timezone: str) -> str",
                "output": "<class 'str'>",
            }
            test_query = "What time is it in America/New_York?"
            base_llm.extract_function_inputs(test_schema, test_query)
