import pytest

from semantic_router.llms import BaseLLM


class TestBaseLLM:

    @pytest.fixture
    def base_llm(self):
        return BaseLLM(name="TestLLM")

    @pytest.fixture
    def mixed_function_schema(self):
        return {
            "name": "test_function",
            "description": "A test function with mixed mandatory and optional parameters.",
            "signature": "(mandatory1, mandatory2: int, optional1=None, optional2: str = 'default')",
        }

    @pytest.fixture
    def mandatory_params(self):
        return ["param1", "param2"]

    @pytest.fixture
    def all_params(self):
        return ["param1", "param2", "optional1"]

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

    def test_mandatory_args_only(self, base_llm, mixed_function_schema):
        inputs = {"mandatory1": "value1", "mandatory2": 42}
        assert base_llm._is_valid_inputs(inputs, mixed_function_schema)  # True is implied

    def test_all_args_provided(self, base_llm, mixed_function_schema):
        inputs = {
            "mandatory1": "value1",
            "mandatory2": 42,
            "optional1": "opt1",
            "optional2": "opt2",
        }
        assert base_llm._is_valid_inputs(inputs, mixed_function_schema)  # True is implied

    def test_missing_mandatory_arg(self, base_llm, mixed_function_schema):
        inputs = {"mandatory1": "value1", "optional1": "opt1", "optional2": "opt2"}
        assert not base_llm._is_valid_inputs(inputs, mixed_function_schema)

    def test_extra_arg_provided(self, base_llm, mixed_function_schema):
        inputs = {
            "mandatory1": "value1",
            "mandatory2": 42,
            "optional1": "opt1",
            "optional2": "opt2",
            "extra": "value",
        }
        assert not base_llm._is_valid_inputs(inputs, mixed_function_schema)

    def test_check_for_mandatory_inputs_all_present(self, base_llm, mandatory_params):
        inputs = {"param1": "value1", "param2": "value2"}
        assert base_llm._check_for_mandatory_inputs(inputs, mandatory_params)  # True is implied

    def test_check_for_mandatory_inputs_missing_one(self, base_llm, mandatory_params):
        inputs = {"param1": "value1"}
        assert not base_llm._check_for_mandatory_inputs(inputs, mandatory_params)

    def test_check_for_extra_inputs_no_extras(self, base_llm, all_params):
        inputs = {"param1": "value1", "param2": "value2"}
        assert base_llm._check_for_extra_inputs(inputs, all_params)  # True is implied

    def test_check_for_extra_inputs_with_extras(self, base_llm, all_params):
        inputs = {"param1": "value1", "param2": "value2", "extra_param": "extra"}
        assert not base_llm._check_for_extra_inputs(inputs, all_params)
