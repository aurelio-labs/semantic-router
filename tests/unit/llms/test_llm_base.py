import pytest
from semantic_router.llms import BaseLLM
from unittest.mock import patch


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
        test_schemas = [
            {
                "name": "get_time",
                "description": 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
                "signature": "(timezone: str) -> str",
                "output": "<class 'str'>",
            }
        ]
        test_inputs = [{"timezone": "America/New_York"}]

        assert base_llm._is_valid_inputs(test_inputs, test_schemas) is True

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

    def test_is_valid_inputs_multiple_inputs(self, base_llm, mocker):
        # Mock the logger to capture the error messages
        mocked_logger = mocker.patch("semantic_router.utils.logger.logger.error")

        # Prepare multiple sets of inputs
        test_inputs = [{"timezone": "America/New_York"}, {"timezone": "Europe/London"}]
        test_schemas = [
            {
                "name": "get_time",
                "description": "Finds the current time in a specific timezone.",
                "signature": "(timezone: str) -> str",
                "output": "<class 'str'>",
            }
        ]

        # Call the method with multiple inputs
        result = base_llm._is_valid_inputs(test_inputs, test_schemas)

        # Assert that the method returns False
        assert (
            not result
        ), "Method should return False when multiple inputs are provided"

        # Check that the appropriate error message was logged
        mocked_logger.assert_called_once_with(
            "Only one set of function inputs is allowed."
        )

    def test_is_valid_inputs_exception_handling(self, base_llm, mocker):
        # Mock the logger to capture the error messages
        mocked_logger = mocker.patch("semantic_router.utils.logger.logger.error")

        # Use patch on the method's full path
        with patch(
            "semantic_router.llms.base.BaseLLM._validate_single_function_inputs",
            side_effect=Exception("Test Exception"),
        ):
            test_inputs = [{"timezone": "America/New_York"}]
            test_schemas = [
                {
                    "name": "get_time",
                    "description": "Finds the current time in a specific timezone.",
                    "signature": "(timezone: str) -> str",
                    "output": "<class 'str'>",
                }
            ]

            # Call the method and expect it to return False due to the exception
            result = base_llm._is_valid_inputs(test_inputs, test_schemas)

            # Assert that the method returns False
            assert not result, "Method should return False when an exception occurs"

            # Check that the appropriate error message was logged
            mocked_logger.assert_called_once_with(
                "Input validation error: Test Exception"
            )

    def test_validate_single_function_inputs_exception_handling(self, base_llm, mocker):
        # Mock the logger to capture the error messages
        mocked_logger = mocker.patch("semantic_router.utils.logger.logger.error")

        # Prepare inputs and a malformed function schema
        test_inputs = {"timezone": "America/New_York"}
        malformed_function_schema = {
            "name": "get_time",
            "description": "Finds the current time in a specific timezone.",
            "signature": "(timezone str)",  # Malformed signature missing colon
            "output": "<class 'str'>",
        }

        # Call the method and expect it to return False due to the exception
        result = base_llm._validate_single_function_inputs(
            test_inputs, malformed_function_schema
        )

        # Assert that the method returns False
        assert not result, "Method should return False when an exception occurs"

        # Check that the appropriate error message was logged
        expected_error_message = "Single input validation error: list index out of range"  # Adjust based on the actual exception message
        mocked_logger.assert_called_once_with(expected_error_message)

    def test_extract_parameter_info_valid(self, base_llm):
        # Test with a valid signature
        signature = "(param1: int, param2: str = 'default')"
        expected_names = ["param1", "param2"]
        expected_types = ["int", "str"]
        param_names, param_types = base_llm._extract_parameter_info(signature)
        assert param_names == expected_names, "Parameter names did not match expected"
        assert param_types == expected_types, "Parameter types did not match expected"

    def test_extract_parameter_info_malformed(self, base_llm):
        # Test with a malformed signature
        signature = "(param1 int, param2: str = 'default')"
        with pytest.raises(IndexError):
            base_llm._extract_parameter_info(signature)
