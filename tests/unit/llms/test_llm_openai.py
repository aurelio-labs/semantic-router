import pytest

from semantic_router.llms.openai import OpenAILLM, get_schemas_openai
from semantic_router.schema import Message


@pytest.fixture
def openai_llm(mocker):
    mocker.patch("openai.Client")
    return OpenAILLM(openai_api_key="test_api_key")


get_user_data_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_user_data",
            "description": "Function to fetch user data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user.",
                    }
                },
                "required": ["user_id"],
            },
        },
    }
]

example_function_schema = {
    "parameters": {
        "properties": {
            "user_id": {"type": "string", "description": "The ID of the user."}
        },
        "required": ["user_id"],
    }
}


class TestOpenAILLM:
    def test_openai_llm_init_with_api_key(self, openai_llm):
        assert openai_llm._client is not None, "Client should be initialized"
        assert openai_llm.name == "gpt-4o", "Default name not set correctly"

    def test_openai_llm_init_success(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = OpenAILLM()
        assert llm._client is not None

    def test_openai_llm_init_without_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            OpenAILLM()

    def test_openai_llm_call_uninitialized_client(self, openai_llm):
        # Set the client to None to simulate an uninitialized client
        openai_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role="user", content="test")]
            openai_llm(llm_input)
        assert "OpenAI client is not initialized." in str(e.value)

    def test_openai_llm_init_exception(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("openai.OpenAI", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            OpenAILLM()
        assert (
            "OpenAI API client failed to initialize. Error: Initialization error"
            in str(e.value)
        )

    def test_openai_llm_call_success(self, openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = "test"

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch.object(
            openai_llm._client.chat.completions, "create", return_value=mock_completion
        )
        llm_input = [Message(role="user", content="test")]
        output = openai_llm(llm_input)
        assert output == "test"

    def test_get_schemas_openai_with_valid_callable(self):
        def sample_function(param1: int, param2: str = "default") -> str:
            """Sample function for testing."""
            return f"param1: {param1}, param2: {param2}"

        expected_schema = [
            {
                "type": "function",
                "function": {
                    "name": "sample_function",
                    "description": "Sample function for testing.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param1": {
                                "type": "number",
                                "description": "No description available.",
                            },
                            "param2": {
                                "type": "string",
                                "description": "No description available.",
                            },
                        },
                        "required": ["param1"],
                    },
                },
            }
        ]
        schema = get_schemas_openai([sample_function])
        assert schema == expected_schema, "Schema did not match expected output."

    def test_get_schemas_openai_with_non_callable(self):
        non_callable = "I am not a function"
        with pytest.raises(ValueError):
            get_schemas_openai([non_callable])

    # def test_openai_llm_call_with_function_schema(self, openai_llm, mocker):
    #     mock_completion = mocker.MagicMock()
    #     mock_completion.choices[0].message.tool_calls = [
    #         mocker.MagicMock(function=mocker.MagicMock(arguments="result"))
    #     ]
    #     mocker.patch.object(
    #         openai_llm._client.chat.completions, "create", return_value=mock_completion
    #     )
    #     llm_input = [Message(role="user", content="test")]
    #     function_schemas = [{"type": "function", "name": "sample_function"}]
    #     output = openai_llm(llm_input, function_schemas)
    #     assert (
    #         output == "result"
    #     ), "Output did not match expected result with function schema"

    def test_openai_llm_call_with_function_schema(self, openai_llm, mocker):
        # Mocking the tool call with valid JSON arguments and setting the function name explicitly
        mock_function = mocker.MagicMock(arguments='{"timezone":"America/New_York"}')
        mock_function.name = "sample_function"  # Set the function name explicitly here
        mock_tool_call = mocker.MagicMock(function=mock_function)
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.tool_calls = [mock_tool_call]

        mocker.patch.object(
            openai_llm._client.chat.completions, "create", return_value=mock_completion
        )

        llm_input = [Message(role="user", content="test")]
        function_schemas = [{"type": "function", "name": "sample_function"}]
        output = openai_llm(llm_input, function_schemas)
        assert (
            output
            == "[{'function_name': 'sample_function', 'arguments': {'timezone': 'America/New_York'}}]"
        ), "Output did not match expected result with function schema"

    def test_openai_llm_call_with_invalid_tool_calls(self, openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.tool_calls = None
        mocker.patch.object(
            openai_llm._client.chat.completions, "create", return_value=mock_completion
        )
        llm_input = [Message(role="user", content="test")]
        function_schemas = [{"type": "function", "name": "sample_function"}]

        with pytest.raises(Exception) as exc_info:
            openai_llm(llm_input, function_schemas)

        expected_error_message = "LLM error: Invalid output, expected a tool call."
        actual_error_message = str(exc_info.value)
        assert expected_error_message in actual_error_message, (
            f"Expected error message: '{expected_error_message}', but got: '{actual_error_message}'"
        )

    def test_openai_llm_call_with_no_arguments_in_tool_calls(self, openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.tool_calls = [
            mocker.MagicMock(function=mocker.MagicMock(arguments=None))
        ]
        mocker.patch.object(
            openai_llm._client.chat.completions, "create", return_value=mock_completion
        )
        llm_input = [Message(role="user", content="test")]
        function_schemas = [{"type": "function", "name": "sample_function"}]

        with pytest.raises(Exception) as exc_info:
            openai_llm(llm_input, function_schemas)

        expected_error_message = "LLM error: Invalid output, expected arguments to be specified for each tool call."
        actual_error_message = str(exc_info.value)
        assert expected_error_message in actual_error_message, (
            f"Expected error message: '{expected_error_message}', but got: '{actual_error_message}'"
        )

    def test_extract_function_inputs(self, openai_llm, mocker):
        query = "fetch user data"
        function_schemas = get_user_data_schema

        # Mock the __call__ method to return a JSON string as expected
        mocker.patch.object(
            OpenAILLM,
            "__call__",
            return_value='[{"function_name": "get_user_data", "arguments": {"user_id": "123"}}]',
        )
        result = openai_llm.extract_function_inputs(query, function_schemas)

        # Ensure the __call__ method is called with the correct parameters
        expected_messages = [
            Message(
                role="system",
                content="You are an intelligent AI. Given a command or request from the user, call the function to complete the request.",
            ),
            Message(role="user", content=query),
        ]
        openai_llm.__call__.assert_called_once_with(
            messages=expected_messages, function_schemas=function_schemas
        )

        # Check if the result is as expected
        assert result == [
            {"function_name": "get_user_data", "arguments": {"user_id": "123"}}
        ], "The function inputs should match the expected dictionary."

    def test_openai_llm_call_with_no_tool_calls_specified(self, openai_llm, mocker):
        # Mocking the completion object to simulate no tool calls being specified
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.tool_calls = []  # Empty list of tool calls

        # Patching the completions.create method to return the mocked completion
        mocker.patch.object(
            openai_llm._client.chat.completions, "create", return_value=mock_completion
        )

        # Input message list
        llm_input = [Message(role="user", content="test")]
        # Example function schema
        function_schemas = [{"type": "function", "name": "sample_function"}]

        # Expecting a generic Exception to be raised due to no tool calls being specified
        with pytest.raises(Exception) as exc_info:
            openai_llm(llm_input, function_schemas)

        # Check if the raised Exception contains the expected message
        expected_error_message = (
            "LLM error: Invalid output, expected at least one tool to be specified."
        )
        assert str(exc_info.value) == expected_error_message, (
            f"Expected error message: '{expected_error_message}', but got: '{str(exc_info.value)}'"
        )

    def test_extract_function_inputs_no_output(self, openai_llm, mocker):
        query = "fetch user data"
        function_schemas = [{"type": "function", "name": "get_user_data"}]

        # Mock the __call__ method to return an empty string
        mocker.patch.object(OpenAILLM, "__call__", return_value="")

        # Expecting an Exception due to no output
        with pytest.raises(Exception) as exc_info:
            openai_llm.extract_function_inputs(query, function_schemas)

        assert (
            str(exc_info.value) == "No output generated for extract function input"
        ), "Expected exception message not found"

    def test_extract_function_inputs_invalid_output(self, openai_llm, mocker):
        query = "fetch user data"
        function_schemas = [{"type": "function", "name": "get_user_data"}]

        # Mock the __call__ method to return a JSON string
        mocker.patch.object(
            OpenAILLM,
            "__call__",
            return_value='[{"function_name": "get_user_data", "arguments": {"user_id": "123"}}]',
        )

        # Mock _is_valid_inputs to return False
        mocker.patch.object(OpenAILLM, "_is_valid_inputs", return_value=False)

        # Expecting a ValueError due to invalid inputs
        with pytest.raises(ValueError) as exc_info:
            openai_llm.extract_function_inputs(query, function_schemas)

        assert str(exc_info.value) == "Invalid inputs", (
            "Expected exception message not found"
        )

    def test_is_valid_inputs_missing_function_name(self, openai_llm, mocker):
        # Mock the logger to capture the error messages
        mocked_logger = mocker.patch("semantic_router.utils.logger.logger.error")

        # Input where 'function_name' is missing
        inputs = [{"arguments": {"user_id": "123"}}]
        function_schemas = get_user_data_schema

        # Call the method with the test inputs
        result = openai_llm._is_valid_inputs(inputs, function_schemas)

        # Assert that the method returns False due to missing 'function_name'
        assert not result, (
            "The method should return False when 'function_name' is missing"
        )

        # Check that the appropriate error message was logged
        mocked_logger.assert_called_once_with(
            "Missing 'function_name' or 'arguments' in inputs"
        )

    def test_is_valid_inputs_missing_arguments(self, openai_llm, mocker):
        # Mock the logger to capture the error messages
        mocked_logger = mocker.patch("semantic_router.utils.logger.logger.error")

        # Input where 'arguments' is missing but 'function_name' is present
        inputs = [{"function_name": "get_user_data"}]
        function_schemas = get_user_data_schema

        # Call the method with the test inputs
        result = openai_llm._is_valid_inputs(inputs, function_schemas)

        # Assert that the method returns False due to missing 'arguments'
        assert not result, "The method should return False when 'arguments' is missing"

        # Check that the appropriate error message was logged
        mocked_logger.assert_called_once_with(
            "Missing 'function_name' or 'arguments' in inputs"
        )

    def test_is_valid_inputs_no_matching_schema(self, openai_llm, mocker):
        # Mock the logger to capture the error messages
        mocked_logger = mocker.patch("semantic_router.utils.logger.logger.error")

        # Input where 'function_name' does not match any schema
        inputs = [
            {
                "function_name": "name_that_does_not_exist_in_schema",
                "arguments": {"user_id": "123"},
            }
        ]
        function_schemas = get_user_data_schema

        # Call the method with the test inputs
        result = openai_llm._is_valid_inputs(inputs, function_schemas)

        # Assert that the method returns False due to no matching function schema
        assert not result, (
            "The method should return False when no matching function schema is found"
        )

        # Check that the appropriate error message was logged
        expected_error_message = "No matching function schema found for function name: name_that_does_not_exist_in_schema"
        mocked_logger.assert_called_once_with(expected_error_message)

    def test_is_valid_inputs_validation_failed(self, openai_llm, mocker):
        # Mock the logger to capture the error messages
        mocked_logger = mocker.patch("semantic_router.utils.logger.logger.error")

        # Input where 'arguments' do not meet the schema requirements
        inputs = [
            {"function_name": "get_user_data", "arguments": {"user_id": 123}}
        ]  # user_id should be a string, not an integer
        function_schemas = get_user_data_schema

        # Mock the _validate_single_function_inputs method to return False
        mocker.patch.object(
            OpenAILLM, "_validate_single_function_inputs", return_value=False
        )

        # Call the method with the test inputs
        result = openai_llm._is_valid_inputs(inputs, function_schemas)

        # Assert that the method returns False due to validation failure
        assert not result, "The method should return False when validation fails"

        # Check that the appropriate error message was logged
        expected_error_message = "Validation failed for function name: get_user_data"
        mocked_logger.assert_called_once_with(expected_error_message)

    def test_is_valid_inputs_exception_handling(self, openai_llm, mocker):
        # Mock the logger to capture the error messages
        mocked_logger = mocker.patch("semantic_router.utils.logger.logger.error")

        # Create test inputs that are valid but mock an internal method to raise an exception
        inputs = [{"function_name": "get_user_data", "arguments": {"user_id": "123"}}]
        function_schemas = get_user_data_schema

        # Mock a method used within _is_valid_inputs to raise an Exception
        mocker.patch.object(
            OpenAILLM,
            "_validate_single_function_inputs",
            side_effect=Exception("Test exception"),
        )

        # Call the method with the test inputs
        result = openai_llm._is_valid_inputs(inputs, function_schemas)

        # Assert that the method returns False due to exception
        assert not result, "The method should return False when an exception occurs"

        # Check that the appropriate error message was logged
        mocked_logger.assert_called_once_with("Input validation error: Test exception")

    def test_validate_single_function_inputs_missing_required_param(
        self, openai_llm, mocker
    ):
        # Mock the logger to capture the error messages
        mocked_logger = mocker.patch("semantic_router.utils.logger.logger.error")

        # Define the function schema with a required parameter
        function_schema = example_function_schema

        # Input dictionary missing the required 'user_id' parameter
        inputs = {}

        # Call the method with the test inputs
        result = openai_llm._validate_single_function_inputs(inputs, function_schema)

        # Assert that the method returns False due to missing required parameter
        assert not result, (
            "The method should return False when a required parameter is missing"
        )

        # Check that the appropriate error message was logged
        expected_error_message = "Required input 'user_id' missing from query"
        mocked_logger.assert_called_once_with(expected_error_message)

    def test_validate_single_function_inputs_incorrect_type(self, openai_llm, mocker):
        # Mock the logger to capture the error messages
        mocked_logger = mocker.patch("semantic_router.utils.logger.logger.error")

        # Define the function schema with type specifications
        function_schema = example_function_schema

        # Input dictionary with incorrect type for 'user_id'
        inputs = {"user_id": 123}  # user_id should be a string, not an integer

        # Call the method with the test inputs
        result = openai_llm._validate_single_function_inputs(inputs, function_schema)

        # Assert that the method returns False due to incorrect type
        assert not result, "The method should return False when input type is incorrect"

        # Check that the appropriate error message was logged
        expected_error_message = "Input type for 'user_id' is not string"
        mocked_logger.assert_called_once_with(expected_error_message)

    def test_validate_single_function_inputs_exception_handling(
        self, openai_llm, mocker
    ):
        # Mock the logger to capture the error messages
        mocked_logger = mocker.patch("semantic_router.utils.logger.logger.error")

        # Create a custom class that raises an exception when any attribute is accessed
        class SchemaSimulator:
            def __getitem__(self, item):
                raise Exception("Test exception")

        # Replace the function_schema with an instance of this custom class
        function_schema = SchemaSimulator()

        # Call the method with the test inputs
        result = openai_llm._validate_single_function_inputs(
            {"user_id": "123"}, function_schema
        )

        # Assert that the method returns False due to exception
        assert not result, "The method should return False when an exception occurs"

        # Check that the appropriate error message was logged
        mocked_logger.assert_called_once_with(
            "Single input validation error: Test exception"
        )
