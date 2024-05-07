import pytest

from semantic_router.llms.openai import OpenAILLM, get_schemas_openai
from semantic_router.schema import Message


@pytest.fixture
def openai_llm(mocker):
    mocker.patch("openai.Client")
    return OpenAILLM(openai_api_key="test_api_key")


class TestOpenAILLM:
    def test_openai_llm_init_with_api_key(self, openai_llm):
        assert openai_llm.client is not None, "Client should be initialized"
        assert openai_llm.name == "gpt-3.5-turbo", "Default name not set correctly"

    def test_openai_llm_init_success(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        llm = OpenAILLM()
        assert llm.client is not None

    def test_openai_llm_init_without_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            OpenAILLM()

    def test_openai_llm_call_uninitialized_client(self, openai_llm):
        # Set the client to None to simulate an uninitialized client
        openai_llm.client = None
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
            openai_llm.client.chat.completions, "create", return_value=mock_completion
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
    #         openai_llm.client.chat.completions, "create", return_value=mock_completion
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
            openai_llm.client.chat.completions, "create", return_value=mock_completion
        )

        llm_input = [Message(role="user", content="test")]
        function_schemas = [{"type": "function", "name": "sample_function"}]
        output = openai_llm(llm_input, function_schemas)
        assert (
            output == "[{'function_name': 'sample_function', 'arguments': {'timezone': 'America/New_York'}}]"
        ), "Output did not match expected result with function schema"

    def test_openai_llm_call_with_invalid_tool_calls(self, openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.tool_calls = None
        mocker.patch.object(
            openai_llm.client.chat.completions, "create", return_value=mock_completion
        )
        llm_input = [Message(role="user", content="test")]
        function_schemas = [{"type": "function", "name": "sample_function"}]

        with pytest.raises(Exception) as exc_info:
            openai_llm(llm_input, function_schemas)

        expected_error_message = "LLM error: Invalid output, expected a tool call."
        actual_error_message = str(exc_info.value)
        assert (
            expected_error_message in actual_error_message
        ), f"Expected error message: '{expected_error_message}', but got: '{actual_error_message}'"

    def test_openai_llm_call_with_no_arguments_in_tool_calls(self, openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.tool_calls = [
            mocker.MagicMock(function=mocker.MagicMock(arguments=None))
        ]
        mocker.patch.object(
            openai_llm.client.chat.completions, "create", return_value=mock_completion
        )
        llm_input = [Message(role="user", content="test")]
        function_schemas = [{"type": "function", "name": "sample_function"}]

        with pytest.raises(Exception) as exc_info:
            openai_llm(llm_input, function_schemas)

        expected_error_message = (
            "LLM error: Invalid output, expected arguments to be specified for each tool call."
        )
        actual_error_message = str(exc_info.value)
        assert (
            expected_error_message in actual_error_message
        ), f"Expected error message: '{expected_error_message}', but got: '{actual_error_message}'"


    def test_extract_function_inputs(self, openai_llm, mocker):
        query = "fetch user data"
        function_schemas = [
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
                                "description": "The ID of the user."
                            }
                        },
                        "required": ["user_id"]
                    }
                }
            }
        ]

        # Mock the __call__ method to return a JSON string as expected
        mocker.patch.object(OpenAILLM, "__call__", return_value='[{"function_name": "get_user_data", "arguments": {"user_id": "123"}}]')
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
        assert result == [{"function_name": "get_user_data", "arguments": {"user_id": "123"}}], "The function inputs should match the expected dictionary."