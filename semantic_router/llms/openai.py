import os
from typing import List, Optional, Any, Callable, Dict, Union

import openai
from openai._types import NotGiven, NOT_GIVEN

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger
import json
from semantic_router.utils.function_call import (
    get_schema,
    convert_python_type_to_json_type,
)
import inspect
import re
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)


class OpenAILLM(BaseLLM):
    client: Optional[openai.OpenAI]
    temperature: Optional[float]
    max_tokens: Optional[int]

    def __init__(
        self,
        name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        if name is None:
            name = EncoderDefault.OPENAI.value["language_model"]
        super().__init__(name=name)
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key cannot be 'None'.")
        try:
            self.client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            raise ValueError(
                f"OpenAI API client failed to initialize. Error: {e}"
            ) from e
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _extract_tool_calls_info(
        self, tool_calls: List[ChatCompletionMessageToolCall]
    ) -> List[Dict[str, Any]]:
        tool_calls_info = []
        for tool_call in tool_calls:
            if tool_call.function.arguments is None:
                raise ValueError(
                    "Invalid output, expected arguments to be specified for each tool call."
                )
            tool_calls_info.append(
                {
                    "function_name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                }
            )
        return tool_calls_info

    def __call__(
        self,
        messages: List[Message],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if self.client is None:
            raise ValueError("OpenAI client is not initialized.")
        try:
            tools: Union[List[Dict[str, Any]], NotGiven] = (
                function_schemas if function_schemas is not None else NOT_GIVEN
            )

            completion = self.client.chat.completions.create(
                model=self.name,
                messages=[m.to_openai() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,  # type: ignore # We pass a list of dicts which get interpreted as Iterable[ChatCompletionToolParam].
            )

            if function_schemas:
                tool_calls = completion.choices[0].message.tool_calls
                if tool_calls is None:
                    raise ValueError("Invalid output, expected a tool call.")
                if len(tool_calls) < 1:
                    raise ValueError(
                        "Invalid output, expected at least one tool to be specified."
                    )

                # Collecting multiple tool calls information
                output = str(
                    self._extract_tool_calls_info(tool_calls)
                )  # str in keeping with base type.
            else:
                content = completion.choices[0].message.content
                if content is None:
                    raise ValueError("Invalid output, expected content.")
                output = content
            return output

        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise Exception(f"LLM error: {e}") from e

    def extract_function_inputs(
        self, query: str, function_schemas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        system_prompt = "You are an intelligent AI. Given a command or request from the user, call the function to complete the request."
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query),
        ]
        output = self(messages=messages, function_schemas=function_schemas)
        if not output:
            raise Exception("No output generated for extract function input")
        output = output.replace("'", '"')
        function_inputs = json.loads(output)
        logger.info(f"Function inputs: {function_inputs}")
        if not self._is_valid_inputs(function_inputs, function_schemas):
            raise ValueError("Invalid inputs")
        return function_inputs

    def _is_valid_inputs(
        self, inputs: List[Dict[str, Any]], function_schemas: List[Dict[str, Any]]
    ) -> bool:
        """Determine if the functions chosen by the LLM exist within the function_schemas,
        and if the input arguments are valid for those functions."""
        try:
            for input_dict in inputs:
                # Check if 'function_name' and 'arguments' keys exist in each input dictionary
                if "function_name" not in input_dict or "arguments" not in input_dict:
                    logger.error("Missing 'function_name' or 'arguments' in inputs")
                    return False

                function_name = input_dict["function_name"]
                arguments = input_dict["arguments"]

                # Find the matching function schema based on function_name
                matching_schema = next(
                    (
                        schema["function"]
                        for schema in function_schemas
                        if schema["function"]["name"] == function_name
                    ),
                    None,
                )
                if not matching_schema:
                    logger.error(
                        f"No matching function schema found for function name: {function_name}"
                    )
                    return False

                # Validate the inputs against the function schema
                if not self._validate_single_function_inputs(
                    arguments, matching_schema
                ):
                    logger.error(
                        f"Validation failed for function name: {function_name}"
                    )
                    return False

            return True
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            return False

    def _validate_single_function_inputs(
        self, inputs: Dict[str, Any], function_schema: Dict[str, Any]
    ) -> bool:
        """Validate the extracted inputs against the function schema"""
        try:
            # Access the parameters and their properties from the function schema directly
            parameters = function_schema["parameters"]["properties"]
            required_params = function_schema["parameters"].get("required", [])

            # Check if all required parameters are present in the inputs
            for param_name in required_params:
                if param_name not in inputs:
                    logger.error(f"Required input '{param_name}' missing from query")
                    return False

            # Check if the types of the inputs match the expected types (if type checking is needed)
            for param_name, param_info in parameters.items():
                if param_name in inputs:
                    expected_type = param_info["type"]
                    # This is a simple type check, consider expanding it based on your needs
                    if expected_type == "string" and not isinstance(
                        inputs[param_name], str
                    ):
                        logger.error(
                            f"Input type for '{param_name}' is not {expected_type}"
                        )
                        return False

            return True
        except Exception as e:
            logger.error(f"Single input validation error: {str(e)}")
            return False


def get_schemas_openai(items: List[Callable]) -> List[Dict[str, Any]]:
    schemas = []
    for item in items:
        if not callable(item):
            raise ValueError("Provided item must be a callable function.")

        # Use the existing get_schema function to get the basic schema
        basic_schema = get_schema(item)

        # Initialize the function schema with basic details
        function_schema = {
            "name": basic_schema["name"],
            "description": basic_schema["description"],
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

        # Extract parameter details from the signature
        signature = inspect.signature(item)
        docstring = inspect.getdoc(item)
        param_doc_regex = re.compile(r":param (\w+):(.*?)\n(?=:\w|$)", re.S)
        doc_params = param_doc_regex.findall(docstring) if docstring else []

        for param_name, param in signature.parameters.items():
            param_type = (
                param.annotation.__name__
                if param.annotation != inspect.Parameter.empty
                else "Any"
            )
            param_description = "No description available."
            param_required = param.default is inspect.Parameter.empty

            # Find the parameter description in the docstring
            for doc_param_name, doc_param_desc in doc_params:
                if doc_param_name == param_name:
                    param_description = doc_param_desc.strip()
                    break

            function_schema["parameters"]["properties"][param_name] = {
                "type": convert_python_type_to_json_type(param_type),
                "description": param_description,
            }

            if param_required:
                function_schema["parameters"]["required"].append(param_name)

        schemas.append({"type": "function", "function": function_schema})

    return schemas
