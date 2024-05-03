import json
from typing import Any, List, Optional

from pydantic.v1 import BaseModel

from semantic_router.schema import Message
from semantic_router.utils.logger import logger


class BaseLLM(BaseModel):
    name: str

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)

    def __call__(self, messages: List[Message]) -> Optional[str]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def _is_valid_inputs(
        self, inputs: list[dict[str, Any]], function_schemas: list[dict[str, Any]]
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
                matching_schema = next((schema for schema in function_schemas if schema["name"] == function_name), None)
                if not matching_schema:
                    logger.error(f"No matching function schema found for function name: {function_name}")
                    return False

                # Validate the inputs against the function schema
                if not self._validate_single_function_inputs(arguments, matching_schema):
                    return False

            return True
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            return False

    def _validate_single_function_inputs(self, inputs: dict[str, Any], function_schema: dict[str, Any]) -> bool:
        """Validate the extracted inputs against the function schema"""
        try:
            # Extract parameter names and types from the signature string
            signature = function_schema["signature"]
            param_info = [param.strip() for param in signature[1:-1].split(",")]
            param_names = [info.split(":")[0].strip() for info in param_info]
            param_types = [
                info.split(":")[1].strip().split("=")[0].strip() for info in param_info
            ]
            for name, type_str in zip(param_names, param_types):
                if name not in inputs:
                    logger.error(f"Input {name} missing from query")
                    return False
            return True
        except Exception as e:
            logger.error(f"Single input validation error: {str(e)}")
            return False

    def _extract_parameter_info(self, signature: str) -> tuple[list[str], list[str]]:
        """Extract parameter names and types from the function signature."""
        param_info = [param.strip() for param in signature[1:-1].split(",")]
        param_names = [info.split(":")[0].strip() for info in param_info]
        param_types = [
            info.split(":")[1].strip().split("=")[0].strip() for info in param_info
        ]
        return param_names, param_types

    def extract_function_inputs(
        self, query: str, function_schemas: list[dict[str, Any]]
    ) -> dict:
        logger.info("Extracting function input...")

        prompt = f"""
You are a precise program designed to output valid JSON based on Python function schemas and a given QUERY. Follow these steps:

1. Review the FUNCTION_SCHEMAS provided below:
### FUNCTION_SCHEMAS Start ###
{json.dumps(function_schemas, indent=4)}
### FUNCTION_SCHEMAS End ###

2. Analyze the QUERY:
### QUERY Start ###
{query}
### QUERY End ###

3. Select the most relevant function schema(s) based on:
- The function's output aligning with the information requested in the QUERY.
- The availability of required input information in the QUERY.

4. Format the JSON output as follows:
### FORMATTING_INSTRUCTIONS Start ###
[
    {{
        "function_name": "FUNCTION_NAME",
        "arguments": {{
            "ARGUMENT_NAME": "VALUE",
            ...
        }}
    }},
    ...
]
### FORMATTING_INSTRUCTIONS End ###

Output the JSON now:
"""
        llm_input = [Message(role="user", content=prompt)]
        output = self(llm_input)

        if not output:
            raise Exception("No output generated for extract function input")

        output = output.replace("'", '"').strip().rstrip(",")
        logger.info(f"LLM output: {output}")
        function_inputs = json.loads(output)
        logger.info(f"Function inputs: {function_inputs}")
        if not self._is_valid_inputs(function_inputs, function_schemas):
            raise ValueError("Invalid inputs")
        return function_inputs