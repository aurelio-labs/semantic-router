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
        self, inputs: dict[str, Any], function_schema: dict[str, Any]
    ) -> bool:
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
            logger.error(f"Input validation error: {str(e)}")
            return False

    def extract_function_inputs(
        self, query: str, function_schema: dict[str, Any]
    ) -> dict:
        logger.info("Extracting function input...")

        prompt = f"""
        You are a helpful assistant designed to output JSON.
        Given the following function schema
        << {function_schema} >>
        and query
        << {query} >>
        extract the parameters values from the query, in a valid JSON format.
        Example:
        Input:
        query: "How is the weather in Hawaii right now in International units?"
        schema:
        {{
            "name": "get_weather",
            "description": "Useful to get the weather in a specific location",
            "signature": "(location: str, degree: str) -> str",
            "output": "<class 'str'>",
        }}

        Result: {{
            "location": "London",
            "degree": "Celsius",
        }}

        Input:
        query: {query}
        schema: {function_schema}
        Result:
        """
        llm_input = [Message(role="user", content=prompt)]
        output = self(llm_input)

        if not output:
            raise Exception("No output generated for extract function input")

        output = output.replace("'", '"').strip().rstrip(",")
        logger.info(f"LLM output: {output}")
        function_inputs = json.loads(output)
        logger.info(f"Function inputs: {function_inputs}")
        if not self._is_valid_inputs(function_inputs, function_schema):
            raise ValueError("Invalid inputs")
        return function_inputs
