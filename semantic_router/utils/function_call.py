import inspect
import json
from typing import Any, Callable, Union

from pydantic import BaseModel

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message, RouteChoice
from semantic_router.utils.logger import logger


def get_schema(item: Union[BaseModel, Callable]) -> dict[str, Any]:
    if isinstance(item, BaseModel):
        signature_parts = []
        for field_name, field_model in item.__annotations__.items():
            field_info = item.__fields__[field_name]
            default_value = field_info.default

            if default_value:
                default_repr = repr(default_value)
                signature_part = (
                    f"{field_name}: {field_model.__name__} = {default_repr}"
                )
            else:
                signature_part = f"{field_name}: {field_model.__name__}"

            signature_parts.append(signature_part)
        signature = f"({', '.join(signature_parts)}) -> str"
        schema = {
            "name": item.__class__.__name__,
            "description": item.__doc__,
            "signature": signature,
        }
    else:
        schema = {
            "name": item.__name__,
            "description": str(inspect.getdoc(item)),
            "signature": str(inspect.signature(item)),
            "output": str(inspect.signature(item).return_annotation),
        }
    return schema


def extract_function_inputs(
    query: str, llm: BaseLLM, function_schema: dict[str, Any]
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
    output = llm(llm_input)
    if not output:
        raise Exception("No output generated for extract function input")

    output = output.replace("'", '"').strip().rstrip(",")

    function_inputs = json.loads(output)
    if not is_valid_inputs(function_inputs, function_schema):
        raise ValueError("Invalid inputs")
    return function_inputs


def is_valid_inputs(inputs: dict[str, Any], function_schema: dict[str, Any]) -> bool:
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


# TODO: Add route layer object to the input, solve circular import issue
async def route_and_execute(
    query: str, llm: BaseLLM, functions: list[Callable], layer
) -> Any:
    route_choice: RouteChoice = layer(query)

    for function in functions:
        if function.__name__ == route_choice.name:
            if route_choice.function_call:
                return function(**route_choice.function_call)

    logger.warning("No function found, calling LLM.")
    llm_input = [Message(role="user", content=query)]
    return llm(llm_input)
