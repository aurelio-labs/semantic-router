import inspect
from typing import Any, Callable, Dict, List, Union

from pydantic.v1 import BaseModel

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message, RouteChoice
from semantic_router.utils.logger import logger
import re


def get_schemas(items: List[Union[BaseModel, Callable]]) -> List[Dict[str, Any]]:
    schemas = []
    for item in items:
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
        schemas.append(schema)
    return schemas


def convert_param_type_to_json_type(param_type: str) -> str:
    if param_type == "int":
        return "number"
    if param_type == "float":
        return "number"
    if param_type == "str":
        return "string"
    if param_type == "bool":
        return "boolean"
    if param_type == "NoneType":
        return "null"
    if param_type == "list":
        return "array"
    else:
        return "object"


def get_schemas_openai(items: List[Callable]) -> List[Dict[str, Any]]:
    schemas = []
    for item in items:
        if not callable(item):
            raise ValueError("Provided item must be a callable function.")

        docstring = inspect.getdoc(item)
        signature = inspect.signature(item)

        schema = {
            "type": "function",
            "function": {
                "name": item.__name__,
                "description": docstring if docstring else "No description available.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

        for param_name, param in signature.parameters.items():
            param_type = (
                param.annotation.__name__
                if param.annotation != inspect.Parameter.empty
                else "Any"
            )
            param_description = "No description available."
            param_required = param.default is inspect.Parameter.empty

            # Attempt to extract the parameter description from the docstring
            if docstring:
                param_doc_regex = re.compile(rf":param {param_name}:(.*?)\n(?=:\w|$)", re.S)
                match = param_doc_regex.search(docstring)
                if match:
                    param_description = match.group(1).strip()

            schema["function"]["parameters"]["properties"][param_name] = {
                "type": convert_param_type_to_json_type(param_type),
                "description": param_description,
            }

            if param_required:
                schema["function"]["parameters"]["required"].append(param_name)

        schemas.append(schema)

    return schemas


# TODO: Add route layer object to the input, solve circular import issue
async def route_and_execute(
    query: str, llm: BaseLLM, functions: List[Callable], layer
) -> Any:
    route_choice: RouteChoice = layer(query)

    for function in functions:
        if function.__name__ == route_choice.name:
            if route_choice.function_call:
                return function(**route_choice.function_call)

    logger.warning("No function found, calling LLM.")
    llm_input = [Message(role="user", content=query)]
    return llm(llm_input)
