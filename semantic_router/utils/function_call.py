import inspect
from typing import Any, Callable, Dict, List, Union

from pydantic.v1 import BaseModel

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message, RouteChoice
from semantic_router.utils.logger import logger


def get_schema(item: Union[BaseModel, Callable]) -> Dict[str, Any]:
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
