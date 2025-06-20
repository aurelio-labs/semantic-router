import inspect
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message, RouteChoice
from semantic_router.utils.logger import logger


class Parameter(BaseModel):
    """Parameter for a function.

    :param name: The name of the parameter.
    :type name: str
    :param description: The description of the parameter.
    :type description: Optional[str]
    :param type: The type of the parameter.
    :type type: str
    :param default: The default value of the parameter.
    :type default: Any
    :param required: Whether the parameter is required.
    :type required: bool
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="The name of the parameter")
    description: Optional[str] = Field(
        default=None, description="The description of the parameter"
    )
    type: str = Field(description="The type of the parameter")
    default: Any = Field(description="The default value of the parameter")
    required: bool = Field(description="Whether the parameter is required")

    def to_ollama(self):
        """Convert the parameter to a dictionary for an Ollama-compatible function schema.

        :return: The parameter in dictionary format.
        :rtype: Dict[str, Any]
        """
        return {
            self.name: {
                "description": self.description,
                "type": self.type,
            }
        }


class FunctionSchema:
    """Class that consumes a function and can return a schema required by
    different LLMs for function calling.
    """

    name: str = Field(description="The name of the function")
    description: str = Field(description="The description of the function")
    signature: str = Field(description="The signature of the function")
    output: str = Field(description="The output of the function")
    parameters: List[Parameter] = Field(description="The parameters of the function")

    def __init__(self, function: Union[Callable, BaseModel]):
        """Initialize the FunctionSchema.

        :param function: The function to consume.
        :type function: Union[Callable, BaseModel]
        """
        self.function = function
        if callable(function):
            self._process_function(function)
        elif isinstance(function, BaseModel):
            raise NotImplementedError("Pydantic BaseModel not implemented yet.")
        else:
            raise TypeError("Function must be a Callable or BaseModel")

    def _process_function(self, function: Callable):
        """Process the function to get the name, description, signature, and output.

        :param function: The function to process.
        :type function: Callable
        """
        self.name = function.__name__
        self.description = str(inspect.getdoc(function))
        self.signature = str(inspect.signature(function))
        self.output = str(inspect.signature(function).return_annotation)
        parameters = []
        for param in inspect.signature(function).parameters.values():
            parameters.append(
                Parameter(
                    name=param.name,
                    type=param.annotation.__name__,
                    default=param.default,
                    required=False if param.default is param.empty else True,
                )
            )
        self.parameters = parameters

    def to_ollama(self):
        """Convert the FunctionSchema to an Ollama-compatible function schema dictionary.

        :return: The function schema in dictionary format.
        :rtype: Dict[str, Any]
        """
        schema_dict = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param.name: {
                            "description": (
                                param.description
                                if isinstance(param.description, str)
                                else None
                            ),
                            "type": self._ollama_type_mapping(param.type),
                        }
                        for param in self.parameters
                    },
                    "required": [
                        param.name for param in self.parameters if param.required
                    ],
                },
            },
        }
        return schema_dict

    def _ollama_type_mapping(self, param_type: str) -> str:
        """Map the parameter type to an Ollama-compatible type.

        :param param_type: The type of the parameter.
        :type param_type: str
        :return: The Ollama-compatible type.
        :rtype: str
        """
        if param_type == "int":
            return "number"
        elif param_type == "float":
            return "number"
        elif param_type == "str":
            return "string"
        elif param_type == "bool":
            return "boolean"
        else:
            return "object"


def get_schema_list(items: List[Union[BaseModel, Callable]]) -> List[Dict[str, Any]]:
    """Get a list of function schemas from a list of functions or Pydantic BaseModels.

    :param items: The functions or BaseModels to get the schemas for.
    :type items: List[Union[BaseModel, Callable]]
    :return: A list of function schemas.
    :rtype: List[Dict[str, Any]]
    """
    schemas = []
    for item in items:
        schema = get_schema(item)
        schemas.append(schema)
    return schemas


def get_schema(item: Union[BaseModel, Callable]) -> Dict[str, Any]:
    """Get a function schema from a function or Pydantic BaseModel.

    :param item: The function or BaseModel to get the schema for.
    :type item: Union[BaseModel, Callable]
    :return: The function schema.
    :rtype: Dict[str, Any]
    """
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


def convert_python_type_to_json_type(param_type: str) -> str:
    """Convert a Python type to a JSON type.

    :param param_type: The type of the parameter.
    :type param_type: str
    :return: The JSON type.
    :rtype: str
    """
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


# TODO: Add route layer object to the input, solve circular import issue
async def route_and_execute(
    query: str, llm: BaseLLM, functions: List[Callable], layer
) -> Any:
    """Route and execute a function.

    :param query: The query to route and execute.
    :type query: str
    :param llm: The LLM to use.
    :type llm: BaseLLM
    :param functions: The functions to execute.
    :type functions: List[Callable]
    :param layer: The layer to use.
    :type layer: Layer
    :return: The result of the function.
    :rtype: Any
    """
    route_choice: RouteChoice = layer(query)

    for function in functions:
        if function.__name__ == route_choice.name:
            if route_choice.function_call:
                return function(**route_choice.function_call[0])

    logger.warning("No function found, calling LLM.")
    llm_input = [Message(role="user", content=query)]
    return llm(llm_input)
