import json
import re
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message, RouteChoice
from semantic_router.utils import function_call
from semantic_router.utils.logger import logger


def is_valid(route_config: str) -> bool:
    """Check if the route config is valid.

    :param route_config: The route config to check.
    :type route_config: str
    :return: Whether the route config is valid.
    :rtype: bool
    """
    try:
        output_json = json.loads(route_config)
        required_keys = ["name", "utterances"]

        if isinstance(output_json, list):
            for item in output_json:
                missing_keys = [key for key in required_keys if key not in item]
                if missing_keys:
                    logger.warning(
                        f"Missing keys in route config: {', '.join(missing_keys)}"
                    )
                    return False
            return True
        else:
            missing_keys = [key for key in required_keys if key not in output_json]
            if missing_keys:
                logger.warning(
                    f"Missing keys in route config: {', '.join(missing_keys)}"
                )
                return False
            else:
                return True
    except json.JSONDecodeError as e:
        logger.error(e)
        return False


class Route(BaseModel):
    """A route for the semantic router.

    :param name: The name of the route.
    :type name: str
    :param utterances: The utterances of the route.
    :type utterances: Union[List[str], List[Any]]
    :param description: The description of the route.
    :type description: Optional[str]
    :param function_schemas: The function schemas of the route.
    :type function_schemas: Optional[List[Dict[str, Any]]]
    :param llm: The LLM to use.
    :type llm: Optional[BaseLLM]
    :param score_threshold: The score threshold of the route.
    :type score_threshold: Optional[float]
    :param metadata: The metadata of the route.
    :type metadata: Optional[Dict[str, Any]]
    """

    name: str
    utterances: Union[List[str], List[Any]]
    description: Optional[str] = None
    function_schemas: Optional[List[Dict[str, Any]]] = None
    llm: Optional[BaseLLM] = None
    score_threshold: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = {}

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self, query: Optional[str] = None) -> RouteChoice:
        """Call the route. If dynamic routes have been provided the query must have been
        provided and the llm attribute must be set.

        :param query: The query to pass to the route.
        :type query: Optional[str]
        :return: The route choice.
        :rtype: RouteChoice
        """
        if self.function_schemas:
            if not self.llm:
                raise ValueError(
                    "LLM is required for dynamic routes. Please ensure the `llm` "
                    "attribute is set."
                )
            elif query is None:
                raise ValueError(
                    "Query is required for dynamic routes. Please ensure the `query` "
                    "argument is passed."
                )
            # if a function schema is provided we generate the inputs
            try:
                extracted_inputs = self.llm.extract_function_inputs(
                    query=query, function_schemas=self.function_schemas
                )
                func_call = extracted_inputs
            except Exception:
                logger.error("Error extracting function inputs", exc_info=True)
                func_call = None
        else:
            # otherwise we just pass None for the call
            func_call = None
        return RouteChoice(name=self.name, function_call=func_call)

    async def acall(self, query: Optional[str] = None) -> RouteChoice:
        """Asynchronous call the route. If dynamic routes have been provided the query
        must have been provided and the llm attribute must be set.

        :param query: The query to pass to the route.
        :type query: Optional[str]
        :return: The route choice.
        :rtype: RouteChoice
        """
        if self.function_schemas:
            if not self.llm:
                raise ValueError(
                    "LLM is required for dynamic routes. Please ensure the `llm` "
                    "attribute is set."
                )
            elif query is None:
                raise ValueError(
                    "Query is required for dynamic routes. Please ensure the `query` "
                    "argument is passed."
                )
            # if a function schema is provided we generate the inputs
            try:
                extracted_inputs = await self.llm.async_extract_function_inputs(  # type: ignore # openai-llm
                    query=query, function_schemas=self.function_schemas
                )
                func_call = extracted_inputs
            except Exception:
                logger.error("Error extracting function inputs", exc_info=True)
                func_call = None
        else:
            # otherwise we just pass None for the call
            func_call = None
        return RouteChoice(name=self.name, function_call=func_call)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the route to a dictionary.

        :return: The dictionary representation of the route.
        :rtype: Dict[str, Any]
        """
        data = self.dict()
        if self.llm is not None:
            data["llm"] = {
                "module": self.llm.__module__,
                "class": self.llm.__class__.__name__,
                "model": self.llm.name,
            }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create a Route object from a dictionary.

        :param data: The dictionary to create the route from.
        :type data: Dict[str, Any]
        :return: The created route.
        :rtype: Route
        """
        return cls(**data)

    @classmethod
    def from_dynamic_route(
        cls, llm: BaseLLM, entities: List[Union[BaseModel, Callable]], route_name: str
    ):
        """Generate a dynamic Route object from a list of functions or Pydantic models
        using an LLM.

        :param llm: The LLM to use.
        :type llm: BaseLLM
        :param entities: The entities to use.
        :type entities: List[Union[BaseModel, Callable]]
        :param route_name: The name of the route.
        """
        schemas = function_call.get_schema_list(items=entities)
        dynamic_route = cls._generate_dynamic_route(
            llm=llm, function_schemas=schemas, route_name=route_name
        )
        dynamic_route.function_schemas = schemas
        return dynamic_route

    @classmethod
    def _parse_route_config(cls, config: str) -> str:
        """Parse the route config from the LLM output using regex. Expects the output
        content to be wrapped in <config></config> tags.

        :param config: The LLM output.
        :type config: str
        :return: The parsed route config.
        :rtype: str
        """
        # Regular expression to match content inside <config></config>
        config_pattern = r"<config>(.*?)</config>"
        match = re.search(config_pattern, config, re.DOTALL)

        if match:
            config_content = match.group(1).strip()  # Get the matched content
            return config_content
        else:
            raise ValueError("No <config></config> tags found in the output.")

    @classmethod
    def _generate_dynamic_route(
        cls, llm: BaseLLM, function_schemas: List[Dict[str, Any]], route_name: str
    ):
        """Generate a dynamic Route object from a list of function schemas using an LLM.

        :param llm: The LLM to use.
        :type llm: BaseLLM
        :param function_schemas: The function schemas to use.
        :type function_schemas: List[Dict[str, Any]]
        :param route_name: The name of the route.
        """
        formatted_schemas = "\n".join(
            [json.dumps(schema, indent=4) for schema in function_schemas]
        )
        prompt = f"""
        You are tasked to generate a single JSON configuration for multiple function schemas. 
        Each function schema should contribute five example utterances. 
        Please follow the template below, no other tokens allowed:

        <config>
        {{
            "name": "{route_name}",
            "utterances": [
                "<example_utterance_1>",
                "<example_utterance_2>",
                "<example_utterance_3>",
                "<example_utterance_4>",
                "<example_utterance_5>"]
        }}
        </config>

        Only include the "name" and "utterances" keys in your answer.
        The "name" should match the provided route name and the "utterances"
        should comprise a list of 5 example phrases for each function schema that could be used to invoke
        the functions. Use real values instead of placeholders.

        Input schemas:
        {formatted_schemas}
        """

        llm_input = [Message(role="user", content=prompt)]
        output = llm(llm_input)
        if not output:
            raise Exception("No output generated for dynamic route")

        route_config = cls._parse_route_config(config=output)

        if is_valid(route_config):
            route_config_dict = json.loads(route_config)
            route_config_dict["llm"] = llm
            return Route.from_dict(route_config_dict)
        raise Exception("No config generated")
