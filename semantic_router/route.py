import json
import re
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic.v1 import BaseModel

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message, RouteChoice
from semantic_router.utils import function_call
from semantic_router.utils.logger import logger

try:
    from PIL.Image import Image
except ImportError:
    pass


def is_valid(route_config: str) -> bool:
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
    name: str
    utterances: Union[List[str], List[Union[Any, "Image"]]]
    description: Optional[str] = None
    function_schema: Optional[Dict[str, Any]] = None
    llm: Optional[BaseLLM] = None
    score_threshold: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, query: Optional[str] = None) -> RouteChoice:
        if self.function_schema:
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
            extracted_inputs = self.llm.extract_function_inputs(
                query=query, function_schema=self.function_schema
            )
            func_call = extracted_inputs
        else:
            # otherwise we just pass None for the call
            func_call = None
        return RouteChoice(name=self.name, function_call=func_call)

    # def to_dict(self) -> Dict[str, Any]:
    #     return self.dict()

    def to_dict(self) -> Dict[str, Any]:
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
        return cls(**data)

    @classmethod
    def from_dynamic_route(cls, llm: BaseLLM, entity: Union[BaseModel, Callable]):
        """
        Generate a dynamic Route object from a function or Pydantic model using LLM
        """
        schema = function_call.get_schema(item=entity)
        dynamic_route = cls._generate_dynamic_route(llm=llm, function_schema=schema)
        dynamic_route.function_schema = schema
        return dynamic_route

    @classmethod
    def _parse_route_config(cls, config: str) -> str:
        # Regular expression to match content inside <config></config>
        config_pattern = r"<config>(.*?)</config>"
        match = re.search(config_pattern, config, re.DOTALL)

        if match:
            config_content = match.group(1).strip()  # Get the matched content
            return config_content
        else:
            raise ValueError("No <config></config> tags found in the output.")

    @classmethod
    def _generate_dynamic_route(cls, llm: BaseLLM, function_schema: Dict[str, Any]):
        logger.info("Generating dynamic route...")

        prompt = f"""
        You are tasked to generate a JSON configuration based on the provided
        function schema. Please follow the template below, no other tokens allowed:

        <config>
        {{
            "name": "<function_name>",
            "utterances": [
                "<example_utterance_1>",
                "<example_utterance_2>",
                "<example_utterance_3>",
                "<example_utterance_4>",
                "<example_utterance_5>"]
        }}
        </config>

        Only include the "name" and "utterances" keys in your answer.
        The "name" should match the function name and the "utterances"
        should comprise a list of 5 example phrases that could be used to invoke
        the function. Use real values instead of placeholders.

        Input schema:
        {function_schema}
        """

        llm_input = [Message(role="user", content=prompt)]
        output = llm(llm_input)
        if not output:
            raise Exception("No output generated for dynamic route")

        route_config = cls._parse_route_config(config=output)

        logger.info(f"Generated route config:\n{route_config}")

        if is_valid(route_config):
            route_config_dict = json.loads(route_config)
            route_config_dict["llm"] = llm
            return Route.from_dict(route_config_dict)
        raise Exception("No config generated")
