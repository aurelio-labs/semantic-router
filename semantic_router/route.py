import json
import re
from typing import Any, Callable, Union

from pydantic import BaseModel

from semantic_router.utils import function_call
from semantic_router.utils.llm import llm
from semantic_router.utils.logger import logger
from semantic_router.schema import RouteChoice


class Route(BaseModel):
    name: str
    utterances: list[str]
    description: str | None = None
    function_schema: dict[str, Any] | None = None

    def __call__(self, query: str) -> RouteChoice:
        if self.function_schema:
            # if a function schema is provided we generate the inputs
            extracted_inputs = function_call.extract_function_inputs(
                query=query, function_schema=self.function_schema
            )
            function_call = extracted_inputs
        else:
            # otherwise we just pass None for the call
            function_call = None
        return RouteChoice(
            name=self.name,
            function_call=function_call
        )

    def to_dict(self):
        return self.dict()

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    @classmethod
    async def from_dynamic_route(cls, entity: Union[BaseModel, Callable]):
        """
        Generate a dynamic Route object from a function or Pydantic model using LLM
        """
        schema = function_call.get_schema(item=entity)
        dynamic_route = await cls._generate_dynamic_route(function_schema=schema)
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
    async def _generate_dynamic_route(cls, function_schema: dict[str, Any]):
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

        output = await llm(prompt)
        if not output:
            raise Exception("No output generated for dynamic route")

        route_config = cls._parse_route_config(config=output)

        logger.info(f"Generated route config:\n{route_config}")

        if is_valid(route_config):
            return Route.from_dict(json.loads(route_config))
        raise Exception("No config generated")


