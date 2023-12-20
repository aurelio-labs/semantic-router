import json
import os
import re
from typing import Any, Callable, Union

import yaml
from pydantic import BaseModel

from semantic_router.utils import function_call
from semantic_router.utils.llm import llm
from semantic_router.utils.logger import logger


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
    utterances: list[str]
    description: str | None = None

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


class RouteConfig:
    """
    Generates a RouteConfig object from a list of Route objects
    """

    routes: list[Route] = []

    def __init__(self, routes: list[Route] = []):
        self.routes = routes

    @classmethod
    def from_file(cls, path: str):
        """Load the routes from a file in JSON or YAML format"""
        logger.info(f"Loading route config from {path}")
        _, ext = os.path.splitext(path)
        with open(path, "r") as f:
            if ext == ".json":
                routes = json.load(f)
            elif ext in [".yaml", ".yml"]:
                routes = yaml.safe_load(f)
            else:
                raise ValueError(
                    "Unsupported file type. Only .json and .yaml are supported"
                )

            route_config_str = json.dumps(routes)
            if is_valid(route_config_str):
                routes = [Route.from_dict(route) for route in routes]
                return cls(routes=routes)
            else:
                raise Exception("Invalid config JSON or YAML")

    def to_dict(self):
        return [route.to_dict() for route in self.routes]

    def to_file(self, path: str):
        """Save the routes to a file in JSON or YAML format"""
        logger.info(f"Saving route config to {path}")
        _, ext = os.path.splitext(path)
        with open(path, "w") as f:
            if ext == ".json":
                json.dump(self.to_dict(), f)
            elif ext in [".yaml", ".yml"]:
                yaml.safe_dump(self.to_dict(), f)
            else:
                raise ValueError(
                    "Unsupported file type. Only .json and .yaml are supported"
                )

    def add(self, route: Route):
        self.routes.append(route)
        logger.info(f"Added route `{route.name}`")

    def get(self, name: str) -> Route | None:
        for route in self.routes:
            if route.name == name:
                return route
        logger.error(f"Route `{name}` not found")
        return None

    def remove(self, name: str):
        if name not in [route.name for route in self.routes]:
            logger.error(f"Route `{name}` not found")
        else:
            self.routes = [route for route in self.routes if route.name != name]
            logger.info(f"Removed route `{name}`")
