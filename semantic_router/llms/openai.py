import os
from typing import List, Optional, Any

import openai

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger
import json


class OpenAILLM(BaseLLM):
    client: Optional[openai.OpenAI]
    temperature: Optional[float]
    max_tokens: Optional[int]

    def __init__(
        self,
        name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        if name is None:
            name = EncoderDefault.OPENAI.value["language_model"]
        super().__init__(name=name)
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key cannot be 'None'.")
        try:
            self.client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            raise ValueError(
                f"OpenAI API client failed to initialize. Error: {e}"
            ) from e
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(
        self,
        messages: List[Message],
        function_schema: Optional[dict[str, Any]] = None,
    ) -> str:
        if self.client is None:
            raise ValueError("OpenAI client is not initialized.")
        try:
            if function_schema:
                tools = [function_schema]
            else:
                tools = None

            completion = self.client.chat.completions.create(
                model=self.name,
                messages=[m.to_openai() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,  # type: ignore # MyPy expecting Iterable[ChatCompletionToolParam] | NotGiven, but dict is accepted by OpenAI.
            )

            if function_schema:
                tool_calls = completion.choices[0].message.tool_calls
                if tool_calls is None:
                    raise ValueError("Invalid output, expected a tool call.")
                if len(tool_calls) != 1:
                    raise ValueError(
                        "Invalid output, expected a single tool to be specified."
                    )
                arguments = tool_calls[0].function.arguments
                if arguments is None:
                    raise ValueError("Invalid output, expected arguments to be specified.")
                output = str(arguments)  # str to keep MyPy happy.
            else:
                content = completion.choices[0].message.content
                if content is None:
                    raise ValueError("Invalid output, expected content.")
                output = str(content) # str to keep MyPy happy.
                
            return output

        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise Exception(f"LLM error: {e}") from e
        
    def __call__(self, messages: List[Message]) -> str:
        if self.client is None:
            raise ValueError("OpenAI client is not initialized.")
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                messages=[m.to_openai() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            output = completion.choices[0].message.content

            if not output:
                raise Exception("No output generated")
            return output
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise Exception(f"LLM error: {e}") from e

    def extract_function_inputs(
        self, query: str, function_schema: dict[str, Any]
    ) -> dict:
        messages = []
        system_prompt = "You are an intelligent AI. Given a command or request from the user, call the function to complete the request."
        messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=query))
        function_inputs_str = self(messages=messages, function_schema=function_schema)
        function_inputs = json.loads(function_inputs_str)
        return function_inputs
