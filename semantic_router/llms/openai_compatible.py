import os
from typing import List, Optional

import openai
from pydantic import PrivateAttr

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.logger import logger


class OpenAICompatibleLLM(BaseLLM):
    """LLM for any OpenAI compatible chat completions endpoint, such as LLM
    routers and proxies. Configure the base URL and the name of the environment
    variable that holds the API key, for example
    OpenAICompatibleLLM(name="provider/model", base_url="https://api.example.com/v1",
    api_key_var_name="EXAMPLE_API_KEY")."""

    _client: Optional[openai.OpenAI] = PrivateAttr(default=None)
    _base_url: Optional[str] = PrivateAttr(default=None)

    def __init__(
        self,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_var_name: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        """Initialize the OpenAICompatibleLLM.

        :param name: The name of the model to use.
        :type name: Optional[str]
        :param base_url: The base URL of the OpenAI compatible API.
        :type base_url: Optional[str]
        :param api_key: The API key. Takes precedence over api_key_var_name.
        :type api_key: Optional[str]
        :param api_key_var_name: The name of the environment variable holding
            the API key, used when api_key is not provided.
        :type api_key_var_name: Optional[str]
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        """
        if name is None:
            raise ValueError("Model name cannot be 'None'.")
        if base_url is None:
            raise ValueError("Base URL cannot be 'None'.")
        super().__init__(name=name)
        self._base_url = base_url
        if api_key is None and api_key_var_name is not None:
            api_key = os.getenv(api_key_var_name)
        if api_key is None:
            raise ValueError(
                "API key cannot be 'None'. Pass api_key or set the environment "
                f"variable named by api_key_var_name ({api_key_var_name})."
            )
        try:
            self._client = openai.OpenAI(api_key=api_key, base_url=self._base_url)
        except Exception as e:
            raise ValueError(
                f"OpenAI compatible API client failed to initialize. Error: {e}"
            ) from e
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages: List[Message]) -> str:
        """Call the OpenAICompatibleLLM.

        :param messages: The messages to pass to the OpenAICompatibleLLM.
        :type messages: List[Message]
        :return: The response from the OpenAICompatibleLLM.
        :rtype: str
        """
        if self._client is None:
            raise ValueError("OpenAI compatible client is not initialized.")
        try:
            completion = self._client.chat.completions.create(
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
