import os
from typing import List, Optional

import openai
from pydantic import PrivateAttr

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.logger import logger


class OpenRouterLLM(BaseLLM):
    """LLM for OpenRouter. Requires an OpenRouter API key, see here for more information
    https://openrouter.ai/docs/api-reference/authentication#using-an-api-key"""

    _client: Optional[openai.OpenAI] = PrivateAttr(default=None)
    _base_url: str = PrivateAttr(default="https://openrouter.ai/api/v1")

    def __init__(
        self,
        name: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        """Initialize the OpenRouterLLM.

        :param name: The name of the OpenRouter model to use.
        :type name: Optional[str]
        :param openrouter_api_key: The OpenRouter API key.
        :type openrouter_api_key: Optional[str]
        :param base_url: The base URL for the OpenRouter API.
        :type base_url: str
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        """
        if name is None:
            name = os.getenv(
                "OPENROUTER_CHAT_MODEL_NAME", "mistralai/mistral-7b-instruct"
            )
        super().__init__(name=name)
        self._base_url = base_url
        api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OpenRouter API key cannot be 'None'.")
        try:
            self._client = openai.OpenAI(api_key=api_key, base_url=self._base_url)
        except Exception as e:
            raise ValueError(
                f"OpenRouter API client failed to initialize. Error: {e}"
            ) from e
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages: List[Message]) -> str:
        """Call the OpenRouterLLM.

        :param messages: The messages to pass to the OpenRouterLLM.
        :type messages: List[Message]
        :return: The response from the OpenRouterLLM.
        :rtype: str
        """
        if self._client is None:
            raise ValueError("OpenRouter client is not initialized.")
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
