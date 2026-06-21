import os
from typing import List, Optional

import openai
from pydantic import PrivateAttr

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.logger import logger


class RequestyLLM(BaseLLM):
    """LLM for Requesty. Requires a Requesty API key, see here for more information
    https://docs.requesty.ai/quickstart"""

    _client: Optional[openai.OpenAI] = PrivateAttr(default=None)
    _base_url: str = PrivateAttr(default="https://router.requesty.ai/v1")

    def __init__(
        self,
        name: Optional[str] = None,
        requesty_api_key: Optional[str] = None,
        base_url: str = "https://router.requesty.ai/v1",
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        """Initialize the RequestyLLM.

        :param name: The name of the Requesty model to use.
        :type name: Optional[str]
        :param requesty_api_key: The Requesty API key.
        :type requesty_api_key: Optional[str]
        :param base_url: The base URL for the Requesty API.
        :type base_url: str
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        """
        if name is None:
            name = os.getenv("REQUESTY_CHAT_MODEL_NAME", "openai/gpt-4o-mini")
        super().__init__(name=name)
        self._base_url = base_url
        api_key = requesty_api_key or os.getenv("REQUESTY_API_KEY")
        if api_key is None:
            raise ValueError("Requesty API key cannot be 'None'.")
        try:
            self._client = openai.OpenAI(api_key=api_key, base_url=self._base_url)
        except Exception as e:
            raise ValueError(
                f"Requesty API client failed to initialize. Error: {e}"
            ) from e
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages: List[Message]) -> str:
        """Call the RequestyLLM.

        :param messages: The messages to pass to the RequestyLLM.
        :type messages: List[Message]
        :return: The response from the RequestyLLM.
        :rtype: str
        """
        if self._client is None:
            raise ValueError("Requesty client is not initialized.")
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
