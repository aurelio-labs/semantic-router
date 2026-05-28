import os
import re
from typing import List, Optional

import openai
from pydantic import PrivateAttr

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.logger import logger


class MiniMaxLLM(BaseLLM):
    """LLM for MiniMax. Uses the OpenAI-compatible API at https://api.minimax.io/v1.
    Requires a MiniMax API key from https://platform.minimaxi.com/"""

    _client: Optional[openai.OpenAI] = PrivateAttr(default=None)
    _async_client: Optional[openai.AsyncOpenAI] = PrivateAttr(default=None)
    _base_url: str = PrivateAttr(default="https://api.minimax.io/v1")

    def __init__(
        self,
        name: Optional[str] = None,
        minimax_api_key: Optional[str] = None,
        base_url: str = "https://api.minimax.io/v1",
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        """Initialize the MiniMaxLLM.

        :param name: The name of the MiniMax model to use.
        :type name: Optional[str]
        :param minimax_api_key: The MiniMax API key.
        :type minimax_api_key: Optional[str]
        :param base_url: The base URL for the MiniMax API.
        :type base_url: str
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        """
        if name is None:
            name = os.getenv("MINIMAX_CHAT_MODEL_NAME", "MiniMax-M2.5")
        super().__init__(name=name)
        self._base_url = base_url
        api_key = minimax_api_key or os.getenv("MINIMAX_API_KEY")
        if api_key is None:
            raise ValueError("MiniMax API key cannot be 'None'.")
        try:
            self._client = openai.OpenAI(api_key=api_key, base_url=self._base_url)
            self._async_client = openai.AsyncOpenAI(
                api_key=api_key, base_url=self._base_url
            )
        except Exception as e:
            raise ValueError(
                f"MiniMax API client failed to initialize. Error: {e}"
            ) from e
        # MiniMax temperature must be in (0.0, 1.0]
        self.temperature = max(0.01, min(temperature, 1.0))
        self.max_tokens = max_tokens

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Strip <think>...</think> tags from MiniMax model responses.

        :param text: The text to strip think tags from.
        :type text: str
        :return: The text with think tags removed.
        :rtype: str
        """
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def __call__(self, messages: List[Message]) -> str:
        """Call the MiniMaxLLM.

        :param messages: The messages to pass to the MiniMaxLLM.
        :type messages: List[Message]
        :return: The response from the MiniMaxLLM.
        :rtype: str
        """
        if self._client is None:
            raise ValueError("MiniMax client is not initialized.")
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
            return self._strip_think_tags(output)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise Exception(f"LLM error: {e}") from e

    async def acall(self, messages: List[Message]) -> str:
        """Call the MiniMaxLLM asynchronously.

        :param messages: The messages to pass to the MiniMaxLLM.
        :type messages: List[Message]
        :return: The response from the MiniMaxLLM.
        :rtype: str
        """
        if self._async_client is None:
            raise ValueError("MiniMax async client is not initialized.")
        try:
            completion = await self._async_client.chat.completions.create(
                model=self.name,
                messages=[m.to_openai() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            output = completion.choices[0].message.content

            if not output:
                raise Exception("No output generated")
            return self._strip_think_tags(output)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise Exception(f"LLM error: {e}") from e
