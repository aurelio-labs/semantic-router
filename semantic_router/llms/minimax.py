import os
import re
from typing import List, Optional

import openai
from pydantic import PrivateAttr

from semantic_router.llms.openai_compatible import OpenAICompatibleLLM
from semantic_router.schema import Message
from semantic_router.utils.logger import logger


class MiniMaxLLM(OpenAICompatibleLLM):
    """LLM for MiniMax, a preset of OpenAICompatibleLLM pointing at MiniMax's
    OpenAI-compatible API (https://api.minimax.io/v1) and reading the API key
    from MINIMAX_API_KEY. Create a key at https://platform.minimax.io.

    MiniMax's models think before they answer. MiniMax-M3 puts its thinking in
    <think> tags inside the reply, which are stripped from the output, and
    MiniMax-M3.1-Flash returns it in a separate field, which is ignored. Thinking
    counts towards max_tokens, so the default leaves room for it.

    Current MiniMax models ignore temperature. It is still clamped to
    (0.0, 1.0], the range older MiniMax models required.

    Limits inherited from semantic-router 0.x:

    - Dynamic routes extract function inputs only when called synchronously;
      an async call returns the route with function_call=None. MiniMaxLLM.acall
      itself works, but routes don't use it for extraction.
    - A router config file saves only the class and model name. Reloading one
      uses the default base_url and MINIMAX_API_KEY, so a custom base_url or a
      key passed in code is lost."""

    _async_client: Optional[openai.AsyncOpenAI] = PrivateAttr(default=None)

    def __init__(
        self,
        name: Optional[str] = None,
        minimax_api_key: Optional[str] = None,
        base_url: str = "https://api.minimax.io/v1",
        temperature: float = 0.01,
        max_tokens: int = 4096,
    ):
        """Initialize the MiniMaxLLM.

        :param name: The name of the MiniMax model to use. Defaults to the
            MINIMAX_CHAT_MODEL_NAME environment variable or MiniMax-M3.1-Flash.
        :type name: Optional[str]
        :param minimax_api_key: The MiniMax API key. Defaults to the
            MINIMAX_API_KEY environment variable.
        :type minimax_api_key: Optional[str]
        :param base_url: The base URL for the MiniMax API.
        :type base_url: str
        :param temperature: The temperature of the LLM, clamped to (0.0, 1.0].
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate, thinking
            included.
        :type max_tokens: int
        """
        if name is None:
            name = os.getenv("MINIMAX_CHAT_MODEL_NAME", "MiniMax-M3.1-Flash")
        api_key = minimax_api_key or os.getenv("MINIMAX_API_KEY")
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            api_key_var_name="MINIMAX_API_KEY",
            temperature=max(0.01, min(temperature, 1.0)),
            max_tokens=max_tokens,
        )
        try:
            self._async_client = openai.AsyncOpenAI(
                api_key=api_key, base_url=self._base_url
            )
        except Exception as e:
            raise ValueError(
                f"MiniMax async client failed to initialize. Error: {e}"
            ) from e

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Strip <think>...</think> blocks from MiniMax model responses,
        including one left open when the model ran out of tokens mid-thought.

        :param text: The text to strip think tags from.
        :type text: str
        :return: The text with think tags removed.
        :rtype: str
        """
        return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()

    def _answer(self, output: str) -> str:
        answer = self._strip_think_tags(output)
        if not answer:
            raise Exception(
                "LLM error: No output generated outside <think> tags. Thinking "
                "counts towards max_tokens, so try a higher max_tokens."
            )
        return answer

    def __call__(self, messages: List[Message]) -> str:
        """Call the MiniMaxLLM.

        :param messages: The messages to pass to the MiniMaxLLM.
        :type messages: List[Message]
        :return: The response from the MiniMaxLLM, without its thinking.
        :rtype: str
        """
        return self._answer(super().__call__(messages))

    async def acall(self, messages: List[Message]) -> str:
        """Call the MiniMaxLLM asynchronously.

        :param messages: The messages to pass to the MiniMaxLLM.
        :type messages: List[Message]
        :return: The response from the MiniMaxLLM, without its thinking.
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
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise Exception(f"LLM error: {e}") from e
        return self._answer(output)
