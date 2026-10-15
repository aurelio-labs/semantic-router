import os
from typing import List, Optional

import openai
from pydantic import PrivateAttr

from semantic_router.llms.base import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.logger import logger


class AtlasCloudLLM(BaseLLM):
    """LLM for Atlas Cloud. Requires an Atlas Cloud API key, see here for more information
    https://docs.atlascloud.ai/"""

    _client: Optional[openai.OpenAI] = PrivateAttr(default=None)
    _base_url: str = PrivateAttr(default="https://api.atlascloud.ai/v1")

    def __init__(
        self,
        name: Optional[str] = None,
        atlascloud_api_key: Optional[str] = None,
        base_url: str = "https://api.atlascloud.ai/v1",
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        """Initialize the AtlasCloudLLM.

        :param name: The name of the Atlas Cloud model to use.
        :type name: Optional[str]
        :param atlascloud_api_key: The Atlas Cloud API key.
        :type atlascloud_api_key: Optional[str]
        :param base_url: The base URL for the Atlas Cloud API.
        :type base_url: str
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        """
        if name is None:
            name = os.getenv(
                "ATLASCLOUD_CHAT_MODEL_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507"
            )
        super().__init__(name=name)
        self._base_url = base_url
        api_key = atlascloud_api_key or os.getenv("ATLASCLOUD_API_KEY")
        if api_key is None:
            raise ValueError("Atlas Cloud API key cannot be 'None'.")
        try:
            self._client = openai.OpenAI(api_key=api_key, base_url=self._base_url)
        except Exception as e:
            raise ValueError(
                f"Atlas Cloud API client failed to initialize. Error: {e}"
            ) from e
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages: List[Message]) -> str:
        """Call the AtlasCloudLLM.

        :param messages: The messages to pass to the AtlasCloudLLM.
        :type messages: List[Message]
        :return: The response from the AtlasCloudLLM.
        :rtype: str
        """
        if self._client is None:
            raise ValueError("Atlas Cloud client is not initialized.")
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
