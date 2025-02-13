import os
from typing import List, Optional

import openai
from pydantic import PrivateAttr

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


class AzureOpenAILLM(BaseLLM):
    """LLM for Azure OpenAI. Requires an Azure OpenAI API key."""

    _client: Optional[openai.AzureOpenAI] = PrivateAttr(default=None)

    def __init__(
        self,
        name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: int = 200,
        api_version="2023-07-01-preview",
    ):
        """Initialize the AzureOpenAILLM.

        :param name: The name of the Azure OpenAI model to use.
        :type name: Optional[str]
        :param openai_api_key: The Azure OpenAI API key.
        :type openai_api_key: Optional[str]
        :param azure_endpoint: The Azure OpenAI endpoint.
        :type azure_endpoint: Optional[str]
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        :param api_version: The API version to use.
        :type api_version: str
        """
        if name is None:
            name = EncoderDefault.AZURE.value["language_model"]
        super().__init__(name=name)
        api_key = openai_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("AzureOpenAI API key cannot be 'None'.")
        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint is None:
            raise ValueError("Azure endpoint API key cannot be 'None'.")
        try:
            self._client = openai.AzureOpenAI(
                api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version
            )
        except Exception as e:
            raise ValueError(f"AzureOpenAI API client failed to initialize. Error: {e}")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages: List[Message]) -> str:
        """Call the AzureOpenAILLM.

        :param messages: The messages to pass to the AzureOpenAILLM.
        :type messages: List[Message]
        :return: The response from the AzureOpenAILLM.
        :rtype: str
        """
        if self._client is None:
            raise ValueError("AzureOpenAI client is not initialized.")
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
