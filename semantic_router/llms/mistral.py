import os
from typing import Any, List, Optional

from pydantic import PrivateAttr

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


class MistralAILLM(BaseLLM):
    """LLM for MistralAI. Requires a MistralAI API key from https://console.mistral.ai/api-keys/"""

    _client: Any = PrivateAttr()
    _mistralai: Any = PrivateAttr()

    def __init__(
        self,
        name: Optional[str] = None,
        mistralai_api_key: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        """Initialize the MistralAILLM.

        :param name: The name of the MistralAI model to use.
        :type name: Optional[str]
        :param mistralai_api_key: The MistralAI API key.
        :type mistralai_api_key: Optional[str]
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        """
        if name is None:
            name = EncoderDefault.MISTRAL.value["language_model"]
        super().__init__(name=name)
        self._client, self._mistralai = self._initialize_client(mistralai_api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _initialize_client(self, api_key):
        """Initialize the MistralAI client.

        :param api_key: The MistralAI API key.
        :type api_key: Optional[str]
        :return: The MistralAI client.
        :rtype: MistralClient
        """
        try:
            import mistralai
            from mistralai.client import MistralClient
        except ImportError:
            raise ImportError(
                "Please install MistralAI to use MistralAI LLM. "
                "You can install it with: "
                "`pip install 'semantic-router[mistralai]'`"
            )
        api_key = api_key or os.getenv("MISTRALAI_API_KEY")
        if api_key is None:
            raise ValueError("MistralAI API key cannot be 'None'.")
        try:
            client = MistralClient(api_key=api_key)
        except Exception as e:
            raise ValueError(
                f"MistralAI API client failed to initialize. Error: {e}"
            ) from e
        return client, mistralai

    def __call__(self, messages: List[Message]) -> str:
        """Call the MistralAILLM.

        :param messages: The messages to pass to the MistralAILLM.
        :type messages: List[Message]
        :return: The response from the MistralAILLM.
        :rtype: str
        """
        if self._client is None:
            raise ValueError("MistralAI client is not initialized.")
        chat_messages = [
            self._mistralai.models.chat_completion.ChatMessage(
                role=m.role, content=m.content
            )
            for m in messages
        ]
        try:
            completion = self._client.chat(
                model=self.name,
                messages=chat_messages,
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
