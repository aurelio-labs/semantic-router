import os
from typing import List, Optional, Any


from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger

from pydantic.v1 import PrivateAttr


class MistralAILLM(BaseLLM):
    _client: Any = PrivateAttr()
    temperature: Optional[float]
    max_tokens: Optional[int]

    def __init__(
        self,
        name: Optional[str] = None,
        mistralai_api_key: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        if name is None:
            name = EncoderDefault.MISTRAL.value["language_model"]
        super().__init__(name=name)
        self._client = self._initialize_client(mistralai_api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _initialize_client(self, api_key):
        try:
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
        return client

    def __call__(self, messages: List[Message]) -> str:
        if self._client is None:
            raise ValueError("MistralAI client is not initialized.")
        try:
            completion = self._client.chat(
                model=self.name,
                messages=[m.to_mistral() for m in messages],
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
