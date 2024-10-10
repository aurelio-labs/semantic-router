import os
from typing import Any, List, Optional

from pydantic.v1 import PrivateAttr

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message


class CohereLLM(BaseLLM):
    _client: Any = PrivateAttr()

    def __init__(
        self,
        name: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
    ):
        if name is None:
            name = os.getenv("COHERE_CHAT_MODEL_NAME", "command")
        super().__init__(name=name)
        self._client = self._initialize_client(cohere_api_key)

    def _initialize_client(self, cohere_api_key: Optional[str] = None):
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Please install Cohere to use CohereLLM. "
                "You can install it with: "
                "`pip install 'semantic-router[cohere]'`"
            )
        cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError("Cohere API key cannot be 'None'.")
        try:
            client = cohere.Client(cohere_api_key)
        except Exception as e:
            raise ValueError(
                f"Cohere API client failed to initialize. Error: {e}"
            ) from e
        return client

    def __call__(self, messages: List[Message]) -> str:
        if self._client is None:
            raise ValueError("Cohere client is not initialized.")
        try:
            completion = self._client.chat(
                model=self.name,
                chat_history=[m.to_cohere() for m in messages[:-1]],
                message=messages[-1].content,
            )

            output = completion.text

            if not output:
                raise Exception("No output generated")
            return output

        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e
