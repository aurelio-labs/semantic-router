import os
from typing import List, Optional

import cohere

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message


class CohereLLM(BaseLLM):
    client: Optional[cohere.Client] = None

    def __init__(
        self,
        name: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
    ):
        if name is None:
            name = os.getenv("COHERE_CHAT_MODEL_NAME", "command")
        super().__init__(name=name)
        cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError("Cohere API key cannot be 'None'.")
        try:
            self.client = cohere.Client(cohere_api_key)
        except Exception as e:
            raise ValueError(
                f"Cohere API client failed to initialize. Error: {e}"
            ) from e

    def __call__(self, messages: List[Message]) -> str:
        if self.client is None:
            raise ValueError("Cohere client is not initialized.")
        try:
            completion = self.client.chat(
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
