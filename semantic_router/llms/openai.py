import os

import openai

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.logger import logger


class OpenAILLM(BaseLLM):
    client: openai.OpenAI | None
    temperature: float | None
    max_tokens: int | None

    def __init__(
        self,
        name: str | None = None,
        openai_api_key: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 200,
    ):
        if name is None:
            name = os.getenv("OPENAI_CHAT_MODEL_NAME", "gpt-3.5-turbo")
        super().__init__(name=name)
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key cannot be 'None'.")
        try:
            self.client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            raise ValueError(f"OpenAI API client failed to initialize. Error: {e}")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages: list[Message]) -> str:
        if self.client is None:
            raise ValueError("OpenAI client is not initialized.")
        try:
            completion = self.client.chat.completions.create(
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
            raise Exception(f"LLM error: {e}")
