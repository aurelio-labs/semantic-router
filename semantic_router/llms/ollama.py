from typing import List, Optional

import requests

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.logger import logger


class OllamaLLM(BaseLLM):
    stream: bool = False

    def __init__(
        self,
        name: str = "openhermes",
        temperature: float = 0.2,
        max_tokens: Optional[int] = 200,
        stream: bool = False,
    ):
        super().__init__(name=name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream

    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
    ) -> str:
        # Use instance defaults if not overridden
        temperature = temperature if temperature is not None else self.temperature
        name = name if name is not None else self.name
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        stream = stream if stream is not None else self.stream

        try:
            payload = {
                "model": name,
                "messages": [m.to_openai() for m in messages],
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "format": "json",
                "stream": stream,
            }
            response = requests.post("http://localhost:11434/api/chat", json=payload)
            output = response.json()["message"]["content"]

            return output
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise Exception(f"LLM error: {e}") from e
