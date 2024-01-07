import os
from time import sleep

import openai
from openai import OpenAIError
from openai.types import CreateEmbeddingResponse

from semantic_router.encoders import BaseEncoder
from semantic_router.utils.logger import logger


class OpenAIEncoder(BaseEncoder):
    client: openai.Client | None
    type: str = "openai"

    def __init__(
        self,
        name: str | None = None,
        openai_api_key: str | None = None,
        score_threshold: float = 0.82,
    ):
        if name is None:
            name = os.getenv("OPENAI_MODEL_NAME", "text-embedding-ada-002")
        super().__init__(name=name, score_threshold=score_threshold)
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key cannot be 'None'.")
        try:
            self.client = openai.Client(api_key=api_key)
        except Exception as e:
            raise ValueError(f"OpenAI API client failed to initialize. Error: {e}")

    def __call__(self, docs: list[str]) -> list[list[float]]:
        if self.client is None:
            raise ValueError("OpenAI client is not initialized.")
        embeds = None
        error_message = ""

        # Exponential backoff
        for j in range(3):
            try:
                embeds = self.client.embeddings.create(input=docs, model=self.name)
                if embeds.data:
                    break
            except OpenAIError as e:
                sleep(2**j)
                error_message = str(e)
                logger.warning(f"Retrying in {2**j} seconds...")
            except Exception as e:
                logger.error(f"OpenAI API call failed. Error: {error_message}")
                raise ValueError(f"OpenAI API call failed. Error: {e}")

        if (
            not embeds
            or not isinstance(embeds, CreateEmbeddingResponse)
            or not embeds.data
        ):
            raise ValueError(f"No embeddings returned. Error: {error_message}")

        embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
        return embeddings
