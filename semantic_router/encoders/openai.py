import os
from time import sleep

import openai
from openai.error import OpenAIError, RateLimitError, ServiceUnavailableError

from semantic_router.encoders import BaseEncoder
from semantic_router.utils.logger import logger


class OpenAIEncoder(BaseEncoder):
    def __init__(self, name: str, openai_api_key: str | None = None):
        super().__init__(name=name)
        openai.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            raise ValueError("OpenAI API key cannot be 'None'.")

    def __call__(self, docs: list[str]) -> list[list[float]]:
        """Encode a list of texts using the OpenAI API. Returns a list of
        vector embeddings.
        """
        res = None
        error_message = ""

        # exponential backoff
        for j in range(5):
            try:
                logger.info(f"Encoding {len(docs)} documents...")
                res = openai.Embedding.create(input=docs, engine=self.name)
                if isinstance(res, dict) and "data" in res:
                    break
            except (RateLimitError, ServiceUnavailableError, OpenAIError) as e:
                logger.warning(f"Retrying in {2**j} seconds...")
                sleep(2**j)
                error_message = str(e)
        if not res or not isinstance(res, dict) or "data" not in res:
            raise ValueError(f"OpenAI API call failed. Error: {error_message}")

        embeds = [r["embedding"] for r in res["data"]]
        return embeds
