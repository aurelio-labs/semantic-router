import os

from decision_layer.encoders import BaseEncoder
import openai
from time import time


class OpenAIEncoder(BaseEncoder):
    def __init__(self, name: str, openai_api_key: str | None = None):
        super().__init__(name=name)
        openai.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            raise ValueError("OpenAI API key cannot be 'None'.")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts using the OpenAI API. Returns a list of
        vector embeddings.
        """
        passed = False
        # exponential backoff in case of RateLimitError
        for j in range(5):
            try:
                # create embeddings
                res = openai.Embedding.create(
                    input=texts, engine=self.name
                )
                passed = True
            except openai.error.RateLimitError:
                time.sleep(2 ** j)
        if not passed:
            raise openai.error.RateLimitError
        # get embeddings
        embeds = [r["embedding"] for r in res["data"]]
        return embeds
        