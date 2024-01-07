import os

import cohere

from semantic_router.encoders import BaseEncoder


class CohereEncoder(BaseEncoder):
    client: cohere.Client | None = None
    type: str = "cohere"

    def __init__(
        self,
        name: str | None = None,
        cohere_api_key: str | None = None,
        score_threshold: float = 0.3,
    ):
        if name is None:
            name = os.getenv("COHERE_MODEL_NAME", "embed-english-v3.0")
        super().__init__(name=name, score_threshold=score_threshold)
        cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError("Cohere API key cannot be 'None'.")
        try:
            self.client = cohere.Client(cohere_api_key)
        except Exception as e:
            raise ValueError(f"Cohere API client failed to initialize. Error: {e}")

    def __call__(self, docs: list[str]) -> list[list[float]]:
        if self.client is None:
            raise ValueError("Cohere client is not initialized.")
        try:
            embeds = self.client.embed(docs, input_type="search_query", model=self.name)
            return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}")
