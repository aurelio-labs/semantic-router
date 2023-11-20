import os

import cohere

from semantic_router.retrievers import BaseRetriever


class CohereRetriever(BaseRetriever):
    client: cohere.Client | None

    def __init__(
        self, name: str = "embed-english-v3.0", cohere_api_key: str | None = None
    ):
        super().__init__(name=name)
        cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError("Cohere API key cannot be 'None'.")
        self.client = cohere.Client(cohere_api_key)

    def __call__(self, docs: list[str]) -> list[list[float]]:
        if self.client is None:
            raise ValueError("Cohere client is not initialized.")
        if len(docs) == 1:
            input_type = "search_query"
        else:
            input_type = "search_document"
        embeds = self.client.embed(docs, input_type=input_type, model=self.name)
        return embeds.embeddings