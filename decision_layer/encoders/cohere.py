import os
import cohere
from decision_layer.encoders import BaseEncoder

class CohereEncoder(BaseEncoder):
    client: cohere.Client | None
    def __init__(self, name: str, cohere_api_key: str | None = None):
        super().__init__(name=name, client=None)
        cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError("Cohere API key cannot be 'None'.")
        self.client = cohere.Client(cohere_api_key)

    def __call__(self, texts: list[str]) -> list[float]:
        if len(texts) == 1:
            input_type = "search_query"
        else:
            input_type = "search_document"
        embeds = self.client.embed(
            texts, input_type=input_type, model="embed-english-v3.0"
        )
        return embeds.embeddings