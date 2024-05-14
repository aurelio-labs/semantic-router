import os
from typing import List, Optional

import cohere
from cohere.types.embed_response import (
    EmbedResponse_EmbeddingsFloats,
    EmbedResponse_EmbeddingsByType,
)

from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault


class CohereEncoder(BaseEncoder):
    client: Optional[cohere.Client] = None
    type: str = "cohere"
    input_type: Optional[str] = "search_query"

    def __init__(
        self,
        name: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        score_threshold: float = 0.3,
        input_type: Optional[str] = "search_query",
    ):
        if name is None:
            name = EncoderDefault.COHERE.value["embedding_model"]
        super().__init__(
            name=name,
            score_threshold=score_threshold,
            input_type=input_type,  # type: ignore
        )
        self.input_type = input_type
        cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError("Cohere API key cannot be 'None'.")
        try:
            self.client = cohere.Client(cohere_api_key)
        except Exception as e:
            raise ValueError(
                f"Cohere API client failed to initialize. Error: {e}"
            ) from e

    def __call__(self, docs: List[str]) -> List[List[float]]:
        if self.client is None:
            raise ValueError("Cohere client is not initialized.")
        try:
            embeds = self.client.embed(
                texts=docs, input_type=self.input_type, model=self.name
            )
            # Check the type of response and handle accordingly
            # Only EmbedResponse_EmbeddingsFloats has embeddings of type List[List[float]]
            if isinstance(embeds, EmbedResponse_EmbeddingsFloats):
                return embeds.embeddings
            elif isinstance(embeds, EmbedResponse_EmbeddingsByType):
                raise NotImplementedError(
                    "Handling of EmbedByTypeResponseEmbeddings is not implemented."
                )
            else:
                raise ValueError(f"Unexpected response type from Cohere API: type(embeds) = {type(embeds)}")
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e
