import os
from typing import Any, List, Optional

from pydantic.v1 import PrivateAttr

from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault


class CohereEncoder(BaseEncoder):
    _client: Any = PrivateAttr()
    _embed_type: Any = PrivateAttr()
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
        self._client = self._initialize_client(cohere_api_key)

    def _initialize_client(self, cohere_api_key: Optional[str] = None):
        """Initializes the Cohere client.

        :param cohere_api_key: The API key for the Cohere client, can also
        be set via the COHERE_API_KEY environment variable.

        :return: An instance of the Cohere client.
        """
        try:
            import cohere
            from cohere.types.embed_response import EmbeddingsByTypeEmbedResponse

            self._embed_type = EmbeddingsByTypeEmbedResponse
        except ImportError:
            raise ImportError(
                "Please install Cohere to use CohereEncoder. "
                "You can install it with: "
                "`pip install 'semantic-router[cohere]'`"
            )
        cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError("Cohere API key cannot be 'None'.")
        try:
            client = cohere.Client(cohere_api_key)
        except Exception as e:
            raise ValueError(
                f"Cohere API client failed to initialize. Error: {e}"
            ) from e
        return client

    def __call__(self, docs: List[str]) -> List[List[float]]:
        if self._client is None:
            raise ValueError("Cohere client is not initialized.")
        try:
            embeds = self._client.embed(
                texts=docs, input_type=self.input_type, model=self.name
            )
            # Check for unsupported type.
            if isinstance(embeds, self._embed_type):
                raise NotImplementedError(
                    "Handling of EmbedByTypeResponseEmbeddings is not implemented."
                )
            else:
                return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e
