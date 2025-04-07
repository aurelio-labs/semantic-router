import os
from typing import Any

import litellm
from pydantic import PrivateAttr
from typing_extensions import deprecated

from semantic_router.encoders import DenseEncoder
from semantic_router.encoders.base import AsymmetricDenseMixin
from semantic_router.encoders.litellm import litellm_to_list
from semantic_router.utils.defaults import EncoderDefault


class CohereEncoder(DenseEncoder, AsymmetricDenseMixin):
    """Dense encoder that uses Cohere API to embed documents. Supports text only. Requires
    a Cohere API key from https://dashboard.cohere.com/api-keys.
    """

    _client: Any = PrivateAttr()
    _async_client: Any = PrivateAttr()
    _embed_type: Any = PrivateAttr()
    type: str = "cohere"

    def __init__(
        self,
        name: str | None = None,
        cohere_api_key: str | None = None,
        score_threshold: float = 0.3,
    ):
        """Initialize the Cohere encoder.

        :param name: The name of the embedding model to use.
        :type name: str
        :param cohere_api_key: The API key for the Cohere client, can also
            be set via the COHERE_API_KEY environment variable.
        :type cohere_api_key: str
        :param score_threshold: The threshold for the score of the embedding.
        :type score_threshold: float
        :param input_type: The type of input to embed.
        :type input_type: str
        """
        if name is None:
            name = EncoderDefault.COHERE.value["embedding_model"]
        super().__init__(
            name=name,
            score_threshold=score_threshold,
        )
        if cohere_api_key is None:
            cohere_api_key = os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError(
                "Cohere API key must be provided via `cohere_api_key` parameter or "
                "`COHERE_API_KEY` environment variable."
            )
        self._client = None
        self._async_client = None

    @deprecated("_initialize_client method no longer required")
    def _initialize_client(self, cohere_api_key: str | None = None):
        """Initializes the Cohere client.

        :param cohere_api_key: The API key for the Cohere client, can also
            be set via the COHERE_API_KEY environment variable.
        :type cohere_api_key: str
        :return: An instance of the Cohere client.
        :rtype: cohere.Client
        """
        cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError("Cohere API key cannot be 'None'.")
        return None, None

    def __call__(self, docs: list[str]) -> list[list[float]]:
        """Embed a list of documents. Supports text only.

        :param docs: The documents to embed.
        :type docs: List[str]
        :return: The vector embeddings of the documents.
        :rtype: List[List[float]]
        """
        return self.encode_queries(docs)

    async def acall(self, docs: list[Any]) -> list[list[float]]:
        """Embed a list of documents asynchronously. Supports text only.

        :param docs: The documents to embed.
        :type docs: List[str]
        :return: The vector embeddings of the documents.
        :rtype: List[List[float]]
        """
        return await self.aencode_queries(docs)

    def encode_queries(self, docs: list[str]) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs, input_type="search_query", model=f"{self.type}/{self.name}"
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    def encode_documents(self, docs: list[str]) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs,
                input_type="search_document",
                model=f"{self.type}/{self.name}",
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    async def aencode_queries(self, docs: list[str]) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs, input_type="search_query", model=f"{self.type}/{self.name}"
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    async def aencode_documents(self, docs: list[str]) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs,
                input_type="search_document",
                model=f"{self.type}/{self.name}",
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e
