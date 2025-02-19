import os
from typing import Any, List, Optional

from pydantic import PrivateAttr

from semantic_router.encoders import DenseEncoder
from semantic_router.encoders.base import AsymmetricDenseMixin
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
        name: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
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
        self._client, self._async_client = self._initialize_client(cohere_api_key)

    def _initialize_client(self, cohere_api_key: Optional[str] = None):
        """Initializes the Cohere client.

        :param cohere_api_key: The API key for the Cohere client, can also
            be set via the COHERE_API_KEY environment variable.
        :type cohere_api_key: str
        :return: An instance of the Cohere client.
        :rtype: cohere.Client
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
            async_client = cohere.AsyncClient(cohere_api_key)
        except Exception as e:
            raise ValueError(
                f"Cohere API client failed to initialize. Error: {e}"
            ) from e
        return client, async_client

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Embed a list of documents. Supports text only.

        :param docs: The documents to embed.
        :type docs: List[str]
        :return: The vector embeddings of the documents.
        :rtype: List[List[float]]
        """
        return self.encode_queries(docs)

    async def acall(self, docs: List[Any]) -> List[List[float]]:
        return await self.aencode_queries(docs)

    def encode_queries(self, docs: List[str]) -> List[List[float]]:
        if self._client is None:
            raise ValueError("Cohere client is not initialized.")

        try:
            embeds = self._client.embed(
                texts=docs, input_type="search_query", model=self.name
            )
            if isinstance(embeds, self._embed_type):
                raise NotImplementedError(
                    "Handling of EmbedByTypeResponseEmbeddings is not implemented."
                )
            else:
                return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    def encode_documents(self, docs: List[str]) -> List[List[float]]:
        if self._client is None:
            raise ValueError("Cohere client is not initialized.")

        try:
            embeds = self._client.embed(
                texts=docs, input_type="search_document", model=self.name
            )
            if isinstance(embeds, self._embed_type):
                raise NotImplementedError(
                    "Handling of EmbedByTypeResponseEmbeddings is not implemented."
                )
            else:
                return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    async def aencode_queries(self, docs: List[str]) -> List[List[float]]:
        if self._async_client is None:
            raise ValueError("Cohere client is not initialized.")

        try:
            embeds = await self._async_client.embed(
                texts=docs, input_type="search_query", model=self.name
            )
            if isinstance(embeds, self._embed_type):
                raise NotImplementedError(
                    "Handling of EmbedByTypeResponseEmbeddings is not implemented."
                )
            else:
                return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    async def aencode_documents(self, docs: List[str]) -> List[List[float]]:
        if self._async_client is None:
            raise ValueError("Cohere client is not initialized.")

        try:
            embeds = await self._async_client.embed(
                texts=docs, input_type="search_document", model=self.name
            )
            if isinstance(embeds, self._embed_type):
                raise NotImplementedError(
                    "Handling of EmbedByTypeResponseEmbeddings is not implemented."
                )
            else:
                return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e
