import os
from typing import Any

import cohere
from pydantic import PrivateAttr
from typing_extensions import deprecated

from semantic_router.encoders.base import AsymmetricDenseMixin, DenseEncoder
from semantic_router.utils.defaults import EncoderDefault


def docs2cohere_embed_input(docs: list[str]) -> list[dict[str, Any]]:
    """Convert a list of texts into Cohere's ``inputs`` format.

    The Cohere ``embed`` endpoint expects one ``EmbedInput`` per embedding, each
    holding its own ``content`` array, so a list of N texts becomes N inputs.

    :param docs: The texts to embed.
    :type docs: list[str]
    :return: One embed input per text.
    :rtype: list[dict[str, Any]]
    """
    return [{"content": [{"type": "text", "text": d}]} for d in docs]


class CohereEncoder(DenseEncoder, AsymmetricDenseMixin):
    """Dense encoder that uses Cohere API and SDK to embed documents.
    Supports text only.
    Requires a Cohere API key from https://dashboard.cohere.com/api-keys.
    """

    _client: Any = PrivateAttr()  # TODO: deprecated, to remove in v0.2.0
    _async_client: Any = PrivateAttr()  # TODO: deprecated, to remove in v0.2.0
    _embed_type: Any = PrivateAttr()  # TODO: deprecated, to remove in v0.2.0
    type: str = "cohere"

    def __init__(
        self,
        name: str | None = None,
        cohere_api_key: str | None = None,  # TODO: rename to api_key in v0.2.0
        score_threshold: float = 0.3,
    ):
        """Initialize the Cohere encoder.

        :param name: The name of the embedding model to use such as "embed-english-v3.0" or
            "embed-multilingual-v3.0".
        :type name: str
        :param cohere_api_key: The API key for the Cohere client, can also
            be set via the COHERE_API_KEY environment variable.
        :type cohere_api_key: str
        :param score_threshold: The threshold for the score of the embedding.
        :type score_threshold: float
        :raises ValueError: If no API key is provided or found in the environment.
        """
        # get default model name if none provided
        if name is None:
            name = f"{EncoderDefault.COHERE.value['embedding_model']}"
        super().__init__(name=name, score_threshold=score_threshold)
        if cohere_api_key is None:
            cohere_api_key = os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError(
                "Expected API key via `cohere_api_key` parameter or "
                "`COHERE_API_KEY` environment variable."
            )
        self._client = cohere.ClientV2(api_key=cohere_api_key)
        self._async_client = cohere.AsyncClientV2(api_key=cohere_api_key)

    # TODO: deprecated, to remove in v0.2.0
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

    def __call__(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using Cohere.

        :param docs: List of text documents to encode.
        :type docs: list[Any]
        :return: List of embeddings for each document.
        :rtype: list[list[float]]
        """
        return self.encode_queries(docs, **kwargs)

    async def acall(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using Cohere asynchronously.

        :param docs: List of text documents to encode.
        :type docs: list[Any]
        :return: List of embeddings for each document.
        :rtype: list[list[float]]
        """
        return await self.aencode_queries(docs, **kwargs)

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            cohere_embeds = self._client.embed(
                inputs=docs2cohere_embed_input(docs),
                input_type="search_query",
                model=self.name,
                embedding_types=["float"],
                **kwargs,
            )
            values = cohere_embeds.embeddings.float_
            if values is None:
                raise ValueError("Cohere API call returned None.")
            return values
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            cohere_embeds = self._client.embed(
                inputs=docs2cohere_embed_input(docs),
                input_type="search_document",
                model=self.name,
                embedding_types=["float"],
                **kwargs,
            )
            values = cohere_embeds.embeddings.float_
            if values is None:
                raise ValueError("Cohere API call returned None.")
            return values
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            cohere_embeds = await self._async_client.embed(
                inputs=docs2cohere_embed_input(docs),
                input_type="search_query",
                model=self.name,
                embedding_types=["float"],
                **kwargs,
            )
            values = cohere_embeds.embeddings.float_
            if values is None:
                raise ValueError("Cohere API call returned None.")
            return values
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            cohere_embeds = await self._async_client.embed(
                inputs=docs2cohere_embed_input(docs),
                input_type="search_document",
                model=self.name,
                embedding_types=["float"],
                **kwargs,
            )
            values = cohere_embeds.embeddings.float_
            if values is None:
                raise ValueError("Cohere API call returned None.")
            return values
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e
