"""This file contains the VoyageEncoder class which is used to encode text using Voyage"""

import os
from typing import Any

from pydantic import PrivateAttr
from voyageai.client import Client
from voyageai.client_async import AsyncClient

from semantic_router.encoders.base import AsymmetricDenseMixin, DenseEncoder
from semantic_router.utils.defaults import EncoderDefault


class VoyageEncoder(DenseEncoder, AsymmetricDenseMixin):
    """Class to encode text using Voyage and VoyageAI SDK.
    Requires a Voyage API key from https://voyageai.com/api-keys/
    """

    _client: Any = PrivateAttr()
    _async_client: Any = PrivateAttr()
    type: str = "voyage"

    def __init__(
        self,
        name: str | None = None,
        api_key: str | None = None,
        score_threshold: float = 0.4,
    ):
        """Initialize the VoyageEncoder.

        :param name: The name of the embedding model to use such as "voyage-3-lite".
        :type name: str
        :param api_key: The Voyage API key, can also be set via the VOYAGE_API_KEY
            environment variable.
        :type api_key: str
        :param score_threshold: The score threshold for the embeddings.
        :type score_threshold: float
        :raises ValueError: If no API key is provided or found in the environment.
        """
        # get default model name if none provided
        if name is None:
            name = f"{EncoderDefault.VOYAGE.value['embedding_model']}"
        super().__init__(name=name, score_threshold=score_threshold)
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        if api_key is None:
            raise ValueError(
                "Expected API key via `api_key` parameter or "
                "`VOYAGE_API_KEY` environment variable."
            )
        # NOTE: voyageai clients accept a missing key and only fail on the first
        # request, so the check above is required to fail fast here instead.
        self._client = Client(api_key=api_key)
        self._async_client = AsyncClient(api_key=api_key)

    def __call__(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using Voyage.

        :param docs: List of text documents to encode.
        :type docs: list[Any]
        :return: List of embeddings for each document.
        :rtype: list[list[float]]
        """
        return self.encode_queries(docs, **kwargs)

    async def acall(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using Voyage asynchronously.

        :param docs: List of text documents to encode.
        :type docs: list[Any]
        :return: List of embeddings for each document.
        :rtype: list[list[float]]
        """
        return await self.aencode_queries(docs, **kwargs)

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = self._client.embed(
                texts=docs,
                model=self.name,
                input_type="query",
                **kwargs,
            )
            return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Voyage API call failed. Error: {e}") from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = self._client.embed(
                texts=docs,
                model=self.name,
                input_type="document",
                **kwargs,
            )
            return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Voyage API call failed. Error: {e}") from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await self._async_client.embed(
                texts=docs,
                model=self.name,
                input_type="query",
                **kwargs,
            )
            return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Voyage API call failed. Error: {e}") from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await self._async_client.embed(
                texts=docs,
                model=self.name,
                input_type="document",
                **kwargs,
            )
            return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Voyage API call failed. Error: {e}") from e
