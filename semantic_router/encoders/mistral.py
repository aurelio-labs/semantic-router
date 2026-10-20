"""This file contains the MistralEncoder class which is used to encode text using MistralAI"""

import os
from typing import Any

from mistralai import Mistral
from mistralai.models import EmbeddingResponse
from pydantic import PrivateAttr
from typing_extensions import deprecated

from semantic_router.encoders.base import AsymmetricDenseMixin, DenseEncoder
from semantic_router.utils.defaults import EncoderDefault


def mistral_to_list(embeds: EmbeddingResponse) -> list[list[float]]:
    """Convert a Mistral embedding response to a list of embeddings.

    :param embeds: The Mistral embedding response.
    :type embeds: EmbeddingResponse
    :return: One embedding per input document.
    :rtype: list[list[float]]
    """
    if not embeds or not embeds.data:
        raise ValueError("No embeddings found in Mistral embedding response.")
    embeddings: list[list[float]] = []
    for entry in embeds.data:
        if entry.embedding is None:
            raise ValueError("Mistral embedding response contained an empty entry.")
        embeddings.append(entry.embedding)
    return embeddings


class MistralEncoder(DenseEncoder, AsymmetricDenseMixin):
    """Class to encode text using MistralAI and SDK.
    Requires a MistralAI API key from https://console.mistral.ai/api-keys/
    """

    _client: Any = PrivateAttr()  # TODO: deprecated, to remove in v0.2.0
    _mistralai: Any = PrivateAttr()  # TODO: deprecated, to remove in v0.2.0
    type: str = "mistral"

    def __init__(
        self,
        name: str | None = None,
        mistralai_api_key: str | None = None,  # TODO: rename to api_key in v0.2.0
        score_threshold: float = 0.4,
    ):
        """Initialize the MistralEncoder.

        :param name: The name of the embedding model to use such as "mistral-embed".
        :type name: str
        :param mistralai_api_key: The MistralAI API key.
        :type mistralai_api_key: str
        :param score_threshold: The score threshold for the embeddings.
        """
        # get default model name if none provided
        if name is None:
            name = f"{EncoderDefault.MISTRAL.value['embedding_model']}"
        super().__init__(name=name, score_threshold=score_threshold)
        if mistralai_api_key is None:
            mistralai_api_key = os.getenv("MISTRALAI_API_KEY")
        if mistralai_api_key is None:
            mistralai_api_key = os.getenv("MISTRAL_API_KEY")
        if mistralai_api_key is None:
            raise ValueError(
                "Expected API key via `mistralai_api_key` parameter or "
                "`MISTRALAI_API_KEY`/`MISTRAL_API_KEY` environment variable."
            )
        self._client = Mistral(api_key=mistralai_api_key)

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            mistral_embeds = self._client.embeddings.create(
                inputs=docs,
                model=self.name,
                encoding_format="float",
                **kwargs,
            )
            return mistral_to_list(mistral_embeds)
        except Exception as e:
            raise ValueError(f"Mistral API call failed. Error: {e}") from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        return self.encode_queries(docs, **kwargs)

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            mistral_embeds = await self._client.embeddings.create_async(
                inputs=docs,
                model=self.name,
                encoding_format="float",
                **kwargs,
            )
            return mistral_to_list(mistral_embeds)
        except Exception as e:
            raise ValueError(f"Mistral API call failed. Error: {e}") from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        return await self.aencode_queries(docs, **kwargs)

    def __call__(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using MistralAI.

        :param docs: List of text documents to encode.
        :type docs: list[Any]
        :return: List of embeddings for each document.
        :rtype: list[list[float]]
        """
        return self.encode_queries(docs, **kwargs)

    async def acall(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using MistralAI asynchronously.

        :param docs: List of text documents to encode.
        :type docs: list[Any]
        :return: List of embeddings for each document.
        :rtype: list[list[float]]
        """
        return await self.aencode_queries(docs, **kwargs)

    # TODO: deprecated, to remove in v0.2.0
    @deprecated("_initialize_client method no longer required")
    def _initialize_client(self, api_key):
        """Initialize the MistralAI client.

        :param api_key: The MistralAI API key.
        :type api_key: str
        :return: None
        :rtype: None
        """
        api_key = (
            api_key or os.getenv("MISTRALAI_API_KEY") or os.getenv("MISTRAL_API_KEY")
        )
        if api_key is None:
            raise ValueError("Mistral API key not provided")
        return None
