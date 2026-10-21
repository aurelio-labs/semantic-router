"""This file contains the JinaEncoder class which is used to encode text using Jina"""

import os
from typing import Any

import openai
from openai.types import CreateEmbeddingResponse
from pydantic import PrivateAttr

from semantic_router.encoders.base import AsymmetricDenseMixin, DenseEncoder
from semantic_router.utils.defaults import EncoderDefault

JINA_BASE_URL = "https://api.jina.ai/v1"


def jina_to_list(embeds: CreateEmbeddingResponse) -> list[list[float]]:
    """Convert a Jina embedding response to a list of embeddings.

    :param embeds: The embedding response returned by the Jina API.
    :type embeds: CreateEmbeddingResponse
    :return: One embedding per input document.
    :rtype: list[list[float]]
    """
    if not embeds or not embeds.data:
        raise ValueError("No embeddings found in Jina embedding response.")
    return [entry.embedding for entry in embeds.data]


class JinaEncoder(DenseEncoder, AsymmetricDenseMixin):
    """Class to encode text using Jina. Requires a Jina API key from
    https://jina.ai/api-keys/

    Jina exposes an OpenAI-compatible embeddings endpoint, so this encoder drives it
    with the OpenAI SDK pointed at ``https://api.jina.ai/v1`` rather than depending on
    LiteLLM.
    """

    _client: Any = PrivateAttr()
    _async_client: Any = PrivateAttr()
    type: str = "jina"

    def __init__(
        self,
        name: str | None = None,
        api_key: str | None = None,
        score_threshold: float = 0.4,
        base_url: str | None = None,
    ):
        """Initialize the JinaEncoder.

        :param name: The name of the embedding model to use such as
            "jina-embeddings-v3".
        :type name: str
        :param api_key: The Jina API key, can also be set via the JINA_API_KEY or
            JINA_AI_API_KEY environment variable.
        :type api_key: str
        :param score_threshold: The score threshold for the embeddings.
        :type score_threshold: float
        :param base_url: Override the Jina API base URL, can also be set via the
            JINA_BASE_URL or JINA_AI_API_BASE environment variable.
        :type base_url: str
        :raises ValueError: If no API key is provided or found in the environment.
        """
        # get default model name if none provided
        if name is None:
            name = f"{EncoderDefault.JINA.value['embedding_model']}"
        super().__init__(name=name, score_threshold=score_threshold)
        if api_key is None:
            api_key = os.getenv("JINA_API_KEY")
        if api_key is None:
            # JINA_AI_API_KEY is what the previous litellm-based encoder read, so it
            # stays supported for backwards compatibility.
            api_key = os.getenv("JINA_AI_API_KEY")
        if api_key is None:
            raise ValueError(
                "Expected API key via `api_key` parameter or "
                "`JINA_API_KEY`/`JINA_AI_API_KEY` environment variable."
            )
        # JINA_AI_API_BASE is what the previous litellm-based encoder read.
        base_url = (
            base_url
            or os.getenv("JINA_BASE_URL")
            or os.getenv("JINA_AI_API_BASE")
            or JINA_BASE_URL
        )
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    def __call__(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using Jina.

        :param docs: List of text documents to encode.
        :type docs: list[Any]
        :return: List of embeddings for each document.
        :rtype: list[list[float]]
        """
        return self.encode_queries(docs, **kwargs)

    async def acall(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using Jina asynchronously.

        :param docs: List of text documents to encode.
        :type docs: list[Any]
        :return: List of embeddings for each document.
        :rtype: list[list[float]]
        """
        return await self.aencode_queries(docs, **kwargs)

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = self._client.embeddings.create(
                input=docs,
                model=self.name,
                **kwargs,
            )
            return jina_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Jina API call failed. Error: {e}") from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        return self.encode_queries(docs, **kwargs)

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await self._async_client.embeddings.create(
                input=docs,
                model=self.name,
                **kwargs,
            )
            return jina_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Jina API call failed. Error: {e}") from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        return await self.aencode_queries(docs, **kwargs)
