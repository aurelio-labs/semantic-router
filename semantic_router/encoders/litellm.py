import os
from typing import Any

import litellm

from semantic_router.encoders import DenseEncoder
from semantic_router.encoders.base import AsymmetricDenseMixin
from semantic_router.utils.defaults import EncoderDefault


def litellm_to_list(embeds: litellm.EmbeddingResponse) -> list[list[float]]:
    """Convert a LiteLLM embedding response to a list of embeddings.

    :param embeds: The LiteLLM embedding response.
    :return: A list of embeddings.
    """
    if (
        not embeds
        or not isinstance(embeds, litellm.EmbeddingResponse)
        or not embeds.data
    ):
        raise ValueError("No embeddings found in LiteLLM embedding response.")
    return [x["embedding"] for x in embeds.data]


class LiteLLMEncoder(DenseEncoder, AsymmetricDenseMixin):
    """LiteLLM encoder class for generating embeddings using LiteLLM.

    The LiteLLMEncoder class is a subclass of DenseEncoder and utilizes the LiteLLM SDK
    to generate embeddings for given documents. It supports all encoders supported by LiteLLM
    and supports customization of the score threshold for filtering or processing the embeddings.
    """

    type: str = "litellm"

    def __init__(
        self,
        name: str | None = None,
        score_threshold: float | None = None,
        api_key: str | None = None,
    ):
        """Initialize the LiteLLMEncoder.

        :param name: The name of the embedding model to use. Must use LiteLLM naming
            convention (e.g. "openai/text-embedding-3-small" or "mistral/mistral-embed").
        :type name: str
        :param score_threshold: The score threshold for the embeddings.
        :type score_threshold: float
        """
        if name is None:
            # defaults to default openai model if none provided
            name = "openai/" + EncoderDefault.OPENAI.value["embedding_model"]
        super().__init__(
            name=name,
            score_threshold=score_threshold if score_threshold is not None else 0.3,
        )
        self.type, self.name = self.name.split("/", 1)
        if api_key is None:
            api_key = os.getenv(self.type.upper() + "_API_KEY")
        if api_key is None:
            raise ValueError(
                "Expected API key via `api_key` parameter or `{self.type.upper()}_API_KEY` "
                "environment variable."
            )
        os.environ[self.type.upper() + "_API_KEY"] = api_key

    def __call__(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using LiteLLM.

        :param docs: List of text documents to encode.
        :return: List of embeddings for each document."""
        return self.encode_queries(docs, **kwargs)

    async def acall(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of documents into embeddings using LiteLLM asynchronously.

        :param docs: List of documents to encode.
        :return: List of embeddings for each document."""
        return await self.aencode_queries(docs, **kwargs)

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs, model=f"{self.type}/{self.name}", **kwargs
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(
                f"{self.type.capitalize()} API call failed. Error: {e}"
            ) from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs, model=f"{self.type}/{self.name}", **kwargs
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(
                f"{self.type.capitalize()} API call failed. Error: {e}"
            ) from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs, model=f"{self.type}/{self.name}", **kwargs
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(
                f"{self.type.capitalize()} API call failed. Error: {e}"
            ) from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs, model=f"{self.type}/{self.name}", **kwargs
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(
                f"{self.type.capitalize()} API call failed. Error: {e}"
            ) from e
