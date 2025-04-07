from typing import Any

import litellm

from semantic_router.encoders import DenseEncoder
from semantic_router.utils.defaults import EncoderDefault


class LiteLLMEncoder(DenseEncoder):
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
    ):
        """Initialize the OpenAIEncoder.

        :param name: The name of the embedding model to use.
        :type name: str
        :param score_threshold: The score threshold for the embeddings.
        :type score_threshold: float
        """
        if name is None:
            self.name = "openai/" + EncoderDefault.OPENAI.value["embedding_model"]
        else:
            self.name = name
        if score_threshold is None:
            self.score_threshold = EncoderDefault.OPENAI.value["threshold"]
        else:
            self.score_threshold = score_threshold

    def __call__(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using LiteLLM.

        :param docs: List of text documents to encode.
        :return: List of embeddings for each document."""
        embeds = litellm.embedding(
            input=docs,
            model=self.name,
            **kwargs
        )
        if (
            not embeds
            or not isinstance(embeds, litellm.EmbeddingResponse)
            or not embeds.data
        ):
            raise ValueError("No embeddings returned")
        embeddings = [x["embedding"] for x in embeds.data]
        return embeddings

    async def acall(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of documents into embeddings using LiteLLM asynchronously.

        :param docs: List of documents to encode.
        :return: List of embeddings for each document."""
        embeds = await litellm.aembedding(
            input=docs,
            model=self.name,
            **kwargs
        )
        if (
            not embeds
            or not isinstance(embeds, litellm.EmbeddingResponse)
            or not embeds.data
        ):
            raise ValueError("No embeddings returned.")

        embeddings = [x["embedding"] for x in embeds.data]
        return embeddings
