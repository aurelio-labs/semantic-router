"""This file contains the MistralEncoder class which is used to encode text using MistralAI"""

import os
from typing import Any

import litellm
from pydantic import PrivateAttr
from typing_extensions import deprecated

from semantic_router.encoders import DenseEncoder
from semantic_router.encoders.litellm import litellm_to_list
from semantic_router.utils.defaults import EncoderDefault


class MistralEncoder(DenseEncoder):
    """Class to encode text using MistralAI. Requires a MistralAI API key from
    https://console.mistral.ai/api-keys/"""

    _client: Any = PrivateAttr()
    _mistralai: Any = PrivateAttr()
    type: str = "mistral"

    def __init__(
        self,
        name: str | None = None,
        mistralai_api_key: str | None = None,
        score_threshold: float = 0.82,
    ):
        """Initialize the MistralEncoder.

        :param name: The name of the embedding model to use.
        :type name: str
        :param mistralai_api_key: The MistralAI API key.
        :type mistralai_api_key: str
        :param score_threshold: The score threshold for the embeddings.
        """
        if name is None:
            name = EncoderDefault.MISTRAL.value["embedding_model"]
        super().__init__(
            name=name,
            score_threshold=score_threshold,
        )
        if mistralai_api_key is None:
            mistralai_api_key = os.getenv("MISTRALAI_API_KEY")
        if mistralai_api_key is None:
            mistralai_api_key = os.getenv("MISTRAL_API_KEY")
        if mistralai_api_key is None:
            raise ValueError(
                "MistralAI API key must be provided via `mistralai_api_key` parameter or "
                "`MISTRALAI_API_KEY` environment variable."
            )

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

    def __call__(self, docs: list[str]) -> list[list[float]]:
        """Embed a list of documents. Supports text only.

        :param docs: The documents to embed.
        :type docs: List[str]
        :return: The vector embeddings of the documents.
        :rtype: List[List[float]]
        """
        return self.encode_queries(docs)

    async def acall(self, docs: list[str]) -> list[list[float]]:
        """Embed a list of documents asynchronously. Supports text only.

        :param docs: The documents to embed.
        :type docs: List[str]
        :return: The vector embeddings of the documents.
        :rtype: List[List[float]]
        """
        return await self.aencode_queries(docs)

    def encode_queries(self, docs: list[str]) -> list[list[float]]:
        try:
            embeds = litellm.embedding(input=docs, model=f"{self.type}/{self.name}")
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Mistral API call failed. Error: {e}") from e

    def encode_documents(self, docs: list[str]) -> list[list[float]]:
        try:
            embeds = litellm.embedding(input=docs, model=f"{self.type}/{self.name}")
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Mistral API call failed. Error: {e}") from e

    async def aencode_queries(self, docs: list[str]) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs, model=f"{self.type}/{self.name}"
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Mistral API call failed. Error: {e}") from e

    async def aencode_documents(self, docs: list[str]) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs, model=f"{self.type}/{self.name}"
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Mistral API call failed. Error: {e}") from e
