"""This file contains the NimEncoder class which is used to encode text using Nim"""

import os
from typing import Any

import openai
from openai.types import CreateEmbeddingResponse
from pydantic import PrivateAttr

from semantic_router.encoders.base import AsymmetricDenseMixin, DenseEncoder
from semantic_router.utils.defaults import EncoderDefault

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def nim_to_list(embeds: CreateEmbeddingResponse) -> list[list[float]]:
    """Convert a NVIDIA NIM embedding response to a list of embeddings.

    :param embeds: The embedding response returned by the NIM API.
    :type embeds: CreateEmbeddingResponse
    :return: One embedding per input document.
    :rtype: list[list[float]]
    """
    if not embeds or not embeds.data:
        raise ValueError("No embeddings found in Nim embedding response.")
    return [entry.embedding for entry in embeds.data]


class NimEncoder(DenseEncoder, AsymmetricDenseMixin):
    """Class to encode text using Nvidia NIM. Requires a Nim API key from
    https://build.nvidia.com/

    NVIDIA NIM exposes an OpenAI-compatible embeddings endpoint, so this encoder drives
    it with the OpenAI SDK pointed at ``https://integrate.api.nvidia.com/v1`` rather than
    depending on LiteLLM. Point ``base_url`` at your own host to use a self-hosted NIM.

    NVIDIA's ``nv-embedqa`` models are asymmetric: queries must be embedded with
    ``input_type="query"`` and stored documents with ``input_type="passage"``, so
    :meth:`encode_queries` and :meth:`encode_documents` differ accordingly.
    """

    _client: Any = PrivateAttr()
    _async_client: Any = PrivateAttr()
    type: str = "nvidia_nim"

    def __init__(
        self,
        name: str | None = None,
        api_key: str | None = None,
        score_threshold: float = 0.4,
        base_url: str | None = None,
    ):
        """Initialize the NimEncoder.

        :param name: The name of the embedding model to use such as
            "nvidia/nv-embedqa-e5-v5".
        :type name: str
        :param api_key: The Nim API key, can also be set via the NVIDIA_NIM_API_KEY
            environment variable.
        :type api_key: str
        :param score_threshold: The score threshold for the embeddings.
        :type score_threshold: float
        :param base_url: Override the NIM API base URL, can also be set via the
            NVIDIA_NIM_API_BASE environment variable. Use this to target a
            self-hosted NIM container.
        :type base_url: str
        :raises ValueError: If no API key is provided or found in the environment.
        """
        # get default model name if none provided
        if name is None:
            name = f"{EncoderDefault.NVIDIA_NIM.value['embedding_model']}"
        super().__init__(name=name, score_threshold=score_threshold)
        if api_key is None:
            api_key = os.getenv("NVIDIA_NIM_API_KEY")
        if api_key is None:
            raise ValueError(
                "Expected API key via `api_key` parameter or "
                "`NVIDIA_NIM_API_KEY` environment variable."
            )
        base_url = base_url or os.getenv("NVIDIA_NIM_API_BASE") or NIM_BASE_URL
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    def __call__(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using NVIDIA NIM.

        :param docs: List of text documents to encode.
        :type docs: list[Any]
        :return: List of embeddings for each document.
        :rtype: list[list[float]]
        """
        return self.encode_queries(docs, **kwargs)

    async def acall(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using NVIDIA NIM
        asynchronously.

        :param docs: List of text documents to encode.
        :type docs: list[Any]
        :return: List of embeddings for each document.
        :rtype: list[list[float]]
        """
        return await self.aencode_queries(docs, **kwargs)

    @staticmethod
    def _extra_body(input_type: str, kwargs: dict) -> dict:
        """Build the NIM-specific ``extra_body``, letting callers override defaults.

        ``input_type`` is not an OpenAI parameter, so NIM expects it in ``extra_body``.

        :param input_type: The NIM input type, either "query" or "passage".
        :type input_type: str
        :param kwargs: The keyword arguments passed to an encode method. Any
            ``extra_body`` entry is removed and merged over the defaults.
        :type kwargs: dict
        :return: The body to send alongside the OpenAI parameters.
        :rtype: dict
        """
        return {"input_type": input_type, **kwargs.pop("extra_body", {})}

    def _embed(self, docs: list[str], input_type: str, **kwargs) -> list[list[float]]:
        """Embed documents, sending NIM's ``input_type`` through ``extra_body``.

        :param docs: The documents to embed.
        :type docs: list[str]
        :param input_type: The NIM input type, either "query" or "passage".
        :type input_type: str
        :return: One embedding per document.
        :rtype: list[list[float]]
        """
        extra_body = self._extra_body(input_type, kwargs)
        try:
            embeds = self._client.embeddings.create(
                input=docs,
                model=self.name,
                extra_body=extra_body,
                **kwargs,
            )
            return nim_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Nim API call failed. Error: {e}") from e

    async def _aembed(
        self, docs: list[str], input_type: str, **kwargs
    ) -> list[list[float]]:
        """Async version of :meth:`_embed`.

        :param docs: The documents to embed.
        :type docs: list[str]
        :param input_type: The NIM input type, either "query" or "passage".
        :type input_type: str
        :return: One embedding per document.
        :rtype: list[list[float]]
        """
        extra_body = self._extra_body(input_type, kwargs)
        try:
            embeds = await self._async_client.embeddings.create(
                input=docs,
                model=self.name,
                extra_body=extra_body,
                **kwargs,
            )
            return nim_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Nim API call failed. Error: {e}") from e

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        return self._embed(docs, input_type="query", **kwargs)

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        return self._embed(docs, input_type="passage", **kwargs)

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        return await self._aembed(docs, input_type="query", **kwargs)

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        return await self._aembed(docs, input_type="passage", **kwargs)
