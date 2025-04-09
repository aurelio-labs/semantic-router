import os
from typing import Any

import litellm
from pydantic import PrivateAttr
from typing_extensions import deprecated

from semantic_router.encoders.litellm import LiteLLMEncoder, litellm_to_list
from semantic_router.utils.defaults import EncoderDefault


class CohereEncoder(LiteLLMEncoder):
    """Dense encoder that uses Cohere API to embed documents. Supports text only. Requires
    a Cohere API key from https://dashboard.cohere.com/api-keys.
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
        """
        # get default model name if none provided and convert to litellm format
        if name is None:
            name = f"cohere/{EncoderDefault.COHERE.value['embedding_model']}"
        elif not name.startswith("cohere/"):
            name = f"cohere/{name}"
        super().__init__(
            name=name,
            score_threshold=score_threshold,
            api_key=cohere_api_key,
        )
        self._client = None  # TODO: deprecated, to remove in v0.2.0
        self._async_client = None  # TODO: deprecated, to remove in v0.2.0

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

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs,
                input_type="search_query",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs,
                input_type="search_document",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs,
                input_type="search_query",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs,
                input_type="search_document",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Cohere API call failed. Error: {e}") from e
