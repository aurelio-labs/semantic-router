"""This file contains the JinaEncoder class which is used to encode text using Jina"""

import litellm

from semantic_router.encoders.litellm import LiteLLMEncoder, litellm_to_list
from semantic_router.utils.defaults import EncoderDefault


class JinaEncoder(LiteLLMEncoder):
    """Class to encode text using Jina. Requires a Jina API key from
    https://jina.ai/api-keys/"""

    type: str = "jina"

    def __init__(
        self,
        name: str | None = None,
        api_key: str | None = None,
        score_threshold: float = 0.4,
    ):
        """Initialize the JinaEncoder.

        :param name: The name of the embedding model to use such as "jina-embeddings-v3".
        :param jina_api_key: The Jina API key.
        :type jina_api_key: str
        """

        if name is None:
            name = f"jina_ai/{EncoderDefault.JINA.value['embedding_model']}"
        elif not name.startswith("jina_ai/"):
            name = f"jina_ai/{name}"
        super().__init__(
            name=name,
            score_threshold=score_threshold,
            api_key=api_key,
        )

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs,
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Jina API call failed. Error: {e}") from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs,
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Jina API call failed. Error: {e}") from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs,
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Jina API call failed. Error: {e}") from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs,
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Jina API call failed. Error: {e}") from e
