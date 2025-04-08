"""This file contains the VoyageEncoder class which is used to encode text using Voyage"""

import litellm

from semantic_router.encoders.litellm import LiteLLMEncoder, litellm_to_list
from semantic_router.utils.defaults import EncoderDefault


class VoyageEncoder(LiteLLMEncoder):
    """Class to encode text using Voyage. Requires a Voyage API key from
    https://voyageai.com/api-keys/"""

    type: str = "voyage"

    def __init__(
        self,
        name: str | None = None,
        api_key: str | None = None,
        score_threshold: float = 0.4,
    ):
        """Initialize the VoyageEncoder.

        :param name: The name of the embedding model to use such as "voyage-embed".
        :type name: str
        :param voyage_api_key: The Voyage API key.
        :type voyage_api_key: str
        """

        if name is None:
            name = f"voyage/{EncoderDefault.VOYAGE.value['embedding_model']}"
        elif not name.startswith("voyage/"):
            name = f"voyage/{name}"
        super().__init__(
            name=name,
            score_threshold=score_threshold,
            api_key=api_key,
        )

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs,
                input_type="query",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Voyage API call failed. Error: {e}") from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs,
                input_type="document",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Voyage API call failed. Error: {e}") from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs,
                input_type="query",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Voyage API call failed. Error: {e}") from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs,
                input_type="document",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Voyage API call failed. Error: {e}") from e
