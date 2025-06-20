"""This file contains the NimEncoder class which is used to encode text using Nim"""

import litellm

from semantic_router.encoders.litellm import LiteLLMEncoder, litellm_to_list
from semantic_router.utils.defaults import EncoderDefault


class NimEncoder(LiteLLMEncoder):
    """Class to encode text using Nvidia NIM. Requires a Nim API key from
    https://nim.ai/api-keys/"""

    type: str = "nvidia_nim"

    def __init__(
        self,
        name: str | None = None,
        api_key: str | None = None,
        score_threshold: float = 0.4,
    ):
        """Initialize the NimEncoder.

        :param name: The name of the embedding model to use such as "nv-embedqa-e5-v5".
        :type name: str
        :param nim_api_key: The Nim API key.
        :type nim_api_key: str
        """

        if name is None:
            name = f"nvidia_nim/{EncoderDefault.NVIDIA_NIM.value['embedding_model']}"
        elif not name.startswith("nvidia_nim/"):
            name = f"nvidia_nim/{name}"
        super().__init__(
            name=name,
            score_threshold=score_threshold,
            api_key=api_key,
        )

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs,
                input_type="passage",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Nim API call failed. Error: {e}") from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(
                input=docs,
                input_type="passage",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Nim API call failed. Error: {e}") from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs,
                input_type="passage",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Nim API call failed. Error: {e}") from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(
                input=docs,
                input_type="passage",
                model=f"{self.type}/{self.name}",
                **kwargs,
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f"Nim API call failed. Error: {e}") from e
