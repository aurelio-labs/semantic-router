"""This file contains the MistralEncoder class which is used to encode text using MistralAI"""

import os
from time import sleep
from typing import Any, List, Optional

from pydantic.v1 import PrivateAttr

from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault


class MistralEncoder(BaseEncoder):
    """Class to encode text using MistralAI"""

    _client: Any = PrivateAttr()
    _mistralai: Any = PrivateAttr()
    type: str = "mistral"

    def __init__(
        self,
        name: Optional[str] = None,
        mistralai_api_key: Optional[str] = None,
        score_threshold: float = 0.82,
    ):
        if name is None:
            name = EncoderDefault.MISTRAL.value["embedding_model"]
        super().__init__(name=name, score_threshold=score_threshold)
        self._client, self._mistralai = self._initialize_client(mistralai_api_key)

    def _initialize_client(self, api_key):
        try:
            import mistralai
            from mistralai.client import MistralClient
        except ImportError:
            raise ImportError(
                "Please install MistralAI to use MistralEncoder. "
                "You can install it with: "
                "`pip install 'semantic-router[mistralai]'`"
            )

        api_key = api_key or os.getenv("MISTRALAI_API_KEY")
        if api_key is None:
            raise ValueError("Mistral API key not provided")
        try:
            client = MistralClient(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Unable to connect to MistralAI {e.args}: {e}") from e
        return client, mistralai

    def __call__(self, docs: List[str]) -> List[List[float]]:
        if self._client is None:
            raise ValueError("Mistral client not initialized")
        embeds = None
        error_message = ""

        # Exponential backoff
        for _ in range(3):
            try:
                embeds = self._client.embeddings(model=self.name, input=docs)
                if embeds.data:
                    break
            except self._mistralai.exceptions.MistralException as e:
                sleep(2**_)
                error_message = str(e)
            except Exception as e:
                raise ValueError(f"Unable to connect to MistralAI {e.args}: {e}") from e

        if (
            not embeds
            or not isinstance(
                embeds, self._mistralai.models.embeddings.EmbeddingResponse
            )
            or not embeds.data
        ):
            raise ValueError(f"No embeddings returned from MistralAI: {error_message}")
        embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
        return embeddings
