import os
from time import sleep
from typing import Any, List, Optional

from pydantic.v1 import PrivateAttr

from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


class VoyageAIEncoder(BaseEncoder):
    _client: Any = PrivateAttr()
    type: str = "voyageai"

    def __init__(
        self,
        name: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        score_threshold: float = 0.82,
    ):
        if name is None:
            name = EncoderDefault.VOYAGE.value["embedding_model"]
        super().__init__(name=name, score_threshold=score_threshold)
        self._client = self._initialize_client(api_key=voyage_api_key)

    def _initialize_client(self, api_key: Optional[str] = None):
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "Please install VoyageAI to use VoyageAIEncoder. "
                "You can install it with: "
                "`pip install 'semantic-router[voyageai]'`"
            )

        api_key = api_key or os.getenv("VOYAGEAI_API_KEY")
        if api_key is None:
            raise ValueError("VoyageAI API key not provided")
        try:
            client = voyageai.Client(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Unable to connect to VoyageAI {e.args}: {e}") from e
        return client

    def __call__(self, docs: List[str]) -> List[List[float]]:
        if self._client == PrivateAttr():
            raise ValueError("VoyageAI client is not initialized.")
        embeds = None
        error_message = ""

        # Exponential backoff
        for j in range(1, 7):
            try:
                embeds = self._client.embed(
                    texts=docs,
                    model=self.name,
                    input_type="query",  # query or document
                )
                if embeds.embeddings:
                    break
                else:
                    sleep(2**j)
                    logger.warning(f"Retrying in {2**j} seconds...")

            except Exception as e:
                logger.error(f"VoyageAI API call failed. Error: {error_message}")
                raise ValueError(f"VoyageAI API call failed. Error: {e}") from e

        if not embeds or not embeds.embeddings:
            raise ValueError("VoyageAI API call failed. Error: No embeddings found.")

        return embeds.embeddings
