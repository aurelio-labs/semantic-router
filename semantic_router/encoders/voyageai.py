import os
from time import sleep
from typing import List, Optional

import voyageai
from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


class VoyageAIEncoder(BaseEncoder):
    client: Optional[voyageai.Client]
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
        api_key = voyage_api_key or os.environ.get("VOYAGE_API_KEY")
        if api_key is None:
            raise ValueError("VOYAGEAI API key cannot be 'None'.")
        try:
            self.client = voyageai.Client(api_key)
        except Exception as e:
            raise ValueError(
                f"VOYAGE API client failed to initialize. Error: {e}"
            ) from e

    def __call__(self, docs: List[str]) -> List[List[float]]:
        if self.client is None:
            raise ValueError("VoyageAI client is not initialized.")
        embeds = None
        error_message = ""

        # Exponential backoff
        for j in range(1, 7):
            try:
                embeds = self.client.embed(
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
