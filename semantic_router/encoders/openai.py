import os
from time import sleep
from typing import Any, List, Optional, Union
from pydantic.v1 import PrivateAttr

import openai
from openai import OpenAIError
from openai._types import NotGiven
from openai.types import CreateEmbeddingResponse
import tiktoken

from semantic_router.encoders import BaseEncoder
from semantic_router.schema import EncoderInfo
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


model_configs = {
    "text-embedding-ada-002": EncoderInfo(
        name="text-embedding-ada-002", token_limit=8192
    ),
    "text-embed-3-small": EncoderInfo(name="text-embed-3-small", token_limit=8192),
    "text-embed-3-large": EncoderInfo(name="text-embed-3-large", token_limit=8192),
}


class OpenAIEncoder(BaseEncoder):
    client: Optional[openai.Client]
    dimensions: Union[int, NotGiven] = NotGiven()
    token_limit: int = 8192  # default value, should be replaced by config
    _token_encoder: Any = PrivateAttr()
    type: str = "openai"

    def __init__(
        self,
        name: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_org_id: Optional[str] = None,
        score_threshold: float = 0.82,
        dimensions: Union[int, NotGiven] = NotGiven(),
    ):
        if name is None:
            name = EncoderDefault.OPENAI.value["embedding_model"]
        super().__init__(name=name, score_threshold=score_threshold)
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        openai_org_id = openai_org_id or os.getenv("OPENAI_ORG_ID")
        if api_key is None:
            raise ValueError("OpenAI API key cannot be 'None'.")
        try:
            self.client = openai.Client(
                base_url=base_url, api_key=api_key, organization=openai_org_id
            )
        except Exception as e:
            raise ValueError(
                f"OpenAI API client failed to initialize. Error: {e}"
            ) from e
        # set dimensions to support openai embed 3 dimensions param
        self.dimensions = dimensions
        # if model name is known, set token limit
        if name in model_configs:
            self.token_limit = model_configs[name].token_limit
        # get token encoder
        self._token_encoder = tiktoken.encoding_for_model(name)

    def __call__(self, docs: List[str], truncate: bool = True) -> List[List[float]]:
        """Encode a list of text documents into embeddings using OpenAI API.

        :param docs: List of text documents to encode.
        :param truncate: Whether to truncate the documents to token limit. If
            False and a document exceeds the token limit, an error will be
            raised.
        :return: List of embeddings for each document."""
        if self.client is None:
            raise ValueError("OpenAI client is not initialized.")
        embeds = None
        error_message = ""

        if truncate:
            # check if any document exceeds token limit and truncate if so
            for i in range(len(docs)):
                logger.info(f"Document {i+1} length: {len(docs[i])}")
                docs[i] = self._truncate(docs[i])
                logger.info(f"Document {i+1} trunc length: {len(docs[i])}")

        # Exponential backoff
        for j in range(1, 7):
            try:
                embeds = self.client.embeddings.create(
                    input=docs,
                    model=self.name,
                    dimensions=self.dimensions,
                )
                if embeds.data:
                    break
            except OpenAIError as e:
                sleep(2**j)
                error_message = str(e)
                logger.warning(f"Retrying in {2**j} seconds...")
            except Exception as e:
                logger.error(f"OpenAI API call failed. Error: {error_message}")
                raise ValueError(f"OpenAI API call failed. Error: {e}") from e

        if (
            not embeds
            or not isinstance(embeds, CreateEmbeddingResponse)
            or not embeds.data
        ):
            logger.info(f"Returned embeddings: {embeds}")
            raise ValueError(f"No embeddings returned. Error: {error_message}")

        embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
        return embeddings

    def _truncate(self, text: str) -> str:
        tokens = self._token_encoder.encode(text)
        if len(tokens) > self.token_limit:
            logger.warning(
                f"Document exceeds token limit: {len(tokens)} > {self.token_limit}"
                "\nTruncating document..."
            )
            text = self._token_encoder.decode(tokens[: self.token_limit - 1])
            logger.info(f"Trunc length: {len(self._token_encoder.encode(text))}")
            return text
        return text
