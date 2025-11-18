import os
from asyncio import sleep as asleep
from time import sleep
from typing import Any, List, Optional, Union

import openai
import tiktoken
from openai import OpenAIError
from openai._types import NotGiven
from openai.types import CreateEmbeddingResponse
from pydantic import PrivateAttr

from semantic_router.encoders import DenseEncoder
from semantic_router.schema import EncoderInfo
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger

model_configs = {
    "text-embedding-ada-002": EncoderInfo(
        name="text-embedding-ada-002",
        token_limit=8192,
        threshold=0.82,
    ),
    "text-embedding-3-small": EncoderInfo(
        name="text-embedding-3-small",
        token_limit=8192,
        threshold=0.3,
    ),
    "text-embedding-3-large": EncoderInfo(
        name="text-embedding-3-large",
        token_limit=8192,
        threshold=0.3,
    ),
}


class OpenAIEncoder(DenseEncoder):
    """OpenAI encoder class for generating embeddings using OpenAI API.

    The OpenAIEncoder class is a subclass of DenseEncoder and utilizes the OpenAI API
    to generate embeddings for given documents. It requires an OpenAI API key and
    supports customization of the score threshold for filtering or processing the embeddings.
    """

    _client: Optional[openai.Client] = PrivateAttr(default=None)
    _async_client: Optional[openai.AsyncClient] = PrivateAttr(default=None)
    dimensions: Union[int, NotGiven] = NotGiven()
    token_limit: int = 8192  # default value, should be replaced by config
    _token_encoder: Any = PrivateAttr()
    type: str = "openai"
    max_retries: int = 3

    def __init__(
        self,
        name: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_org_id: Optional[str] = None,
        score_threshold: Optional[float] = None,
        dimensions: Union[int, NotGiven] = NotGiven(),
        max_retries: int = 3,
    ):
        """Initialize the OpenAIEncoder.

        :param name: The name of the embedding model to use.
        :type name: str
        :param openai_base_url: The base URL for the OpenAI API.
        :type openai_base_url: str
        :param openai_api_key: The OpenAI API key.
        :type openai_api_key: str
        :param openai_org_id: The OpenAI organization ID.
        :type openai_org_id: str
        :param score_threshold: The score threshold for the embeddings.
        :type score_threshold: float
        :param dimensions: The dimensions of the embeddings.
        :type dimensions: int
        :param max_retries: The maximum number of retries for the OpenAI API call.
        :type max_retries: int
        """
        if name is None:
            name = EncoderDefault.OPENAI.value["embedding_model"]
        if score_threshold is None and name in model_configs:
            set_score_threshold = model_configs[name].threshold
        elif score_threshold is None:
            logger.warning(
                f"Score threshold not set for model: {name}. Using default value."
            )
            set_score_threshold = 0.82
        else:
            set_score_threshold = score_threshold
        super().__init__(
            name=name,
            score_threshold=set_score_threshold,
        )
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        openai_org_id = openai_org_id or os.getenv("OPENAI_ORG_ID")
        if api_key is None or api_key.strip() == "":
            raise ValueError("OpenAI API key cannot be 'None' or empty.")
        if max_retries is not None:
            self.max_retries = max_retries
        try:
            self._client = openai.Client(
                base_url=base_url, api_key=api_key, organization=openai_org_id
            )
            self._async_client = openai.AsyncClient(
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
        if self._client is None:
            raise ValueError("OpenAI client is not initialized.")
        embeds = None

        if truncate:
            # check if any document exceeds token limit and truncate if so
            docs = [self._truncate(doc) for doc in docs]

        # Exponential backoff
        for j in range(self.max_retries + 1):
            try:
                logger.debug(f"Creating embeddings for {len(docs)} docs")
                embeds = self._client.embeddings.create(
                    input=docs,
                    model=self.name,
                    dimensions=self.dimensions,  # type: ignore[arg-type]  # NotGiven vs Omit - ignore type errors between openai SDK <2.0.0 and >=2.0.0
                )
                if embeds.data:
                    break
            except OpenAIError as e:
                logger.error("Exception occurred", exc_info=True)
                if self.max_retries != 0 and j < self.max_retries:
                    sleep(2**j)
                    logger.warning(
                        f"Retrying in {2**j} seconds due to OpenAIError: {e}"
                    )
                else:
                    raise

            except Exception as e:
                logger.error(f"OpenAI API call failed. Error: {e}")
                raise ValueError(f"OpenAI API call failed. Error: {str(e)}") from e

        if (
            not embeds
            or not isinstance(embeds, CreateEmbeddingResponse)
            or not embeds.data
        ):
            logger.info(f"Returned embeddings: {embeds}")
            raise ValueError("No embeddings returned.")

        embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
        return embeddings

    def _truncate(self, text: str) -> str:
        """Truncate a document to the token limit.

        :param text: The document to truncate.
        :type text: str
        :return: The truncated document.
        :rtype: str
        """
        # we use encode_ordinary as faster equivalent to encode(text, disallowed_special=())
        tokens = self._token_encoder.encode_ordinary(text)
        if len(tokens) > self.token_limit:
            logger.warning(
                f"Document exceeds token limit: {len(tokens)} > {self.token_limit}"
                "\nTruncating document..."
            )
            text = self._token_encoder.decode(tokens[: self.token_limit - 1])
            logger.info(f"Trunc length: {len(self._token_encoder.encode(text))}")
            return text
        return text

    async def acall(self, docs: List[str], truncate: bool = True) -> List[List[float]]:
        """Encode a list of text documents into embeddings using OpenAI API asynchronously.

        :param docs: List of text documents to encode.
        :param truncate: Whether to truncate the documents to token limit. If
            False and a document exceeds the token limit, an error will be
            raised.
        :return: List of embeddings for each document."""
        if self._async_client is None:
            raise ValueError("OpenAI async client is not initialized.")
        embeds = None

        if truncate:
            # check if any document exceeds token limit and truncate if so
            docs = [self._truncate(doc) for doc in docs]

        # Exponential backoff
        for j in range(self.max_retries + 1):
            try:
                embeds = await self._async_client.embeddings.create(
                    input=docs,
                    model=self.name,
                    dimensions=self.dimensions,  # type: ignore[arg-type]  # NotGiven vs Omit - ignore type errors between openai SDK <2.0.0 and >=2.0.0
                )
                if embeds.data:
                    break
            except OpenAIError as e:
                logger.error("Exception occurred", exc_info=True)
                if self.max_retries != 0 and j < self.max_retries:
                    await asleep(2**j)
                    logger.warning(
                        f"Retrying in {2**j} seconds due to OpenAIError: {e}"
                    )
                else:
                    raise

            except Exception as e:
                logger.error(f"OpenAI API call failed. Error: {e}")
                raise ValueError(f"OpenAI API call failed. Error: {e}") from e

        if (
            not embeds
            or not isinstance(embeds, CreateEmbeddingResponse)
            or not embeds.data
        ):
            logger.info(f"Returned embeddings: {embeds}")
            raise ValueError("No embeddings returned.")

        embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
        return embeddings
