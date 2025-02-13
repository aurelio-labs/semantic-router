import os
from typing import Any, List, Optional

from aurelio_sdk import AsyncAurelioClient, AurelioClient, EmbeddingResponse
from pydantic import Field

from semantic_router.encoders.base import SparseEncoder
from semantic_router.schema import SparseEmbedding


class AurelioSparseEncoder(SparseEncoder):
    """Sparse encoder using Aurelio Platform's embedding API. Requires an API key from
    https://platform.aurelio.ai
    """
    model: Optional[Any] = None
    client: AurelioClient = Field(default_factory=AurelioClient, exclude=True)
    async_client: AsyncAurelioClient = Field(
        default_factory=AsyncAurelioClient, exclude=True
    )
    type: str = "sparse"

    def __init__(
        self,
        name: str | None = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the AurelioSparseEncoder.

        :param name: The name of the model to use.
        :type name: str | None
        :param api_key: The API key to use.
        :type api_key: str | None
        """
        if name is None:
            name = "bm25"
        super().__init__(name=name)
        if api_key is None:
            api_key = os.getenv("AURELIO_API_KEY")
        if api_key is None:
            raise ValueError("AURELIO_API_KEY environment variable is not set.")
        self.client = AurelioClient(api_key=api_key)
        self.async_client = AsyncAurelioClient(api_key=api_key)

    def __call__(self, docs: list[str]) -> list[SparseEmbedding]:
        """Encode a list of documents using the Aurelio Platform embedding API. Documents
        must be strings, sparse encoders do not support other types.

        :param docs: The documents to encode.
        :type docs: list[str]
        :return: The encoded documents.
        :rtype: list[SparseEmbedding]
        """
        res: EmbeddingResponse = self.client.embedding(input=docs, model=self.name)
        embeds = [SparseEmbedding.from_aurelio(r.embedding) for r in res.data]
        return embeds

    async def acall(self, docs: list[str]) -> list[SparseEmbedding]:
        """Asynchronously encode a list of documents using the Aurelio Platform
        embedding API. Documents must be strings, sparse encoders do not support other
        types.

        :param docs: The documents to encode.
        :type docs: list[str]
        :return: The encoded documents.
        :rtype: list[SparseEmbedding]
        """
        res: EmbeddingResponse = await self.async_client.embedding(
            input=docs, model=self.name
        )
        embeds = [SparseEmbedding.from_aurelio(r.embedding) for r in res.data]
        return embeds

    def fit(self, docs: List[str]):
        """Fit the encoder to a list of documents. AurelioSparseEncoder does not support
        fit yet.

        :param docs: The documents to fit the encoder to.
        :type docs: list[str]
        """
        raise NotImplementedError("AurelioSparseEncoder does not support fit.")
