import os
from typing import Any, List, Optional
from pydantic import Field

from aurelio_sdk import AurelioClient, AsyncAurelioClient, EmbeddingResponse

from semantic_router.encoders.base import SparseEncoder
from semantic_router.schema import SparseEmbedding


class AurelioSparseEncoder(SparseEncoder):
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
        res: EmbeddingResponse = self.client.embedding(input=docs, model=self.name)
        embeds = [SparseEmbedding.from_aurelio(r.embedding) for r in res.data]
        return embeds

    async def acall(self, docs: list[str]) -> list[SparseEmbedding]:
        res: EmbeddingResponse = await self.async_client.embedding(
            input=docs, model=self.name
        )
        embeds = [SparseEmbedding.from_aurelio(r.embedding) for r in res.data]
        return embeds

    def fit(self, docs: List[str]):
        raise NotImplementedError("AurelioSparseEncoder does not support fit.")
