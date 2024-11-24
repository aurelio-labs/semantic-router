import os
from typing import Any, Dict, List, Optional
from pydantic.v1 import Field

from aurelio_sdk import AurelioClient, AsyncAurelioClient, EmbeddingResponse

from semantic_router.encoders import BaseEncoder


class AurelioSparseEncoder(BaseEncoder):
    model: Optional[Any] = None
    idx_mapping: Optional[Dict[int, int]] = None
    client: AurelioClient = Field(default_factory=AurelioClient, exclude=True)
    async_client: AsyncAurelioClient = Field(default_factory=AsyncAurelioClient, exclude=True)
    type: str = "sparse"

    def __init__(
        self,
        name: str = "bm25",
        score_threshold: float = 1.0,
        api_key: Optional[str] = None,
    ):
        super().__init__(name=name, score_threshold=score_threshold)
        if api_key is None:
            api_key = os.getenv("AURELIO_API_KEY")
        if api_key is None:
            raise ValueError("AURELIO_API_KEY environment variable is not set.")
        self.client = AurelioClient(api_key=api_key)
        self.async_client = AsyncAurelioClient(api_key=api_key)

    def __call__(self, docs: list[str]) -> list[dict[int, float]]:
        res: EmbeddingResponse = self.client.embedding(input=docs, model=self.name)
        embeds = [r.embedding.model_dump() for r in res.data]
        # convert sparse vector to {index: value} format
        sparse_dicts = [{i: v for i, v in zip(e["indices"], e["values"])} for e in embeds]
        return sparse_dicts
    
    async def acall(self, docs: list[str]) -> list[dict[int, float]]:
        res: EmbeddingResponse = await self.async_client.embedding(input=docs, model=self.name)
        embeds = [r.embedding.model_dump() for r in res.data]
        # convert sparse vector to {index: value} format
        sparse_dicts = [{i: v for i, v in zip(e["indices"], e["values"])} for e in embeds]
        return sparse_dicts

    def fit(self, docs: List[str]):
        raise NotImplementedError("AurelioSparseEncoder does not support fit.")
