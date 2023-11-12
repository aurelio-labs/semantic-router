import os

import cohere

from semantic_router.rankers import BaseReranker


class CohereRanker(BaseReranker):
    client: cohere.Client | None

    def __init__(
        self, name: str = "rerank-english-v2.0",
        top_n: int = 5,
        cohere_api_key: str | None = None
    ):
        super().__init__(name=name, top_n=top_n)
        cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if cohere_api_key is None:
            raise ValueError("Cohere API key cannot be 'None'.")
        self.client = cohere.Client(cohere_api_key)

    def __call__(self, query: str, docs: list[str]) -> list[str]:
        # get top_n results
        results = self.client.rerank(
            query=query, documents=docs, top_n=self.top_n,
            model=self.name
        )
        # get indices of entries that are ranked highest by cohere
        top_idx = [r.index for r in results]
        top_docs = [docs[i] for i in top_idx]
        return top_idx, top_docs