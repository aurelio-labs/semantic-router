import os

from pinecone_text import BM25Encoder

from semantic_router.retrievers import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, name: str = "bm25"):
        super().__init__(name=name)
        self.model = BM25Encoder()

    def __call__(self, docs: list[str]) -> list[list[float]]:
        if self.params is None:
            raise ValueError("BM25 model not trained, must call `.fit` first.")
        embeds = self.model.encode_doocuments(docs)
        return embeds.embeddings

    def fit(self, docs: list[str]):
        params = self.model.fit(docs)
        self.model.set_params(**params)