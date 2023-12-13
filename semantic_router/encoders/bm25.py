from typing import Any

from pinecone_text.sparse import BM25Encoder as encoder

from semantic_router.encoders import BaseEncoder


class BM25Encoder(BaseEncoder):
    model: Any | None = None
    idx_mapping: dict[int, int] | None = None

    def __init__(self, name: str = "bm25"):
        super().__init__(name=name)
        self.model = encoder.default()

        params = self.model.get_params()
        doc_freq = params["doc_freq"]
        if isinstance(doc_freq, dict):
            indices = doc_freq["indices"]
            self.idx_mapping = {int(idx): i for i, idx in enumerate(indices)}
        else:
            raise TypeError("Expected a dictionary for 'doc_freq'")

    def __call__(self, docs: list[str]) -> list[list[float]]:
        if self.model is None or self.idx_mapping is None:
            raise ValueError("Model or index mapping is not initialized.")
        if len(docs) == 1:
            sparse_dicts = self.model.encode_queries(docs)
        elif len(docs) > 1:
            sparse_dicts = self.model.encode_documents(docs)
        else:
            raise ValueError("No documents to encode.")

        embeds = [[0.0] * len(self.idx_mapping)] * len(docs)
        for i, output in enumerate(sparse_dicts):
            indices = output["indices"]
            values = output["values"]
            for idx, val in zip(indices, values):
                if idx in self.idx_mapping:
                    position = self.idx_mapping[idx]
                    embeds[i][position] = val
        return embeds

    def fit(self, docs: list[str]):
        if self.model is None:
            raise ValueError("Model is not initialized.")
        self.model.fit(docs)
