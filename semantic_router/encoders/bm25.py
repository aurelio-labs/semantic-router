from pinecone_text.sparse import BM25Encoder as encoder

from semantic_router.encoders import BaseEncoder


class BM25Encoder(BaseEncoder):
    model: encoder | None = None
    idx_mapping: dict[int, int] | None = None

    def __init__(self, name: str = "bm25"):
        super().__init__(name=name)
        # initialize BM25 encoder with default params (trained on MSMarco)
        self.model = encoder.default()
        self.idx_mapping = {
            idx: i
            for i, idx in enumerate(self.model.get_params()["doc_freq"]["indices"])
        }

    def __call__(self, docs: list[str]) -> list[list[float]]:
        if len(docs) == 1:
            sparse_dicts = self.model.encode_queries(docs)
        elif len(docs) > 1:
            sparse_dicts = self.model.encode_documents(docs)
        else:
            raise ValueError("No documents to encode.")
        # convert sparse dict to sparse vector
        embeds = [[0.0] * len(self.idx_mapping)] * len(docs)
        for i, output in enumerate(sparse_dicts):
            indices = output["indices"]
            values = output["values"]
            for idx, val in zip(indices, values):
                if idx in self.idx_mapping:
                    position = self.idx_mapping[idx]
                    embeds[i][position] = val
                else:
                    print(idx, "not in encoder.idx_mapping")
        return embeds

    def fit(self, docs: list[str]):
        self.model.fit(docs)
