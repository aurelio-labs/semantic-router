from typing import Any, Dict, List, Optional

from semantic_router.encoders.tfidf import TfidfEncoder
from semantic_router.utils.logger import logger
from semantic_router.schema import SparseEmbedding


class BM25Encoder(TfidfEncoder):
    model: Optional[Any] = None
    idx_mapping: Optional[Dict[int, int]] = None
    type: str = "sparse"

    def __init__(
        self,
        name: str | None = None,
        use_default_params: bool = True,
    ):
        if name is None:
            name = "bm25"
        super().__init__(name=name)
        try:
            from pinecone_text.sparse import BM25Encoder as encoder
        except ImportError:
            raise ImportError(
                "Please install pinecone-text to use BM25Encoder. "
                "You can install it with: `pip install 'semantic-router[hybrid]'`"
            )

        self.model = encoder()

        if use_default_params:
            logger.info("Downloading and initializing default sBM25 model parameters.")
            self.model = encoder.default()
            self._set_idx_mapping()

    def _set_idx_mapping(self):
        params = self.model.get_params()
        doc_freq = params["doc_freq"]
        if isinstance(doc_freq, dict):
            indices = doc_freq["indices"]
            self.idx_mapping = {int(idx): i for i, idx in enumerate(indices)}
        else:
            raise TypeError("Expected a dictionary for 'doc_freq'")

    def __call__(self, docs: List[str]) -> list[SparseEmbedding]:
        if self.model is None or self.idx_mapping is None:
            raise ValueError("Model or index mapping is not initialized.")
        if len(docs) == 1:
            sparse_dicts = self.model.encode_queries(docs)
        elif len(docs) > 1:
            sparse_dicts = self.model.encode_documents(docs)
        else:
            raise ValueError("No documents to encode.")

        embeds = []
        for i, output in enumerate(sparse_dicts):
            if isinstance(output, dict):
                embeds.append(SparseEmbedding.from_pinecone_dict(output))
        return embeds
