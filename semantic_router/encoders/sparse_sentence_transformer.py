from typing import Any, List, Optional

from pydantic import PrivateAttr

from semantic_router.encoders.base import SparseEncoder
from semantic_router.schema import SparseEmbedding


class SparseSentenceTransformerEncoder(SparseEncoder):
    """Local sparse encoder using sentence-transformers' SparseEncoder (e.g., SPLADE, CSR) for efficient local sparse embeddings."""

    name: str = "naver/splade-v3"
    type: str = "sparse_sentence_transformer"
    device: Optional[str] = None
    batch_size: int = 32
    _model: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from sentence_transformers import SparseEncoder as STSparseEncoder
        except ImportError:
            raise ImportError(
                "Please install sentence-transformers >=v5 to use SparseSentenceTransformerEncoder. "
                "You can install it with: `pip install sentence-transformers`"
            )
        self._model = STSparseEncoder(self.name)
        if self.device:
            self._model.to(self.device)
        else:
            # Auto-detect device
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            self._model.to(self.device)

    def __call__(self, docs: List[str]) -> List[SparseEmbedding]:
        # The model.encode returns a numpy array (batch, vocab_size) sparse matrix
        sparse_embeddings = self._model.encode(docs, batch_size=self.batch_size)
        # Convert to List[SparseEmbedding] using the helper from base.py
        return self._array_to_sparse_embeddings(sparse_embeddings)
