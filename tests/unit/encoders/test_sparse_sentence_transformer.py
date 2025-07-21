import pytest

from semantic_router.encoders import LocalSparseEncoder
from semantic_router.schema import SparseEmbedding

_ = pytest.importorskip("sentence_transformers")


class TestLocalSparseEncoder:
    def test_sparse_local_encoder(self):
        # Use a public SPLADE model for testing
        encoder = LocalSparseEncoder(name="naver/splade-cocondenser-ensembledistil")
        test_docs = ["This is a test", "This is another test"]
        embeddings = encoder(test_docs)
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(test_docs)
        assert all(isinstance(embedding, SparseEmbedding) for embedding in embeddings)
