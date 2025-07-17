import pytest

from semantic_router.encoders import SparseSentenceTransformerEncoder
from semantic_router.schema import SparseEmbedding

_ = pytest.importorskip("sentence_transformers")


class TestSparseSentenceTransformerEncoder:
    def test_sparse_sentence_transformer_encoder(self):
        encoder = SparseSentenceTransformerEncoder()
        test_docs = ["This is a test", "This is another test"]
        embeddings = encoder(test_docs)
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(test_docs)
        assert all(isinstance(embedding, SparseEmbedding) for embedding in embeddings)
