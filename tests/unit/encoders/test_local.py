import pytest

from semantic_router.encoders import LocalEncoder

_ = pytest.importorskip("sentence_transformers")


class TestLocalEncoder:
    def test_local_encoder(self):
        encoder = LocalEncoder()
        test_docs = ["This is a test", "This is another test"]
        embeddings = encoder(test_docs)
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(test_docs)
        assert all(isinstance(embedding, list) for embedding in embeddings)
