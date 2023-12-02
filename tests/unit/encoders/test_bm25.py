import pytest

from semantic_router.encoders import BM25Encoder


@pytest.fixture
def bm25_encoder():
    return BM25Encoder()


class TestBM25Encoder:
    def test_initialization(self, bm25_encoder):
        assert len(bm25_encoder.idx_mapping) != 0

    def test_fit(self, bm25_encoder):
        bm25_encoder.fit([
            "some docs",
            "and more docs",
            "and even more docs"
        ])
        assert len(bm25_encoder.idx_mapping) != 0

    def test_call_method(self, bm25_encoder):
        result = bm25_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sublist, list) for sublist in result
        ), "Each item in result should be a list"
