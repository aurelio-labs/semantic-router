import os

import numpy as np
import pytest

from semantic_router.encoders import BM25Encoder
from semantic_router.route import Route

UTTERANCES = [
    "Hello we need this text to be a little longer for our sparse encoders",
    "In this case they need to learn from recurring tokens, ie words.",
    "We give ourselves several examples from our encoders to learn from.",
    "But given this is only an example we don't need too many",
    "Just enough to test that our sparse encoders work as expected",
]


@pytest.fixture
@pytest.mark.skipif(
    os.environ.get("RUN_HF_TESTS") is None,
    reason="Set RUN_HF_TESTS=1 to run. This test downloads models from Hugging Face which can time out in CI.",
)
def bm25_encoder():
    sparse_encoder = BM25Encoder(use_default_params=True)
    sparse_encoder.fit(
        [
            Route(
                name="test_route",
                utterances=[
                    "The quick brown fox",
                    "jumps over the lazy dog",
                    "Hello, world!",
                ],
            )
        ]
    )
    return sparse_encoder


@pytest.fixture
def routes():
    return [
        Route(name="Route 1", utterances=[UTTERANCES[0], UTTERANCES[1]]),
        Route(name="Route 2", utterances=[UTTERANCES[2], UTTERANCES[3], UTTERANCES[4]]),
    ]


class TestBM25Encoder:
    def test_initialization(self, bm25_encoder):
        assert bm25_encoder._tokenizer is not None

    def test_fit(self, bm25_encoder, routes):
        bm25_encoder.fit(routes)
        assert bm25_encoder._tokenizer is not None

    def test_fit_with_strings(self, bm25_encoder):
        route_strings = ["test a", "test b", "test c"]
        with pytest.raises(TypeError):
            bm25_encoder.fit(route_strings)

    def test_call_method(self, bm25_encoder):
        result = bm25_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sparse_emb.embedding, np.ndarray) for sparse_emb in result
        ), "Each item in result should be an array"

    def test_call_method_no_docs_bm25_encoder(self, bm25_encoder):
        with pytest.raises(ValueError):
            bm25_encoder([])

    def test_call_method_no_word(self, bm25_encoder):
        result = bm25_encoder(["doc with fake word gta5jabcxyz"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sparse_emb.embedding, np.ndarray) for sparse_emb in result
        ), "Each item in result should be an array"

    def test_call_method_with_uninitialized_model_or_mapping(self, bm25_encoder):
        bm25_encoder._tokenizer = None
        with pytest.raises(ValueError):
            bm25_encoder(["test"])

    def test_fit_with_uninitialized_model(self, bm25_encoder, routes):
        bm25_encoder._tokenizer = None
        with pytest.raises(ValueError):
            bm25_encoder.fit(routes)

    def test_encode_queries(self, bm25_encoder):
        queries = ["quick brown", "lazy dog", "hello world"]
        results = bm25_encoder.encode_queries(queries)

        assert len(results) == len(queries)
        assert all([isinstance(result.embedding, np.ndarray) for result in results])

    def test_encode_queries_empty_list(self, bm25_encoder):
        with pytest.raises(ValueError, match="No documents provided for encoding"):
            bm25_encoder.encode_queries([])

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None,
        reason="Set RUN_HF_TESTS=1 to run. This test downloads models from Hugging Face which can time out in CI.",
    )
    def test_encode_queries_unfitted(self):
        encoder = BM25Encoder(use_default_params=True)
        with pytest.raises(ValueError, match="Encoder not fitted"):
            encoder.encode_queries(["test query"])

    def test_encode_documents(self, bm25_encoder):
        documents = ["quick brown", "lazy dog", "hello world"]
        results = bm25_encoder.encode_documents(documents)

        assert len(results) == len(documents)
        assert all([isinstance(result.embedding, np.ndarray) for result in results])

    def test_encode_documents_empty_list(self, bm25_encoder):
        with pytest.raises(ValueError, match="No documents provided for encoding"):
            bm25_encoder.encode_documents([])

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None,
        reason="Set RUN_HF_TESTS=1 to run. This test downloads models from Hugging Face which can time out in CI.",
    )
    def test_encode_documents_unfitted(self):
        encoder = BM25Encoder(use_default_params=True)
        with pytest.raises(ValueError, match="Encoder not fitted"):
            encoder.encode_documents(["test document"])

    def test_encode_documents_batch_size(self, bm25_encoder):
        documents = ["quick brown", "lazy dog", "hello world", "test document"]
        batch_size = 2
        results = bm25_encoder.encode_documents(documents, batch_size=batch_size)

        assert len(results) == len(documents)
        assert all(isinstance(result.embedding, np.ndarray) for result in results)
