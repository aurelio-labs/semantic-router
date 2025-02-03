import numpy as np
import pytest

from semantic_router import Route
from semantic_router.encoders.bm25 import BM25Encoder

UTTERANCES = [
    "A high weight in tf–idf is reached by a high term frequency",
    "(in the given document) and a low document frequency of the term",
    "in the whole collection of documents; the weights hence tend to filter",
    "out common terms. Since the ratio inside the idf's log function is always",
    "greater than or equal to 1, the value of idf (and tf–idf) is greater than or equal",
    "to 0. As a term appears in more documents, the ratio inside the logarithm approaches",
    "1, bringing the idf and tf–idf closer to 0.",
]
QUERIES = ["weights", "ratio logarithm"]


@pytest.fixture
def bm25_encoder():
    sparse_encoder = BM25Encoder(use_default_params=True)
    sparse_encoder.fit([Route(name="test_route", utterances=UTTERANCES)])
    return sparse_encoder


class TestBM25Encoder:
    def _sparse_to_vector(self, sparse_embedding, vocab_size):
        """Re-constructs the full (sparse_embedding.shape[0], vocab_size) array"""
        return (
            np.eye(vocab_size)[sparse_embedding[:, 0].astype(np.uint).tolist()]
            * np.atleast_2d(sparse_embedding[:, 1]).T
        ).sum(axis=0)

    def test_bm25_scoring(self, bm25_encoder):
        vocab_size = bm25_encoder._tokenizer.vocab_size
        expected = np.array(
            [
                [0.0000, 0.0000, 1.8475, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 2.3539, 0.0000, 6.6934, 0.0000],
            ]
        )
        q_e = np.stack(
            [
                self._sparse_to_vector(v.embedding, vocab_size=vocab_size)
                for v in bm25_encoder.encode_queries(QUERIES)
            ]
        )
        d_e = np.stack(
            [
                self._sparse_to_vector(v.embedding, vocab_size=vocab_size)
                for v in bm25_encoder.encode_documents(UTTERANCES)
            ]
        )
        scores = q_e @ d_e.T
        assert np.allclose(scores, expected, rtol=1e-4), expected
